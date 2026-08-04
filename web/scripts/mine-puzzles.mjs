// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2019-2026 Jean-David Moisan
//
// Mine tactics puzzles from the self-play dataset.
//
// A good puzzle is a position with exactly ONE clearly-winning move that isn't
// obvious. The dataset already tells us where to look: every record carries the
// search's value `v` (side-to-move view), so a blunder shows up as consecutive
// plies of one game where the new side to move is suddenly winning (their view
// jumped from -prev.v to v). The position AFTER the blunder is the puzzle — the
// solver punishes it. The scan only nominates candidates; workers
// (puzzleWorker.mjs) then verify each one at full depth: unique winning move,
// solution line walked while it stays forced, variant themes tagged (jumps,
// jump-blocks, knight path-blocks…), and difficulty = the shallowest depth that
// finds the move. See the worker header for the details.
//
// Output is web/public/puzzles.json — a static catalog the app fetches at
// runtime (same pattern as the nn/ net catalog), so adding puzzles needs no
// rebuild. With --append, freshly-mined puzzles merge into the existing catalog
// (deduped by position) instead of replacing it.
//
// Usage (run from web/):
//   node scripts/mine-puzzles.mjs [--depth=6] [--jobs=N] [--eval=nn|handcrafted]
//     [--weights=FILE] [--in=FILE] [--out=FILE] [--seed=1] [--append]
//     [--swing=400]    blunder size (prev.v + v) that nominates a candidate
//     [--pre-floor=-200]  the blunderer's eval BEFORE the blunder must be at least
//                      this (cp, their view). A side that was already lost isn't
//                      blundering, it's delaying — "finish the long-decided mate"
//                      isn't a puzzle. This also guarantees the solver wasn't
//                      already winning, so the key move CREATES the win.
//     [--min-move=10]  puzzle position must be at this fullmove or later — the
//                      generator opens every game with random plies (gen default 8,
//                      i.e. through move 4), and "punish the random opening move"
//                      makes for boring puzzles nobody would blunder into for real
//     [--min-engine-depth=2]  both sides of the game must have searched at least this
//                      deep. A depth-1 search can't even see the recapture, so the
//                      ranking pool's depth-1 ladder games hang a piece every few moves
//                      and flood the candidate list with sound-but-dull "the queen is
//                      en prise" positions — 46% of the raw pool, and 3.6x the
//                      one-move-win rejects. 0 keeps every game.
//     [--per-game=3]   cap on candidates taken from one game (seeded pick), so a
//                      single collapse doesn't crowd out other games. 0 = no cap.
//     [--win=500]      best move must score at least this (cp) — or mate
//     [--second=150]   runner-up move must score at most this (uniqueness)
//     [--line-gap=250] continuation moves stay in the solution while they beat the
//                      runner-up by this margin — clearly best, not necessarily the
//                      only win, so the line plays THROUGH the payoff (capturing
//                      the cornered queen) instead of stopping just before it
//     [--save-floor=-150]  defense puzzles: the only-move save must keep the solver
//                      at least this (cp). Each candidate is mined from both sides
//                      of the blunder — a WIN puzzle from the position after it
//                      ('razor-edge'-tagged when every alternative outright loses)
//                      and a DEFENSE (only-move) puzzle from the position before it,
//                      where the game's own engine fell off the tightrope
//     [--verify-depth=D+2]  re-run the root gates this deep before accepting. The app
//                      plays the position out with a deeper engine than the mining
//                      depth and only accepts the scripted move, so a "unique win"
//                      a stronger search disagrees with is a puzzle that marks the
//                      solver wrong for solving it. 0 disables.
//     [--max-candidates=20000] cap on candidates verified (seeded sample). 0 = all.
//     [--limit=N]              stop after this many accepted puzzles (default: none —
//                      accept every candidate that verifies; --max-candidates still
//                      bounds how many are checked)
//     [--min-difficulty=1]     drop puzzles solved at a shallower depth
//     [--min-win-moves=2]      minimum solver moves for a win/mate puzzle — a
//                      one-move "win" is just grabbing a hanging piece, not a tactic
//                      (also drops mate-in-1). Defense (only-move) puzzles are exempt.
//                      Set to 1 to keep one-move wins.
//     [--max-solver-moves=4]   solution length cap (forced-mate finishes exempt:
//                      a line that walks into a mate runs to checkmate)
// Defaults: jobs = CPU cores, eval = nn with weights = ./src/nn-weights.json
// (handcrafted if the weights don't load), in = ../training/data/selfplay.jsonl,
// out = public/puzzles.json.

import { Worker } from 'node:worker_threads';
import { createReadStream, existsSync, readFileSync, writeFileSync, statSync } from 'node:fs';
import { createInterface } from 'node:readline';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { cpus } from 'node:os';

import { toFen } from '../src/board.js';
import { ensureWasm } from './wasmEngine.mjs';
import { fmtDur, fmtNum, fmtMB, liveStatus, everyMs } from './fmt.mjs';
import { isGameRecord, expandPositions } from './gameRecord.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const args = Object.fromEntries(process.argv.slice(2).map((a) => {
  const m = a.replace(/^--/, '').split('='); return [m[0], m.length > 1 ? m[1] : true];
}));
const num = (k, d) => (args[k] !== undefined ? Number(args[k]) : d);

const depth = num('depth', 6);
const jobs = Math.max(1, num('jobs', cpus().length));
const seed = num('seed', 1);
const swing = num('swing', 400);
const minMove = num('min-move', 10);
const preFloor = num('pre-floor', -200);
const minEngineDepth = num('min-engine-depth', 2);
const perGame = num('per-game', 3);
const lineGap = num('line-gap', 250);
const saveFloor = num('save-floor', -150);
const win = num('win', 500);
const second = num('second', 150);
const verifyDepth = num('verify-depth', depth + 2);
const maxCandidates = num('max-candidates', 20000) || Infinity;
const limit = num('limit', Infinity); // no accept cap by default; --max-candidates bounds the work
const minDifficulty = num('min-difficulty', 1);
const minWinMoves = num('min-win-moves', 2);
const maxSolverMoves = num('max-solver-moves', 4);
const evalName = typeof args.eval === 'string' ? args.eval : 'nn';
const weights = typeof args.weights === 'string'
  ? resolve(process.cwd(), args.weights) : resolve(here, '../src/nn-weights.json');
const inFile = typeof args.in === 'string'
  ? resolve(process.cwd(), args.in) : resolve(here, '../../training/data/selfplay.jsonl');
const outFile = typeof args.out === 'string'
  ? resolve(process.cwd(), args.out) : resolve(here, '../public/puzzles.json');

if (!existsSync(inFile)) { console.error(`No dataset at ${inFile}`); process.exit(1); }

function mulberry32(a) {
  a >>>= 0;
  return () => { a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
const rng = mulberry32(seed);

const t0 = Date.now();
const status = liveStatus();
const tick = everyMs(1000);
console.log(`mine-puzzles: ${inFile} (${fmtMB(statSync(inFile).size)}) | depth ${depth}${verifyDepth > depth ? ` (verify ${verifyDepth})` : ''} | jobs ${jobs} | seed ${seed}`);
console.log(`  gates: swing>=${swing} | pre>=${preFloor} | move>=${minMove} | engine-depth>=${minEngineDepth} | per-game<=${perGame || '∞'} | win>=${win} | second<=${second} | line-gap>=${lineGap} | difficulty>=${minDifficulty} | win-moves>=${minWinMoves} | limit ${Number.isFinite(limit) ? limit : 'none'} of <=${Number.isFinite(maxCandidates) ? fmtNum(maxCandidates) : 'all'} candidates`);

// --- phase 1: scan the dataset for blunder-punish candidates ----------------------
// Two passes per record, cheapest first. The blunder test reads only the record's `v`
// array — plain arithmetic, no board — so the ~92% of games that nominate nothing are
// dropped without replaying a single move, and a game that does nominate is expanded
// only as far as the last ply that needs a FEN. (Building a FEN for every position of
// every game and throwing almost all of them away cost ~9 min on a 790 MB dataset;
// this way the whole scan is under a minute.)

// The position key ignores the move counters (board + turn + castling) so the same
// position reached in different games is verified only once.
const posKey = (fen) => fen.split(' ').slice(0, 3).join(' ');
// A shallow-search `v` won't match the deep verify exactly; nominate with slack and
// let the worker apply the real gate.
const vMin = Math.max(300, win - 150);

// Ply -> fullmove without replaying: the standard start is fullmove 1 with White to
// move, and a `start` FEN carries its own counter and side.
function fullmoveAt(rec, ply) {
  if (!rec.start) return 1 + (ply >> 1);
  const f = rec.start.split(' ');
  return Number(f[5]) + ((ply + (f[1] === 'b' ? 1 : 0)) >> 1);
}
// Search depth from a vtag: "nn6@c76205" -> 6, "hc2@?" -> 2, unreadable -> 0.
const tagDepth = (t) => (typeof t === 'string' ? Number((/^[a-z]+(\d+)/.exec(t) || [])[1]) || 0 : 0);

const candidates = [];
// Positions already accounted for. Under --append the existing catalog seeds it, so a
// re-mine spends its budget on new positions instead of re-verifying shipped ones.
const seen = new Set();
let existing = [];
if (args.append && existsSync(outFile)) {
  try {
    existing = JSON.parse(readFileSync(outFile, 'utf8')).puzzles || [];
    for (const p of existing) seen.add(posKey(p.fen));
    console.log(`  append: ${fmtNum(existing.length)} puzzles already in the catalog`);
  } catch { console.log('  append: existing catalog unreadable — writing fresh'); }
}
let gamesScanned = 0, gamesWeak = 0, gamesHit = 0, nominated = 0;
{
  const rl = createInterface({ input: createReadStream(inFile), crlfDelay: Infinity });
  for await (const line of rl) {
    if (!line) continue;
    let rec;
    try { rec = JSON.parse(line); } catch { continue; }
    if (!isGameRecord(rec) || !rec.v) continue;
    gamesScanned++;
    // Both engines must have looked deep enough for a mistake to be worth punishing.
    // A depth-1 search doesn't see the recapture, so those games hang something every
    // few moves and the candidate list fills up with sound-but-dull "the queen is en
    // prise" positions — the ranking pool plays a lot of them.
    if (minEngineDepth && (!rec.players
      || Math.min(tagDepth(rec.players.w), tagDepth(rec.players.b)) < minEngineDepth)) { gamesWeak++; continue; }
    // A blunder is a decisive v swing between two consecutive positions: prev's side
    // was fine, then the new side to move sees a big edge (their view went from
    // prev.v to -cur.v).
    const v = rec.v;
    let hits = null;
    for (let i = 1; i < v.length; i++) {
      const prev = v[i - 1], cur = v[i];
      if (typeof prev !== 'number' || typeof cur !== 'number') continue;
      // The blunderer must not have been lost already: a dead-lost side "blundering"
      // is just delaying the end, and the solver was then already winning anyway.
      if (cur < vMin || prev + cur < swing || prev < preFloor) continue;
      // Skip the opening: the generator's first plies are RANDOM moves, and punishing
      // one of those isn't a tactic anyone needs to learn.
      if (fullmoveAt(rec, i) < minMove) continue;
      (hits || (hits = [])).push(i);
    }
    if (!hits) continue;
    gamesHit++; nominated += hits.length;
    // One collapse can nominate dozens of near-identical plies. Keep a seeded few so
    // the catalog samples many games instead of a handful of blowouts.
    if (perGame && hits.length > perGame) {
      for (let i = hits.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [hits[i], hits[j]] = [hits[j], hits[i]]; }
      hits.length = perGame;
      hits.sort((a, b) => a - b);
    }
    // Now replay, and only up to the last ply anyone needs. Each candidate wants the
    // blunder position and the two before it (lead-in, and the defense mine's own lead-in).
    const need = new Set();
    for (const i of hits) { need.add(i); need.add(i - 1); if (i >= 2) need.add(i - 2); }
    const last = hits[hits.length - 1];
    const fens = new Map();
    try {
      for (const p of expandPositions(rec)) {
        if (need.has(p.ply)) fens.set(p.ply, toFen(p.state));
        if (p.ply >= last) break;
      }
    } catch { continue; } // a record whose moves don't replay is corrupt — skip the game
    for (const i of hits) {
      const fen = fens.get(i);
      if (!fen) continue;
      const key = posKey(fen);
      if (seen.has(key)) continue;
      seen.add(key);
      // prevFen is the position BEFORE the blunder: the worker derives the blunder move
      // from it for the win puzzle's lead-in, and mines it as a defense (only-move) puzzle
      // of its own — prevPrevFen then serves as THAT puzzle's lead-in, the same way.
      candidates.push({ id: `${rec.g}#${i}`, fen, prevFen: fens.get(i - 1), prevPrevFen: fens.get(i - 2) });
    }
    if (tick()) status.update(`  scanning… ${fmtNum(gamesScanned)} games, ${fmtNum(candidates.length)} candidates`);
  }
}
status.clear();
console.log(`  scan: ${fmtNum(gamesScanned)} games (${fmtNum(gamesWeak)} below engine-depth, ${fmtNum(gamesHit)} nominating) `
  + `-> ${fmtNum(nominated)} plies -> ${fmtNum(candidates.length)} candidate positions in ${fmtDur((Date.now() - t0) / 1000)}`);

// Seeded sample down to the verification budget (shuffle then slice, so reruns with
// the same seed verify the same set).
for (let i = candidates.length - 1; i > 0; i--) {
  const j = Math.floor(rng() * (i + 1));
  [candidates[i], candidates[j]] = [candidates[j], candidates[i]];
}
if (candidates.length > maxCandidates) candidates.length = maxCandidates;
candidates.forEach((c, i) => { c.idx = i; c.seed = (seed + 0x9e3779b9 * (i + 1)) >>> 0; });

// --- phase 2: verify candidates in the worker pool --------------------------------
const B = 4; // candidates per batch: each one can cost many deep searches
const queue = [];
for (let i = 0; i < candidates.length; i += B) queue.push(candidates.slice(i, i + B));

const puzzles = [];
const rejects = {};
let verified = 0, accepted = 0, stopped = false;

const pool = [], idle = [];
let finished = false;

await new Promise((done) => {
  function pump() { while (idle.length && queue.length) idle.pop().postMessage({ type: 'batch', items: queue.shift() }); }
  // Done when nothing is queued and every worker is idle (a worker turning ready
  // can complete that condition, so both message kinds re-check).
  function maybeFinalize() {
    if (finished || queue.length || idle.length !== pool.length) return;
    finished = true;
    for (const w of pool) w.terminate();
    done();
  }
  if (queue.length === 0) { finished = true; done(); return; }
  let reportedEngine = null;
  ensureWasm(); // build the native engine (wasm) once before the workers race for it
  for (let i = 0; i < jobs; i++) {
    const w = new Worker(new URL('./puzzleWorker.mjs', import.meta.url), {
      workerData: { weights, depth, verifyDepth, win, second, lineGap, saveFloor, maxSolverMoves, minWinMoves, eval: evalName },
    });
    pool.push(w);
    w.on('message', (msg) => {
      if (msg.type === 'ready') {
        if (reportedEngine === null) {
          reportedEngine = msg.engine;
          if (evalName === 'nn' && msg.engine !== 'nn') console.log('  note: nn weights did not load — using the handcrafted eval');
          else console.log(`  eval: ${msg.engine}`);
        }
        idle.push(w); pump(); maybeFinalize();
        return;
      }
      if (msg.type !== 'done') return;
      for (const r of msg.results) {
        verified++;
        // Each candidate yields two outcomes: the win mine (post-blunder position)
        // and the defense mine (pre-blunder position). Tally them separately so the
        // reject histogram stays readable ('d-' prefixes the defense side).
        for (const [tag, res] of [['', r.win], ['d-', r.defense]]) {
          if (!res) continue;
          if (res.puzzle) {
            if (res.puzzle.difficulty >= minDifficulty) { puzzles.push(res.puzzle); accepted++; }
            else rejects[tag + 'too-easy'] = (rejects[tag + 'too-easy'] || 0) + 1;
          } else {
            rejects[tag + res.reject] = (rejects[tag + res.reject] || 0) + 1;
            if (res.error) console.error(`\n  worker error on a candidate: ${res.error}`);
          }
        }
      }
      if (accepted >= limit && !stopped) { stopped = true; queue.length = 0; }
      if (tick()) {
        const el = (Date.now() - t0) / 1000;
        status.update(`  ${fmtNum(verified)}/${fmtNum(candidates.length)} verified | ${fmtNum(accepted)} accepted | `
          + `${(verified / Math.max(el, 0.001)).toFixed(1)}/s | ${fmtDur(el)}`);
      }
      idle.push(w); pump(); maybeFinalize();
    });
    w.on('error', (e) => { console.error('\npuzzle worker error:', e); });
  }
});

finish();

function finish() {
  status.clear();

  // One position, one puzzle. Consecutive blunders nominate overlapping plies, so the
  // defense mine of ply i and the win mine of ply i-1 can land on the same board — same
  // solution, different id. Keep the first and drop the rest (--append seeded `seen`
  // with the shipped catalog, so this also covers re-mines).
  const have = new Set();
  const keep = (p) => { const k = posKey(p.fen); return have.has(k) ? false : (have.add(k), true); };
  const kept = existing.filter(keep);
  const fresh = puzzles.filter(keep);
  const all = kept.concat(fresh);
  const dropped = existing.length + puzzles.length - all.length;
  if (dropped) console.log(`  deduped: ${fmtNum(dropped)} repeated position(s)`);
  if (existing.length) console.log(`  append: ${fmtNum(kept.length)} existing + ${fmtNum(fresh.length)} new`);
  all.sort((a, b) => a.difficulty - b.difficulty || a.id.localeCompare(b.id));

  const json = '{\n  "puzzles": [\n'
    + all.map((p) => '    ' + JSON.stringify(p)).join(',\n')
    + '\n  ]\n}\n';
  writeFileSync(outFile, json);

  // Report what the catalog looks like, so gate tuning has something to react to.
  const hist = (key) => {
    const h = {};
    for (const p of all) for (const v of [].concat(p[key])) h[v] = (h[v] || 0) + 1;
    return Object.entries(h).sort().map(([k, n]) => `${k}:${n}`).join(' ');
  };
  console.log(`Done: ${fmtNum(verified)} verified -> ${fmtNum(accepted)} accepted `
    + `(${fmtNum(all.length)} total) in ${fmtDur((Date.now() - t0) / 1000)} -> ${outFile}`);
  const rej = Object.entries(rejects).sort((a, b) => b[1] - a[1]).map(([k, n]) => `${k}:${fmtNum(n)}`).join(' ');
  if (rej) console.log(`  rejected: ${rej}`);
  if (all.length) {
    console.log(`  kinds:      ${hist('kind')}`);
    console.log(`  difficulty: ${hist('difficulty')}`);
    console.log(`  themes:     ${hist('themes')}`);
  }
}
