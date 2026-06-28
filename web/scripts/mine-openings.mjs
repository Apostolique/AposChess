// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2019-2026 Jean-David Moisan
//
// Mine an opening explorer catalog from the self-play dataset.
//
// For every position (up to a ply cutoff) we want the moves that were played
// from it, how each scored, and the engine's eval — a lichess-style explorer for
// the variant. The dataset stores POSITIONS, not moves, so we recover the move
// between two consecutive plies of one game by trying each legal move and matching
// the resulting position key (the same trick the puzzle miner uses), then
// aggregate, per parent position, the child moves with their game counts, White
// win/draw/loss tally, and mean eval.
//
// Two dataset realities shape this (see the gen pipeline + dedup-cap):
//   - Games open with RANDOM plies (gen --openings, default 8), so move POPULARITY
//     in the opening reflects the generator, not engine taste. We lean into it: the
//     random openings give broad coverage, and the per-move eval + win-rate are the
//     real signal (the engine's value of each resulting position). From ply ~9 on
//     the distribution becomes a genuine engine book.
//   - The dataset is dedup-capped, so a common early position keeps only a few
//     copies and an intermediate ply can be removed entirely. A removed ply leaves a
//     GAP between two surviving records, which no single legal move connects — so we
//     only reconstruct a move when the two records are CONSECUTIVE plies (child ply
//     == parent ply + 1). Ply depth comes straight from the FEN, robust to any
//     missing leading plies.
//
// Output is web/public/openings.json — a static catalog the app's Analysis-mode
// opening book fetches at runtime (same no-rebuild pattern as puzzles.json):
//   { maxPly, minGames, positions: { "<posKey>": [ {u,n,w,d,l,ev}, ... ] } }
// keyed by posKey (board + side + castling) so the app looks up a position directly.
//
// Usage (run from web/):
//   node scripts/mine-openings.mjs [--in=FILE] [--out=FILE]
//     [--max-ply=16]    only positions at this ply depth or shallower (0 = start).
//                       Ply = (fullmove-1)*2 + (black to move ? 1 : 0).
//     [--min-games=5]   drop a move played in fewer than this many games (and a
//                       position left with no surviving move).
//     [--top=12]        keep at most this many moves per position (most-played first).
// Defaults: in = ../training/data/selfplay.jsonl, out = public/openings.json.

import { createReadStream, existsSync, writeFileSync, statSync } from 'node:fs';
import { createInterface } from 'node:readline';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

import { toFen } from '../src/board.js';
import { isGameRecord, expandPositions } from './gameRecord.mjs';
import { fmtDur, fmtNum, fmtMB, liveStatus, everyMs } from './fmt.mjs';
import { installStop, printStopHint } from './stop.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const args = Object.fromEntries(process.argv.slice(2).map((a) => {
  const m = a.replace(/^--/, '').split('='); return [m[0], m.length > 1 ? m[1] : true];
}));
const num = (k, d) => (args[k] !== undefined ? Number(args[k]) : d);

const maxPly = num('max-ply', 16);
const minGames = num('min-games', 5);
const top = num('top', 12);
const inFile = typeof args.in === 'string'
  ? resolve(process.cwd(), args.in) : resolve(here, '../../training/data/selfplay.jsonl');
const outFile = typeof args.out === 'string'
  ? resolve(process.cwd(), args.out) : resolve(here, '../public/openings.json');

if (!existsSync(inFile)) { console.error(`No dataset at ${inFile}`); process.exit(1); }

// Position key: board + side to move + castling, ignoring en passant and the move
// counters — the same key the app uses to look up the viewed position, and what
// makes the same position reached via different games collapse to one node.
const posKey = (fen) => fen.split(' ').slice(0, 3).join(' ');

const t0 = Date.now();
const status = liveStatus();
const tick = everyMs(1000);
console.log(`mine-openings: ${inFile} (${fmtMB(statSync(inFile).size)}) | max-ply ${maxPly} | min-games ${minGames} | top ${top}`);
printStopHint();

let stopping = false;
const stopper = installStop(() => { stopping = true; });

// parent posKey -> Map(move -> { n, w, d, l, evSum }). A game record stores its moves
// directly, so each move connects position i to i+1 with no reconstruction needed.
const nodes = new Map();
let games = 0, pairs = 0;

{
  const rl = createInterface({ input: createReadStream(inFile), crlfDelay: Infinity });
  for await (const line of rl) {
    if (stopping) break;
    if (!line) continue;
    let rec;
    try { rec = JSON.parse(line); } catch { continue; }
    if (!isGameRecord(rec)) continue;
    games++;
    // Replay into positions; moves[i] is the move from position i to i+1.
    const ps = [];
    for (const p of expandPositions(rec)) ps.push(p.state);
    for (let i = 0; i < rec.moves.length && i <= maxPly; i++) {
      const child = ps[i + 1];
      if (!child) break;
      pairs++;
      const parentKey = posKey(toFen(ps[i]));
      const u = rec.moves[i];
      let node = nodes.get(parentKey);
      if (!node) nodes.set(parentKey, node = new Map());
      let e = node.get(u);
      if (!e) node.set(u, e = { n: 0, w: 0, d: 0, l: 0, evSum: 0 });
      e.n++;
      // rec.r is the game's White-view result; report every column from White's POV.
      if (rec.r > 0) e.w++; else if (rec.r < 0) e.l++; else e.d++;
      // child's v is side-to-move-relative; fold to White's POV.
      const cv = rec.v ? rec.v[i + 1] : undefined;
      if (typeof cv === 'number') e.evSum += child.turn === 'white' ? cv : -cv;
    }
    if (tick()) status.update(`  scanning… ${fmtNum(games)} games, ${fmtNum(nodes.size)} positions`);
  }
}
stopper.dispose();
status.clear();
console.log(`  scan: ${fmtNum(games)} games -> ${fmtNum(pairs)} move(s) in window in ${fmtDur((Date.now() - t0) / 1000)}`);

// Prune + shape the catalog: drop rare moves, keep the most-played `top`, sort by
// games desc (eval as a tiebreak so equal-popularity moves read best-first).
const positions = {};
let kept = 0, moves = 0;
for (const [key, node] of nodes) {
  const arr = [...node.entries()]
    .filter(([, e]) => e.n >= minGames)
    .map(([u, e]) => ({ u, n: e.n, w: e.w, d: e.d, l: e.l, ev: Math.round(e.evSum / e.n) }));
  if (arr.length === 0) continue;
  arr.sort((a, b) => b.n - a.n || b.ev - a.ev);
  if (arr.length > top) arr.length = top;
  positions[key] = arr;
  kept++; moves += arr.length;
}

const json = '{\n'
  + `  "maxPly": ${maxPly},\n  "minGames": ${minGames},\n`
  + '  "positions": {\n'
  + Object.entries(positions).map(([k, v]) => `    ${JSON.stringify(k)}: ${JSON.stringify(v)}`).join(',\n')
  + '\n  }\n}\n';
writeFileSync(outFile, json);

console.log(`Done: ${fmtNum(kept)} positions, ${fmtNum(moves)} moves, ${fmtMB(statSync(outFile).size)} -> ${outFile}`);
const root = positions[posKey('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')];
if (root) console.log(`  start position: ${root.length} moves, top: ${root.slice(0, 5).map((m) => `${m.u}(${m.n})`).join(' ')}`);
else console.log('  start position: not present (dedup-capped away or below min-games)');
