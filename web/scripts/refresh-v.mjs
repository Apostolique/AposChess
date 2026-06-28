// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2019-2026 Jean-David Moisan
//
// Dataset maintenance: (re)compute the search value `v` on the GAME-PRIMARY dataset
// (scripts/gameRecord.mjs) with the CURRENT champion. `v` is a TD/bootstrap target
// (train.py --lambda), and it goes stale as the champion improves — it's a weaker net's
// opinion of the position. Refreshing it is value iteration: re-bootstrap targets from the
// improved value function. (The game result `r` never goes stale, so only the `v` part
// drifts.) Each game is replayed to recover the position at every ply; the plies that need a
// (re)computation are searched, and their v/vs are written back into the game's arrays
// (vs promotes from a scalar to a per-position array the moment two positions disagree).
//
// Modes:
//   default          fill `v` only on positions that lack it (e.g. random opening plies).
//   --refresh        also recompute `v` on positions that already have one.
//   --frac=P         with --refresh, recompute only a random fraction P (0..1) of the
//                    positions that have a `v` (positions MISSING `v` are always filled).
//   --minutes=M      wall-clock budget (default 10): after M minutes the run stops
//                    gracefully and finalizes a complete dataset. --minutes=0 removes it.
//
// The expensive searches are fanned out across worker threads (refreshWorker.mjs, which
// scores a single FEN — unchanged). The main thread streams the file, plans each game,
// replays it to get the FENs of the plies to recompute, dispatches them, and is the single
// writer — emitting whole game records in input order via a reorder buffer once all of a
// game's searched plies come back. Games needing no recompute pass through unchanged.
//
// Usage (run from web/):
//   node scripts/refresh-v.mjs [--refresh] [--frac=P] [--depth=D] [--jobs=N] [--eval=E]
//                              [--weights=FILE] [--in=FILE] [--out=FILE] [--seed=S]
//                              [--ledger[=FILE]] [--minutes=M] [--band=E]
// Defaults: depth 6, jobs = CPU cores, eval 'nn' (weights = ./src/nn-weights.json), in/out
//           = ../training/data/selfplay.jsonl (atomic replace), seed 1.
//
// Smart weakest-first refresh (--ledger): read the strength ledger from `npm run rank:pool` and
// let it drive the whole refresh. The recompute engine DEFAULTS to the ledger's STRONGEST
// engine (explicit --eval/--weights still win). Each run, in priority order: (1) FILL every
// position with no `v` (never throttled by --frac); (2) REFRESH the WEAKEST BAND of
// positions that DO have a `v` — every label within --band Elo (default 25) of the lowest-Elo
// engine still present (untagged / unrecoverable count as -Inf), at --frac. Re-running it
// walks the dataset upward, weakest band first, until everything carries the best engine's v.

import { Worker } from 'node:worker_threads';
import { createReadStream, createWriteStream, existsSync, rmSync, renameSync, readFileSync, statSync } from 'node:fs';
import { createInterface } from 'node:readline';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { cpus } from 'node:os';

import { toFen } from '../src/board.js';
import { ensureWasm } from './wasmEngine.mjs';
import { fmtDur, fmtNum, fmtMB, liveStatus, everyMs } from './fmt.mjs';
import { vtag as computeVtag, ephemeralElo } from './vtag.mjs';
import { installStop, printStopHint } from './stop.mjs';
import { isGameRecord, vsAt, setVsAt, normalizeVs, serializeGameRecord, expandPositions } from './gameRecord.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const args = Object.fromEntries(process.argv.slice(2).map((a) => {
  const m = a.replace(/^--/, '').split('='); return [m[0], m.length > 1 ? m[1] : true];
}));
const refresh = !!args.refresh;
const frac = args.frac !== undefined ? Number(args.frac) : 1.0;
const band = args.band !== undefined ? Number(args.band) : 25;
const depth = args.depth !== undefined ? Number(args.depth) : 6;
const jobs = Math.max(1, args.jobs !== undefined ? Number(args.jobs) : cpus().length);
const seed = args.seed !== undefined ? Number(args.seed) : 1;
const minutes = args.minutes !== undefined ? Number(args.minutes) : 10;

// --- smart weakest-first refresh (--ledger) --------------------------------------
const ledgerPath = args.ledger === true
  ? resolve(here, '../../training/data/loop/engine-elo.ladder.json')
  : (typeof args.ledger === 'string' ? resolve(process.cwd(), args.ledger) : null);
let eloByVersion = null, eloByTag = null, best = null;
if (ledgerPath) {
  let ledger;
  try { ledger = JSON.parse(readFileSync(ledgerPath, 'utf8')); }
  catch (e) { console.error(`Could not read ledger ${ledgerPath}: ${e.message}. Run 'npm run rank:pool' first.`); process.exit(1); }
  eloByVersion = new Map(); eloByTag = new Map();
  // The material fallback ('?') is excluded — its Elo is only an internal ledger stat, so
  // anything it (or an unrecoverable/untagged source) labeled is treated as weakest (-Inf).
  for (const e of ledger.ranking || []) {
    if (e.elo == null || e.version === '?') continue;
    if (e.tag) eloByTag.set(e.tag, e.elo);
    const prev = eloByVersion.has(e.version) ? eloByVersion.get(e.version) : -Infinity;
    eloByVersion.set(e.version, Math.max(prev, e.elo));
  }
  best = (ledger.ranking || []).filter((e) => e.elo != null && e.version !== '?')
    .reduce((a, b) => (b.elo > a.elo ? b : a), { elo: -Infinity });
  if (!Number.isFinite(best.elo)) best = null;
}

// Which eval recomputes v: explicit --eval/--weights win; otherwise the ledger's best engine
// (when --ledger), else 'nn' with the shipped champion.
const evalName = typeof args.eval === 'string' ? args.eval
  : (best ? (best.eng === 'nn' ? 'nn' : 'handcrafted') : 'nn');
const weights = typeof args.weights === 'string' ? resolve(process.cwd(), args.weights)
  : (best && best.eng === 'nn' && best.file ? best.file : resolve(here, '../src/nn-weights.json'));
const vtag = computeVtag(evalName, depth, weights);

const recomputeEng = evalName.startsWith('nn') ? 'nn' : 'hc';
const recomputeVersion = vtag.slice(vtag.indexOf('@') + 1);
const parseTag = (tag) => { const m = /^(nn|hc)(\d+|t)@(.+)$/.exec(tag || ''); return m ? { eng: m[1], depth: m[2], version: m[3] } : null; };
const eloForTag = (tag) => {
  const t = parseTag(tag);
  if (!t) return -Infinity;
  const eph = ephemeralElo(t.version);
  if (eph !== null) return eph;
  if (eloByTag && eloByTag.has(tag)) return eloByTag.get(tag);
  const e = eloByVersion && eloByVersion.get(t.version);
  return e == null ? -Infinity : e;
};
const alreadyDone = (tag) => { const t = parseTag(tag); if (!t || t.eng !== recomputeEng || t.version !== recomputeVersion) return false; const d = Number(t.depth); return Number.isFinite(d) && d >= depth; };
const cohortName = (elo) => {
  if (elo === -Infinity) return 'untagged / material / unrecoverable';
  const tags = [...eloByTag.entries()].filter(([, e]) => e === elo).map(([t]) => t);
  if (tags.length) return `${tags.join(', ')} (Elo ${elo.toFixed(0)})`;
  const vs = [...eloByVersion.entries()].filter(([, e]) => e === elo).map(([v]) => v);
  return `${vs.length ? vs.join(', ') : 'ephemeral candidate(s)'} (Elo ${elo.toFixed(0)})`;
};
const inFile = typeof args.in === 'string'
  ? resolve(process.cwd(), args.in) : resolve(here, '../../training/data/selfplay.jsonl');
const outFile = typeof args.out === 'string'
  ? resolve(process.cwd(), args.out) : resolve(here, '../../training/data/selfplay.jsonl');

if (!existsSync(inFile)) { console.error(`No dataset at ${inFile}`); process.exit(1); }

const B = 64;                              // positions per worker batch
const CAP = Math.max(jobs * 64, 4000);     // max outstanding (un-emitted) GAMES
const t0 = Date.now();
const status = liveStatus();
const tick = everyMs(1000);

function mulberry32(a) {
  a >>>= 0;
  return () => { a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
const rng = mulberry32(seed);

// Should position `i` of game `rec` be (re)computed? Mirrors the classic + ledger logic, now
// per ply. `targetElo` (ledger mode) is the weakest cohort to refresh; null => fills only.
let targetElo = null;
function shouldRecompute(rec, i) {
  const hasV = rec.v != null && rec.v[i] != null;
  if (!eloByVersion) return !hasV || (refresh && rng() < frac);   // classic mode
  if (!hasV) return true;                                         // ledger: always fill
  const tag = vsAt(rec, i);
  return targetElo !== null && !alreadyDone(tag) && eloForTag(tag) <= targetElo + band && rng() < frac;
}

// --- graceful early stop ----------------------------------------------------------
let stopping = false;
let rl = null;
let budgetTimer = null;
const stopFlag = new Int32Array(new SharedArrayBuffer(4));
function requestStop() {
  stopping = true;
  Atomics.store(stopFlag, 0, 1); // tell in-flight workers to abandon the rest of their batch
  status.clear();
  if (rl) {
    console.log('\n  Stopping early — copying the remaining games through unchanged and finalizing…');
    enqueueBatch();                 // push any partial batch so workers skip-resolve it
    if (!inputEnded) { try { rl.resume(); } catch { /* already closed */ } }
    flush();
    maybeFinalize();
  } else {
    console.log('\n  Stopping early during the pre-scan — the dataset is left unchanged.');
  }
}
const stopper = installStop(requestStop);

// --- pre-scan (--ledger): plan this run in one cheap streaming pass (no search).
let missingCount = 0;          // positions lacking v entirely (always filled)
let toCompute = null;          // estimated positions this run will (re)compute (for the ETA)
if (eloByVersion) {
  let min = Infinity, scanned = 0, gamesScanned = 0;
  const cohortCounts = new Map();
  const sTick = everyMs(500);
  try {
    const srl = createInterface({ input: createReadStream(inFile), crlfDelay: Infinity });
    for await (const line of srl) {
      if (stopping) { srl.close(); break; }
      if (!line) continue;
      let rec; try { rec = JSON.parse(line); } catch { continue; }
      if (!isGameRecord(rec)) continue;
      gamesScanned++;
      const n = rec.v ? rec.v.length : rec.moves.length + 1;
      for (let i = 0; i < n; i++) {
        scanned++;
        if (rec.v == null || rec.v[i] == null) { missingCount++; continue; } // missing v: always filled
        const tag = vsAt(rec, i);
        if (alreadyDone(tag)) continue;
        const elo = eloForTag(tag);
        if (elo < min) min = elo;
        cohortCounts.set(elo, (cohortCounts.get(elo) || 0) + 1);
      }
      if (sTick()) status.update(`  planning refresh... ${fmtNum(gamesScanned)} games`);
    }
  } catch (e) { console.warn(`\n  pre-scan interrupted (${e.message}); proceeding with what was seen.`); }
  status.clear();
  if (stopping) { console.log('Stopped before writing; dataset left unchanged.'); stopper.dispose(); process.exit(); }
  if (missingCount === 0 && min === Infinity) {
    console.log(`Nothing to refresh: every label is already the best engine ${vtag} at depth >= ${depth}.`);
    process.exit(0);
  }
  targetElo = min === Infinity ? null : min;
  const cohort = targetElo === null ? 0
    : [...cohortCounts].reduce((s, [e, n]) => s + (e <= targetElo + band ? n : 0), 0);
  toCompute = missingCount + Math.round(frac * cohort);
}

const budgeted = minutes > 0 && Number.isFinite(minutes);
console.log(`refresh-v: ${inFile} (${fmtMB(statSync(inFile).size)}) | depth ${depth} | jobs ${jobs} | `
  + `eval ${evalName}${evalName === 'nn' ? ` (${weights.replace(/^.*[\\/]/, '')})` : ''} | seed ${seed}`
  + ` | budget ${budgeted ? `${minutes}m` : 'none'}`);
if (eloByVersion) {
  const recomputeElo = eloForTag(vtag);
  const isBest = best && best.tag === vtag;
  console.log(`  ledger: ${ledgerPath.replace(/^.*[\\/]/, '')} | recompute with ${vtag}`
    + `${Number.isFinite(recomputeElo) ? ` (Elo ${recomputeElo.toFixed(0)})` : ''}`
    + `${best && !isBest ? `  [ledger best: ${best.tag}, Elo ${best.elo.toFixed(0)}]` : ''}`);
  const fillPart = missingCount > 0 ? `fill ${fmtNum(missingCount)} missing v (all)` : null;
  const refreshPart = targetElo !== null
    ? `refresh ${(frac * 100).toFixed(0)}% of weakest band: ${cohortName(targetElo)}`
      + `${band > 0 && targetElo !== -Infinity ? ` … +${band} Elo` : ''}`
    : null;
  console.log(`  target: ${[fillPart, refreshPart].filter(Boolean).join(' + ')} -> ${outFile}`);
} else {
  console.log(`  mode: ${refresh ? `refresh ${(frac * 100).toFixed(0)}% of existing v + fill missing` : 'fill missing v only'} -> ${outFile}`);
}
printStopHint();
if (budgeted) console.log(`  Will stop on its own after ${minutes}m and finalize cleanly.`);

const tmp = outFile + '.tmp';
const out = createWriteStream(tmp);

// --- reorder buffer (emit whole games in input order) ----------------------------
let nextEmit = 0, gameIdx = 0, filledPlies = 0, refreshedPlies = 0, passedGames = 0, computed = 0;
const ready = new Map();        // gameIdx -> string (newline-terminated)
const pending = new Map();      // gameIdx -> { rec, remaining }
const itemMeta = new Map();     // itemId -> { gameIdx, ply }
let itemSeq = 0;
function flush() {
  while (ready.has(nextEmit)) { out.write(ready.get(nextEmit)); ready.delete(nextEmit); nextEmit++; }
}
function emitGame(idx, str) { ready.set(idx, str); flush(); }

// --- worker pool -----------------------------------------------------------------
const pool = [], idle = [], queue = [];
let inputEnded = false, finalized = false;
let batch = [];
function enqueueBatch() { if (batch.length) { queue.push(batch); batch = []; pump(); } }
function pump() { while (idle.length && queue.length) idle.pop().postMessage({ type: 'batch', items: queue.shift() }); }

// A game's searched ply has resolved (a real value or a stop-skip): decrement and, when the
// game has no outstanding plies, serialize and emit it.
function resolvePly(itemId, v) {
  const meta = itemMeta.get(itemId); if (!meta) return;
  itemMeta.delete(itemId);
  const p = pending.get(meta.gameIdx); if (!p) return;
  if (v != null) { p.rec.v[meta.ply] = v; setVsAt(p.rec, meta.ply, vtag); computed++; }
  if (--p.remaining === 0) {
    pending.delete(meta.gameIdx);
    normalizeVs(p.rec);
    emitGame(meta.gameIdx, serializeGameRecord(p.rec) + '\n');
  }
}

function maybeFinalize() {
  if (finalized || !inputEnded || batch.length || queue.length || idle.length !== pool.length) return;
  finalized = true;
  if (budgetTimer) clearTimeout(budgetTimer);
  stopper.dispose();
  flush();
  status.clear();
  for (const w of pool) w.terminate();
  out.end(() => {
    if (existsSync(outFile)) rmSync(outFile);
    renameSync(tmp, outFile);
    console.log(`${stopping ? 'Stopped early' : 'Done'}: ${fmtNum(filledPlies)} filled, ${fmtNum(refreshedPlies)} refreshed, ${fmtNum(passedGames)} game(s) unchanged `
      + `(${fmtNum(gameIdx)} games) in ${fmtDur((Date.now() - t0) / 1000)} -> ${outFile} (${fmtMB(statSync(outFile).size)}).`);
    console.log('Re-featurize next:  npm run train:featurize');
  });
}

rl = createInterface({ input: createReadStream(inFile), crlfDelay: Infinity });

if (budgeted) {
  budgetTimer = setTimeout(() => {
    if (stopping) return;
    console.log(`\n  Time budget (${minutes}m) reached.`);
    requestStop();
  }, minutes * 60_000);
  budgetTimer.unref?.();
}

rl.on('line', (line) => {
  if (!line) return;
  const idx = gameIdx++;
  // After a stop, every remaining game is copied through verbatim.
  if (stopping) {
    emitGame(idx, line + '\n'); passedGames++;
    if (!inputEnded && gameIdx - nextEmit > CAP) rl.pause();
    return;
  }
  let rec; try { rec = JSON.parse(line); } catch { emitGame(idx, line + '\n'); passedGames++; return; }
  if (!isGameRecord(rec)) { emitGame(idx, line + '\n'); passedGames++; return; } // legacy/non-game: pass through

  // Plan which plies to recompute (calls rng in deterministic game/ply order).
  const n = rec.v ? rec.v.length : rec.moves.length + 1;
  const need = [];
  for (let i = 0; i < n; i++) if (shouldRecompute(rec, i)) need.push(i);
  if (need.length === 0) { emitGame(idx, line + '\n'); passedGames++; return; }
  if (!Array.isArray(rec.v)) rec.v = new Array(n).fill(null); // a record without v -> fill it

  // Replay the game to get the FEN at each needed ply, and dispatch each as a work item.
  let dispatched = 0;
  try {
    const needSet = new Set(need);
    for (const p of expandPositions(rec)) {
      if (!needSet.has(p.ply)) continue;
      const had = rec.v[p.ply] != null;
      if (had) refreshedPlies++; else filledPlies++;
      const itemId = itemSeq++;
      itemMeta.set(itemId, { gameIdx: idx, ply: p.ply });
      batch.push({ idx: itemId, fen: toFen(p.state) });
      dispatched++;
      if (batch.length >= B) enqueueBatch();
    }
  } catch {
    // Corrupt/unreplayable game: pass through unchanged. Any items already queued for it
    // resolve into a `pending` entry that won't exist, so guard registration on dispatched>0
    // and only register when the replay fully succeeded.
    emitGame(idx, line + '\n'); passedGames++; return;
  }
  pending.set(idx, { rec, remaining: dispatched });

  // Backpressure: never pause while holding an undispatched partial batch nextEmit may be
  // blocked on. Flush whatever's accumulated first so workers can unstick the buffer.
  if (!inputEnded && gameIdx - nextEmit > CAP) { enqueueBatch(); rl.pause(); }
});
rl.on('close', () => { inputEnded = true; enqueueBatch(); maybeFinalize(); });

ensureWasm(); // build the native engine (wasm) once on the main thread before workers race for it
for (let i = 0; i < jobs; i++) {
  const w = new Worker(new URL('./refreshWorker.mjs', import.meta.url), { workerData: { weights, depth, evalName, stopFlag } });
  pool.push(w);
  w.on('message', (msg) => {
    if (msg.type === 'ready') { idle.push(w); pump(); maybeFinalize(); return; }
    if (msg.type !== 'done') return;
    for (const { idx, v } of msg.vs) resolvePly(idx, v);
    if (msg.skipped) for (const idx of msg.skipped) resolvePly(idx, null); // stop-skipped: keep original v
    flush();
    if (tick()) {
      const el = (Date.now() - t0) / 1000;
      const rate = computed / Math.max(el, 0.001);
      const etaWork = (toCompute != null && rate > 0) ? Math.max(0, (toCompute - computed) / rate) : Infinity;
      const etaBudget = budgeted ? Math.max(0, minutes * 60 - el) : Infinity;
      const eta = Math.min(etaWork, etaBudget);
      status.update(`  ${fmtNum(computed)}${toCompute != null ? `/${fmtNum(toCompute)}` : ''} computed | ${rate.toFixed(0)}/s | `
        + `${fmtDur(el)} elapsed${Number.isFinite(eta) ? ` | ETA ${fmtDur(eta)}` : ''} | ${fmtNum(gameIdx)} games read`);
    }
    idle.push(w);
    if (!inputEnded && gameIdx - nextEmit <= CAP) rl.resume();
    pump();
    maybeFinalize();
  });
  w.on('error', (e) => { console.error('\nrefresh worker error:', e); });
}
