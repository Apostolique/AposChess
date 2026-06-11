// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2019-2026 Jean-David Moisan
//
// Dataset maintenance: (re)compute the search value `v` on the position-primary
// dataset with the CURRENT champion. `v` is a TD/bootstrap target (train.py --lambda),
// and it goes stale as the champion improves — it's a weaker net's opinion of the
// position. Refreshing it is value iteration: re-bootstrap targets from the improved
// value function. (The game result `r` never goes stale, so only the `v` part drifts.)
//
// Modes:
//   default          fill `v` only on records that lack it (e.g. random opening plies).
//   --refresh        also recompute `v` on records that already have one.
//   --frac=P         with --refresh, recompute only a random fraction P (0..1) of the
//                    records that have a `v` (records MISSING `v` are always filled).
//                    Partial refresh amortizes cost and keeps average `v`-staleness ~1/P
//                    passes instead of lurching between all-stale and all-fresh.
//
// The expensive searches are fanned out across worker threads (refreshWorker.mjs). The
// main thread streams the file, dispatches batches of positions that need a value, and
// is the single writer — emitting lines in input order via a reorder buffer. Records
// that don't need a (re)computation pass through as raw strings (no parse). Seeded, so
// which records get refreshed is reproducible from --seed.
//
// Usage (run from web/):
//   node scripts/refresh-v.mjs [--refresh] [--frac=P] [--depth=D] [--jobs=N]
//                              [--weights=FILE] [--in=FILE] [--out=FILE] [--seed=S]
// Defaults: depth 6, jobs = CPU cores, weights = ./src/nn-weights.json (the champion),
//           in/out = ../training/data/selfplay.jsonl (atomic replace), seed 1.

import { Worker } from 'node:worker_threads';
import { createReadStream, createWriteStream, existsSync, rmSync, renameSync, readFileSync, statSync } from 'node:fs';
import { createInterface } from 'node:readline';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { cpus } from 'node:os';

import { fmtDur, fmtNum, fmtMB, liveStatus, everyMs } from './fmt.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const args = Object.fromEntries(process.argv.slice(2).map((a) => {
  const m = a.replace(/^--/, '').split('='); return [m[0], m.length > 1 ? m[1] : true];
}));
const refresh = !!args.refresh;
const frac = args.frac !== undefined ? Number(args.frac) : 1.0;
const depth = args.depth !== undefined ? Number(args.depth) : 6;
const jobs = Math.max(1, args.jobs !== undefined ? Number(args.jobs) : cpus().length);
const seed = args.seed !== undefined ? Number(args.seed) : 1;
const weights = typeof args.weights === 'string'
  ? resolve(process.cwd(), args.weights) : resolve(here, '../src/nn-weights.json');
const inFile = typeof args.in === 'string'
  ? resolve(process.cwd(), args.in) : resolve(here, '../../training/data/selfplay.jsonl');
const outFile = typeof args.out === 'string'
  ? resolve(process.cwd(), args.out) : resolve(here, '../../training/data/selfplay.jsonl');

if (!existsSync(inFile)) { console.error(`No dataset at ${inFile}`); process.exit(1); }

const B = 64;                              // positions per worker batch
const CAP = Math.max(jobs * B * 8, 20000); // max outstanding (un-emitted) lines
const t0 = Date.now();
const status = liveStatus();
const tick = everyMs(1000);
console.log(`refresh-v: ${inFile} (${fmtMB(statSync(inFile).size)}) | depth ${depth} | jobs ${jobs} | `
  + `weights ${weights.replace(/^.*[\\/]/, '')} | seed ${seed}`);
console.log(`  mode: ${refresh ? `refresh ${(frac * 100).toFixed(0)}% of existing v + fill missing` : 'fill missing v only'} -> ${outFile}`);

function mulberry32(a) {
  a >>>= 0;
  return () => { a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
const rng = mulberry32(seed);

const tmp = outFile + '.tmp';
const out = createWriteStream(tmp);

// --- reorder buffer (emit in input order) ----------------------------------------
let nextEmit = 0, lineIdx = 0, filled = 0, refreshed = 0, passed = 0, computed = 0;
const ready = new Map();        // lineIdx -> string (newline-terminated)
const pendingRec = new Map();   // lineIdx -> parsed record awaiting its v
function flush() {
  while (ready.has(nextEmit)) { out.write(ready.get(nextEmit)); ready.delete(nextEmit); nextEmit++; }
}

// --- worker pool -----------------------------------------------------------------
const pool = [], idle = [], queue = [];
let inputEnded = false, finalized = false;
let batch = [];
function enqueueBatch() { if (batch.length) { queue.push(batch); batch = []; pump(); } }
function pump() { while (idle.length && queue.length) idle.pop().postMessage({ type: 'batch', items: queue.shift() }); }

function maybeFinalize() {
  if (finalized || !inputEnded || batch.length || queue.length || idle.length !== pool.length) return;
  finalized = true;
  flush();
  status.clear();
  for (const w of pool) w.terminate();
  out.end(() => {
    if (existsSync(outFile)) rmSync(outFile);
    renameSync(tmp, outFile);
    console.log(`Done: ${fmtNum(filled)} filled, ${fmtNum(refreshed)} refreshed, ${fmtNum(passed)} unchanged `
      + `(${fmtNum(lineIdx)} total) in ${fmtDur((Date.now() - t0) / 1000)} -> ${outFile} (${fmtMB(statSync(outFile).size)}).`);
    console.log('Re-featurize next:  npm run train:featurize');
  });
}

const rl = createInterface({ input: createReadStream(inFile), crlfDelay: Infinity });

rl.on('line', (line) => {
  if (!line) return;
  const idx = lineIdx++;
  const hasV = line.includes('"v":');
  const recompute = !hasV || (refresh && rng() < frac);
  if (!recompute) { ready.set(idx, line + '\n'); passed++; flush(); return; }
  const rec = JSON.parse(line);
  if (typeof rec.fen !== 'string') { ready.set(idx, line + '\n'); passed++; flush(); return; }
  if (hasV) refreshed++; else filled++;
  pendingRec.set(idx, rec);
  batch.push({ idx, fen: rec.fen });
  if (batch.length >= B) enqueueBatch();
  if (!inputEnded && lineIdx - nextEmit > CAP) rl.pause(); // backpressure
});
rl.on('close', () => { inputEnded = true; enqueueBatch(); maybeFinalize(); });

for (let i = 0; i < jobs; i++) {
  const w = new Worker(new URL('./refreshWorker.mjs', import.meta.url), { workerData: { weights, depth } });
  pool.push(w);
  w.on('message', (msg) => {
    // A worker turning ready can complete the "all idle" condition, so re-check
    // finalization — otherwise a run where NO record needs a value (input closes
    // before the workers finish booting) waits forever on a 'done' that never comes.
    if (msg.type === 'ready') { idle.push(w); pump(); maybeFinalize(); return; }
    if (msg.type !== 'done') return;
    for (const { idx, v } of msg.vs) {
      const rec = pendingRec.get(idx); pendingRec.delete(idx);
      rec.v = v;
      ready.set(idx, JSON.stringify(rec) + '\n');
    }
    computed += msg.vs.length;
    flush();
    if (tick()) {
      const el = (Date.now() - t0) / 1000;
      status.update(`  ${fmtNum(computed)} values computed | ${(computed / Math.max(el, 0.001)).toFixed(0)}/s | `
        + `${fmtDur(el)} elapsed | ${fmtNum(lineIdx)} lines read`);
    }
    idle.push(w);
    if (!inputEnded && lineIdx - nextEmit <= CAP) rl.resume();
    pump();
    maybeFinalize();
  });
  w.on('error', (e) => { console.error('\nrefresh worker error:', e); });
}
