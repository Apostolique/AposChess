// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2019-2026 Jean-David Moisan
//
// One-off migration: turn the legacy feature-only dataset ({f,r,g}) into the current
// position-primary raw format ({fen,r,g,v}), recovering the TRUE position (colour to
// move + castling + move counters) by replaying each game through the engine, and
// attaching a freshly-searched value `v` (nn eval, fixed depth) as a TD/bootstrap
// target. Records that already have `fen` (freshly generated data) pass through
// verbatim, so this is safe to run on a mixed dataset.
//
// The expensive part (the per-position search) is fanned out across worker threads
// (scripts/backfillWorker.mjs), one game per job. The main thread is the single
// writer and emits units in input order via a small reorder buffer, so the output is
// deterministic regardless of which worker finishes first.
//
// Usage (run from web/):
//   node scripts/backfill-v.mjs [--in=FILE] [--out=FILE] [--depth=D] [--jobs=N] [--weights=FILE] [--total=N]
// Defaults: in/out = ../training/data/selfplay.jsonl (atomic replace via .tmp),
//           depth 4, jobs = CPU cores, weights = ./src/nn-weights.json (the champion).

import { Worker } from 'node:worker_threads';
import { createReadStream, createWriteStream, existsSync, rmSync, renameSync, statSync } from 'node:fs';
import { createInterface } from 'node:readline';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { cpus } from 'node:os';

import { ensureWasm } from './wasmEngine.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const args = Object.fromEntries(process.argv.slice(2).map((a) => {
  const m = a.replace(/^--/, '').split('='); return [m[0], m.length > 1 ? m[1] : true];
}));
const dataDefault = resolve(here, '../../training/data/selfplay.jsonl');
const inFile = typeof args.in === 'string' ? resolve(process.cwd(), args.in) : dataDefault;
const outFile = typeof args.out === 'string' ? resolve(process.cwd(), args.out) : dataDefault;
const depth = args.depth !== undefined ? Number(args.depth) : 4;
const jobs = Math.max(1, args.jobs !== undefined ? Number(args.jobs) : cpus().length);
const total = args.total !== undefined ? Number(args.total) : null; // legacy positions, for ETA
const weights = typeof args.weights === 'string'
  ? resolve(process.cwd(), args.weights) : resolve(here, '../src/nn-weights.json');

if (!existsSync(inFile)) { console.error(`No dataset at ${inFile}`); process.exit(1); }

console.log(`backfill-v: ${inFile}`);
console.log(`  depth ${depth} | jobs ${jobs} | weights ${weights.replace(/^.*[\\/]/, '')} -> ${outFile}`);

const tmp = outFile + '.tmp';
const out = createWriteStream(tmp);
const t0 = Date.now();
const fmt = (s) => { s = Math.round(s); return s < 60 ? `${s}s` : `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, '0')}s`; };

// --- reorder buffer: emit units (games + pass-throughs) in input order ----------
let created = 0, nextEmit = 0, donePos = 0, totalFallbacks = 0, gamesDone = 0, passDone = 0;
const ready = new Map(); // u -> string (newline-terminated), or '' for nothing
function flush() {
  while (ready.has(nextEmit)) {
    const s = ready.get(nextEmit); ready.delete(nextEmit);
    if (s) out.write(s);
    nextEmit++;
  }
}

const CAP = jobs * 4; // bound buffered/in-flight games (memory backpressure)
const pool = [];
const idle = [];
const queue = []; // pending game jobs {type:'game',u,g,recs}
let inputEnded = false, finalized = false;

function pump() {
  while (idle.length && queue.length) idle.pop().postMessage(queue.shift());
}

function log() {
  const el = (Date.now() - t0) / 1000;
  const rate = donePos / Math.max(el, 0.001);
  const eta = total ? ` | ETA ${fmt((total - donePos) / Math.max(rate, 0.001))}` : '';
  const pct = total ? ` (${(100 * donePos / total).toFixed(1)}%)` : '';
  process.stdout.write(`\r  ${donePos}${pct} positions | ${gamesDone}g+${passDone}p | `
    + `${rate.toFixed(0)}/s | ${fmt(el)}${eta}${totalFallbacks ? ` | ${totalFallbacks} fb` : ''}   `);
}

function maybeFinalize() {
  if (finalized) return;
  if (!inputEnded || queue.length || idle.length !== pool.length) return;
  finalized = true;
  flush(); log();
  for (const w of pool) w.terminate();
  out.end(() => {
    if (existsSync(outFile)) rmSync(outFile);
    renameSync(tmp, outFile);
    process.stdout.write('\n');
    console.log(`Done: ${donePos} positions (${gamesDone} games reconstructed, ${passDone} passed through) `
      + `in ${fmt((Date.now() - t0) / 1000)}.`);
    if (totalFallbacks) console.log(`  ${totalFallbacks} positions used heuristic castling (engine chain broke).`);
    console.log(`Wrote ${outFile} (${(statSync(outFile).size / 1e6).toFixed(1)} MB). `
      + `Re-featurize next:  npm run train:featurize`);
  });
}

// --- input: group legacy records by game; pass fen-records through ---------------
const rl = createInterface({ input: createReadStream(inFile), crlfDelay: Infinity });
let curG = null, buf = [];

function flushGame() {
  if (!buf.length) return;
  queue.push({ type: 'game', u: created++, g: curG, recs: buf });
  buf = [];
  pump();
  if (!inputEnded && created - nextEmit > CAP) rl.pause(); // backpressure (never after close)
}

rl.on('line', (line) => {
  if (!line) return;
  const rec = JSON.parse(line);
  if (typeof rec.fen === 'string') {           // already migrated -> verbatim, in order
    flushGame();
    ready.set(created++, line + '\n'); passDone++; flush();
    return;
  }
  if (rec.g !== curG) { flushGame(); curG = rec.g; }
  buf.push({ f: rec.f, r: rec.r });
});
rl.on('close', () => { inputEnded = true; flushGame(); maybeFinalize(); });

// --- workers --------------------------------------------------------------------
ensureWasm(); // build the native engine (wasm) once before the workers race for it
for (let i = 0; i < jobs; i++) {
  const w = new Worker(new URL('./backfillWorker.mjs', import.meta.url), { workerData: { weights, depth } });
  pool.push(w);
  w.on('message', (msg) => {
    // Re-check finalization: if the input closed with nothing to compute before the
    // workers booted, this is the only event left that can complete "all idle".
    if (msg.type === 'ready') { idle.push(w); pump(); maybeFinalize(); return; }
    if (msg.type !== 'done') return;
    ready.set(msg.u, msg.lines); flush();
    donePos += msg.n; gamesDone++; totalFallbacks += msg.fallbacks;
    if ((gamesDone & 31) === 0) log();
    idle.push(w);
    if (created - nextEmit <= CAP && !inputEnded) rl.resume(); // relieve backpressure
    pump();
    maybeFinalize();
  });
  w.on('error', (e) => { console.error('\nworker error:', e); });
}
