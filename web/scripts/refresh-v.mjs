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
//   node scripts/refresh-v.mjs [--refresh] [--frac=P] [--depth=D] [--jobs=N] [--eval=E]
//                              [--weights=FILE] [--in=FILE] [--out=FILE] [--seed=S]
//                              [--ledger[=FILE]]
// Defaults: depth 6, jobs = CPU cores, eval 'nn' (weights = ./src/nn-weights.json, the
//           champion), in/out = ../training/data/selfplay.jsonl (atomic replace), seed 1.
//   --eval=handcrafted recomputes v with the handcrafted eval (no weights) — e.g. to
//   relabel the bootstrap target with its outcome-grounded values.
//
// Smart weakest-first refresh (--ledger): read the strength ledger written by
// `npm run rank` (default training/data/loop/engine-elo.json) and let it drive the whole
// refresh. The recompute engine DEFAULTS to the ledger's STRONGEST engine (so v is always
// relabeled with the best available eval; explicit --eval/--weights still win), and the
// refresh replaces ONLY the single WEAKEST cohort of labels still present — the lowest-Elo
// engine in the data, with records that are missing v / untagged / from an unrecoverable
// engine all counting as weakest. Stronger labels are left alone, so re-running it walks
// the dataset upward — weakest engine first — until everything carries the best engine's
// v. (A pre-scan picks the target each run; --frac can still chunk a very large cohort.)

import { Worker } from 'node:worker_threads';
import { createReadStream, createWriteStream, existsSync, rmSync, renameSync, readFileSync, statSync } from 'node:fs';
import { createInterface } from 'node:readline';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { cpus } from 'node:os';

import { fmtDur, fmtNum, fmtMB, liveStatus, everyMs } from './fmt.mjs';
import { vtag as computeVtag } from './vtag.mjs';
import { installStop, printStopHint } from './stop.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const args = Object.fromEntries(process.argv.slice(2).map((a) => {
  const m = a.replace(/^--/, '').split('='); return [m[0], m.length > 1 ? m[1] : true];
}));
const refresh = !!args.refresh;
const frac = args.frac !== undefined ? Number(args.frac) : 1.0;
const depth = args.depth !== undefined ? Number(args.depth) : 6;
const jobs = Math.max(1, args.jobs !== undefined ? Number(args.jobs) : cpus().length);
const seed = args.seed !== undefined ? Number(args.seed) : 1;
// --- smart weakest-first refresh (--ledger) --------------------------------------
// With --ledger, the strength ledger from `npm run rank` (engine-elo.json) drives the
// refresh end to end: the RECOMPUTE ENGINE defaults to the ledger's STRONGEST engine (so
// v is always relabeled with the best available eval; explicit --eval/--weights still
// win), and the refresh TARGETS exactly the single WEAKEST cohort of labels still present
// — the lowest-Elo engine in the data, with records that are missing v / carry no vs tag
// / were labeled by an unrecoverable engine all counting as weakest (-Inf). Every
// stronger label is left untouched, so running it repeatedly replaces the v values one
// engine at a time, weakest first, until the whole dataset carries the current best
// engine's labels. Without --ledger nothing here changes the classic behavior.
const ledgerPath = args.ledger === true
  ? resolve(here, '../../training/data/loop/engine-elo.json')
  : (typeof args.ledger === 'string' ? resolve(process.cwd(), args.ledger) : null);
let eloByVersion = null, best = null;
if (ledgerPath) {
  let ledger;
  try { ledger = JSON.parse(readFileSync(ledgerPath, 'utf8')); }
  catch (e) { console.error(`Could not read ledger ${ledgerPath}: ${e.message}. Run 'npm run rank' first.`); process.exit(1); }
  eloByVersion = new Map();
  // The material fallback ('?') is excluded from refresh decisions entirely — its Elo is
  // only an internal stat in the ledger (a floor reference for `npm run rank`). So anything
  // it (or an unrecoverable/untagged source) labeled falls into the -Inf "weakest" bucket
  // and refreshes first, rather than being scored at the material level.
  for (const e of ledger.ranking || []) if (e.elo != null && e.version !== '?') eloByVersion.set(e.version, e.elo);
  // Strongest engine = highest Elo (the anchor sits at 0; a contender may beat it).
  best = (ledger.ranking || []).filter((e) => e.elo != null && e.version !== '?')
    .reduce((a, b) => (b.elo > a.elo ? b : a), { elo: -Infinity });
  if (!Number.isFinite(best.elo)) best = null;
}

// Which eval recomputes v: explicit --eval/--weights win; otherwise the ledger's best
// engine (when --ledger), else 'nn' with the shipped champion (the prior default).
// 'nn' uses --weights; 'handcrafted' needs none. Relabeling with the recalibrated
// handcrafted injects its outcome-grounded values where an nn champion's own v may carry
// the net's biases.
const evalName = typeof args.eval === 'string' ? args.eval
  : (best ? (best.eng === 'nn' ? 'nn' : 'handcrafted') : 'nn');
const weights = typeof args.weights === 'string' ? resolve(process.cwd(), args.weights)
  : (best && best.eng === 'nn' && best.file ? best.file : resolve(here, '../src/nn-weights.json'));
// Provenance tag stamped onto every (re)computed v: which eval + depth + version
// produced it (e.g. 'nn6@a3f2c1', 'hc6@2') — keeps the dataset auditable as v drifts.
const vtag = computeVtag(evalName, depth, weights);

// The recompute engine's identity (eng + version), used to skip records it already
// labelled at >= this depth (no improvement to redo them).
const recomputeEng = evalName.startsWith('nn') ? 'nn' : 'hc';
const recomputeVersion = vtag.slice(vtag.indexOf('@') + 1);
const parseTag = (tag) => { const m = /^(nn|hc)(\d+|t)@(.+)$/.exec(tag || ''); return m ? { eng: m[1], depth: m[2], version: m[3] } : null; };
const eloForTag = (tag) => { const t = parseTag(tag); const e = t && eloByVersion.get(t.version); return e == null ? -Infinity : e; };
const alreadyDone = (tag) => { const t = parseTag(tag); if (!t || t.eng !== recomputeEng || t.version !== recomputeVersion) return false; const d = Number(t.depth); return Number.isFinite(d) && d >= depth; };
// Human-readable name for a cohort's Elo (for the banner): the version(s) at that Elo, or
// the catch-all weakest bucket for -Inf (missing v / no tag / material / unrecoverable).
const cohortName = (elo) => {
  if (elo === -Infinity) return 'unknown / missing v / material / unrecoverable';
  const vs = [...eloByVersion.entries()].filter(([, e]) => e === elo).map(([v]) => v);
  return `${vs.length ? vs.join(', ') : '?'} (Elo ${elo.toFixed(0)})`;
};
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

// --- graceful early stop ----------------------------------------------------------
// On a key/Ctrl-C we DON'T abort (that would lose all the recomputed values and leave
// a dangling .tmp). Instead we stop scheduling new recomputes, copy every remaining
// input line through UNCHANGED, let the in-flight searches finish, and then do the
// normal atomic rename — so the output is a complete, valid dataset that's simply
// refreshed up to the point we stopped. `rl` is filled in once the main pass starts;
// a stop during the (read-only) pre-scan just leaves the dataset untouched.
let stopping = false;
let rl = null;
function requestStop() {
  stopping = true;
  status.clear();
  if (rl) {
    console.log('\n  Stopping early — copying the remaining lines through unchanged and finalizing…');
    enqueueBatch();   // flush any half-full batch so the in-flight records still resolve
    // Un-pause if backpressure had paused the reader — but only if it's still open;
    // the whole file may already be read (just the searches outstanding), in which
    // case the readline is closed and resume() would throw.
    if (!inputEnded) { try { rl.resume(); } catch { /* already closed */ } }
    flush();
    maybeFinalize();  // in case everything's already drained, complete the rename now
  } else {
    console.log('\n  Stopping early during the pre-scan — the dataset is left unchanged.');
  }
}
const stopper = installStop(requestStop);

// --- pre-scan (--ledger): find this run's target = the single weakest cohort still
// present. One cheap streaming pass (no search): each record's label Elo is its engine's
// ledger Elo, or -Inf if it's missing v / untagged / from an unrecoverable engine;
// records the recompute engine already produced at >= depth are not refresh candidates.
// The minimum over the candidates is the target — only those records get refreshed.
let targetElo = null;            // null => classic mode (no --ledger)
if (eloByVersion) {
  let min = Infinity, scanned = 0;
  const sTick = everyMs(500);
  try {
    const srl = createInterface({ input: createReadStream(inFile), crlfDelay: Infinity });
    for await (const line of srl) {
      if (stopping) { srl.close(); break; }
      if (!line) continue;
      scanned++;
      const hasV = line.includes('"v":');
      const tag = hasV ? ((line.match(/"vs":"([^"]+)"/) || [])[1] || null) : null;
      if (alreadyDone(tag)) continue;                  // already the best engine at >= depth
      const elo = hasV ? eloForTag(tag) : -Infinity;   // missing v counts as weakest
      if (elo < min) min = elo;
      if (sTick()) status.update(`  scanning for weakest cohort... ${fmtNum(scanned)} lines`);
    }
  } catch (e) { console.warn(`\n  pre-scan interrupted (${e.message}); proceeding with what was seen.`); }
  status.clear();
  if (stopping) { console.log('Stopped before writing; dataset left unchanged.'); stopper.dispose(); process.exit(0); }
  if (min === Infinity) {
    console.log(`Nothing to refresh: every label is already the best engine ${vtag} at depth >= ${depth}.`);
    process.exit(0);
  }
  targetElo = min;
}

console.log(`refresh-v: ${inFile} (${fmtMB(statSync(inFile).size)}) | depth ${depth} | jobs ${jobs} | `
  + `eval ${evalName}${evalName === 'nn' ? ` (${weights.replace(/^.*[\\/]/, '')})` : ''} | seed ${seed}`);
if (eloByVersion) {
  console.log(`  ledger: ${ledgerPath.replace(/^.*[\\/]/, '')} | recompute with ${vtag}`
    + `${best ? ` (best engine, Elo ${best.elo.toFixed(0)})` : ''}`);
  console.log(`  target: weakest cohort still present = ${cohortName(targetElo)} -> ${outFile}`);
} else {
  console.log(`  mode: ${refresh ? `refresh ${(frac * 100).toFixed(0)}% of existing v + fill missing` : 'fill missing v only'} -> ${outFile}`);
}
printStopHint();

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
  stopper.dispose();
  flush();
  status.clear();
  for (const w of pool) w.terminate();
  out.end(() => {
    if (existsSync(outFile)) rmSync(outFile);
    renameSync(tmp, outFile);
    console.log(`${stopper.requested ? 'Stopped early' : 'Done'}: ${fmtNum(filled)} filled, ${fmtNum(refreshed)} refreshed, ${fmtNum(passed)} unchanged `
      + `(${fmtNum(lineIdx)} total) in ${fmtDur((Date.now() - t0) / 1000)} -> ${outFile} (${fmtMB(statSync(outFile).size)}).`);
    console.log('Re-featurize next:  npm run train:featurize');
  });
}

rl = createInterface({ input: createReadStream(inFile), crlfDelay: Infinity });

rl.on('line', (line) => {
  if (!line) return;
  const idx = lineIdx++;
  // After a stop request, every remaining line is copied through verbatim (no
  // recompute) so the output stays a complete, valid dataset.
  if (stopping) { ready.set(idx, line + '\n'); passed++; flush(); return; }
  const hasV = line.includes('"v":');
  let recompute;
  if (targetElo === null) {                     // classic mode (no --ledger)
    recompute = !hasV || (refresh && rng() < frac);
  } else {                                       // --ledger: only the weakest cohort
    const tag = hasV ? ((line.match(/"vs":"([^"]+)"/) || [])[1] || null) : null;
    const elo = hasV ? eloForTag(tag) : -Infinity;
    recompute = !alreadyDone(tag) && elo === targetElo && rng() < frac;
  }
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
  const w = new Worker(new URL('./refreshWorker.mjs', import.meta.url), { workerData: { weights, depth, evalName } });
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
      rec.vs = vtag; // who computed this v (eval+depth)
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
