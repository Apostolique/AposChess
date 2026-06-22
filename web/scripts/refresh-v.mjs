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
//   --minutes=M      wall-clock budget (default 10): after M minutes the run stops
//                    gracefully and finalizes a complete dataset, just like pressing q.
//                    --minutes=0 (or Infinity) removes the budget and runs to completion.
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
//                              [--ledger[=FILE]] [--minutes=M] [--band=E]
// Defaults: depth 6, jobs = CPU cores, eval 'nn' (weights = ./src/nn-weights.json, the
//           champion), in/out = ../training/data/selfplay.jsonl (atomic replace), seed 1.
//   --eval=handcrafted recomputes v with the handcrafted eval (no weights) — e.g. to
//   relabel the bootstrap target with its outcome-grounded values.
//
// Smart weakest-first refresh (--ledger): read the strength ledger written by
// `npm run rank` (default training/data/loop/engine-elo.json) and let it drive the whole
// refresh. The recompute engine DEFAULTS to the ledger's STRONGEST engine (so v is always
// relabeled with the best available eval; explicit --eval/--weights still win). Each run
// does two things, in priority order:
//   1. FILL every record that has no `v` at all — these are useless to the trainer, so
//      they're always filled in full, never throttled by --frac.
//   2. REFRESH the WEAKEST BAND of records that DO have a `v` — every label within --band
//      Elo (default 25) of the lowest-Elo engine still present (untagged / unrecoverable
//      count as -Inf), at --frac. The band (not a single exact Elo) keeps a quasi-continuous
//      spread of ephemeral candidate Elos (nn<d>@elo<E>; see vtag.mjs) draining as one weak
//      cohort per run rather than one exact value at a time. --band=0 restores exact-Elo
//      cohorts (one engine at a time). The -Inf bucket stays its own cohort (-Inf + band = -Inf).
// Stronger labels are left alone, so re-running it walks the dataset upward — missing v
// first, then the weakest band — until everything carries the best engine's v. (A pre-scan
// plans the run; --frac throttles only the cohort relabel, never the fills.)

import { Worker } from 'node:worker_threads';
import { createReadStream, createWriteStream, existsSync, rmSync, renameSync, readFileSync, statSync } from 'node:fs';
import { createInterface } from 'node:readline';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { cpus } from 'node:os';

import { ensureWasm } from './wasmEngine.mjs';
import { fmtDur, fmtNum, fmtMB, liveStatus, everyMs } from './fmt.mjs';
import { vtag as computeVtag, ephemeralElo } from './vtag.mjs';
import { installStop, printStopHint } from './stop.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const args = Object.fromEntries(process.argv.slice(2).map((a) => {
  const m = a.replace(/^--/, '').split('='); return [m[0], m.length > 1 ? m[1] : true];
}));
const refresh = !!args.refresh;
const frac = args.frac !== undefined ? Number(args.frac) : 1.0;
// Weakest-cohort BAND width (Elo, --ledger only). The weakest cohort isn't a single exact
// Elo but every label within `band` Elo of the weakest still present — so a quasi-continuous
// spread of ephemeral candidate Elos (nn<d>@elo<E>, see vtag.mjs) drains as one weak band per
// run instead of one exact value at a time (which would relabel a handful of records yet still
// force a full re-featurize). The −∞ bucket (untagged / material / unrecoverable) stays its own
// cohort, since −∞ + band is still −∞. 0 restores exact-Elo cohorts (one engine at a time).
const band = args.band !== undefined ? Number(args.band) : 25;
const depth = args.depth !== undefined ? Number(args.depth) : 6;
const jobs = Math.max(1, args.jobs !== undefined ? Number(args.jobs) : cpus().length);
const seed = args.seed !== undefined ? Number(args.seed) : 1;
// Wall-clock budget (minutes): when it elapses the run stops gracefully, exactly like a
// q keypress — finalizing a complete, valid dataset. Defaults to 10 so an unattended
// refresh is self-limiting and easy to schedule; --minutes=0 (or Infinity) removes it.
const minutes = args.minutes !== undefined ? Number(args.minutes) : 10;
// --- smart weakest-first refresh (--ledger) --------------------------------------
// With --ledger, the strength ledger from `npm run rank` (engine-elo.json) drives the
// refresh end to end: the RECOMPUTE ENGINE defaults to the ledger's STRONGEST engine (so
// v is always relabeled with the best available eval; explicit --eval/--weights still
// win). Records with NO v are filled first, in full (never throttled by --frac), since a
// record without a value can't train. The relabel then TARGETS exactly the single WEAKEST
// cohort of records that HAVE a v — the lowest-Elo engine in the data, with records that
// carry no vs tag / were labeled by an unrecoverable engine counting as weakest (-Inf) —
// at --frac. Every stronger label is left untouched, so running it repeatedly fills the
// gaps and then replaces the v values one engine at a time, weakest first, until the whole
// dataset carries the current best engine's labels. Without --ledger nothing here changes
// the classic behavior (which likewise always fills missing v and fractions the rest).
const ledgerPath = args.ledger === true
  ? resolve(here, '../../training/data/loop/engine-elo.json')
  : (typeof args.ledger === 'string' ? resolve(process.cwd(), args.ledger) : null);
let eloByVersion = null, eloByTag = null, best = null;
if (ledgerPath) {
  let ledger;
  try { ledger = JSON.parse(readFileSync(ledgerPath, 'utf8')); }
  catch (e) { console.error(`Could not read ledger ${ledgerPath}: ${e.message}. Run 'npm run rank' first.`); process.exit(1); }
  eloByVersion = new Map(); eloByTag = new Map();
  // The material fallback ('?') is excluded from refresh decisions entirely — its Elo is
  // only an internal stat in the ledger (a floor reference for `npm run rank`). So anything
  // it (or an unrecoverable/untagged source) labeled falls into the -Inf "weakest" bucket
  // and refreshes first, rather than being scored at the material level.
  // The ledger holds one entry per engine×depth: eloByTag is the exact strength at a depth;
  // eloByVersion is the engine's BEST measured Elo, the fallback for a depth never ranked.
  for (const e of ledger.ranking || []) {
    if (e.elo == null || e.version === '?') continue;
    if (e.tag) eloByTag.set(e.tag, e.elo);
    const prev = eloByVersion.has(e.version) ? eloByVersion.get(e.version) : -Infinity;
    eloByVersion.set(e.version, Math.max(prev, e.elo));
  }
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
// Exact engine×depth Elo if that depth was ranked, else the engine's best per-version Elo
// (a depth never ranked); unknown engine / no tag -> -Inf (weakest, refreshed first).
const eloForTag = (tag) => {
  const t = parseTag(tag);
  if (!t) return -Infinity;
  // A non-promoted gate candidate carries its strength in the tag itself ("nn6@elo37") —
  // read it straight off, no ledger needed (it's never archived/ranked).
  const eph = ephemeralElo(t.version);
  if (eph !== null) return eph;
  if (eloByTag && eloByTag.has(tag)) return eloByTag.get(tag);
  const e = eloByVersion && eloByVersion.get(t.version);
  return e == null ? -Infinity : e;
};
const alreadyDone = (tag) => { const t = parseTag(tag); if (!t || t.eng !== recomputeEng || t.version !== recomputeVersion) return false; const d = Number(t.depth); return Number.isFinite(d) && d >= depth; };
// Human-readable name for a cohort's Elo (for the banner): the version(s) at that Elo, or
// the catch-all weakest bucket for -Inf (no vs tag / material / unrecoverable). Records
// missing v entirely aren't a cohort here — they're filled first, separately.
const cohortName = (elo) => {
  if (elo === -Infinity) return 'untagged / material / unrecoverable';
  // Prefer the exact engine×depth tag(s) at this Elo; fall back to version(s) for an Elo that
  // only the per-version fallback produced (a dataset depth that wasn't itself ranked).
  const tags = [...eloByTag.entries()].filter(([, e]) => e === elo).map(([t]) => t);
  if (tags.length) return `${tags.join(', ')} (Elo ${elo.toFixed(0)})`;
  const vs = [...eloByVersion.entries()].filter(([, e]) => e === elo).map(([v]) => v);
  // A finite Elo matching no ledger engine comes from an ephemeral candidate tag (nn6@elo<N>),
  // whose strength lives in the tag rather than the ledger.
  return `${vs.length ? vs.join(', ') : 'ephemeral candidate(s)'} (Elo ${elo.toFixed(0)})`;
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
let budgetTimer = null;          // the --minutes wall-clock budget (cleared on finalize)
// Shared with every worker (via workerData). A worker checks it BETWEEN the positions
// of its batch and bails the moment it's set — so an early stop only has to wait out the
// single search each worker is mid-flight on, not the rest of its 64-position batch.
const stopFlag = new Int32Array(new SharedArrayBuffer(4));
function requestStop() {
  stopping = true;
  Atomics.store(stopFlag, 0, 1); // tell in-flight workers to abandon the rest of their batch
  status.clear();
  if (rl) {
    console.log('\n  Stopping early — copying the remaining lines through unchanged and finalizing…');
    // Drop the work that hasn't reached a worker yet (the half-full batch and every
    // queued batch) and emit those records UNCHANGED instead of recomputing them. The
    // reader runs far ahead of the searches (CAP can be thousands of records), so this
    // backlog is what made an earlier stop keep grinding; only the batches already
    // dispatched to a worker are left to finish.
    abandon(batch); batch = [];
    for (const b of queue) abandon(b);
    queue.length = 0;
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

// --- pre-scan (--ledger): plan this run in one cheap streaming pass (no search).
// A record with NO v is useless to the trainer, so filling those is the top priority and
// is NEVER throttled by --frac: every missing v is filled this run, before any relabel.
// Among the records that DO have a v, the run then refreshes the single weakest cohort —
// the lowest ledger Elo still present (untagged / unrecoverable count as -Inf), at --frac
// — walking the dataset upward one engine at a time. Records the recompute engine already
// produced at >= depth aren't candidates. (This mirrors classic --refresh, which has
// always filled all missing v and only fractioned the rest.)
let targetElo = null;            // weakest cohort (with v) to refresh; null => none such
let missingCount = 0;            // records lacking v entirely (always filled)
// Estimated count of records this run will (re)compute, for the live ETA. Known only when
// the pre-scan runs (ledger mode); null otherwise, in which case the ETA falls back to the
// time budget. The cohort term is an expectation: ~frac of the weakest cohort passes the
// random gate, so the real count varies a little run to run.
let toCompute = null;
if (eloByVersion) {
  let min = Infinity, scanned = 0;
  const cohortCounts = new Map();   // elo -> # of candidate records with a v at that elo
  const sTick = everyMs(500);
  try {
    const srl = createInterface({ input: createReadStream(inFile), crlfDelay: Infinity });
    for await (const line of srl) {
      if (stopping) { srl.close(); break; }
      if (!line) continue;
      scanned++;
      const hasV = line.includes('"v":');
      if (!hasV) { missingCount++; continue; }          // missing v: always filled, not a cohort
      const tag = (line.match(/"vs":"([^"]+)"/) || [])[1] || null;
      if (alreadyDone(tag)) continue;                   // already the best engine at >= depth
      const elo = eloForTag(tag);                       // untagged / unrecoverable -> -Inf
      if (elo < min) min = elo;
      cohortCounts.set(elo, (cohortCounts.get(elo) || 0) + 1);
      if (sTick()) status.update(`  planning refresh... ${fmtNum(scanned)} lines`);
    }
  } catch (e) { console.warn(`\n  pre-scan interrupted (${e.message}); proceeding with what was seen.`); }
  status.clear();
  if (stopping) { console.log('Stopped before writing; dataset left unchanged.'); stopper.dispose(); process.exit(); }
  if (missingCount === 0 && min === Infinity) {
    console.log(`Nothing to refresh: every label is already the best engine ${vtag} at depth >= ${depth}.`);
    process.exit(0);
  }
  targetElo = min === Infinity ? null : min;
  // The cohort is the whole weakest BAND — every Elo in [min, min + band] (so a continuous
  // spread of ephemeral Elos drains together, not one exact value per run). −∞ + band is −∞,
  // so the unrecoverable bucket stays its own cohort until it's drained.
  const cohort = targetElo === null ? 0
    : [...cohortCounts].reduce((s, [e, n]) => s + (e <= targetElo + band ? n : 0), 0);
  toCompute = missingCount + Math.round(frac * cohort);  // fills (all) + expected cohort relabels
}

const budgeted = minutes > 0 && Number.isFinite(minutes);
console.log(`refresh-v: ${inFile} (${fmtMB(statSync(inFile).size)}) | depth ${depth} | jobs ${jobs} | `
  + `eval ${evalName}${evalName === 'nn' ? ` (${weights.replace(/^.*[\\/]/, '')})` : ''} | seed ${seed}`
  + ` | budget ${budgeted ? `${minutes}m` : 'none'}`);
if (eloByVersion) {
  console.log(`  ledger: ${ledgerPath.replace(/^.*[\\/]/, '')} | recompute with ${vtag}`
    + `${best ? ` (best engine, Elo ${best.elo.toFixed(0)})` : ''}`);
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
const pendingLine = new Map();  // lineIdx -> raw input line (to pass through on abandon)
function flush() {
  while (ready.has(nextEmit)) { out.write(ready.get(nextEmit)); ready.delete(nextEmit); nextEmit++; }
}

// --- worker pool -----------------------------------------------------------------
const pool = [], idle = [], queue = [];
let inputEnded = false, finalized = false;
let batch = [];
function enqueueBatch() { if (batch.length) { queue.push(batch); batch = []; pump(); } }
function pump() { while (idle.length && queue.length) idle.pop().postMessage({ type: 'batch', items: queue.shift() }); }

// Emit a set of not-yet-dispatched records UNCHANGED (used on early stop). Each was
// pre-counted as filled/refreshed when scheduled, so undo that and count it as passed.
function abandon(items) {
  for (const { idx } of items) {
    if (!pendingRec.has(idx)) continue;
    pendingRec.delete(idx);
    const line = pendingLine.get(idx); pendingLine.delete(idx);
    if (line.includes('"v":')) refreshed--; else filled--;
    ready.set(idx, line + '\n');
    passed++;
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
    console.log(`${stopping ? 'Stopped early' : 'Done'}: ${fmtNum(filled)} filled, ${fmtNum(refreshed)} refreshed, ${fmtNum(passed)} unchanged `
      + `(${fmtNum(lineIdx)} total) in ${fmtDur((Date.now() - t0) / 1000)} -> ${outFile} (${fmtMB(statSync(outFile).size)}).`);
    console.log('Re-featurize next:  npm run train:featurize');
  });
}

rl = createInterface({ input: createReadStream(inFile), crlfDelay: Infinity });

// Arm the wall-clock budget now that the search pass is starting (the pre-scan doesn't
// count against it). Firing just runs the same graceful stop as a keypress. unref() so a
// run that finishes early isn't held open by the pending timer.
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
  const idx = lineIdx++;
  // After a stop request, every remaining line is copied through verbatim (no
  // recompute) so the output stays a complete, valid dataset.
  if (stopping) {
    ready.set(idx, line + '\n'); passed++; flush();
    // Same backpressure as the normal path below: a few depth-N searches still in
    // flight at the front leave a gap at nextEmit that blocks flush, so without this
    // the reader floods the whole rest of the file into `ready` and blows V8's Map
    // size cap on a large dataset. The worker 'done' handler resumes us as the gap
    // drains; once nothing is pending, every line flushes at once and we stream to EOF.
    if (!inputEnded && lineIdx - nextEmit > CAP) rl.pause();
    return;
  }
  const hasV = line.includes('"v":');
  let recompute;
  if (!eloByVersion) {                           // classic mode (no --ledger)
    recompute = !hasV || (refresh && rng() < frac);
  } else if (!hasV) {                            // --ledger: always fill a missing v first
    recompute = true;
  } else {                                       // --ledger: refresh the weakest BAND that has a v
    const tag = (line.match(/"vs":"([^"]+)"/) || [])[1] || null;
    recompute = targetElo !== null && !alreadyDone(tag) && eloForTag(tag) <= targetElo + band && rng() < frac;
  }
  if (!recompute) { ready.set(idx, line + '\n'); passed++; flush(); return; }
  const rec = JSON.parse(line);
  if (typeof rec.fen !== 'string') { ready.set(idx, line + '\n'); passed++; flush(); return; }
  if (hasV) refreshed++; else filled++;
  pendingRec.set(idx, rec);
  pendingLine.set(idx, line);
  batch.push({ idx, fen: rec.fen });
  if (batch.length >= B) enqueueBatch();
  // Backpressure — but NEVER pause while still holding an undispatched partial batch
  // that nextEmit is blocked on. When recompute candidates are sparse near the front of
  // the file, the batch can't reach B before the reader runs CAP lines ahead; pausing
  // here would deadlock (partial batch never sent -> its values never computed -> nextEmit
  // never advances -> reader never resumes). Flush whatever's accumulated first, so the
  // workers always have the blocking records and can unstick the reorder buffer.
  if (!inputEnded && lineIdx - nextEmit > CAP) { enqueueBatch(); rl.pause(); }
});
rl.on('close', () => { inputEnded = true; enqueueBatch(); maybeFinalize(); });

ensureWasm(); // build the native engine (wasm) once on the main thread before workers race for it
for (let i = 0; i < jobs; i++) {
  const w = new Worker(new URL('./refreshWorker.mjs', import.meta.url), { workerData: { weights, depth, evalName, stopFlag } });
  pool.push(w);
  w.on('message', (msg) => {
    // A worker turning ready can complete the "all idle" condition, so re-check
    // finalization — otherwise a run where NO record needs a value (input closes
    // before the workers finish booting) waits forever on a 'done' that never comes.
    if (msg.type === 'ready') { idle.push(w); pump(); maybeFinalize(); return; }
    if (msg.type !== 'done') return;
    for (const { idx, v } of msg.vs) {
      const rec = pendingRec.get(idx); pendingRec.delete(idx);
      pendingLine.delete(idx);
      rec.v = v;
      rec.vs = vtag; // who computed this v (eval+depth)
      ready.set(idx, JSON.stringify(rec) + '\n');
    }
    // Positions the worker skipped because a stop fired mid-batch: pass them through
    // unchanged, exactly like the never-dispatched backlog.
    if (msg.skipped && msg.skipped.length) abandon(msg.skipped.map((idx) => ({ idx })));
    computed += msg.vs.length;
    flush();
    if (tick()) {
      const el = (Date.now() - t0) / 1000;
      const rate = computed / Math.max(el, 0.001);
      // ETA = whichever comes first: finishing the work (known count / current rate) or the
      // wall-clock budget running out. Either may be unknown (classic mode has no count; an
      // unbudgeted run has no deadline) — show it only once at least one is finite.
      const etaWork = (toCompute != null && rate > 0) ? Math.max(0, (toCompute - computed) / rate) : Infinity;
      const etaBudget = budgeted ? Math.max(0, minutes * 60 - el) : Infinity;
      const eta = Math.min(etaWork, etaBudget);
      status.update(`  ${fmtNum(computed)}${toCompute != null ? `/${fmtNum(toCompute)}` : ''} computed | ${rate.toFixed(0)}/s | `
        + `${fmtDur(el)} elapsed${Number.isFinite(eta) ? ` | ETA ${fmtDur(eta)}` : ''} | ${fmtNum(lineIdx)} lines read`);
    }
    idle.push(w);
    if (!inputEnded && lineIdx - nextEmit <= CAP) rl.resume();
    pump();
    maybeFinalize();
  });
  w.on('error', (e) => { console.error('\nrefresh worker error:', e); });
}
