// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2019-2026 Jean-David Moisan
//
// Read-only progress report for train:loop. It re-reads the persisted loop log
// (training/data/loop/loop.log) and turns the per-cycle verdict lines the loop already
// writes into a trend you can act on: is the latest run's candidate climbing toward the
// gate (let it run) or stuck below 50% (restart with different values)?
//
//   npm run train:progress           # latest run in detail + a one-line history + a read
//   npm run train:progress -- --runs=12   # show that many runs in the compact history
//   npm run train:progress -- --all  # detail EVERY run's cycles, not just the latest
//   npm run train:progress -- --log=PATH  # point at a different loop.log
//
// Nothing here writes or spawns — it only parses the log and the champion weights, so it's
// safe to run against a live loop. The numbers it reports are exactly the loop's own
// `cycle N: ...` lines (candidate score %, Elo vs champion, SPRT verdict), grouped by run.

import { existsSync, readFileSync } from 'node:fs';
import { dirname, resolve, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { fmtDur } from './fmt.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const webDir = resolve(here, '..');
const repoDir = resolve(webDir, '..');
const loopDir = resolve(repoDir, 'training', 'data', 'loop');

const args = Object.fromEntries(
  process.argv.slice(2).map((a) => {
    const m = a.replace(/^--/, '').split('=');
    return [m[0], m.length > 1 ? m[1] : true];
  }),
);
const logFile = typeof args.log === 'string' ? resolve(args.log) : join(loopDir, 'loop.log');
const histN = args.runs !== undefined ? Number(args.runs) : 8;
const detailAll = !!args.all;

const championFile = resolve(webDir, 'src', 'nn-weights.json');

if (!existsSync(logFile)) {
  console.error(`No loop log at ${logFile}. Run \`npm run train:loop\` first (or pass --log=PATH).`);
  process.exit(1);
}

// --- Parse the log into a list of runs, each holding its config + cycle verdicts. -------
// The loop writes three line shapes we care about (see train-loop.mjs `log(...)`):
//   [ts] train:loop start — batch N @ depth D | gate Gg @ depth D SPRT(0,E) | candidate hidden=[..] λ=L <warm/cold> ...
//   [ts] cycle N: PROMOTED ✓  candidate P% / Elo +E over champion (G games, cycle took T)...
//   [ts] cycle N: kept champion — candidate P% / Elo E (SPRT V, G games, cycle took T). <tail>
//   [ts] train:loop stopped after K promotion(s) in T...
//   [ts] Discarded stale lineage (reason).
const TS = /^\[([\d-]+ [\d:]+)\] (.*)$/;
const runs = [];
let cur = null;

for (const raw of readFileSync(logFile, 'utf8').split('\n')) {
  const tm = raw.match(TS);
  if (!tm) continue;
  const [, ts, body] = tm;

  if (body.startsWith('train:loop start')) {
    cur = {
      start: ts, end: null, stopped: null, promotions: 0, cycles: [],
      hidden: (body.match(/hidden=\[([^\]]*)\]/) || [, '?'])[1],
      batch: num(body.match(/batch (\d+)/)),
      gateGames: num(body.match(/gate (\d+)g/)),
      gateDepth: num(body.match(/gate \d+g @ depth (\d+)/)),
      elo1: num(body.match(/SPRT\(\d+,(\d+)\)/)),
      lambda: (body.match(/λ=([\d.]+)/) || [, null])[1],
      start_kind: /cold start/.test(body) ? 'cold'
        : /cold first cycle/.test(body) ? 'cold→warm'
        : /resuming lineage/.test(body) ? 'warm (resumed lineage)'
        : 'warm',
      lineageDiscarded: null,
    };
    runs.push(cur);
    continue;
  }
  if (!cur) continue;

  if (body.startsWith('Discarded stale lineage')) {
    cur.lineageDiscarded = (body.match(/\(([^)]+)\)/) || [, '?'])[1];
    continue;
  }
  if (body.startsWith('train:loop stopped')) {
    cur.stopped = (body.match(/after (\d+) promotion/) || [, '?'])[1];
    cur.end = ts;
    continue;
  }

  const cm = body.match(/^cycle (\d+): (PROMOTED|kept champion)/);
  if (cm) {
    const promoted = cm[2] === 'PROMOTED';
    const tail = /kept as lineage/.test(body) ? 'lineage+'
      : /Lineage reset/.test(body) ? 'lineage reset'
      : promoted ? 'PROMOTED' : '';
    cur.cycles.push({
      ts,
      n: Number(cm[1]),
      promoted,
      score: Number((body.match(/candidate ([\d.]+)%/) || [, NaN])[1]),
      elo: Number((body.match(/Elo ([+-]?\d+)/) || [, NaN])[1]),
      sprt: promoted ? 'H1' : (body.match(/SPRT (H0|inconclusive)/) || [, '?'])[1],
      games: num(body.match(/(\d+) games/)),
      dur: (body.match(/cycle took ([^)]+)\)/) || [, '?'])[1],
      tail,
    });
    cur.end = ts;
    if (promoted) cur.promotions++;
  }
}

function num(m) { return m ? Number(m[1]) : null; }

if (!runs.length) {
  console.log('Loop log has no runs yet (no `train:loop start` lines).');
  process.exit(0);
}

// --- Champion shape, to flag a candidate that can't warm-start from it. -----------------
let champHidden = null, champArch = null;
try {
  champArch = JSON.parse(readFileSync(championFile, 'utf8')).arch;
  if (Array.isArray(champArch) && champArch.length >= 3) champHidden = champArch.slice(1, -1).join(',');
} catch { /* placeholder/material champion — no arch */ }

// Least-squares slope of a numeric series (Elo points per cycle), for "climbing vs flat".
function slope(ys) {
  const n = ys.length;
  if (n < 2) return 0;
  const xs = ys.map((_, i) => i);
  const mx = (n - 1) / 2, my = ys.reduce((a, b) => a + b, 0) / n;
  let sxy = 0, sxx = 0;
  for (let i = 0; i < n; i++) { sxy += (xs[i] - mx) * (ys[i] - my); sxx += (xs[i] - mx) ** 2; }
  return sxx ? sxy / sxx : 0;
}
const avg = (a) => a.reduce((x, y) => x + y, 0) / a.length;
// Log timestamps are written in UTC (train-loop's stamp() uses toISOString). Date.parse on a
// no-offset datetime treats it as LOCAL, which on a machine behind UTC pushes the time into the
// future (negative "ago") — so parse it explicitly as UTC, then render local for display.
const parseLogTs = (ts) => new Date(ts.replace(' ', 'T') + 'Z');
const fmtLocal = (d) => {
  const p = (n) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
};
const pad = (s, w) => String(s).padEnd(w);
const padL = (s, w) => String(s).padStart(w);
const signed = (n) => (n >= 0 ? '+' : '') + n;

// --- Detailed cycle table for a run. ----------------------------------------------------
function printCycles(run) {
  if (!run.cycles.length) { console.log('  (no completed cycles)'); return; }
  console.log(`  ${pad('cyc', 5)}${pad('score', 8)}${pad('Elo', 6)}${pad('SPRT', 14)}${pad('games', 7)}${pad('took', 9)}lineage`);
  for (const c of run.cycles) {
    const mark = c.promoted ? '✓ ' : '  ';
    console.log('  ' + mark + pad(c.n, 3)
      + pad(Number.isFinite(c.score) ? c.score.toFixed(1) + '%' : '?', 8)
      + pad(Number.isFinite(c.elo) ? signed(c.elo) : '?', 6)
      + pad(c.sprt, 14)
      + pad(c.games ?? '?', 7)
      + pad(c.dur, 9)
      + c.tail);
  }
}

// --- Output. ----------------------------------------------------------------------------
console.log(`\nAposChess train:loop progress — ${relish(logFile)}`);
if (champHidden) console.log(`Current champion: arch [${champArch.join(',')}]  (hidden [${champHidden}])`);

const latest = runs[runs.length - 1];
const running = isLatestRunning(latest);

console.log(`\n=== Latest run — started ${fmtLocal(parseLogTs(latest.start))}${running ? '  (RUNNING / not yet stopped)' : `  (stopped: ${latest.stopped} promotion(s))`} ===`);
console.log(`  config: hidden=[${latest.hidden}]  gate ${latest.gateGames}g@d${latest.gateDepth} SPRT(0,${latest.elo1})`
  + `  λ=${latest.lambda ?? '?'}  batch ${latest.batch ?? '?'}  ${latest.start_kind} start`);
if (latest.lineageDiscarded) console.log(`  note: lineage discarded at start (${latest.lineageDiscarded}) — accumulation restarted from scratch.`);
console.log('');
printCycles(latest);

// Per-run read: is the candidate trending toward the gate, and can it even warm-start?
const reads = readRun(latest);
console.log('\n  Read:');
for (const r of reads) console.log(`    • ${r}`);

// --- Compact history of recent runs. ----------------------------------------------------
if (runs.length > 1) {
  console.log(`\n=== Run history (last ${Math.min(histN, runs.length)} of ${runs.length}) ===`);
  for (const run of runs.slice(-histN)) {
    const scores = run.cycles.map((c) => c.score).filter(Number.isFinite);
    const best = scores.length ? Math.max(...scores).toFixed(1) + '%' : '—';
    const tag = run === latest && running ? 'RUN' : (run.stopped ?? '·');
    console.log(`  ${pad(fmtLocal(parseLogTs(run.start)), 20)} h[${pad(run.hidden, 9)}] ${pad(run.start_kind, 22)} `
      + `${padL(run.cycles.length, 2)}cyc  best ${padL(best, 6)}  prom ${run.promotions}  ${tag === 'RUN' ? 'running' : tag + ' prom'}`);
  }
  if (detailAll) {
    for (const run of runs.slice(0, -1)) {
      console.log(`\n--- run ${fmtLocal(parseLogTs(run.start))} (hidden=[${run.hidden}], ${run.start_kind}) ---`);
      printCycles(run);
    }
  }
}

// --- Overall: promotions and how long since the last one. -------------------------------
const allCycles = runs.flatMap((r) => r.cycles);
const totalProm = runs.reduce((a, r) => a + r.promotions, 0);
let sincePromo = 0, lastPromo = null;
for (let i = allCycles.length - 1; i >= 0; i--) {
  if (allCycles[i].promoted) { lastPromo = allCycles[i]; break; }
  sincePromo++;
}
console.log('\n=== Overall ===');
console.log(`  ${allCycles.length} cycle(s) across ${runs.length} run(s);  ${totalProm} promotion(s).`);
if (lastPromo) {
  const promoAt = parseLogTs(lastPromo.ts);
  const ago = (Date.now() - promoAt.getTime()) / 1000;
  console.log(`  Last promotion: ${fmtLocal(promoAt)} (Elo ${signed(lastPromo.elo)}) — ${fmtDur(ago)} ago, ${sincePromo} cycle(s) since.`);
} else {
  console.log('  No promotions yet in this log.');
}
console.log('');

// Is the latest run still going? The loop appends a "stopped" line on exit; if the run has
// cycles but no stopped line AND a live pidfile exists, treat it as running.
function isLatestRunning(run) {
  if (run.stopped !== null) return false;
  const pidFile = join(loopDir, 'loop.pid');
  if (!existsSync(pidFile)) return false;
  const pid = Number(readFileSync(pidFile, 'utf8').trim());
  try { process.kill(pid, 0); return true; } catch (e) { return e.code === 'EPERM'; }
}

// Heuristic continue-vs-restart read for a run, from its cycle scores + config. These are
// the same checks worth eyeballing by hand; spelled out so the report is actionable.
function readRun(run) {
  const out = [];
  const scores = run.cycles.map((c) => c.score).filter(Number.isFinite);
  const elos = run.cycles.map((c) => c.elo).filter(Number.isFinite);

  // Shape mismatch: a candidate whose hidden ≠ champion's can't warm-start from the champion,
  // so it relearns from scratch every cycle and tends to sit well below 50%. (Bit us before.)
  if (champHidden && run.hidden !== '?' && run.hidden !== champHidden) {
    out.push(`Candidate hidden=[${run.hidden}] ≠ champion hidden=[${champHidden}]: it can't warm-start `
      + `from the champion, so it trains from scratch against a strong net. Match the shape to let warm cycles accumulate.`);
  }
  if (run.batch === 0) {
    out.push('batch 0 — generation is OFF this run; no fresh self-play data, only gate harvest + v-refresh. '
      + 'Set --batch (e.g. 200) to feed the net new signal.');
  }
  if (run.start_kind === 'cold→warm') {
    out.push('cold-first start: only cycle 1 is random-init — every later cycle warm-starts from the PREVIOUS '
      + 'cycle\'s candidate, so the chain refines one fresh net rather than relearning each cycle. It still never '
      + 'inherits the champion\'s weights, so it can trail a same/larger-arch champion until it catches up.');
  } else if (run.start_kind === 'cold') {
    out.push('cold start (legacy logs, pre-"only cold on first cycle"): every cycle relearned from random init — expect low scores throughout.');
  }

  if (!scores.length) { out.push('No completed cycles yet.'); return out; }
  const recent = scores.slice(-5);
  const a = avg(recent), best = Math.max(...scores);
  const sl = slope(elos.slice(-6)); // Elo points/cycle over the recent window
  const trend = sl > 1.5 ? 'climbing' : sl < -1.5 ? 'falling' : 'flat';
  out.push(`Recent ${recent.length}-cycle avg ${a.toFixed(1)}%, best ${best.toFixed(1)}%, trend ${trend} (${signed(+sl.toFixed(1))} Elo/cycle).`);

  const anyLineage = run.cycles.some((c) => c.tail === 'lineage+');
  if (run.promotions > 0) {
    out.push('Promoted this run — the champion improved. Expect the next candidates to dip (stronger target) before climbing again.');
  } else if (a < 49 && trend !== 'climbing') {
    out.push('Verdict: candidates are losing and not climbing — NOT productive. Restart with the champion\'s shape, warm, generation on.');
  } else if (best >= run.elo1 / 7 + 50 || (anyLineage && trend === 'climbing')) {
    // best% within reach of the gate, or lineage is accumulating upward
    out.push('Verdict: candidates are at/above 50% and accumulating via lineage — productive, let it keep running toward the gate.');
  } else {
    out.push('Verdict: candidates hover near 50% — marginal. Keep the warm/same-shape chain going a few more cycles, or add fresh generation.');
  }
  return out;
}

function relish(p) { try { return p.replace(repoDir + '\\', '').replace(repoDir + '/', ''); } catch { return p; } }
