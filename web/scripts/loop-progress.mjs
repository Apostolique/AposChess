// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2019-2026 Jean-David Moisan
//
// Read-only progress report for train:loop. It re-reads the persisted loop log
// (training/data/loop/loop.log) and turns the per-cycle verdict lines the loop already
// writes into a trend you can act on: is the latest run's candidate climbing toward the
// gate (let it run) or stuck below 50% (restart with different values)?
//
// LESSON FROM 'Mona' (the [128,64,32] champion, promoted 2026-07-12). A new/larger
// architecture can't warm-start from a differently-shaped champion, so its GATE EDGE sits
// below 50% for DOZENS of cycles even while the net is genuinely, steadily getting stronger.
// Mona's experiment track climbed ~434 → 547 in *absolute* Elo across 50 cycles / 3 runs,
// yet lost the gate (negative edge) on 48 of them — right up until the accumulating net
// finally overtook the champion and promoted at +41. So a below-50% run is NOT automatically
// a failed one: for an experiment track it's the expected early shape of a bootstrapping net.
// The real progress signal is the TRACK's absolute-Elo trajectory (experiment-registry.mjs,
// `npm run train:experiments`), which this report now folds in — not the gate score alone,
// which is measured against a moving (strengthening) champion opponent.
//
// RESTARTS ARE MERGED. Ctrl-C'ing the loop and relaunching it later with the same recipe
// writes a fresh `train:loop start` line and restarts the loop's cycle numbering at 1, but
// it is NOT a new experiment: a warm relaunch resumes the same track's lineage and keeps
// refining the same candidate chain. This report therefore groups consecutive log runs into
// one logical run when the newer one warm-starts the same recipe (same track id; for
// pre-registry logs, same hidden+λ), renumbering cycles cumulatively — so stopping the loop
// overnight no longer makes the trend/read start over from "cycle 1".
//
//   npm run train:progress           # latest run in detail + a one-line history + a read
//   npm run train:progress -- --runs=12   # show that many (merged) runs in the history
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
import { suggestRecipes, readHistory, trackDir } from './experiment-registry.mjs';

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
// The registry writes a `Recipe <slug> [<id>] — … (track run #N…)` line just before the
// matching `train:loop start` (train-loop.mjs beginRun). Capture it so each run knows which
// experiment TRACK it belongs to — that's how we reach the track's cross-run absolute-Elo
// trajectory (the real progress signal). Runs before the registry existed have no such line.
const RECIPE = /^Recipe (\S+) \[([0-9a-f]+)\].*\(track run #(\d+)/;
const TS = /^\[([\d-]+ [\d:]+)\] (.*)$/;
const runs = [];
let cur = null;
let pendingTrack = null; // {slug, id, run} awaiting the next `train:loop start`

for (const raw of readFileSync(logFile, 'utf8').split('\n')) {
  const tm = raw.match(TS);
  if (!tm) continue;
  const [, ts, body] = tm;

  const rm = body.match(RECIPE);
  if (rm) { pendingTrack = { slug: rm[1], id: rm[2], run: Number(rm[3]) }; continue; }

  if (body.startsWith('train:loop start')) {
    cur = {
      start: ts, end: null, stopped: null, promotions: 0, cycles: [],
      track: pendingTrack, // null for pre-registry runs
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
    pendingTrack = null;
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

// --- Merge warm same-recipe relaunches into logical runs. --------------------------------
// A relaunch continues the previous run's chain when it warm-starts (a cold/cold-first
// relaunch deliberately begins a fresh chain) and trains the same recipe: same experiment
// track id when both runs have one; across the registry boundary we don't guess; for two
// pre-registry runs the best available key is hidden+λ. A legacy lineage-discard note means
// accumulation restarted from scratch, so it breaks the chain too.
function continuesChain(prev, run) {
  if (!run.start_kind.startsWith('warm')) return false;
  if (run.lineageDiscarded) return false;
  if (prev.track && run.track) return prev.track.id === run.track.id;
  if (prev.track || run.track) return false;
  return prev.hidden === run.hidden && prev.lambda === run.lambda;
}
const chains = [];
for (const run of runs) {
  const tail = chains.length ? chains[chains.length - 1] : null;
  if (tail && continuesChain(tail[tail.length - 1], run)) tail.push(run);
  else chains.push([run]);
}
// Flatten a chain to the shape printCycles/readRun consume: cycles concatenated across the
// segments; start/start_kind from the FIRST launch (how the chain began); config + stopped
// state from the LATEST launch (what's on disk/running now).
function chainView(segs) {
  const first = segs[0], last = segs[segs.length - 1];
  return {
    segs,
    cycles: segs.flatMap((r) => r.cycles),
    start: first.start, end: last.end, stopped: last.stopped,
    promotions: segs.reduce((a, r) => a + r.promotions, 0),
    track: last.track || first.track,
    hidden: last.hidden, batch: last.batch, gateGames: last.gateGames,
    gateDepth: last.gateDepth, elo1: last.elo1, lambda: last.lambda,
    start_kind: first.start_kind,
    lineageDiscarded: first.lineageDiscarded,
  };
}
const views = chains.map(chainView);

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

// The cross-run absolute-Elo trajectory for a run's experiment track. Unlike the gate score
// (which is measured against a champion that strengthens over time, so it can't be compared
// cycle-to-cycle), the track's per-cycle `absElo` — championLedgerElo + gate edge — is a stable
// yardstick: a candidate can be steadily improving in absolute Elo while still losing the gate,
// which is exactly the shape of a new-architecture bootstrap (the 'Mona' story). Returns null
// when the run predates the registry or the track has no absElo yet.
function trackTrajectory(run) {
  if (!run.track) return null;
  const hist = readHistory(trackDir(loopDir, run.track.id));
  const abs = hist.map((h) => h.absElo).filter(Number.isFinite);
  if (abs.length < 2) return null;
  return {
    slug: run.track.slug,
    id: run.track.id,
    cycles: hist.length,
    runs: new Set(hist.map((h) => h.run)).size,
    promotions: hist.filter((h) => h.promoted).length,
    firstAbs: abs[0],
    latestAbs: abs[abs.length - 1],
    bestAbs: Math.max(...abs),
    absSlope: slope(abs), // whole-track Elo/cycle
    recentAbsSlope: slope(abs.slice(-8)), // recent-window Elo/cycle
  };
}
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

// --- Detailed cycle table for a (merged) run. --------------------------------------------
// Cycle numbers are CUMULATIVE across the chain's launches; a separator marks where each
// warm relaunch picked the chain back up. The loop now logs track-cumulative numbers itself
// (a warm relaunch continues where the track left off), so a logged number that moves FORWARD
// is trusted as-is — matching what the console showed live — while a launch that resets to 1
// (logs from before the change) is renumbered to continue the chain.
function printCycles(view) {
  if (!view.cycles.length) { console.log('  (no completed cycles)'); return; }
  console.log(`  ${pad('cyc', 5)}${pad('score', 8)}${pad('Elo', 6)}${pad('SPRT', 14)}${pad('games', 7)}${pad('took', 9)}lineage`);
  let k = 0;
  view.segs.forEach((run, i) => {
    if (i > 0) console.log(`  ── resumed ${fmtLocal(parseLogTs(run.start))}${run.cycles.length ? '' : ' (no completed cycles)'} ──`);
    for (const c of run.cycles) {
      k = Math.max(k + 1, c.n);
      const mark = c.promoted ? '✓ ' : '  ';
      console.log('  ' + mark + pad(k, 3)
        + pad(Number.isFinite(c.score) ? c.score.toFixed(1) + '%' : '?', 8)
        + pad(Number.isFinite(c.elo) ? signed(c.elo) : '?', 6)
        + pad(c.sprt, 14)
        + pad(c.games ?? '?', 7)
        + pad(c.dur, 9)
        + c.tail);
    }
  });
}

// --- Output. ----------------------------------------------------------------------------
console.log(`\nAposChess train:loop progress — ${relish(logFile)}`);
if (champHidden) console.log(`Current champion: arch [${champArch.join(',')}]  (hidden [${champHidden}])`);

const latest = views[views.length - 1];
const running = isLatestRunning(latest.segs[latest.segs.length - 1]);

console.log(`\n=== Latest run — started ${fmtLocal(parseLogTs(latest.start))}`
  + `${latest.segs.length > 1 ? `, resumed ${latest.segs.length - 1}× (warm same-recipe relaunches merged)` : ''}`
  + `${running ? '  (RUNNING / not yet stopped)' : `  (stopped: ${latest.promotions} promotion(s))`} ===`);
console.log(`  config: hidden=[${latest.hidden}]  gate ${latest.gateGames}g@d${latest.gateDepth} SPRT(0,${latest.elo1})`
  + `  λ=${latest.lambda ?? '?'}  batch ${latest.batch ?? '?'}  ${latest.start_kind} start`);
const drifted = ['batch', 'gateGames', 'gateDepth', 'elo1']
  .filter((key) => new Set(latest.segs.map((r) => String(r[key]))).size > 1);
if (drifted.length) console.log(`  note: ${drifted.join(', ')} changed between relaunches — config above is the latest launch's.`);
if (latest.lineageDiscarded) console.log(`  note: lineage discarded at start (${latest.lineageDiscarded}) — accumulation restarted from scratch.`);
console.log('');
printCycles(latest);

// Track trajectory: the cross-run absolute-Elo climb this run is contributing to. This is the
// real progress yardstick (see the 'Mona' note at the top) — a run can look flat/losing at the
// gate while its track's absolute Elo climbs steadily toward overtaking the champion.
const latestTraj = trackTrajectory(latest);
if (latestTraj) {
  const t = latestTraj;
  const dir = t.absSlope > 1.0 ? 'climbing' : t.absSlope < -1.0 ? 'falling' : 'flat';
  console.log(`\n  Track [${t.id}] ${t.slug} — ${t.runs} run(s), ${t.cycles} cycle(s) accumulated:`);
  console.log(`    absolute Elo ${t.firstAbs.toFixed(0)} → ${t.latestAbs.toFixed(0)} `
    + `(best ${t.bestAbs.toFixed(0)}), ${signed(+t.absSlope.toFixed(1))} Elo/cycle overall, ${dir}.`);
  console.log('    Absolute Elo is the real signal for a warm-starting track — the gate edge above can stay');
  console.log('    negative for dozens of cycles while this climbs (that is how \'Mona\' promoted).');
}

// Per-run read: is the candidate trending toward the gate, and can it even warm-start?
const reads = readRun(latest);
console.log('\n  Read:');
for (const r of reads) console.log(`    • ${r}`);

// --- Compact history of recent (merged) runs. --------------------------------------------
if (views.length > 1) {
  console.log(`\n=== Run history (last ${Math.min(histN, views.length)} of ${views.length}; warm same-recipe relaunches merged) ===`);
  const shown = views.slice(-histN);
  const hiddenW = Math.max(9, ...shown.map((v) => String(v.hidden).length));
  for (const view of shown) {
    const scores = view.cycles.map((c) => c.score).filter(Number.isFinite);
    const best = scores.length ? Math.max(...scores).toFixed(1) + '%' : '—';
    const state = view === latest && running ? 'running' : `${view.promotions} prom`;
    console.log(`  ${pad(fmtLocal(parseLogTs(view.start)), 20)} h[${pad(view.hidden, hiddenW)}] ${pad(view.start_kind, 22)} `
      + `${padL(view.segs.length, 2)}×  ${padL(view.cycles.length, 3)}cyc  best ${padL(best, 6)}  ${state}`);
  }
  if (detailAll) {
    for (const view of views.slice(0, -1)) {
      console.log(`\n--- run ${fmtLocal(parseLogTs(view.start))} (hidden=[${view.hidden}], ${view.start_kind}`
        + `${view.segs.length > 1 ? `, ${view.segs.length} launches` : ''}) ---`);
      printCycles(view);
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
console.log(`  ${allCycles.length} cycle(s) across ${views.length} run(s)`
  + `${runs.length !== views.length ? ` (${runs.length} loop launches — warm same-recipe relaunches merged)` : ''};  ${totalProm} promotion(s).`);
if (lastPromo) {
  const promoAt = parseLogTs(lastPromo.ts);
  const ago = (Date.now() - promoAt.getTime()) / 1000;
  console.log(`  Last promotion: ${fmtLocal(promoAt)} (Elo ${signed(lastPromo.elo)}) — ${fmtDur(ago)} ago, ${sincePromo} cycle(s) since.`);
} else {
  console.log('  No promotions yet in this log.');
}
console.log('');

// When the champion has stalled (many cycles since the last promotion, or none yet), point at
// the experiment registry for concrete next moves: promising-but-stalled recipes worth reviving
// (they warm-start from a saved best) and architectures with no track yet. Read-only, pulled
// from training/data/loop/experiments — the full view is `npm run train:experiments`.
const stalled = totalProm === 0 ? allCycles.length >= 3 : sincePromo >= 6;
if (stalled) {
  const sugg = suggestRecipes(loopDir, { champHidden });
  const resume = sugg.filter((s) => s.kind === 'resume').slice(0, 2);
  const fresh = sugg.filter((s) => s.kind === 'new').slice(0, 2);
  if (resume.length || fresh.length) {
    console.log('=== Stalled? Ideas to try (full list: `npm run train:experiments`) ===');
    for (const s of resume) console.log(`  • revive ${s.slug} — ${s.reason}\n      ${s.cmd}`);
    for (const s of fresh) console.log(`  • try ${s.slug} (architecture never tried)\n      ${s.cmd}`);
    console.log('');
  }
}

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
  const traj = trackTrajectory(run);
  // Is the track's absolute Elo climbing? That's the productivity signal for a below-50% run.
  const trackClimbing = traj && traj.absSlope > 1.0;

  // Shape mismatch: a candidate whose hidden ≠ champion's can't warm-start FROM THE CHAMPION.
  // Pre-registry that meant relearning from scratch every cycle (a real stall). With the
  // experiment registry it instead warm-starts from its OWN track's lineage/best and accumulates
  // — so sub-50% is the EXPECTED early shape of a bootstrapping new-arch track, not a failure.
  // (This exact case produced 'Mona': [128,64,32] bootstrapped over 50 cycles vs a [64,32,16]
  // champion, losing the gate throughout until its absolute Elo finally overtook.)
  if (champHidden && run.hidden !== '?' && run.hidden !== champHidden) {
    if (run.track) {
      out.push(`Candidate hidden=[${run.hidden}] ≠ champion hidden=[${champHidden}]: it can't warm-start from the `
        + `champion, but this is a registered experiment track warm-starting from its own lineage/best — so a `
        + `below-50% gate is expected while it bootstraps. Judge it by the track's absolute-Elo trend, not the gate edge.`);
    } else {
      out.push(`Candidate hidden=[${run.hidden}] ≠ champion hidden=[${champHidden}]: it can't warm-start from the `
        + `champion and has no experiment track to accumulate from, so it relearns from scratch and tends to sit well `
        + `below 50%. Run it via train:loop (which keys a track) so warm cycles accumulate, or match the champion's shape.`);
    }
  }
  if (traj) {
    const dir = traj.absSlope > 1.0 ? 'climbing' : traj.absSlope < -1.0 ? 'falling' : 'flat';
    out.push(`Track [${traj.id}]: absolute Elo ${traj.firstAbs.toFixed(0)} → ${traj.latestAbs.toFixed(0)} `
      + `(best ${traj.bestAbs.toFixed(0)}) over ${traj.cycles} cycle(s) / ${traj.runs} run(s), `
      + `${signed(+traj.absSlope.toFixed(1))} Elo/cycle — ${dir}. This, not the gate score, is the progress signal.`);
  }
  if (run.batch === 0) {
    out.push('batch 0 — no dedicated self-play generation; fresh data comes from the gate harvest '
      + '(~2000 near-champion games/cycle) + the ranked pool\'s strong-engine --play games. This is a '
      + 'supported mode (the pool is the generator), not a stall — watch the promotion trend below. '
      + 'Set --batch (e.g. 200) to add a deep champion self-play batch on top.');
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
  } else if (trackClimbing && a < 49) {
    // Below 50% at the gate BUT the track's absolute Elo is climbing — the 'Mona' pattern. Do
    // NOT call this a failed run: a bootstrapping new-arch track loses the gate for dozens of
    // cycles before overtaking. Keep it running (across restarts — the track resumes its lineage).
    out.push(`Verdict: losing the gate but the track's absolute Elo is CLIMBING (${signed(+traj.absSlope.toFixed(1))} Elo/cycle) `
      + '— productive. This is the new-architecture bootstrap pattern that produced \'Mona\'; keep it running (it '
      + 'resumes its lineage across restarts) — it overtakes the champion only once its absolute Elo passes the champion\'s.');
  } else if (a < 49 && trend !== 'climbing' && !traj) {
    out.push('Verdict: candidates are losing and not climbing — NOT productive. Restart with the champion\'s shape, warm, and a fresh data source (a deep --batch generation, or more strong-engine --rank-minutes pool play).');
  } else if (a < 49 && !trackClimbing && traj) {
    out.push(`Verdict: below 50% AND the track's absolute Elo is flat/falling (${signed(+traj.absSlope.toFixed(1))} Elo/cycle) — `
      + 'this track has stopped improving, not just losing a strong gate. Consider a different recipe (see the ideas below) or fresh data.');
  } else if (best >= run.elo1 / 7 + 50 || (anyLineage && trend === 'climbing')) {
    // best% within reach of the gate, or lineage is accumulating upward
    out.push('Verdict: candidates are at/above 50% and accumulating via lineage — productive, let it keep running toward the gate.');
  } else {
    out.push('Verdict: candidates hover near 50% — marginal. Keep the warm/same-shape chain going a few more cycles, or add fresh data (a deep --batch generation, or more strong-engine --rank-minutes pool play).');
  }
  return out;
}

function relish(p) { try { return p.replace(repoDir + '\\', '').replace(repoDir + '/', ''); } catch { return p; } }
