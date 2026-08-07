// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2019-2026 Jean-David Moisan
//
// Read out the shadow-mode low-depth SCREEN experiment (`npm run train:loop -- --screen`).
//
//   npm run screen:report                # every recorded screen/gate pair
//   npm run screen:report -- --elo1=20   # score the rule against a different promotion bound
//   npm run screen:report -- --loop-dir=PATH
//
// THE QUESTION THIS ANSWERS. A full 2000-game gate at depth 6 costs ~3.5-4h — most of a loop
// cycle. Depth 1 is ~414x cheaper per game, and across this engine's champion sequence depth-1
// Elo ranks depth-8 Elo at Spearman 0.974, with the cumulative d1/d6 gain ratio steady at
// ~0.62. So a cheap screen ahead of the gate looks like a 3-4x speedup. But that ratio is a
// POPULATION slope measured on champions separated by 150+ Elo, and a screen has to work on
// near-clones separated by ~10-20. The quantity that decides it is the PER-CANDIDATE residual
// scatter around the transfer line, and the Bradley-Terry ladder cannot measure it: its own
// +/-31 (d1) and +/-35 (d6) Elo margins already account for more scatter than the residual it
// shows. Rejected candidates are never archived, so there is no offline way to get the paired
// data either. Hence shadow mode — the loop measures it in place, one pair per cycle.
//
// WHY THE ANSWER MATTERS EITHER WAY. --gate-futility already stops the clearly-bad candidates
// cheaply (recent cycles stopped at 134 and 613 games). The gate's cost is concentrated on the
// AMBIGUOUS candidates — the 51-52% ones that burn the full cap on a verdict of "inconclusive".
// Those are exactly the candidates a high-scatter screen cannot resolve. So:
//   sigma_transfer small (<~5 Elo)  -> the screen resolves +11 vs +20 in ~5 min: big speedup.
//   sigma_transfer large (~20 Elo)  -> the screen only duplicates futility: not worth wiring in.
// The report prints sigma_transfer with the noise subtracted out, and simulates the rule.
//
// Read-only: it never writes, spawns, or touches the loop's state, so it is safe against a
// live loop.

import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const webDir = resolve(here, '..');
const repoDir = resolve(webDir, '..');

const args = Object.fromEntries(
  process.argv.slice(2).map((a) => {
    const m = a.replace(/^--/, '').split('=');
    return [m[0], m.length > 1 ? m[1] : true];
  }),
);

const loopDir = typeof args['loop-dir'] === 'string'
  ? resolve(args['loop-dir'])
  : resolve(repoDir, 'training', 'data', 'loop');
const ELO1 = args.elo1 !== undefined ? Number(args.elo1) : 20;

// --- Collect the pairs ---------------------------------------------------------------------
// One observation per cycle that ran a screen: the screen's prediction and the gate's verdict
// for the SAME candidate. Tracks are per-recipe directories; a screen pair is valid wherever
// it was recorded, so they all pool.
const expDir = join(loopDir, 'experiments');
if (!existsSync(expDir)) {
  console.error(`No experiment registry at ${expDir}. Nothing to report.`);
  process.exit(1);
}

const pairs = [];
for (const d of readdirSync(expDir, { withFileTypes: true })) {
  if (!d.isDirectory()) continue;
  const hist = join(expDir, d.name, 'history.jsonl');
  if (!existsSync(hist)) continue;
  for (const line of readFileSync(hist, 'utf8').split('\n')) {
    if (!line.trim()) continue;
    let e; try { e = JSON.parse(line); } catch { continue; }
    if (!e.screen || typeof e.edgeElo !== 'number') continue;
    pairs.push({ track: d.name, ...e });
  }
}
pairs.sort((a, b) => String(a.ts).localeCompare(String(b.ts)));

if (!pairs.length) {
  console.log('\nNo shadow-screen observations recorded yet.\n');
  console.log('The screen is on by default, so just run the loop for ~10 cycles:');
  console.log('  npm run train:loop\n');
  console.log('Each cycle adds one (screen prediction, gate verdict) pair at ~5 min of overhead.');
  console.log('Only cycles that reached a gate verdict count — an interrupted cycle records nothing.');
  process.exit(0);
}

// --- Fit the transfer line ------------------------------------------------------------------
// gate edge ~ a + b * screen edge. `b` IS the transfer factor (the reciprocal of the ratio the
// loop assumed), re-estimated from the population a screen actually faces rather than from the
// well-separated champions the ladder offers.
const x = pairs.map((p) => p.screen.elo);
const y = pairs.map((p) => p.edgeElo);
const n = pairs.length;
const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
const mx = mean(x), my = mean(y);
let sxy = 0, sxx = 0, syy = 0;
for (let i = 0; i < n; i++) { sxy += (x[i] - mx) * (y[i] - my); sxx += (x[i] - mx) ** 2; syy += (y[i] - my) ** 2; }
const slope = sxx > 0 ? sxy / sxx : NaN;
const intercept = my - slope * mx;
const r2 = sxx > 0 && syy > 0 ? (sxy * sxy) / (sxx * syy) : NaN;

// Residual scatter, and the part of it that is NOT the two matches' own sampling error.
// Each pair carries its own error bars: sigma = (hi - lo) / (2 * 1.96). The gate's is often the
// bigger of the two (a futility-stopped gate can be a few hundred games), which is exactly why
// subtracting it matters — scoring a screen against a noisy "truth" makes it look worse than
// it is. What survives the subtraction is the per-candidate transfer scatter.
const sd = (lo, hi) => (lo == null || hi == null ? null : (hi - lo) / (2 * 1.96));
let ssRes = 0, noiseVar = 0, nNoise = 0;
for (let i = 0; i < n; i++) {
  ssRes += (y[i] - (intercept + slope * x[i])) ** 2;
  const sg = sd(pairs[i].edgeLo, pairs[i].edgeHi);
  const ss = sd(pairs[i].screen.eloLo, pairs[i].screen.eloHi);
  if (sg != null && ss != null) { noiseVar += sg * sg + (slope * ss) ** 2; nNoise++; }
}
const residRmse = Math.sqrt(ssRes / Math.max(1, n - 2));
const meanNoise = nNoise ? noiseVar / nNoise : null;
const sigmaTransfer = meanNoise == null ? null
  : Math.sqrt(Math.max(0, (ssRes / Math.max(1, n - 2)) - meanNoise));

// --- Report ---------------------------------------------------------------------------------
const f = (v, w = 6, p = 0) => (v == null || Number.isNaN(v) ? '—'.padStart(w) : v.toFixed(p).padStart(w));
console.log(`\nShadow-screen report — ${n} paired observation(s) across ${new Set(pairs.map((p) => p.track)).size} track(s)\n`);
console.log('  screen (cheap)          gate (truth)           screen rule');
console.log('  depth  games   Elo      games   Elo            predicted        would    gate');
console.log('  ─────  ─────  ─────     ─────  ─────           ───────────────  ──────   ──────');
for (const p of pairs) {
  const s = p.screen;
  const promoted = p.promoted;
  console.log(`  d${String(s.depth).padEnd(4)} ${String(s.games).padStart(6)} ${f(s.elo, 6)}    `
    + `${String(p.games ?? '?').padStart(6)} ${f(p.edgeElo, 6)}          `
    + `${f(s.predElo, 5)} [${f(s.predLo, 4)},${f(s.predHi, 5)}]  `
    + `${(s.wouldReject ? 'reject' : 'escal.').padEnd(8)} ${promoted ? 'PROMOTED' : p.sprt}`);
}

// The null the screen has to beat: predicting every candidate's gate edge as the mean of all of
// them, i.e. learning nothing from the screen at all. Same idiom as surrogateReport's
// leave-one-out-vs-mean check in experiment-registry.mjs, and for the same reason — a residual
// only means something next to the baseline it improves on. A screen whose residual matches the
// baseline has told you nothing, however tight its own error bars are.
const baseRmse = Math.sqrt(syy / Math.max(1, n - 1));
const gain = baseRmse > 0 ? 1 - residRmse / baseRmse : 0;

console.log(`\n  Transfer fit:  gate = ${intercept.toFixed(1)} + ${slope.toFixed(2)} x screen   (R² = ${Number.isNaN(r2) ? '—' : r2.toFixed(3)})`);
console.log(`  Implied ratio (screen:gate Elo) = ${(1 / slope).toFixed(2)}`
  + `   [the loop assumed ${pairs[pairs.length - 1].screen.ratio}]`);
console.log(`\n  Predictive scatter — how tightly the screen pins the gate outcome:`);
console.log(`    from the screen      ${residRmse.toFixed(1)} Elo`);
console.log(`    predicting the mean  ${baseRmse.toFixed(1)} Elo   → the screen is `
  + `${gain > 0 ? `${(100 * gain).toFixed(0)}% better` : 'NO better'} than knowing nothing`);
if (meanNoise != null) {
  const floor = Math.sqrt(meanNoise);
  console.log(`    measurement floor    ${floor.toFixed(1)} Elo   (the two matches' own sampling error)`);
  if (sigmaTransfer > 0) {
    console.log(`    → per-candidate transfer scatter σ ≈ ${sigmaTransfer.toFixed(1)} Elo, net of that floor`);
  } else {
    // Flooring at 0 is not a measurement of 0 — it means the pairs agree to within the noise,
    // so all this run can say is "σ is somewhere below the floor". Saying "σ = 0.0" would
    // overclaim in the direction of building the cascade, which is the expensive mistake here.
    console.log('    → transfer scatter is BELOW the measurement floor: the screen and the gate agree');
    console.log(`      to within their own error bars, so σ is bounded by ~${floor.toFixed(1)} Elo, not resolved.`);
    console.log('      Longer gates (--gate-games) or more pairs would tighten it.');
  }
}

// --- Simulate the rule ------------------------------------------------------------------------
// The decision a cascade would make: skip the gate when the screen's 95% upper bound falls short
// of the promotion threshold. Two error rates matter, and they are NOT symmetric. A false reject
// throws away a real promotion (costly, and invisible — you never learn it happened). A gate
// correctly skipped is ~3.5h saved. So the rule is judged on false rejects first.
let rejected = 0, falseRejects = 0, promotable = 0, gateGamesSaved = 0, gateGamesTotal = 0;
for (const p of pairs) {
  const couldPromote = p.promoted || p.edgeElo >= ELO1;
  if (couldPromote) promotable++;
  gateGamesTotal += p.games ?? 0;
  if (p.screen.wouldReject) {
    rejected++;
    gateGamesSaved += p.games ?? 0;
    if (couldPromote) falseRejects++;
  }
}
console.log(`\n  Rule: skip the gate when the screen's 95% upper bound < +${ELO1} Elo`);
console.log(`    would have skipped   ${rejected}/${n} gate(s)  (${((100 * rejected) / n).toFixed(0)}% of cycles, `
  + `${((100 * gateGamesSaved) / Math.max(1, gateGamesTotal)).toFixed(0)}% of gate games)`);
console.log(`    FALSE REJECTS        ${falseRejects}/${promotable} candidate(s) that reached +${ELO1} at the gate`);

// --- Verdict ----------------------------------------------------------------------------------
// Deliberately conservative about its own sample size: with a handful of pairs the scatter
// estimate is itself mostly noise, and this is a decision worth not getting wrong.
console.log('');
if (n < 8) {
  console.log(`  VERDICT: too early — ${n} pair(s). The scatter estimate needs ~10+ to mean anything.`);
  console.log('           Keep running with --screen.');
} else if (falseRejects > 0) {
  // The empirical check outranks every model-based one: a screen that actually threw away a
  // promotion is disqualified whatever the fitted scatter says.
  console.log('  VERDICT: the screen rejected a candidate the gate would have promoted. A cascade');
  console.log('           would have cost a real promotion — do not wire it in on this evidence.');
} else if (gain < 0.15) {
  console.log(`  VERDICT: the screen barely beats predicting the mean (${(100 * gain).toFixed(0)}%), so it is not`);
  console.log('           reading the candidate — it is reading the population. No cascade.');
  console.log('           Worth one retry at --screen-depth=2 (still ~96x cheaper than d6, and it');
  console.log('           transfers with less compression) before abandoning the idea.');
} else if (residRmse <= 8) {
  console.log(`  VERDICT: the screen pins the gate to ±${residRmse.toFixed(1)} Elo and missed no promotion. A cascade`);
  console.log('           looks justified — screen first, gate only on escalation. Keep the gate as');
  console.log('           the promotion decision; the screen only ever skips a gate it can rule out.');
} else if (residRmse > ELO1) {
  console.log(`  VERDICT: the screen predicts the gate to ±${residRmse.toFixed(1)} Elo — wider than the +${ELO1} promotion`);
  console.log('           bound itself, so it cannot resolve the ambiguous candidates that actually');
  console.log('           cost gate time. --gate-futility already handles the clearly-bad ones.');
} else {
  console.log(`  VERDICT: mixed — ±${residRmse.toFixed(1)} Elo is real information but not decisive at a +${ELO1} bound.`);
  console.log('           Usable as a coarse pre-filter, not a replacement for the gate. Consider');
  console.log('           --screen-depth=2 (~96x cheaper than d6, less compression than d1).');
}
console.log('');
