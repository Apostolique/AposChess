// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2019-2026 Jean-David Moisan
//
// Read-only composition audit of the self-play dataset — what the trainer is actually being
// fed, measured against the champion net's own eval `scale` and against the published NNUE
// dataset criteria.
//
//   npm run data:audit                      # full pass over ../training/data/selfplay.jsonl
//   npm run data:audit -- --quick=20000     # stop after N games, for a fast look
//   npm run data:audit -- --sample=200      # replay 1 game in N for the material histogram
//   npm run data:audit -- --data=FILE       # audit a different *.jsonl dataset
//   npm run data:audit -- --weights=FILE    # read `scale` from a different net
//
// WHY THIS EXISTS. Every other report in this repo reads the LOOP's output — gate scores,
// track trajectories, the Elo ledger — and none of them can see the dataset underneath. An
// ad-hoc pass over the 372,459-game / 34,810,437-position corpus (2026-08-08) found three
// things in that blind spot, which is why the pass is a tool now instead of a one-off script:
//
//  1. THE EVAL IS CENSORED AT ±`scale`. The nn eval is `tanh(z) * scale` and every one of the
//     archived champions carries `scale = 600` (it is inherited through every warm start, so
//     `train.py --scale` is inert without `--rescale`). So the eval can never report more than
//     600 cp — less than a queen (900) in a variant where a knight is worth about as much as a
//     rook (500). The recorded search values pile up against that ceiling: measured in 50 cp
//     bins, 6.5% in 400-450, 11.1% in 450-500, 13.1% in 500-550, 10.1% in 550-600, and then
//     0.05% above it. Above the ceiling the label carries no information about HOW far ahead
//     the position is, and just below it the head's own local gain — 1 − (v/scale)², the
//     derivative of the tanh squash at that output — is down to 0.55 at 0.67·scale and to 0.02
//     at 0.99·scale, so those labels contribute loss but very little gradient. 41.3% of
//     positions sit in that band, plus 6.8% mate scores, which are outside the scale entirely.
//     This report reads `scale` off the weights file rather than hardcoding 600, so it stays
//     correct when the scale moves.
//
//  2. THE CORPUS IS FAR TOO DECIDED for the criteria in Tan & Watkinson, "Study of the Proper
//     NNUE Dataset" (arXiv:2412.17948), which are stated on eval values from the side-to-move
//     view: roughly 50% positive / 50% negative, at least 50% of positions within ±100 cp, and
//     at least 40% materially imbalanced. The sign split is fine (53.1% / 44.4%, the rest
//     exactly zero) and the material criterion passes comfortably, but only 22.2% of positions
//     land within ±100 cp against a ≥50% target — 77.8% are further out than that.
//
//  3. 40.3% OF ALL RECORDED PLIES COME AFTER THE GAME IS ALREADY DECIDED. For each game this
//     finds the earliest ply from which |v| stays past a threshold with one consistent sign all
//     the way to the end, and counts what follows. At ±400 cp, 96.0% of games reach such a
//     point, 14,023,838 plies (40.3%) come after it, and that sign agrees with the final result
//     99.3% of the time. Those plies are what pushes the ±100 cp share down and the saturated
//     share up. Note the ceiling interacts with the measure itself: at a threshold of 600 the
//     figure collapses to 5.8% of plies, not because the games are less decided but because an
//     nn label cannot exceed the ceiling — which is why several thresholds are reported.
//
// WHERE THIS TOOL DISAGREES WITH THAT AD-HOC PASS. The corpus totals, the ±100 cp share, the
// sign split, the post-decision numbers and the result/length figures all reproduce it exactly.
// Three of its saturation figures do not, and the difference is in how the ceiling is counted,
// so the tool's definitions are spelled out rather than left implicit:
//   - Saturation zone 41.3% here vs 32.3% there. 32.3% is what the ad-hoc histogram's own bins
//     sum to from 450 up (0.75·scale); 41.3% is the stated |v| > 0.67·scale = 402 threshold,
//     which is the definition this tool implements.
//   - "7.7% mate scores" is really every label off the tanh scale: mates are 6.77% and the
//     0.89% above the ceiling brings it to 7.66%. Both rows are printed, separately.
//   - "0.11% in 600-650" mixes two different things. `tanh(z)·600` ROUNDS to exactly ±600 for a
//     confident-enough position — 61,598 labels (0.18%) do — and those are clamped nn evals, not
//     values above the ceiling. Counting them in the bin below it, only 16,226 labels (0.05%)
//     genuinely exceed 600, and none of those can have come from an nn eval.
//   - The sampled material figures differ in the third digit (34.7% vs 35.4% within ±100 cp of
//     equality) because the two passes replayed different games; ordinary sampling variation.
//
// It reports the levers where a criterion fails, but it is careful about what it is claiming.
// The distribution is a MEASUREMENT. "This is why the loop plateaued" is a HYPOTHESIS: the only
// thing that settles it is training a net under a different scale / a decided-ply cut and
// gating it head-to-head with `npm run match`, like any other strength change.
//
// Nothing here writes, spawns, or mutates anything — it streams the JSONL line by line (the
// file is ~889 MB and must never be loaded whole) and reads the champion weights, so it is safe
// to run against a live `train:loop`. A full pass is ~18 s over the 889 MB file (plus the
// sampled replay); progress goes to stderr so stdout stays a clean report you can redirect.

import { createReadStream, existsSync, readFileSync, statSync } from 'node:fs';
import { createInterface } from 'node:readline';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { fmtDur, fmtNum, fmtMB, printWrapped, liveStatus, everyMs } from './fmt.mjs';
import { expandPositions } from './gameRecord.mjs';
import { _internal, SEARCH_ERA } from '../src/ai.js';

const here = dirname(fileURLToPath(import.meta.url));
const webDir = resolve(here, '..');
const repoDir = resolve(webDir, '..');

const args = Object.fromEntries(
  process.argv.slice(2).map((a) => {
    const m = a.replace(/^--/, '').split('=');
    return [m[0], m.length > 1 ? m[1] : true];
  }),
);
const dataFile = typeof args.data === 'string'
  ? resolve(args.data)
  : resolve(repoDir, 'training', 'data', 'selfplay.jsonl');
const weightsFile = typeof args.weights === 'string'
  ? resolve(args.weights)
  : resolve(webDir, 'src', 'nn-weights.json');
const sampleEvery = args.sample !== undefined ? Math.max(1, Math.round(Number(args.sample))) : 500;
const quickGames = args.quick !== undefined ? Math.max(1, Math.round(Number(args.quick))) : 0;

// Scores past this magnitude encode a forced mate rather than an evaluation, so they are
// outside the tanh scale and are counted separately everywhere below. Imported rather than
// mirrored — ai.js exposes it through its `_internal` bundle, the same way puzzleWorker.mjs
// takes it.
const { MATE_THRESH } = _internal;

// MIRRORED from web/src/ai.js, which keeps `VALUE` module-private (only the eval functions
// that consume it are exported, and importing one of those would mean running a search just to
// count material). The knight = 500 ≈ rook entry is the load-bearing part — it is a measured
// property of this variant, not a typo for the standard-chess 300 — so if that table ever
// changes, change it here too or this section silently reports standard-chess material.
const VALUE = { p: 100, n: 500, b: 330, r: 500, q: 900, k: 0 };

if (!existsSync(dataFile)) {
  console.error(`No dataset at ${dataFile}. Generate self-play first (or pass --data=FILE).`);
  process.exit(1);
}

// --- The net's eval scale, the yardstick for the whole saturation section. ----------------
// Read from the weights rather than hardcoded: `scale` is what bounds the eval, and the point
// of the audit is to stay correct if it ever moves. A placeholder/material-fallback weights
// file has no scale, and guessing one would quietly report the wrong ceiling — so fail loudly
// and let the caller point at a real net.
let scale = null;
let arch = null;
try {
  const w = JSON.parse(readFileSync(weightsFile, 'utf8'));
  if (typeof w.scale === 'number' && w.scale > 0) scale = w.scale;
  arch = Array.isArray(w.arch) ? w.arch : null;
} catch (e) {
  console.error(`Could not read ${weightsFile}: ${e.message}`);
  process.exit(1);
}
if (scale == null) {
  console.error(`${weightsFile} has no positive \`scale\` (an untrained placeholder?).`);
  console.error('The saturation report is measured against that scale, so point --weights= at a trained net.');
  process.exit(1);
}

// Where tanh's squash has taken most of the head's local gain: at |v| = 0.67·scale the
// derivative factor 1 − (v/scale)² is already down to 0.55, and it reaches 0 at the ceiling.
const SAT = 0.67 * scale;
const localGain = (a) => Math.max(0, 1 - (a / scale) ** 2);

// Thresholds for the post-decision measure. `scale` itself is always included: a threshold at
// or above the ceiling can only be met by mate scores and by non-nn (handcrafted) labels, so
// having that row in the table is what makes the ceiling effect visible instead of hidden.
const DECIDE_T = [...new Set([100, 200, 300, 400, 500, 600, 800, scale])].sort((a, b) => a - b);

// 50 cp bins from 0 up past the ceiling, then one overflow bucket for non-mate outliers.
const CP_BIN = 50;
const nBins = Math.ceil((scale + 200) / CP_BIN);
// The bin the ceiling closes. `tanh(z) * scale` rounds to exactly ±scale for any position the
// net is confident enough about (61,598 labels do, measured 2026-08-08), and those are clamped
// nn evals, not values above the ceiling — so they belong in the bin below it. Keeping them
// there is what makes every bin ABOVE the ceiling mean "no nn eval could have produced this".
const ceilAligned = scale % CP_BIN === 0;
const ceilBin = ceilAligned ? scale / CP_BIN - 1 : Math.floor(scale / CP_BIN);
const binOf = (a) => (ceilAligned && a === scale ? ceilBin : Math.min(Math.floor(a / CP_BIN), nBins));

// --- Accumulators. ------------------------------------------------------------------------
const bins = new Array(nBins + 1).fill(0);
let games = 0, plies = 0, valued = 0, missingV = 0;
let resW = 0, resD = 0, resL = 0;
const eraGames = new Map(), eraPlies = new Map(); // search era -> games / recorded plies
let sgnPos = 0, sgnNeg = 0, sgnZero = 0;
let within100 = 0, mates = 0, saturated = 0, beyond = 0, atCeiling = 0;
let gainSum = 0; // Σ 1 − (v/scale)² over non-mate labels
const dec = DECIDE_T.map(() => ({ games: 0, plies: 0, agree: 0 }));
let badLines = 0, nonGameLines = 0, replayFail = 0;
// Sampled material replay (mover's view, |cp| apart).
const MAT_EDGES = [0, 100, 400, 800]; // buckets: =0, 1-100, 101-400, 401-800, >800
const matBuckets = new Array(MAT_EDGES.length + 1).fill(0);
let sampled = 0, matPositions = 0, matWithin100 = 0;

// The earliest recorded ply from which |v| (WHITE view) stays ≥ T with one consistent sign
// through the end of the game, per threshold in the ascending DECIDE_T — or `n` when the game
// never reaches such a ply. `sgn` is the sign of that decided run (the last position's), which
// is what gets compared against the game's own result.
//
// One backward walk serves every threshold: the qualifying run for a larger T is always a
// suffix of the run for a smaller one, so a threshold closes the moment the walk drops below
// it and never reopens.
function decisionRun(v, startWhite, n) {
  const k = new Array(DECIDE_T.length).fill(n);
  const lastRaw = n > 0 ? v[n - 1] : null;
  if (lastRaw == null) return { k, sgn: 0 };
  // `v` is side-to-move-relative; fold to White's view so one sign spans the whole game. Which
  // colour moves at ply 0 comes from the record's `start` FEN (harvest games can open on
  // either side); a standard-start game is White at every even ply.
  const white = (i) => (((i & 1) === 0) === startWhite ? v[i] : -v[i]);
  const last = white(n - 1);
  if (!last) return { k, sgn: 0 };
  const sgn = last > 0 ? 1 : -1;
  let open = DECIDE_T.length;
  let i = n - 1;
  for (; i >= 0; i--) {
    if (v[i] == null) break;
    const x = white(i);
    if ((x > 0 ? 1 : x < 0 ? -1 : 0) !== sgn) break;
    const a = x < 0 ? -x : x;
    while (open > 0 && a < DECIDE_T[open - 1]) { k[open - 1] = i + 1; open--; }
    if (open === 0) break;
  }
  for (let j = 0; j < open; j++) k[j] = i + 1; // ran back to ply 0, or to the break at ply i
  return { k, sgn };
}

// --- Scan. --------------------------------------------------------------------------------
// The read is bounded to the file's size AT OPEN. `apos-gen`/`apos-match` append to this same
// file while a loop runs, and an unbounded read would mix games appended mid-report into totals
// the header already printed a byte count for — so the audit is of one snapshot, always.
const bytesTotal = statSync(dataFile).size;
const t0 = Date.now();
const status = liveStatus(process.stderr);
const tick = everyMs(400);
let bytesRead = 0;

const rl = createInterface({
  input: createReadStream(dataFile, { end: Math.max(0, bytesTotal - 1) }),
  crlfDelay: Infinity,
});
for await (const line of rl) {
  bytesRead += Buffer.byteLength(line) + 1;
  if (line.trim()) {
    let rec;
    try { rec = JSON.parse(line); } catch { badLines++; continue; }
    if (!Array.isArray(rec.moves) || typeof rec.r !== 'number') { nonGameLines++; continue; }
    const v = Array.isArray(rec.v) ? rec.v : null;
    const n = v ? v.length : rec.moves.length + 1;
    games++;
    plies += n;
    if (rec.r > 0) resW++; else if (rec.r < 0) resL++; else resD++;
    // Which search played it. A depth is worth a different amount under a different search, so
    // this is the split that says how much of the corpus describes the engine running today.
    const era = rec.se == null ? 1 : rec.se;
    eraGames.set(era, (eraGames.get(era) || 0) + 1);
    eraPlies.set(era, (eraPlies.get(era) || 0) + n);

    if (v) {
      for (let i = 0; i < n; i++) {
        const x = v[i];
        if (x == null) { missingV++; continue; }
        valued++;
        if (x > 0) sgnPos++; else if (x < 0) sgnNeg++; else sgnZero++;
        const a = x < 0 ? -x : x;
        if (a <= 100) within100++;
        if (a >= MATE_THRESH) { mates++; continue; }
        bins[binOf(a)]++;
        gainSum += localGain(a);
        if (a > SAT) saturated++;
        if (a === scale) atCeiling++;
        if (a > scale) beyond++;
      }
      const startWhite = typeof rec.start === 'string' ? rec.start.split(' ')[1] !== 'b' : true;
      const { k, sgn } = decisionRun(v, startWhite, n);
      const rSgn = Math.sign(rec.r);
      for (let j = 0; j < DECIDE_T.length; j++) {
        if (k[j] >= n) continue; // this game never settles at that threshold
        dec[j].games++;
        dec[j].plies += n - 1 - k[j];
        if (sgn === rSgn) dec[j].agree++;
      }
    } else {
      missingV += n;
    }

    // Real material balance needs the board, which means replaying the game — so it is
    // sampled rather than measured on every game (1 in `--sample`, deterministic by game
    // index so a re-run audits the same games).
    if ((games - 1) % sampleEvery === 0) {
      try {
        for (const { state } of expandPositions(rec)) {
          let mat = 0;
          for (const p of state.board) {
            if (!p) continue;
            mat += p.color === state.turn ? VALUE[p.role] : -VALUE[p.role];
          }
          const a = mat < 0 ? -mat : mat;
          matPositions++;
          if (a <= 100) matWithin100++;
          let b = MAT_EDGES.length;
          for (let e = 0; e < MAT_EDGES.length; e++) { if (a <= MAT_EDGES[e]) { b = e; break; } }
          matBuckets[b]++;
        }
        sampled++;
      } catch { replayFail++; } // a corrupt/illegal move token — counted, never fatal
    }
  }

  if (tick()) {
    const done = bytesRead / Math.max(1, bytesTotal);
    const secs = (Date.now() - t0) / 1000;
    status.update(`  scanning ${(100 * done).toFixed(1)}%  ${fmtNum(games)} games  ${fmtNum(plies)} plies`
      + `  ${fmtMB(bytesRead)}/${fmtMB(bytesTotal)}  ${fmtDur(secs)} elapsed`
      + (done > 0.02 ? `, ~${fmtDur(secs / done - secs)} left` : ''));
  }
  if (quickGames && games >= quickGames) break;
}
rl.close();
status.clear();
const elapsed = (Date.now() - t0) / 1000;

// --- Report. ------------------------------------------------------------------------------
const pct = (n, d) => (d ? (100 * n) / d : 0);
const p1 = (n, d) => `${pct(n, d).toFixed(1)}%`;
const p2 = (n, d) => `${pct(n, d).toFixed(2)}%`;
const padL = (s, w) => String(s).padStart(w);
const relish = (p) => p.replace(repoDir + '\\', '').replace(repoDir + '/', '');
const verdict = (ok) => (ok ? 'PASS' : 'FAIL');

console.log(`\nAposChess dataset audit — ${relish(dataFile)}`);
console.log(`  ${fmtNum(games)} games, ${fmtNum(plies)} recorded positions, ${fmtMB(bytesRead)} of `
  + `${fmtMB(bytesTotal)} read in ${fmtDur(elapsed)}${quickGames ? `  (--quick=${fmtNum(quickGames)})` : ''}.`);
console.log(`  Eval scale from ${relish(weightsFile)}: ${scale} cp`
  + `${arch ? `  (arch [${arch.length > 6 ? `${arch.slice(0, 3).join(',')},…,${arch[arch.length - 1]}` : arch.join(',')}])` : ''}`
  + ` — tanh(z)·${scale} can never report more than ±${scale} cp.`);
if (quickGames) {
  // Worth spelling out: --quick reads the OLDEST games in the file, which carry older and
  // weaker-engine labels, so it is a prefix rather than a sample of the corpus.
  console.log('  NOTE: --quick reads the oldest games in the file, not a random sample — the numbers below');
  console.log('        describe that prefix (older, weaker-engine labels), not the corpus. Use it to check');
  console.log('        the tool; run the full pass to read the data.');
}
if (badLines || nonGameLines || missingV || replayFail) {
  console.log(`  Skipped: ${fmtNum(badLines)} unparseable line(s), ${fmtNum(nonGameLines)} non-game-record line(s), `
    + `${fmtNum(missingV)} position(s) with no v, ${fmtNum(replayFail)} game(s) that failed to replay.`);
}

// --- Corpus shape: results and length. ----------------------------------------------------
const whiteScore = games ? (resW + resD / 2) / games : 0;
// The usual logistic conversion, so the first-move advantage reads in the same units as a gate.
const scoreElo = whiteScore > 0 && whiteScore < 1 ? -400 * Math.log10(1 / whiteScore - 1) : NaN;
console.log('\n=== Corpus shape ===');
console.log(`  Recorded plies per game: ${(plies / Math.max(1, games)).toFixed(1)} mean`);
console.log(`  Results (White view):    win ${p1(resW, games)}   draw ${p1(resD, games)}   loss ${p1(resL, games)}`);
console.log(`    → White scores ${(100 * whiteScore).toFixed(1)}% (win=1, draw=½)`
  + `${Number.isFinite(scoreElo) ? `, about ${scoreElo >= 0 ? '+' : ''}${scoreElo.toFixed(0)} Elo of first-move advantage` : ''}.`);
{
  // Search era (record `se`, absent = 1). The rating pool rates one era at a time, so this is
  // how much evidence the CURRENT engine's ladder actually has to work with.
  const eras = [...eraGames.keys()].sort((a, b) => a - b);
  const parts = eras.map((e) => `era ${e} ${p1(eraGames.get(e), games)}`
    + ` (${fmtNum(eraPlies.get(e))} plies)${e === SEARCH_ERA ? ' ← this engine' : ''}`);
  console.log(`  Search era:             ${parts.join('   ')}`);
}

// --- 1. Saturation against the net's own scale. -------------------------------------------
console.log(`\n=== 1. Eval saturation against the net's own scale (${scale} cp) ===\n`);
console.log(`  |v| (mover view, ${CP_BIN} cp bins), share of the ${fmtNum(valued)} labeled position(s):`);
const maxShare = Math.max(...bins.map((c) => pct(c, valued)), pct(mates, valued));
const bar = (share) => '█'.repeat(Math.max(share > 0 ? 1 : 0, Math.round((share / Math.max(1e-9, maxShare)) * 34)));
const satBin = Math.floor(SAT / CP_BIN);
for (let b = 0; b <= nBins; b++) {
  const label = b < nBins ? `${padL(b * CP_BIN, 5)}–${padL((b + 1) * CP_BIN, 4)}` : `${padL(nBins * CP_BIN, 5)}+     `;
  const share = pct(bins[b], valued);
  const note = b === satBin ? `  ← saturation zone starts (|v| > ${SAT.toFixed(0)} = 0.67·scale)`
    : b === ceilBin ? `  ← the ceiling closes here (incl. the ${fmtNum(atCeiling)} labels clamped to exactly ±${scale})`
    : b === ceilBin + 1 ? '  ← above the ceiling: an nn eval cannot get here'
    : b === nBins ? '  (non-mate outliers)'
    : '';
  console.log(`    ${label}  ${padL(fmtNum(bins[b]), 12)}  ${padL(share.toFixed(2), 6)}%  ${bar(share)}${note}`);
}
console.log(`    ${padL('mate', 10)}  ${padL(fmtNum(mates), 12)}  ${padL(pct(mates, valued).toFixed(2), 6)}%  `
  + `${bar(pct(mates, valued))}  (|v| ≥ ${fmtNum(MATE_THRESH)} — outside the scale entirely)`);
console.log('');
const satRows = [
  [`Saturation zone   |v| > ${SAT.toFixed(0)} (0.67·scale), non-mate`, p1(saturated, valued),
    'local gain 1 − (v/scale)² is ≤ 0.55 here'],
  [`At the ceiling    |v| = ${scale} exactly`, p2(atCeiling, valued),
    'the eval clamped — it has nothing more to say'],
  [`Past the ceiling  |v| > ${scale}, non-mate`, p2(beyond, valued),
    `an nn eval is bounded by ±${scale}, so these are handcrafted/material labels`],
  [`Mate scores       |v| ≥ ${fmtNum(MATE_THRESH)}`, p1(mates, valued), ''],
  [`Off the tanh scale entirely (|v| > ${scale}, mates included)`, p1(beyond + mates, valued), ''],
  ['Saturated or off the scale — the whole low-gain band', p1(saturated + mates, valued), ''],
];
const satW = Math.max(...satRows.map(([name]) => name.length));
for (const [name, share, detail] of satRows) {
  console.log(`  ${name.padEnd(satW)}  ${padL(share, 6)}${detail ? `  — ${detail}` : ''}`);
}
console.log(`  Mean local gain 1 − (v/scale)² over non-mate labels: `
  + `${(gainSum / Math.max(1, valued - mates)).toFixed(3)}  (1.000 = the full tanh gain at v = 0)`);
console.log('');
printWrapped(`The cliff between the ${ceilBin * CP_BIN}–${scale} bin and everything above it is the ceiling, not the `
  + `games: tanh(z)·${scale} is bounded, so an nn label physically cannot say "up a queen" (900) — or up two `
  + `rooks — in a variant where a knight is already worth ${VALUE.n}. Everything past the ceiling is a `
  + `handcrafted or material label from an older cohort. Below it the resolution is already going: above `
  + `${SAT.toFixed(0)} distinct positions compress into the same few centipawns, and ${p2(atCeiling, valued)} of `
  + `labels have run out of scale altogether and read exactly ±${scale}.`, '  ', '  ');
console.log('');
printWrapped('LEVER: the scale is a trainer knob and a recipe knob, so it keys its own experiment track — '
  + '`npm run train:loop -- --scale=1200` (note train.py ADOPTS a warm-start init\'s scale and ignores '
  + '--scale unless you also pass --rescale, which is why every archived champion still reads 600). '
  + 'CAVEAT: the numbers above are a measurement of the label distribution. That the censoring is what '
  + 'stalled the loop is a hypothesis — settle it by training at a wider scale and gating it head-to-head '
  + '(`npm run match -- --eval-a=nn --eval-b=nn`), like any other strength change.', '  ', '  ');

// --- 2. The published criteria. ------------------------------------------------------------
console.log('\n=== 2. Composition vs Tan & Watkinson, "Study of the Proper NNUE Dataset" (arXiv:2412.17948) ===\n');
printWrapped('The paper\'s criteria for a healthy NNUE training set, stated on eval values from the '
  + 'side-to-move view — which is exactly what the recorded `v` is:', '  ', '  ');
console.log('');
// "~50/50" has no stated tolerance in the paper; a 40/60 split or better is read as balanced,
// which is loose enough that only a genuinely lopsided corpus fails it.
const balanceOk = Math.min(pct(sgnPos, valued), pct(sgnNeg, valued)) >= 40;
const nearOk = pct(within100, valued) >= 50;
const imbalanced = matPositions - matWithin100;
const imbOk = matPositions > 0 && pct(imbalanced, matPositions) >= 40;
const criteria = [
  ['~50% positive / ~50% negative',
    `${p1(sgnPos, valued)} pos / ${p1(sgnNeg, valued)} neg (${p1(sgnZero, valued)} exactly 0)`, balanceOk],
  ['≥ 50% within ±100 cp', p1(within100, valued), nearOk],
  ['≥ 40% materially imbalanced',
    matPositions ? `${p1(imbalanced, matPositions)} more than 100 cp apart (sampled, section 4)` : '— (no games sampled)',
    imbOk],
];
const critW = Math.max(...criteria.map(([, m]) => m.length));
for (const [name, measured, ok] of criteria) {
  console.log(`    ${name.padEnd(32)} ${measured.padEnd(critW)}   ${verdict(ok)}`);
}
console.log('');
if (!nearOk) {
  printWrapped(`FAIL on the quiet-position criterion: ${p1(within100, valued)} within ±100 cp against a ≥50% `
    + `target, so ${p1(valued - within100, valued)} of the corpus is further out than a pawn. The paper's `
    + 'argument is that a net trained mostly on already-decided positions never has to learn the fine '
    + 'distinctions that decide games — it only has to agree about which side is winning.', '  ', '  ');
  console.log('');
  printWrapped('LEVER: section 3 sizes the prize. The bulk of the far-out positions are plies played out '
    + 'after the game was already settled, so the two levers are (a) stop generating them — an adjudication '
    + 'rule in `apos-gen`/`apos-match` that ends a game once |v| holds past a threshold, which also buys '
    + 'back the search time, and (b) stop training on them — a featurize-time filter, the same shape as '
    + '`--drop-conflicts`, which costs nothing to try because the raw games are never deleted. Both are '
    + 'strength changes: gate them.', '  ', '  ');
} else {
  printWrapped(`The quiet-position criterion passes at ${p1(within100, valued)} within ±100 cp.`, '  ', '  ');
}

// --- 3. Post-decision plies. --------------------------------------------------------------
console.log('\n=== 3. Post-decision plies — the generation-waste measure ===\n');
printWrapped('For each game: the earliest ply from which |v| (folded to White\'s view) stays past the '
  + 'threshold with one consistent sign all the way to the end, and how many plies follow it. "sign vs '
  + 'result" is how often that sign matches the game\'s own outcome — the check that the run really was '
  + 'decided and not a horizon mirage (a draw counts as a disagreement, since its result has no sign).',
'  ', '  ');
console.log('');
console.log('    |v| ≥      games decided     plies after   share of all plies   sign vs result');
console.log('    ─────      ─────────────   ─────────────   ──────────────────   ──────────────');
for (let j = 0; j < DECIDE_T.length; j++) {
  const d = dec[j];
  const mark = DECIDE_T[j] === scale ? '  ← = scale (the ceiling)' : '';
  console.log(`    ${padL(DECIDE_T[j], 5)}      ${padL(p1(d.games, games), 6)}         `
    + `${padL(fmtNum(d.plies), 13)}   ${padL(p1(d.plies, plies), 18)}   ${padL(p1(d.agree, d.games), 14)}${mark}`);
}
console.log('');
const d400 = dec[DECIDE_T.indexOf(400)] ?? dec[0];
printWrapped(`At ±400 cp, ${p1(d400.plies, plies)} of every recorded ply comes after the game is decided, and the `
  + `decision's sign is right ${p1(d400.agree, d400.games)} of the time — so those plies are not ambiguous positions `
  + 'the net needs to study, they are the same won game restated dozens of times. They cost search time to '
  + 'generate, they are most of what fails the ±100 cp criterion above, and they are where the saturated '
  + 'labels in section 1 live.', '  ', '  ');
console.log('');
printWrapped(`Read the ${scale}-cp row as a property of the EVAL, not of the games: a threshold at or above the `
  + 'ceiling can only be met by mate scores and by the handcrafted-labeled cohort, so the number collapses '
  + 'even though the games are exactly as decided as they were one row up. That is the censoring from '
  + 'section 1 showing up inside a second measurement.', '  ', '  ');

// --- 4. Real material balance (sampled). --------------------------------------------------
console.log(`\n=== 4. Real material balance (mover's view, sampled 1 game in ${fmtNum(sampleEvery)}) ===\n`);
if (!matPositions) {
  console.log('  No games sampled (raise --quick, or lower --sample).');
} else {
  console.log(`  ${fmtNum(sampled)} game(s) replayed → ${fmtNum(matPositions)} position(s), `
    + `piece values p=${VALUE.p} n=${VALUE.n} b=${VALUE.b} r=${VALUE.r} q=${VALUE.q} (mirrored from src/ai.js):`);
  console.log('');
  const labels = ['equal (0 cp)', '1–100 cp', '101–400 cp', '401–800 cp', 'over 800 cp'];
  const maxMat = Math.max(...matBuckets.map((c) => pct(c, matPositions)));
  for (let b = 0; b < matBuckets.length; b++) {
    const share = pct(matBuckets[b], matPositions);
    console.log(`    ${labels[b].padEnd(13)} ${padL(fmtNum(matBuckets[b]), 10)}  ${padL(share.toFixed(1), 5)}%  `
      + '█'.repeat(Math.max(share > 0 ? 1 : 0, Math.round((share / Math.max(1e-9, maxMat)) * 34))));
  }
  console.log('');
  console.log(`  Within ±100 cp of material equality: ${p1(matWithin100, matPositions)}`
    + `   |   more than 100 cp apart: ${p1(imbalanced, matPositions)}`);
  console.log('');
  printWrapped('The paper does not put a number on "imbalanced", so this report reads it as more than a pawn '
    + `apart, which passes at ${p1(imbalanced, matPositions)} against the ≥40% target. Worth reading alongside `
    + `section 3 rather than as good news on its own: ${p1(matBuckets[4], matPositions)} of positions are more `
    + 'than 800 cp apart — a queen and change — and those are the same played-out won games.', '  ', '  ');
}

// --- Summary. -----------------------------------------------------------------------------
console.log('\n=== Summary ===\n');
const lines = [];
lines.push(`Eval censored at ±${scale} cp: ${p1(saturated, valued)} of labels sit above 0.67·scale where the head's `
  + `local gain is ≤ 0.55, ${p1(mates, valued)} are mate scores, and only ${p2(beyond, valued)} exceed the ceiling `
  + 'at all (those can only be non-nn labels). Lever: a wider `--scale` on its own track, with `--rescale`.');
lines.push(`Tan & Watkinson: sign balance ${verdict(balanceOk)}, material imbalance ${verdict(imbOk)}, `
  + `±100 cp ${verdict(nearOk)} (${p1(within100, valued)} vs ≥50%).`);
lines.push(`${p1(d400.plies, plies)} of all recorded plies come after the game is decided at ±400 cp `
  + `(${p1(d400.games, games)} of games get there; the sign is right ${p1(d400.agree, d400.games)} of the time). `
  + 'Lever: adjudicate in generation, or filter at featurize time.');
lines.push('All of the above is a measurement of the data. Whether fixing any of it makes the engine stronger '
  + 'is a `npm run match` question, and nothing here has answered it.');
for (const l of lines) printWrapped(l, '  •', '    ');
console.log('');
