// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2019-2026 Jean-David Moisan
//
// Experiment registry for train:loop. A "recipe" is the set of TRAINING-AFFECTING knobs
// that turn the shared self-play dataset into a candidate net — architecture (--hidden),
// the TD mix (--lambda), the quiet-position filter (--quiet-only), quantization, and the
// trainer knobs (--scale/--lr/--wd) — plus a free-form --recipe-extra namespace for future
// training systems. It is deliberately NOT the generation/gate/refresh knobs, which shape
// the shared dataset or the decision rather than a candidate's identity.
//
// Each distinct recipe gets its own TRACK on disk, keyed by a stable content hash of the
// recipe, under training/data/loop/experiments/<id>/. A track persists:
//   recipe.json   the recipe + id + slug + created stamp (immutable identity)
//   lineage.json  the accumulated sub-threshold warm-start net (was loop/lineage.json,
//                 but now per-recipe so switching recipes is NON-DESTRUCTIVE)
//   best.json     the strongest net this recipe ever produced (by estimated absolute Elo),
//                 even if it never promoted — the safe warm-start when resuming after a gap
//   history.jsonl one line per cycle (score / Elo / SPRT / promoted / champion hash)
//   state.json    mutable rollup (runs, cycles, promotions, best metadata)
//
// So trying a different architecture (or quiet-games, or any other recipe knob) no longer
// clobbers the previous track's accumulated progress: come back to the same recipe later
// — even after running others in between — and the loop finds this directory and resumes
// its lineage/best automatically (same recipe -> same id, deterministically).
//
// All under training/data/ (git-ignored). This module is pure library code shared by
// train-loop.mjs (writer) and loop-experiments.mjs / loop-progress.mjs (readers); nothing
// here spawns a process, so the readers are safe against a live loop.

import {
  existsSync, mkdirSync, readFileSync, writeFileSync, appendFileSync, readdirSync,
} from 'node:fs';
import { join } from 'node:path';
import { createHash } from 'node:crypto';

export function experimentsDir(loopDir) { return join(loopDir, 'experiments'); }

// --- Recipe construction & identity --------------------------------------------------

// Canonicalize the raw knobs into a recipe object. hidden/lambda/quietOnly/quant are ALWAYS
// present (the loop controls their defaults); scale/lr/wd/filterWeak/dropConflicts/extra are
// included only when the caller explicitly set them (pass `undefined` otherwise), so an unset
// optional never fragments the id or drifts if a tool's own default changes — and adding a
// new optional knob never re-keys the existing tracks.
export function buildRecipe(raw) {
  const r = {
    hidden: String(raw.hidden),
    lambda: Number(raw.lambda),
    quietOnly: !!raw.quietOnly,
    quant: !!raw.quant,
  };
  if (raw.scale !== undefined) r.scale = Number(raw.scale);
  if (raw.lr !== undefined) r.lr = Number(raw.lr);
  if (raw.wd !== undefined) r.wd = Number(raw.wd);
  // Dataset filters applied at featurize time (train-loop --filter-weak / --drop-conflicts):
  // they change what the candidate trains on, so they are part of its identity.
  if (raw.filterWeak !== undefined) r.filterWeak = Number(raw.filterWeak);
  if (raw.dropConflicts !== undefined) r.dropConflicts = Number(raw.dropConflicts);
  if (raw.extra && Object.keys(raw.extra).length) {
    r.extra = {};
    for (const k of Object.keys(raw.extra).sort()) r.extra[k] = String(raw.extra[k]);
  }
  return r;
}

// Parse a --recipe-extra=k=v,k2=v2 string into a sorted plain object (empty -> {}).
export function parseRecipeExtra(spec) {
  const out = {};
  if (typeof spec !== 'string') return out;
  for (const pair of spec.split(',')) {
    const s = pair.trim();
    if (!s) continue;
    const eq = s.indexOf('=');
    if (eq < 0) { out[s] = 'on'; continue; } // bare key = presence flag
    out[s.slice(0, eq).trim()] = s.slice(eq + 1).trim();
  }
  return out;
}

// Stable serialization (fixed key order) so the same recipe always hashes the same.
function canonical(recipe) {
  const ordered = {};
  for (const k of ['hidden', 'lambda', 'quietOnly', 'quant', 'scale', 'lr', 'wd', 'filterWeak', 'dropConflicts']) {
    if (recipe[k] !== undefined) ordered[k] = recipe[k];
  }
  if (recipe.extra) ordered.extra = recipe.extra; // buildRecipe already sorted its keys
  return JSON.stringify(ordered);
}

export function recipeId(recipe) {
  return createHash('sha1').update(canonical(recipe)).digest('hex').slice(0, 8);
}

// Short filesystem-safe-ish label for display (the on-disk dir uses the id, not this).
export function recipeSlug(recipe) {
  const parts = [`h${recipe.hidden.replace(/,/g, '-')}`];
  if (recipe.lambda !== 1) parts.push(`l${recipe.lambda}`);
  if (recipe.quietOnly) parts.push('quiet');
  if (!recipe.quant) parts.push('float');
  if (recipe.scale !== undefined) parts.push(`s${recipe.scale}`);
  if (recipe.lr !== undefined) parts.push(`lr${recipe.lr}`);
  if (recipe.wd !== undefined) parts.push(`wd${recipe.wd}`);
  if (recipe.filterWeak !== undefined) parts.push(`fw${recipe.filterWeak}`);
  if (recipe.dropConflicts !== undefined) parts.push(`dc${recipe.dropConflicts}`);
  if (recipe.extra) for (const [k, v] of Object.entries(recipe.extra)) parts.push(`${k}=${v}`);
  return parts.join('_');
}

// Pretty one-line human description.
export function recipeLabel(recipe) {
  const bits = [`hidden=[${recipe.hidden}]`, `λ=${recipe.lambda}`];
  bits.push(recipe.quietOnly ? 'quiet-only' : 'all-positions');
  bits.push(recipe.quant ? 'quantized' : 'float');
  if (recipe.scale !== undefined) bits.push(`scale=${recipe.scale}`);
  if (recipe.lr !== undefined) bits.push(`lr=${recipe.lr}`);
  if (recipe.wd !== undefined) bits.push(`wd=${recipe.wd}`);
  if (recipe.filterWeak !== undefined) bits.push(`filter-weak=${recipe.filterWeak}`);
  if (recipe.dropConflicts !== undefined) bits.push(`drop-conflicts=${recipe.dropConflicts}`);
  if (recipe.extra) bits.push(...Object.entries(recipe.extra).map(([k, v]) => `${k}=${v}`));
  return bits.join(', ');
}

// The `npm run train:loop -- …` flags that reproduce (resume) a recipe. Only knobs that
// differ from the loop's defaults are emitted, so the command is minimal and copy-pasteable.
export function recipeToFlags(recipe) {
  const f = [`--hidden=${recipe.hidden}`];
  if (recipe.lambda !== 1) f.push(`--lambda=${recipe.lambda}`);
  if (recipe.quietOnly) f.push('--quiet-only');
  if (!recipe.quant) f.push('--float');
  if (recipe.scale !== undefined) f.push(`--scale=${recipe.scale}`);
  if (recipe.lr !== undefined) f.push(`--lr=${recipe.lr}`);
  if (recipe.wd !== undefined) f.push(`--wd=${recipe.wd}`);
  if (recipe.filterWeak !== undefined) f.push(`--filter-weak=${recipe.filterWeak}`);
  if (recipe.dropConflicts !== undefined) f.push(`--drop-conflicts=${recipe.dropConflicts}`);
  if (recipe.extra) {
    const spec = Object.entries(recipe.extra).map(([k, v]) => `${k}=${v}`).join(',');
    if (spec) f.push(`--recipe-extra=${spec}`);
  }
  return f.join(' ');
}

export function recipeResumeCmd(recipe) {
  return `npm run train:loop -- ${recipeToFlags(recipe)}`.trim();
}

// --- Track paths & lifecycle ---------------------------------------------------------

export function trackDir(loopDir, id) { return join(experimentsDir(loopDir), id); }

export function trackPaths(dir) {
  return {
    dir,
    recipe: join(dir, 'recipe.json'),
    lineage: join(dir, 'lineage.json'),
    best: join(dir, 'best.json'),
    history: join(dir, 'history.jsonl'),
    state: join(dir, 'state.json'),
  };
}

export function readState(dir) {
  try { return JSON.parse(readFileSync(trackPaths(dir).state, 'utf8')); } catch { return null; }
}
export function writeState(dir, st) {
  writeFileSync(trackPaths(dir).state, JSON.stringify(st, null, 2) + '\n');
}

// Ensure the track directory exists and its immutable recipe.json is written. Returns the
// track handle { id, dir, slug, paths, isNew }. Idempotent — re-running for the same recipe
// just returns the existing track.
export function ensureTrack(loopDir, recipe, ts) {
  const id = recipeId(recipe);
  const dir = trackDir(loopDir, id);
  mkdirSync(dir, { recursive: true });
  const paths = trackPaths(dir);
  const isNew = !existsSync(paths.recipe);
  if (isNew) {
    writeFileSync(paths.recipe,
      JSON.stringify({ id, slug: recipeSlug(recipe), recipe, createdTs: ts }, null, 2) + '\n');
  }
  return { id, dir, slug: recipeSlug(recipe), paths, isNew };
}

// Note a new loop run against a track (bumps the per-track run counter). Returns the run
// number of this session, which the caller stamps onto each cycle's history entry.
export function beginRun(dir, ts) {
  let st = readState(dir);
  if (!st) st = { cycles: 0, promotions: 0, runs: 0, best: null, createdTs: ts };
  st.runs = (st.runs || 0) + 1;
  st.lastRunTs = ts;
  writeState(dir, st);
  return st.runs;
}

// Record one gate cycle: append the history line and roll it into state. `entry` fields:
//   { run, cycle, ts, score, edgeElo, absElo, sprt, promoted, div, championHash, datasetBytes, hash }
// "best" is tracked by estimated ABSOLUTE Elo (championLedgerElo + gate edge) when available
// — a raw gate score isn't comparable across cycles because the champion opponent strengthens
// over time. Falls back to raw score when no ledger Elo is known yet. Returns { st, isBest };
// the caller copies the candidate weights over best.json when isBest is true.
export function recordCycle(dir, entry) {
  const paths = trackPaths(dir);
  appendFileSync(paths.history, JSON.stringify(entry) + '\n');
  let st = readState(dir) || { cycles: 0, promotions: 0, runs: 0, best: null, createdTs: entry.ts };
  st.cycles = (st.cycles || 0) + 1;
  if (entry.promoted) st.promotions = (st.promotions || 0) + 1;
  st.lastRunTs = entry.ts;
  st.lastChampionHash = entry.championHash;
  const cur = st.best;
  const isBest = !cur
    || (entry.absElo != null && cur.absElo != null
      ? entry.absElo > cur.absElo
      : (entry.score ?? -Infinity) > (cur.score ?? -Infinity));
  if (isBest) {
    st.best = {
      absElo: entry.absElo ?? null, score: entry.score, edgeElo: entry.edgeElo ?? null,
      sprt: entry.sprt, run: entry.run, cycle: entry.cycle, ts: entry.ts, hash: entry.hash ?? null,
    };
  }
  writeState(dir, st);
  return { st, isBest };
}

// --- Reading the whole registry ------------------------------------------------------

export function readAllTracks(loopDir) {
  const dir = experimentsDir(loopDir);
  let ids = [];
  try { ids = readdirSync(dir, { withFileTypes: true }).filter((d) => d.isDirectory()).map((d) => d.name); }
  catch { return []; } // no registry yet
  const tracks = [];
  for (const id of ids) {
    const d = trackDir(loopDir, id);
    let recipe;
    try { recipe = JSON.parse(readFileSync(trackPaths(d).recipe, 'utf8')).recipe; } catch { continue; }
    if (!recipe) continue;
    tracks.push({ id, dir: d, recipe, slug: recipeSlug(recipe), state: readState(d) });
  }
  return tracks;
}

export function readHistory(dir) {
  const out = [];
  let text;
  try { text = readFileSync(trackPaths(dir).history, 'utf8'); } catch { return out; }
  for (const line of text.split('\n')) {
    if (!line) continue;
    try { out.push(JSON.parse(line)); } catch { /* skip a torn line */ }
  }
  return out;
}

// --- Re-anchoring a stored absElo onto today's ledger --------------------------------
// A cycle's stored absElo = (champion's ledger Elo THAT cycle) + gate edge. But the BT pool is
// re-fit every cycle, so a FIXED champion's ledger rating drifts over time (observed ~40 Elo across
// two days, champion unchanged) — which means an old stored absElo is not comparable to today's
// scale, and a cycle that scored 49% at the gate against the CURRENT champion can carry a stale
// absElo that reads as 55% "vs current champion". Re-anchor at read time by placing the candidate on
// TODAY's ledger via the champion it actually played: currentLedgerElo(itsChampion) + gate edge. For
// a cycle gated against the current champion this collapses to champEloNow + edge — exactly the gate
// score, no drift. These helpers are the single source of truth for the reports (loop-progress /
// loop-experiments / suggestRecipes), so the fix can't drift back apart across them.

// Best CURRENT ledger Elo per engine version (nn weights hash), across depths. Read once, passed in.
export function ledgerBestByVersion(loopDir) {
  const map = new Map();
  const file = join(loopDir, 'engine-elo.ladder.json');
  if (!existsSync(file)) return map;
  let ledger;
  try { ledger = JSON.parse(readFileSync(file, 'utf8')); } catch { return map; }
  for (const e of ledger.ranking || []) {
    if (e.eng !== 'nn' || e.version == null || e.elo == null) continue;
    const cur = map.get(e.version);
    if (cur == null || e.elo > cur) map.set(e.version, e.elo);
  }
  return map;
}

// Re-anchor one history entry's absElo onto today's ledger (see block comment). Falls back to the
// stored absElo only when that cycle's champion is no longer rated (old, dropped-off tracks).
export function reAnchoredAbsElo(h, ledgerElo) {
  if (h && h.championHash && Number.isFinite(h.edgeElo) && ledgerElo && ledgerElo.has(h.championHash)) {
    return ledgerElo.get(h.championHash) + h.edgeElo;
  }
  return Number.isFinite(h?.absElo) ? h.absElo : NaN;
}

// The strongest cycle a track ever produced, ranked by RE-ANCHORED absElo (comparable across cycles,
// unlike the drift-prone snapshot in state.best). Returns a best-shaped object
// { absElo, score, edgeElo, sprt, run, cycle, ts, hash } (absElo is the re-anchored value), or null.
// `runFilter`, when given, restricts to those per-track run numbers (loop-progress scopes a merged
// run to its own launches).
export function trackBestAbs(dir, ledgerElo, runFilter = null) {
  let best = null, bestElo = -Infinity;
  for (const h of readHistory(dir)) {
    if (runFilter && !runFilter.has(h.run)) continue;
    const a = reAnchoredAbsElo(h, ledgerElo);
    if (!Number.isFinite(a) || a <= bestElo) continue;
    bestElo = a;
    best = {
      absElo: a, score: h.score, edgeElo: h.edgeElo ?? null, sprt: h.sprt,
      run: h.run, cycle: h.cycle, ts: h.ts, hash: h.hash ?? null,
    };
  }
  return best;
}

// --- Suggestions ---------------------------------------------------------------------
//
// The suggester is a small sequential-design loop over recipe space, not a fixed list. Every
// finished track is one observation — recipe features in, best re-anchored absElo out — and a
// kernel-regression surrogate fitted to those observations scores a generated pool of untried
// recipes. Candidates are ranked by an upper-confidence bound (predicted Elo + kappa · sigma), so
// the ranking rewards BOTH a promising prediction and a region the registry knows little about.
// As tracks accumulate, sigma shrinks where you've measured and the suggestions concentrate on
// the parts of the space that still look good — the "gets better as it learns" behaviour.
//
// This is deliberately modest machinery. With a handful of tracks the surrogate is barely more
// than "a distance-weighted average of what you already ran", so `surrogateReport` cross-validates
// it (leave-one-out) against the do-nothing baseline of predicting the global mean, and
// `suggestRecipes` falls back to a pure space-filling design when the model can't beat that
// baseline. The tool says which mode it's in rather than dressing up noise as a prediction.

// Layer widths of a hidden spec ("64,32,16" -> [64,32,16]). Invalid entries are dropped.
export function hiddenWidths(hidden) {
  return String(hidden ?? '').split(',')
    .map((s) => Number(s.trim()))
    .filter((n) => Number.isFinite(n) && n > 0);
}

// Total weight count of a net built from this hidden spec. Dominated by the sparse input layer
// (NUM_FEATURES x w0), which is why w0 gets its own feature below — it's the capacity knob that
// actually costs parameters, while extra narrow layers are nearly free.
export function recipeParams(hidden, numFeatures = 768) {
  const w = hiddenWidths(hidden);
  if (!w.length) return numFeatures;
  let p = numFeatures * w[0] + w[0];
  for (let i = 1; i < w.length; i++) p += w[i - 1] * w[i] + w[i];
  return p + w[w.length - 1] + 1; // scalar head
}

// Multiply-accumulates per EVALUATED NODE — the cost that actually shows up as ns/node, and a
// different quantity from the parameter count. The 768 x w0 input layer is excluded on purpose:
// the Zig search maintains it as an incrementally-updated accumulator (the "U" in NNUE, see
// web/engine/README.md), so a wide first layer is nearly free at inference. What costs nodes is
// the DENSE stack behind it. That distinction matters a lot for candidate generation: a
// [256 x 12] net is ~60x the per-node work of the current champion while a [256, 16, 16] net is
// cheaper than it, and a search that halves its nps has to be far more accurate just to break
// even at the loop's fixed-depth gate.
export function denseCost(hidden) {
  const w = hiddenWidths(hidden);
  if (w.length < 1) return 1;
  let c = 0;
  for (let i = 1; i < w.length; i++) c += w[i - 1] * w[i];
  return c + w[w.length - 1] + 1; // + scalar head
}

// The surrogate's input space. Log2 on everything multiplicative so "twice as wide" is one unit
// in every direction and the kernel's single bandwidth means the same thing per axis.
// `cycles` is a feature, not a covariate to ignore: a track that ran 53 cycles had far more
// chances to hit a high best-absElo than one that ran 2, so leaving it out would credit the
// architecture for what was really just a bigger sample. Predictions for untried recipes are
// made at a fixed probe budget, which makes them comparable to each other.
export const FEATURE_KEYS = [
  'logParams', 'logDenseCost', 'depth', 'logW0', 'logLast', 'logTaper', 'logNeck',
  'lambda', 'quiet', 'quant', 'filterWeak', 'dropConflicts', 'logCycles',
];

export function recipeFeatures(recipe, cycles = 1) {
  const w = hiddenWidths(recipe.hidden);
  const depth = w.length || 1;
  const w0 = w[0] || 1;
  const last = w[depth - 1] || 1;
  const neck = w.length ? Math.min(...w) : 1;
  return {
    logParams: Math.log2(recipeParams(recipe.hidden)),
    logDenseCost: Math.log2(denseCost(recipe.hidden)),
    depth,
    logW0: Math.log2(w0),
    logLast: Math.log2(last),
    logTaper: Math.log2(w0 / last),   // how hard the stack narrows
    logNeck: Math.log2(neck),         // the tightest bottleneck anywhere in the stack
    lambda: Number(recipe.lambda ?? 1),
    quiet: recipe.quietOnly ? 1 : 0,
    quant: recipe.quant === false ? 0 : 1,
    filterWeak: recipe.filterWeak === undefined ? 0 : 1,
    dropConflicts: recipe.dropConflicts === undefined ? 0 : 1,
    logCycles: Math.log2(Math.max(1, cycles)),
  };
}

const featureVector = (recipe, cycles) => {
  const f = recipeFeatures(recipe, cycles);
  return FEATURE_KEYS.map((k) => f[k]);
};

// Every track that recorded at least one rated cycle, as a surrogate observation.
export function surrogateObservations(loopDir) {
  const ledgerElo = ledgerBestByVersion(loopDir);
  const out = [];
  for (const t of readAllTracks(loopDir)) {
    const best = trackBestAbs(t.dir, ledgerElo);
    if (!best || !Number.isFinite(best.absElo)) continue;
    const cycles = (t.state && t.state.cycles) || readHistory(t.dir).length || 1;
    out.push({
      id: t.id, slug: t.slug, recipe: t.recipe, cycles,
      promotions: (t.state && t.state.promotions) || 0,
      y: best.absElo, x: featureVector(t.recipe, cycles),
    });
  }
  return out;
}

// Fit a Nadaraya-Watson kernel regression over z-scored features. Returns a predictor giving
// { mean, sigma, ess, nearest } at any point, or null when there's nothing to fit.
//
// sigma is the honest part: it combines the residual spread of the observations with the
// EFFECTIVE sample size at the query point (ess = (sum w)^2 / sum w^2). Far from every observed
// recipe the kernel weights collapse onto the single nearest neighbour, ess -> 1, and sigma
// widens to the full spread — which is exactly the "we know nothing here" signal the acquisition
// function needs to go exploring.
export function fitSurrogate(obs, opts = {}) {
  if (obs.length < 2) return null;
  const d = FEATURE_KEYS.length;
  const mu = [], sd = [];
  for (let j = 0; j < d; j++) {
    const col = obs.map((o) => o.x[j]);
    const m = col.reduce((a, b) => a + b, 0) / col.length;
    const v = col.reduce((a, b) => a + (b - m) ** 2, 0) / col.length;
    mu.push(m);
    sd.push(Math.sqrt(v) || 1); // a constant feature carries no information; 1 keeps it inert
  }
  const z = (x) => x.map((v, j) => (v - mu[j]) / sd[j]);
  const Z = obs.map((o) => z(o.x));
  const ys = obs.map((o) => o.y);
  const yMean = ys.reduce((a, b) => a + b, 0) / ys.length;
  const ySpread = Math.sqrt(ys.reduce((a, b) => a + (b - yMean) ** 2, 0) / ys.length) || 1;
  const dist2 = (a, b) => a.reduce((s, v, j) => s + (v - b[j]) ** 2, 0);

  // Bandwidth by leave-one-out CV over a log grid. Too small and every prediction is its own
  // nearest neighbour; too large and everything is the global mean.
  const grid = [0.5, 0.75, 1, 1.5, 2, 3, 4, 6];
  const looErr = (h) => {
    let se = 0;
    for (let i = 0; i < Z.length; i++) {
      let sw = 0, swy = 0;
      for (let k = 0; k < Z.length; k++) {
        if (k === i) continue;
        const w = Math.exp(-dist2(Z[i], Z[k]) / (2 * h * h));
        sw += w; swy += w * ys[k];
      }
      const pred = sw > 1e-12 ? swy / sw : yMean;
      se += (pred - ys[i]) ** 2;
    }
    return Math.sqrt(se / Z.length);
  };
  const h = opts.bandwidth ?? grid.reduce((bestH, cand) =>
    (looErr(cand) < looErr(bestH) ? cand : bestH), grid[0]);

  const predict = (recipe, cycles) => {
    const q = z(featureVector(recipe, cycles));
    let sw = 0, sw2 = 0, swy = 0, nearest = Infinity;
    for (let k = 0; k < Z.length; k++) {
      const dd = dist2(q, Z[k]);
      nearest = Math.min(nearest, Math.sqrt(dd));
      const w = Math.exp(-dd / (2 * h * h));
      sw += w; sw2 += w * w; swy += w * ys[k];
    }
    if (sw <= 1e-12) return { mean: yMean, sigma: ySpread, ess: 0, nearest };
    const ess = (sw * sw) / sw2;                       // 1 = one point dominates, n = all equal
    const mean = swy / sw;
    const sigma = ySpread / Math.sqrt(Math.max(1, ess));
    return { mean, sigma, ess, nearest };
  };

  return {
    predict, bandwidth: h, n: obs.length, yMean, ySpread,
    looRmse: looErr(h),
    baselineRmse: ySpread, // predicting the global mean for everything
    normalize: z,
  };
}

// Human-readable read on whether the surrogate has learned anything yet. `useful` is the switch
// suggestRecipes reads: false means fall back to a space-filling design.
export function surrogateReport(loopDir, opts = {}) {
  const obs = surrogateObservations(loopDir);
  const model = fitSurrogate(obs);
  if (!model) {
    return { obs, model: null, useful: false, n: obs.length,
      verdict: `${obs.length} rated track(s) — need at least 2 to fit anything. Suggestions are pure exploration.` };
  }
  const gain = 1 - model.looRmse / model.baselineRmse; // >0 means it beats predicting the mean
  // Trust the surrogate only when it has both enough observations and a gain that isn't noise.
  // At n<MIN_OBS a leave-one-out win of a few percent is well inside the sampling error of the
  // gate scores the absElo comes from, and the bandwidth search will happily overfit to it — the
  // fit above picks the smallest bandwidth in the grid (nearest-neighbour) on tiny samples. Both
  // bars have to clear, and until they do the right move is to go collect observations anyway,
  // which is exactly what the space-filling fallback does.
  const MIN_OBS = Number.isFinite(opts.minObs) ? opts.minObs : 12;
  const MIN_GAIN = Number.isFinite(opts.minGain) ? opts.minGain : 0.15;
  const useful = obs.length >= MIN_OBS && gain > MIN_GAIN;
  const rmse = `leave-one-out RMSE ${model.looRmse.toFixed(1)} Elo vs ${model.baselineRmse.toFixed(1)} for predicting the mean`;
  return {
    obs, model, useful, n: obs.length, gain, minObs: MIN_OBS,
    verdict: useful
      ? `${rmse} (${(gain * 100).toFixed(0)}% better) over ${obs.length} track(s) — weak but informative, ranking by predicted Elo + uncertainty.`
      : `${rmse} (${(gain * 100).toFixed(0)}% better) over ${obs.length} track(s). `
        + (obs.length < MIN_OBS
          ? `Too few to trust — ${MIN_OBS - obs.length} more track(s) before predictions are ranked on. `
          : 'Not enough of a gain over the mean to rank on. ')
        + 'Suggestions are a space-filling design: maximise what the next track teaches.',
  };
}

// The candidate pool: plausible architectures, generated rather than listed, so the search space
// grows with the width/depth set instead of being capped by whatever was hand-written here.
// Shapes are non-increasing (every champion so far tapers or holds flat) and built from two
// families that between them cover what has actually won: a geometric taper from w0 down to a
// tail width, and a "block" shape (a wide head, then k identical narrow layers — Olga's
// 64,32x6 and Nash's 64,64,64,16,16).
const WIDTHS = [16, 32, 64, 96, 128, 192, 256];
const snap = (v) => WIDTHS.reduce((a, b) => (Math.abs(b - v) < Math.abs(a - v) ? b : a), WIDTHS[0]);

// Coverage buckets for the exploration design. Ranking untried recipes by raw distance in the
// 13-dim z-scored feature space is degenerate while the registry is small: with a handful of
// observations, every untried recipe is "far", and the ranking is decided by whichever axis
// happens to have the smallest sample spread — which is how a pure-distance criterion returned
// four 12-layer 256-wide nets, and then four single-layer nets, instead of a spread of shapes.
// So in explore mode the design works on three interpretable axes instead, the ones this engine's
// own history says matter: how deep the stack is, how wide its first layer is, and what it costs
// per node. A candidate is scored by how EMPTY its bucket is, and the picks are forced into
// distinct buckets — a real space-filling design over the region, not a march to the corners.
const DEPTH_BUCKETS = [[1, 1], [2, 3], [4, 6], [7, 10], [11, 99]];
const W0_BUCKETS = [[0, 32], [33, 64], [65, 128], [129, 9999]];
const COST_BUCKETS = [[0, 2e3], [2e3, 8e3], [8e3, 32e3], [32e3, Infinity]];
const bucketOf = (v, buckets) => Math.max(0, buckets.findIndex(([lo, hi]) => v >= lo && v <= hi));

export function coverageKey(hidden) {
  const w = hiddenWidths(hidden);
  return [
    bucketOf(w.length || 1, DEPTH_BUCKETS),
    bucketOf(w[0] || 1, W0_BUCKETS),
    bucketOf(denseCost(hidden), COST_BUCKETS),
  ].join('/');
}

// `costBudget` bounds the DESIGN REGION, and it is what keeps this from degenerating. A
// space-filling criterion over an unbounded box always picks the far corner, so without a budget
// every suggestion comes back as the widest, deepest net in the grid — maximally novel, and
// maximally unbuildable (a 12-layer 256-wide stack is ~60x the champion's per-node work, which
// the fixed-depth gate would never reward). Bounding by dense cost relative to the reigning shape
// keeps exploration inside the region where a candidate could actually win.
// `w0Floor` bounds the region from BELOW, and it matters as much as the cost ceiling. The first
// layer is the feature detector for a 768-dim sparse input, and every champion this engine has
// produced used 64 or 128 there. Without a floor the coverage design walks straight into the
// narrow corner — a simulated week of rotations spent six consecutive tracks on 16-wide first
// layers, because those cells are unmeasured — which is true but not worth a week to confirm.
// Half the reigning width keeps the plausible range open (a smaller net is a real question) while
// refusing the severe-bottleneck end.
export function candidateArchitectures(maxDepth = 12, costBudget = Infinity, w0Floor = 0) {
  const out = new Set();
  const keep = (spec) => { if (denseCost(spec) <= costBudget) out.add(spec); };
  for (const w0 of WIDTHS) {
    if (w0 < w0Floor) continue;
    keep(String(w0));
    for (const tail of WIDTHS) {
      if (tail > w0) continue;
      for (let depth = 2; depth <= maxDepth; depth++) {
        // Geometric taper w0 -> tail across `depth` layers.
        const geo = [];
        for (let i = 0; i < depth; i++) {
          const t = depth === 1 ? 0 : i / (depth - 1);
          geo.push(snap(w0 * (tail / w0) ** t));
        }
        if (geo.every((v, i) => i === 0 || v <= geo[i - 1])) keep(geo.join(','));
        // Block: one w0 head, then depth-1 layers at `tail`.
        if (depth >= 2) keep([w0, ...Array(depth - 1).fill(tail)].join(','));
      }
    }
  }
  return [...out];
}

// Suggest what to try next, drawing on the registry:
//   kind 'resume' — a PAST recipe whose best net was promising (high estimated abs Elo) but
//                   that stalled (few promotions); resuming warm-starts from its saved best.
//   kind 'new'    — an architecture with NO track yet (never explored).
// Returns a ranked, de-duplicated list of { kind, id?, slug, recipe, reason, cmd, bestElo? }.
export function suggestRecipes(loopDir, opts = {}) {
  const tracks = readAllTracks(loopDir);
  const triedHidden = new Set(tracks.map((t) => t.recipe.hidden));
  const ledgerElo = ledgerBestByVersion(loopDir);
  const out = [];

  // Promising-but-stalled past recipes: rank by best RE-ANCHORED absolute Elo (today's-ledger scale,
  // not the drift-prone stored snapshot), prefer ones that never promoted (their gains never landed)
  // and haven't been touched recently.
  const promising = tracks
    .map((t) => ({ t, best: trackBestAbs(t.dir, ledgerElo) }))
    .filter((x) => x.best && Number.isFinite(x.best.absElo))
    .map((x) => ({ t: x.t, elo: x.best.absElo }))
    .sort((a, b) => b.elo - a.elo);
  for (const { t, elo } of promising) {
    if (out.filter((o) => o.kind === 'resume').length >= 3) break;
    const st = t.state || {};
    const promoted = st.promotions || 0;
    // Skip a recipe that's clearly the reigning line (recently promoted a lot) — it's not
    // "stalled". Everything else with a decent best is a candidate to revive.
    const stalled = promoted === 0 || (st.cycles || 0) >= 4;
    if (!stalled) continue;
    out.push({
      kind: 'resume', id: t.id, slug: t.slug, recipe: t.recipe, bestElo: elo,
      reason: `past best ≈ ${elo.toFixed(0)} Elo`
        + ` over ${st.cycles || 0} cycle(s), ${promoted} promotion(s)`
        + ` — has a saved best to warm-start from`,
      cmd: recipeResumeCmd(t.recipe),
    });
  }

  // Dataset-filter recipes (see featurize --min-elo / --drop-conflicts) not yet tried on the
  // reigning shape: refresh-v repairs stale `v` labels, but who PLAYED a game — hence its
  // position distribution and its result label — is fixed forever, and these filter that at
  // featurize time. Suggested against the most-promoted track's architecture (else the first).
  const FILTER_TRIALS = [
    { knobs: { filterWeak: 700 },
      reason: 'drop games whose weaker player is ≥700 Elo below the champion — off-distribution positions with blunder-decided result labels' },
    { knobs: { dropConflicts: 600 },
      reason: 'drop positions whose recorded search value (≥600cp) contradicts the game result — the result label is noise there' },
  ];
  const reigning = tracks.slice().sort((a, b) =>
    ((b.state && b.state.promotions) || 0) - ((a.state && a.state.promotions) || 0))[0];
  if (reigning) {
    const ids = new Set(tracks.map((t) => t.id));
    for (const trial of FILTER_TRIALS) {
      if (out.filter((o) => o.kind === 'new').length >= 4) break;
      const recipe = buildRecipe({ ...reigning.recipe, ...trial.knobs });
      if (ids.has(recipeId(recipe))) continue;
      out.push({
        kind: 'new', slug: recipeSlug(recipe), recipe,
        reason: trial.reason,
        cmd: recipeResumeCmd(recipe),
      });
    }
  }

  // Untried recipes, scored by the surrogate. The pool is generated (candidateArchitectures)
  // rather than listed, and inherits the reigning track's knobs so a suggestion differs from
  // what's working in ONE axis — the architecture — unless it's deliberately a knob trial.
  const rep = surrogateReport(loopDir, opts);
  const probe = Number(opts.probeCycles) || 5; // predict every candidate at the same budget
  const kappa = Number.isFinite(opts.kappa) ? Number(opts.kappa) : 1.5; // exploration weight
  const knobs = reigning
    ? { lambda: reigning.recipe.lambda, quietOnly: reigning.recipe.quietOnly, quant: reigning.recipe.quant,
        filterWeak: reigning.recipe.filterWeak, dropConflicts: reigning.recipe.dropConflicts }
    : { lambda: 1, quietOnly: false, quant: true };

  // Design region: dense cost within `costMult` of the reigning shape's. Anything pricier can't
  // pay for itself at a fixed-depth gate, and anything far cheaper is the distillation question
  // (a separate experiment — a smaller net's payoff is nps, which fixed depth doesn't measure).
  const costMult = Number(opts.costMult) || 3;
  const budget = reigning ? denseCost(reigning.recipe.hidden) * costMult : Infinity;
  const w0Floor = reigning ? (hiddenWidths(reigning.recipe.hidden)[0] || 0) / 2 : 0;
  const pool = [];
  for (const hidden of candidateArchitectures(12, budget, w0Floor)) {
    if (triedHidden.has(hidden)) continue;
    pool.push(buildRecipe({ ...knobs, hidden }));
  }
  // Knob trials are NOT added here: the FILTER_TRIALS block above already proposes them, and the
  // coverage design below keys on architecture alone (coverageKey reads `hidden`), so a knob
  // variation on the reigning shape lands in an already-occupied bucket and would never surface.
  // Keeping the two mechanisms separate means the arch axis gets a real design and the knob axis
  // keeps its explicit trials, instead of the two quietly cancelling out.

  // Coverage counted MARGINALLY (per axis), not per joint cell. With a handful of tracks the joint
  // cells are almost all empty, so joint-cell emptiness barely discriminates and the design happily
  // suggests five shapes that differ on one axis — a simulated rotation sequence went depth
  // 1/2/4/7/9/11 all at the same width, because each was its own untouched cell. Marginal counts
  // say instead "you have measured depth-1 already, go look at a depth you haven't", which is the
  // question worth answering at this sample size. The joint cell stays in as a weak tiebreak.
  const occD = new Map(), occW = new Map(), occC = new Map(), occCell = new Map();
  const bump = (m, k) => m.set(k, (m.get(k) || 0) + 1);
  for (const t of tracks) {
    const key = coverageKey(t.recipe.hidden);
    const [d, w, c] = key.split('/').map(Number);
    bump(occD, d); bump(occW, w); bump(occC, c); bump(occCell, key);
  }
  const coverageCost = (key) => {
    const [d, w, c] = key.split('/').map(Number);
    return 3 * (occD.get(d) || 0) + 2 * (occW.get(w) || 0) + (occC.get(c) || 0)
      + 0.5 * (occCell.get(key) || 0);
  };

  const known = new Set(tracks.map((t) => t.id));
  const scored = [];
  for (const recipe of pool) {
    const id = recipeId(recipe);
    if (known.has(id)) continue;
    known.add(id); // the pool can generate the same recipe twice (taper == block at some depths)
    const p = rep.model ? rep.model.predict(recipe, probe) : null;
    const bucket = coverageKey(recipe.hidden);
    const occupied = coverageCost(bucket);
    // With a trusted model, rank by UCB (promise + uncertainty). Without one, rank by how unmeasured
    // the candidate's REGION is, tie-broken by feature distance so the pick sits in the middle of
    // the empty bucket rather than on its edge.
    const novelty = p ? p.nearest : 0;
    const ucb = p ? p.mean + kappa * p.sigma : 0;
    scored.push({
      recipe, id, pred: p, novelty, bucket, occupied,
      score: rep.useful ? ucb : -occupied * 1e6 + novelty,
    });
  }
  scored.sort((a, b) => b.score - a.score);

  // Greedy diverse pick: after each choice, drop candidates that sit almost on top of it in
  // feature space. Five near-identical suggestions would waste the exploration budget.
  const picked = [];
  // Separation floor, auto-scaled: distances in this z-scored space depend on how spread the few
  // observations are, so a hard-coded threshold is meaningless (it was letting through four
  // near-identical shapes). Use the mean nearest-neighbour distance among the OBSERVED tracks —
  // suggestions should be at least as far apart as the experiments already run.
  const MIN_SEP = (() => {
    if (!rep.model || rep.obs.length < 2) return 0;
    const Z = rep.obs.map((o) => rep.model.normalize(o.x));
    const dists = Z.map((a, i) => Math.min(...Z.map((b, k) => (k === i ? Infinity
      : Math.sqrt(a.reduce((s, v, j) => s + (v - b[j]) ** 2, 0))))));
    return dists.reduce((a, b) => a + b, 0) / dists.length;
  })();
  if (!rep.useful) {
    // Explore mode: greedy coverage. Requiring only DISTINCT buckets isn't enough — the four
    // depth-1 buckets are all empty, so a naive pass fills every slot with single-layer nets that
    // differ solely in width. So after each pick, penalise reusing that pick's value on any one
    // axis (depth weighted hardest, being the most structurally distinct choice). The result
    // spreads the suggestions across depths AND widths AND cost bands.
    const useD = new Map(), useW = new Map(), useC = new Map();
    const bump = (m, k) => m.set(k, (m.get(k) || 0) + 1);
    const remaining = scored.slice();
    while (picked.length < 3 && remaining.length) {
      let bestI = 0, bestScore = -Infinity;
      for (let i = 0; i < remaining.length; i++) {
        const [d, w, cst] = remaining[i].bucket.split('/').map(Number);
        const penalty = remaining[i].occupied + 3 * (useD.get(d) || 0)
          + 2 * (useW.get(w) || 0) + (useC.get(cst) || 0);
        // Tiny novelty term only breaks ties within an equally-uncovered region.
        const s = -penalty + remaining[i].novelty * 1e-3;
        if (s > bestScore) { bestScore = s; bestI = i; }
      }
      const [c] = remaining.splice(bestI, 1);
      const [d, w, cst] = c.bucket.split('/').map(Number);
      bump(useD, d); bump(useW, w); bump(useC, cst);
      picked.push(c);
    }
  }
  for (const c of rep.useful ? scored : []) {
    if (picked.length >= 3) break;
    if (rep.model) {
      const cz = rep.model.normalize(featureVector(c.recipe, probe));
      const tooClose = picked.some((p) => {
        const pz = rep.model.normalize(featureVector(p.recipe, probe));
        return Math.sqrt(cz.reduce((s, v, j) => s + (v - pz[j]) ** 2, 0)) < MIN_SEP;
      });
      if (tooClose) continue;
    }
    picked.push(c);
  }

  for (const c of picked) {
    const parts = [];
    if (rep.useful && c.pred) parts.push(`predicts ≈ ${c.pred.mean.toFixed(0)} ±${c.pred.sigma.toFixed(0)} Elo after ${probe} cycles`);
    else if (c.bucket !== undefined) {
      const w = hiddenWidths(c.recipe.hidden);
      parts.push(`${c.occupied === 0 ? 'no track yet' : `only ${c.occupied} track(s)`} at depth ${w.length}`
        + ` / first layer ${w[0]} / ${denseCost(c.recipe.hidden).toLocaleString('en-US')} mults per node`);
    }
    out.push({
      kind: 'new', slug: recipeSlug(c.recipe), recipe: c.recipe,
      predElo: c.pred ? c.pred.mean : null,
      sigma: c.pred ? c.pred.sigma : null,
      novelty: c.novelty,
      mode: rep.useful ? 'ucb' : 'explore',
      reason: parts.join(', '),
      cmd: recipeResumeCmd(c.recipe),
    });
  }
  return out;
}
