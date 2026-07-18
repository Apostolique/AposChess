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

// --- Suggestions ---------------------------------------------------------------------

// A small ladder of architectures worth exploring, roughly increasing capacity. Used to
// suggest shapes with no track yet. (Strength is signal-limited, not capacity-limited —
// docs/first-layer-strategy.md — so wider isn't automatically better; these are exploration
// candidates to be gated the usual way, not recommendations.)
const ARCH_LADDER = ['32', '64', '128', '256', '128,32', '256,32', '256,64', '256,64,16', '384', '512,64'];

// Suggest what to try next, drawing on the registry:
//   kind 'resume' — a PAST recipe whose best net was promising (high estimated abs Elo) but
//                   that stalled (few promotions); resuming warm-starts from its saved best.
//   kind 'new'    — an architecture with NO track yet (never explored).
// Returns a ranked, de-duplicated list of { kind, id?, slug, recipe, reason, cmd, bestElo? }.
export function suggestRecipes(loopDir, opts = {}) {
  const tracks = readAllTracks(loopDir);
  const triedHidden = new Set(tracks.map((t) => t.recipe.hidden));
  const out = [];

  // Promising-but-stalled past recipes: rank by best estimated absolute Elo, prefer ones that
  // never promoted (their gains never landed) and haven't been touched recently.
  const promising = tracks
    .filter((t) => t.state && t.state.best)
    .map((t) => ({ t, elo: t.state.best.absElo }))
    .filter((x) => x.elo != null)
    .sort((a, b) => b.elo - a.elo);
  for (const { t, elo } of promising) {
    if (out.filter((o) => o.kind === 'resume').length >= 3) break;
    const st = t.state;
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

  // Never-tried architectures from the ladder (keep near-champion knobs otherwise).
  for (const h of ARCH_LADDER) {
    if (triedHidden.has(h)) continue;
    if (out.filter((o) => o.kind === 'new').length >= 4) break;
    const recipe = buildRecipe({ hidden: h, lambda: 1, quietOnly: false, quant: true });
    out.push({
      kind: 'new', slug: recipeSlug(recipe), recipe,
      reason: 'architecture never tried',
      cmd: recipeResumeCmd(recipe),
    });
  }
  return out;
}
