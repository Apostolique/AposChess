// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2019-2026 Jean-David Moisan
//
// Read-only browser for the train:loop EXPERIMENT REGISTRY (experiment-registry.mjs). Each
// distinct training recipe (architecture + λ + quiet filter + quant + trainer knobs) has a
// persistent track under training/data/loop/experiments/, so recipes you've tried aren't lost
// and can be warm-started again later. This lists them, details one, and — the point of the
// "what to try next" ask — SUGGESTS recipes when things stall: promising-but-stalled past
// recipes to revive (they carry a saved best net to warm-start from) and architectures never
// tried. Nothing here writes or spawns, so it's safe against a live loop.
//
// The "what to try next" list is a sequential design over recipe space, not a fixed ladder: each
// finished track is an observation, a surrogate fitted to them scores a generated candidate pool,
// and the ranking switches from space-filling exploration to predicted-Elo once the surrogate
// actually beats predicting the mean. See suggestRecipes in experiment-registry.mjs.
//
//   npm run train:experiments                 # table of all tracks + suggestions
//   npm run train:experiments -- --show=<id>  # one track's recipe + per-cycle history
//                                             #   (<id> matches the 8-char id or the slug)
//   npm run train:experiments -- --suggest    # just the "what to try next" suggestions
//   npm run train:experiments -- --min-obs=N --min-gain=G   # move the surrogate's trust bars
//   npm run train:experiments -- --cost-mult=M --kappa=K    # design region / exploration weight

import { existsSync, readFileSync } from 'node:fs';
import { dirname, resolve, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { weightsHash } from './vtag.mjs';
import {
  readAllTracks, readHistory, recipeLabel, recipeResumeCmd, suggestRecipes, trackPaths,
  ledgerBestByVersion, reAnchoredAbsElo, trackBestAbs, surrogateReport,
} from './experiment-registry.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const webDir = resolve(here, '..');
const repoDir = resolve(webDir, '..');
const loopDir = resolve(repoDir, 'training', 'data', 'loop');
const championFile = resolve(webDir, 'src', 'nn-weights.json');

const args = Object.fromEntries(
  process.argv.slice(2).map((a) => {
    const m = a.replace(/^--/, '').split('=');
    return [m[0], m.length > 1 ? m[1] : true];
  }),
);

const pad = (s, w) => String(s).padEnd(w);
const padL = (s, w) => String(s).padStart(w);
const signed = (n) => (Number.isFinite(n) ? (n >= 0 ? '+' : '') + n.toFixed(0) : '—');
const parseTs = (ts) => (ts ? new Date(String(ts).replace(' ', 'T') + 'Z') : null);
const fmtLocal = (d) => {
  if (!d) return '—';
  const p = (n) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`;
};

const champHash = weightsHash(championFile);

// Did this track produce the CURRENT champion? True if any promoted cycle in its history has
// the champion's content hash (best-effort — histories are small).
function producedChampion(dir) {
  if (champHash === '?') return false;
  for (const h of readHistory(dir)) if (h.promoted && h.hash === champHash) return true;
  return false;
}

const tracks = readAllTracks(loopDir);

// Re-anchor every track's stored absElo onto today's ledger (the BT pool re-fits each cycle, so a
// fixed champion's rating — hence an old stored absElo — drifts; see experiment-registry.mjs). Ranked
// "best" per track is recomputed from history via that same re-anchoring, not read from state.best.
const ledgerElo = ledgerBestByVersion(loopDir);
const bestByTrack = new Map(tracks.map((t) => [t.id, trackBestAbs(t.dir, ledgerElo)]));

// --- Suggestions block (shared by default view and --suggest). --------------------------
// Tuning knobs for the suggester, all optional. --min-obs/--min-gain move the bars the surrogate
// has to clear before predictions are ranked on (lower them to inspect the UCB ranking early);
// --cost-mult widens or tightens the design region; --kappa weights exploration in UCB mode.
const suggestOpts = {};
for (const [flag, key] of [['min-obs', 'minObs'], ['min-gain', 'minGain'],
  ['cost-mult', 'costMult'], ['kappa', 'kappa'], ['probe-cycles', 'probeCycles']]) {
  if (args[flag] !== undefined && args[flag] !== true) suggestOpts[key] = Number(args[flag]);
}

function printSuggestions() {
  const sugg = suggestRecipes(loopDir, suggestOpts);
  const rep = surrogateReport(loopDir, suggestOpts);
  console.log('\n=== What to try next ===');
  // Say what the suggester actually knows before it makes a claim — the ranking means something
  // different in each mode (predicted Elo vs pure information gain).
  console.log(`  Model: ${rep.verdict}`);
  if (!sugg.length) {
    console.log('  (no suggestions — no registry yet; just run `npm run train:loop -- --hidden=…`).');
    return;
  }
  const resume = sugg.filter((s) => s.kind === 'resume');
  const fresh = sugg.filter((s) => s.kind === 'new');
  if (resume.length) {
    console.log('\n  Revive a promising-but-stalled recipe (warm-starts from its saved best):');
    for (const s of resume) {
      console.log(`    • ${pad(s.slug, 22)} ${s.reason}`);
      console.log(`        ${s.cmd}`);
    }
  }
  if (fresh.length) {
    console.log(`\n  ${rep.useful ? 'Highest upper-confidence bound (predicted Elo + uncertainty)' : 'Most informative untried recipes (space-filling)'}:`);
    for (const s of fresh) {
      console.log(`    • ${pad(s.slug, 22)} ${s.reason}`);
      console.log(`        ${s.cmd}`);
    }
  }
  console.log('\n  Probe a suggestion for a few cycles, then stop — a track that hasn\'t promoted by');
  console.log('  ~8-10 cycles is far less likely to than a fresh shape is on its first (see the');
  console.log('  registry: 2 of 4 promotions landed on a new track\'s cycle 1). Every finished track');
  console.log('  is one more observation, so short probes teach the model faster than long grinds.');
  console.log('\n  (Strength is signal-limited, not capacity-limited — docs/first-layer-strategy.md —');
  console.log('   so a bigger net isn\'t automatically better; gate every recipe head-to-head as usual.)');
}

if (args.suggest && !args.show) {
  printSuggestions();
  process.exit(0);
}

// --- Detail one track (--show=<id|slug>). ----------------------------------------------
if (args.show) {
  const key = String(args.show);
  const t = tracks.find((x) => x.id === key || x.slug === key)
    || tracks.find((x) => x.id.startsWith(key));
  if (!t) {
    console.error(`No track matching '${key}'. Run \`npm run train:experiments\` to list them.`);
    process.exit(1);
  }
  const st = t.state || {};
  console.log(`\nTrack ${t.slug}  [${t.id}]${producedChampion(t.dir) ? '   ← produced the current champion' : ''}`);
  console.log(`  recipe: ${recipeLabel(t.recipe)}`);
  console.log(`  runs ${st.runs ?? 0} · cycles ${st.cycles ?? 0} · promotions ${st.promotions ?? 0}`
    + ` · created ${fmtLocal(parseTs(st.createdTs))} · last run ${fmtLocal(parseTs(st.lastRunTs))}`);
  const best = bestByTrack.get(t.id); // re-anchored best, not the drift-prone state.best snapshot
  if (best) {
    console.log(`  best net: abs Elo ${Number.isFinite(best.absElo) ? best.absElo.toFixed(0) : '—'}  (gate ${(best.score * 100).toFixed(1)}%,`
      + ` edge ${signed(best.edgeElo)}, ${best.sprt}, run ${best.run} cycle ${best.cycle})`
      + `  → ${existsSync(trackPaths(t.dir).best) ? 'best.json saved' : 'best.json missing'}`);
  }
  console.log(`  resume: ${recipeResumeCmd(t.recipe)}`);

  const hist = readHistory(t.dir);
  console.log(`\n  History (${hist.length} cycle(s)):`);
  if (!hist.length) console.log('    (none yet)');
  else {
    console.log(`    ${pad('run/cyc', 9)}${pad('score', 8)}${pad('edge', 7)}${pad('absElo', 8)}${pad('SPRT', 14)}${pad('when', 18)}`);
    for (const h of hist) {
      console.log('    ' + pad(`${h.run ?? '?'}/${h.cycle ?? '?'}`, 9)
        + pad(Number.isFinite(h.score) ? (h.score * 100).toFixed(1) + '%' : '?', 8)
        + pad(signed(h.edgeElo), 7)
        + pad((() => { const a = reAnchoredAbsElo(h, ledgerElo); return Number.isFinite(a) ? a.toFixed(0) : '—'; })(), 8)
        + pad(h.promoted ? 'H1 PROMOTED' : h.sprt, 14)
        + pad(fmtLocal(parseTs(h.ts)), 18));
    }
  }
  process.exit(0);
}

// --- Default view: table of all tracks + suggestions. -----------------------------------
console.log(`\nAposChess train:loop experiment registry — ${tracks.length} track(s)`);
console.log(`Current champion: ${champHash === '?' ? '(none/material)' : champHash}`
  + (() => { try { const a = JSON.parse(readFileSync(championFile, 'utf8')).arch; return Array.isArray(a) ? `  arch [${a.join(',')}]` : ''; } catch { return ''; } })());

if (!tracks.length) {
  console.log('\nNo experiment tracks yet. A track is created automatically the first time you run');
  console.log('`npm run train:loop` with a given recipe (architecture / λ / --quiet-only / --float / …).');
  printSuggestions();
  process.exit(0);
}

// Sort by best RE-ANCHORED abs Elo desc (unknown last), then by cycles.
const rows = tracks.slice().sort((a, b) => {
  const ea = bestByTrack.get(a.id)?.absElo ?? -Infinity, eb = bestByTrack.get(b.id)?.absElo ?? -Infinity;
  if (eb !== ea) return eb - ea;
  return (b.state?.cycles ?? 0) - (a.state?.cycles ?? 0);
});

console.log('');
console.log(`  ${pad('recipe', 24)}${pad('id', 10)}${padL('runs', 5)} ${padL('cyc', 4)} ${padL('prom', 5)} `
  + `${padL('best-absElo', 12)} ${padL('best%', 7)}  ${pad('last run', 18)}`);
for (const t of rows) {
  const st = t.state || {};
  const best = bestByTrack.get(t.id); // re-anchored, not the drift-prone state.best snapshot
  const star = producedChampion(t.dir) ? ' ★' : '';
  console.log('  ' + pad(t.slug, 24) + pad(t.id, 10)
    + padL(st.runs ?? 0, 5) + ' ' + padL(st.cycles ?? 0, 4) + ' ' + padL(st.promotions ?? 0, 5) + ' '
    + padL(best && Number.isFinite(best.absElo) ? best.absElo.toFixed(0) : '—', 12) + ' '
    + padL(best && Number.isFinite(best.score) ? (best.score * 100).toFixed(1) + '%' : '—', 7)
    + '  ' + pad(fmtLocal(parseTs(st.lastRunTs)), 18) + star);
}
console.log('\n  ★ = produced the current champion.  Detail one: `npm run train:experiments -- --show=<id>`.');

printSuggestions();
