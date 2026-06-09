// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2019-2026 Jean-David Moisan
//
// One-command training pipeline for the neural-net evaluation. Runs the whole
// loop so you don't have to chain the steps by hand:
//
//   generate positions  ->  featurize  ->  train (auto epochs)  ->  [match]  ->  [build]
//
// With --generations=N it repeats gen+train N times, and from the 2nd generation
// on it generates data with the freshly trained net (--eval=nn), so the net
// bootstraps off its own improving play. The epoch count is NOT set here — the
// trainer stops itself via early stopping (lowest validation loss wins).
//
// Usage (run from web/):
//   npm run train -- [options]
// Options:
//   --games=N         self-play games per generation (default 300)
//   --jobs=N          parallel worker threads for gen + match (default: all cores)
//   --depth=D         search depth per move while generating (default 4)
//   --generations=N   gen+train cycles; gen 2+ use the trained net (default 1)
//   --eval=NAME       engine for the FIRST generation's data: 'handcrafted'
//                     (default) or 'nn' (reuse existing weights)
//   --fresh           delete the accumulated dataset before starting
//   --match=N         after training, play N games vs the handcrafted eval to
//                     report strength (0 = skip, default 0)
//   --no-build        skip the production build at the end
//   --hidden=H        hidden-layer size passed to the trainer
//   --patience=N      early-stopping patience passed to the trainer

import { spawnSync } from 'node:child_process';
import { existsSync, rmSync, copyFileSync, readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const webDir = resolve(here, '..');
const repoDir = resolve(webDir, '..');
const genScript = resolve(here, 'gen-selfplay.mjs');
const featurizeScript = resolve(here, 'featurize.mjs');
const matchScript = resolve(here, 'selfplay.mjs');
const trainPy = resolve(repoDir, 'training', 'train.py');
const dataFile = resolve(repoDir, 'training', 'data', 'selfplay.jsonl');
const weightsFile = resolve(webDir, 'src', 'nn-weights.json');
const backupFile = resolve(webDir, 'src', 'nn-weights.bak.json');

const args = Object.fromEntries(
  process.argv.slice(2).map((a) => {
    const m = a.replace(/^--/, '').split('=');
    return [m[0], m.length > 1 ? m[1] : true];
  }),
);
const num = (v, d) => (v === undefined ? d : Number(v));
const cfg = {
  games: num(args.games, 300),
  jobs: args.jobs, // forwarded as-is to gen/match; undefined -> their default (all cores)
  depth: num(args.depth, 4),
  generations: num(args.generations, 1),
  evalFirst: typeof args.eval === 'string' ? args.eval : 'handcrafted',
  fresh: !!args.fresh,
  match: num(args.match, 0),
  build: !args['no-build'],
  hidden: args.hidden,
  patience: args.patience,
};

// Run a Node script directly (shell:false so paths with spaces are safe).
function runNode(label, script, scriptArgs, cwd = webDir) {
  console.log(`\n=== ${label} ===`);
  const r = spawnSync(process.execPath, [script, ...scriptArgs], { stdio: 'inherit', cwd });
  if (r.status !== 0) fail(label, r.status);
}
// Run a shell command (for python / npm). Paths are quoted for spaces.
function runShell(label, cmd, cwd = repoDir) {
  console.log(`\n=== ${label} ===\n> ${cmd}`);
  const r = spawnSync(cmd, { stdio: 'inherit', shell: true, cwd });
  if (r.status !== 0) fail(label, r.status);
}
function fail(label, status) {
  console.error(`\n${label} failed (exit ${status}). Stopping pipeline.`);
  process.exit(status ?? 1);
}

function findPython() {
  for (const c of ['python', 'py', 'python3']) {
    const r = spawnSync(c, ['--version'], { shell: true });
    if (r.status === 0) return c;
  }
  console.error('No Python found on PATH (tried python, py, python3).\n'
    + 'Install Python and the trainer deps: pip install -r training/requirements.txt');
  process.exit(1);
}

const python = findPython();
console.log(`AposChess training pipeline: ${cfg.generations} generation(s), `
  + `${cfg.games} games/gen at depth ${cfg.depth}, python="${python}".`);

if (cfg.fresh && existsSync(dataFile)) {
  rmSync(dataFile);
  console.log('Removed existing dataset (--fresh).');
}

// Safety net: training overwrites nn-weights.json with a freshly trained net, and
// improvement isn't guaranteed. Back up the current (real) weights first so a
// regressing run can be rolled back — restore with:
//   copy src\nn-weights.bak.json src\nn-weights.json   (then npm run build)
if (existsSync(weightsFile)) {
  try {
    if (JSON.parse(readFileSync(weightsFile, 'utf8')).arch) {
      copyFileSync(weightsFile, backupFile);
      console.log(`Backed up current weights -> ${backupFile}`);
    }
  } catch { /* unreadable/placeholder weights: nothing worth backing up */ }
}

const t0 = Date.now();
for (let g = 0; g < cfg.generations; g++) {
  const evalName = g === 0 ? cfg.evalFirst : 'nn';
  console.log(`\n############ Generation ${g + 1}/${cfg.generations}  (data from: ${evalName}) ############`);

  // 1. Generate data (appends to the dataset). The generator reads/writes the
  //    weights file in src/ directly, so no build is needed between generations.
  const genArgs = [`--games=${cfg.games}`, `--depth=${cfg.depth}`, `--eval=${evalName}`];
  if (cfg.jobs !== undefined) genArgs.push(`--jobs=${cfg.jobs}`);
  runNode('Generate self-play data', genScript, genArgs);

  // 2. Featurize the raw positions into training inputs for the current net
  //    (selfplay.jsonl -> selfplay.features.jsonl) so a feature change is picked up.
  runNode('Featurize positions', featurizeScript, []);

  // 3. Train (early stopping picks the epoch count; exports src/nn-weights.json).
  let trainCmd = `${python} "${trainPy}"`;
  if (cfg.hidden) trainCmd += ` --hidden=${cfg.hidden}`;
  if (cfg.patience) trainCmd += ` --patience=${cfg.patience}`;
  runShell('Train network', trainCmd);
}

// 4. Optional strength check against the handcrafted eval.
if (cfg.match > 0) {
  const matchArgs = [`--games=${cfg.match}`, `--depth=${cfg.depth}`, '--eval-a=nn'];
  if (cfg.jobs !== undefined) matchArgs.push(`--jobs=${cfg.jobs}`);
  runNode('Measure vs handcrafted', matchScript, matchArgs);
}

// 5. Bundle the new weights into the deployable app.
if (cfg.build) runShell('Build web app', 'npm run build', webDir);

const mins = ((Date.now() - t0) / 60000).toFixed(1);
console.log(`\n✅ Pipeline done in ${mins} min. New weights: web/src/nn-weights.json`);
console.log('   Pick "Engine → Neural net" in the app (npm run dev reloads weights; '
  + 'the production build is already updated).');
if (cfg.match === 0) {
  console.log(`   To gauge strength: npm run match -- --eval-a=nn --depth=${cfg.depth} --games=100`);
}
if (existsSync(backupFile)) {
  console.log('   If this run came out WORSE than before, roll back the previous net:');
  console.log('     copy src\\nn-weights.bak.json src\\nn-weights.json   (then npm run build)');
}
