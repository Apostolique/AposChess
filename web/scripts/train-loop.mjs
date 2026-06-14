// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2019-2026 Jean-David Moisan
//
// Gated self-play improvement loop ("expert iteration"). Each cycle:
//   generate games with the CHAMPION  ->  featurize  ->  train a CANDIDATE
//   ->  play CANDIDATE vs CHAMPION (SPRT)  ->  promote the candidate ONLY if it wins.
// Promotion is gated on a statistically-significant head-to-head win, so the champion
// can NEVER regress — it either improves or stays put. Runs until --cycles or Ctrl-C.
//
// The champion is web/src/nn-weights.json (what `gen --eval=nn` plays with, and the
// Node-tools default). On each promotion it's also published to the web catalog as
// 'loop-champion', so you can play the current champion in the app (rebuild for the
// production bundle; `npm run dev` serves it live).
//
// Reality check: this makes the loop SAFE (no regression), but improvement is not
// guaranteed — the net is signal-limited, so the gate may rarely fire unless the
// labels get better. That's why generation defaults to a DEEPER search than the eval
// alone sees (better outcomes = better labels). Matches are slow; expect a handful of
// cycles per hour. See training/README.md.
//
// Usage (run from web/):
//   npm run train:loop -- [options]
// Options:
//   --batch=N       games generated per cycle (default 200)
//   --depth=D       search depth while generating (default 6 — deeper = better labels)
//   --cycles=N      stop after N cycles (default: run forever until Ctrl-C)
//   --gate-games=N  max games in the candidate-vs-champion match (default 800 — mature
//                   gains are small, and small edges need more games to clear the SPRT)
//   --gate-depth=D  search depth for the gating match (default 4)
//   --elo1=E        SPRT H1 promotion threshold in Elo (default 20; elo0 is 0). This
//                   is the SMALLEST gain worth promoting; it must be wide enough that
//                   the SPRT can actually decide within --gate-games. A too-small band
//                   (e.g. [0,5] over 400 games) needs a candidate ~+170 Elo to fire,
//                   so real improvements get rejected — keep elo1 vs gate-games sane.
//   --lambda=L      TD/bootstrap target mix for training the candidate (default 1 =
//                   pure game result; <1 leans on the champion's own search value,
//                   an unbiased bootstrap — recorded because generation uses the net)
//   --hidden=H      candidate architecture (default: same shape as the champion)
//   --cold          train each candidate from random init instead of warm-starting
//                   from the champion (warm start fine-tunes in a few epochs and
//                   starts at champion strength; cold occasionally explores a
//                   different basin but relearns everything each cycle)
//   --skip-gen      skip generation on the FIRST cycle and go straight to
//                   featurize -> train -> gate on the dataset as it stands. Use it
//                   to resume after interrupting a run mid-generation: completed
//                   games were already flushed to the dataset, so this gates them
//                   instead of generating a whole new batch first. Later cycles
//                   generate normally.
//   --no-harvest    don't save the gate match's games into the dataset. By default
//                   the gate's games (up to --gate-games per cycle — comparable
//                   volume to generation, already paid for) are appended to
//                   selfplay.jsonl via the match runner's --save-games, with the
//                   search value `v` kept only from the engine the gate proved
//                   stronger; the next cycle's (incremental) featurize folds them
//                   in. Note they're played at --gate-depth, shallower labels than
//                   generation's --depth.
//   --jobs=N        parallel workers for gen + match
//   --fresh         clear the dataset before the first cycle (clean deep-search start)
//   --refresh-frac=P  after each PROMOTION, recompute `v` on a random fraction P of the
//                   dataset with the new champion (value iteration; 0 = off, default).
//                   Only runs on promotion — between promotions the champion (hence v)
//                   is unchanged, so a refresh would just recompute identical values.
//                   Cost scales with P × depth; e.g. P=0.2 touches the whole set every
//                   ~5 promotions. Re-featurize happens next cycle, so it flows in.
//   --refresh-depth=D  search depth for the refresh (default 3 — cheap, like the
//                   backfilled majority; a depth-6 refresh of a big fraction is hours)
//   --refresh-cycle=P  EVERY cycle (between generation and featurize), recompute `v` on
//                   a random fraction P of the dataset with the current champion
//                   (default 0.01; 0 = off). Unlike --refresh-frac this helps between
//                   promotions too: most records carry `v` from OLDER champions (or
//                   shallower gate-harvest/backfill searches), so re-labeling them with
//                   the current champion steadily upgrades the TD target even while the
//                   champion is unchanged. Rule of thumb: 1% of a ~2.5M-position set at
//                   depth 6 costs about as much as generating a 200-game batch.
//   --refresh-cycle-depth=D  search depth for the per-cycle refresh (default: --depth,
//                   so the re-labels match generation's deep-label quality)
//   --no-refresh    skip ALL value refreshing (both --refresh-cycle and --refresh-frac),
//                   regardless of their values — shorthand for --refresh-cycle=0 with no
//                   promotion refresh, for when you want the fastest possible cycles and
//                   accept the staler `v` targets
//
// Candidate lineage (automatic, disabled by --cold): when the gate is inconclusive but
// the candidate scored >= 50%, the candidate is KEPT (loop/lineage.json) and the next
// cycle's candidate warm-starts from IT instead of the champion — so sub-threshold
// gains (+10-ish Elo, real but below the SPRT's resolution) accumulate across cycles
// until the lineage clears the gate, instead of being re-derived and discarded every
// cycle. The champion is still protected by the gate; a candidate scoring < 50% (or a
// decided H0) resets the lineage back to the champion.

import { spawnSync } from 'node:child_process';
import {
  existsSync, rmSync, mkdirSync, copyFileSync, readFileSync, writeFileSync, appendFileSync, statSync,
} from 'node:fs';
import { dirname, resolve, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { fmtDur, fmtMB } from './fmt.mjs';
import { weightsHash } from './vtag.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const webDir = resolve(here, '..');
const repoDir = resolve(webDir, '..');
const dataDir = resolve(repoDir, 'training', 'data');
const loopDir = join(dataDir, 'loop');
mkdirSync(loopDir, { recursive: true });

const genScript = resolve(here, 'gen-selfplay.mjs');
const featurizeScript = resolve(here, 'featurize.mjs');
const matchScript = resolve(here, 'selfplay.mjs');
const refreshScript = resolve(here, 'refresh-v.mjs');
const rankScript = resolve(here, 'rank-engines.mjs');
const trainPy = resolve(repoDir, 'training', 'train.py');

const rawFile = join(dataDir, 'selfplay.jsonl');
const featFile = join(dataDir, 'selfplay.features.jsonl');
const champion = resolve(webDir, 'src', 'nn-weights.json');
const candidate = join(loopDir, 'candidate.json');
const lineage = join(loopDir, 'lineage.json'); // rejected-but-positive candidate, warm-start source
const prevChampion = join(loopDir, 'champion-prev.json');
// Archive of every champion that has labelled data, keyed by its content hash (= the
// nn `vs` version stamped onto that data). Lets historical v-contributors be
// re-instantiated for the strength-ranking that drives smart weakest-first v refresh
// (see scripts/vtag.mjs). git-ignored (under training/data).
const championsDir = join(loopDir, 'champions');
// Copy `file` (a champion) into the archive under its hash; returns the hash.
function archiveChampion(file) {
  const hash = weightsHash(file);
  if (hash === '?') return hash;
  mkdirSync(championsDir, { recursive: true });
  const dest = join(championsDir, `${hash}.json`);
  if (!existsSync(dest)) copyFileSync(file, dest);
  return hash;
}
const resultFile = join(loopDir, 'match.json');
const ledgerFile = join(loopDir, 'engine-elo.json'); // engine-strength ledger (npm run rank)
const logFile = join(loopDir, 'loop.log');
const publicNN = resolve(webDir, 'public', 'nn');
const manifestFile = join(publicNN, 'manifest.json');
const loopChampPub = join(publicNN, 'loop-champion.json');

const args = Object.fromEntries(
  process.argv.slice(2).map((a) => {
    const m = a.replace(/^--/, '').split('=');
    return [m[0], m.length > 1 ? m[1] : true];
  }),
);
const num = (v, d) => (v === undefined ? d : Number(v));
const cfg = {
  batch: num(args.batch, 200),
  depth: num(args.depth, 6),
  cycles: args.cycles !== undefined ? Number(args.cycles) : Infinity,
  gateGames: num(args['gate-games'], 800),
  gateDepth: num(args['gate-depth'], 4),
  elo1: num(args.elo1, 20), // wide enough that SPRT can decide within --gate-games
  lam: num(args.lambda, 1), // TD target mix passed to train.py (1 = pure result)
  hidden: typeof args.hidden === 'string' ? args.hidden : null,
  jobs: args.jobs,
  fresh: !!args.fresh,
  cold: !!args.cold,
  skipGen: !!args['skip-gen'],
  harvest: !args['no-harvest'],
  // After each PROMOTION (the only time the champion — hence the `v` target — changes),
  // recompute `v` on a random fraction of the dataset with the new champion (value
  // iteration). 0 = off. Partial keeps cost amortized and average staleness ~1/frac
  // promotions. Refresh search depth defaults to the gen depth.
  // --no-refresh zeroes both, overriding any explicit fractions.
  refreshFrac: args['no-refresh'] ? 0 : num(args['refresh-frac'], 0),
  // Cheap by default (depth 3, like the backfilled majority): a depth-6 refresh of a
  // big fraction is many hours. Raise it to trade speed for value accuracy.
  refreshDepth: num(args['refresh-depth'], 3),
  // Per-cycle refresh: a small slice of the dataset re-labeled with the current champion
  // every cycle. Helps between promotions too — most `v` in the set came from older
  // champions or shallower searches, so "unchanged champion" does NOT mean "nothing to
  // refresh"; only records the current champion already labeled at this depth are no-ops.
  refreshCycle: args['no-refresh'] ? 0 : num(args['refresh-cycle'], 0.01),
  refreshCycleDepth: num(args['refresh-cycle-depth'], num(args.depth, 6)),
  // Engine ranking for smart weakest-first v refresh. On by default: each promotion adds
  // the new champion to the Elo ledger (incrementally — one matchup vs the stable hc
  // anchor), and the refreshes below read the ledger to relabel the WEAKEST engine's `v`
  // first instead of a blind random fraction. --no-rank reverts to the classic refresh.
  rank: !args['no-rank'],
  rankDepth: num(args['rank-depth'], 4),
  // Default to the gate's game count: the ledger needs enough games to resolve the real
  // ordering (champions sit only ~elo1 apart), so it matches the gate's match length.
  rankGames: num(args['rank-games'], num(args['gate-games'], 800)),
};

function findPython() {
  for (const c of ['python', 'py', 'python3']) {
    if (spawnSync(`${c} --version`, { shell: true }).status === 0) return c; // string form: no arg-escaping warning
  }
  console.error('No Python found (tried python, py, python3). pip install -r training/requirements.txt');
  process.exit(1);
}
const python = findPython();

// The candidate keeps the champion's shape unless overridden, so the gate measures
// "did the new data help?", not "is a different net bigger?".
function championHidden() {
  if (cfg.hidden) return cfg.hidden;
  try {
    const a = JSON.parse(readFileSync(champion, 'utf8')).arch;
    if (Array.isArray(a) && a.length >= 3) return a.slice(1, -1).join(',');
  } catch { /* fall through */ }
  return '64';
}

const stamp = () => new Date().toISOString().replace('T', ' ').slice(0, 19);
const hms = () => new Date().toTimeString().slice(0, 8);
// Console lines carry a local time-of-day stamp: the loop runs unattended for
// hours, so "when did this happen" matters when scrolling back. The log file
// keeps the full ISO stamp.
function log(line) {
  console.log(`[${hms()}] ${line}`);
  appendFileSync(logFile, `[${stamp()}] ${line}\n`);
}

let stopping = false;
process.on('SIGINT', () => { stopping = true; console.log('\n  Ctrl-C: stopping after this cycle…'); });

// Run a step; return true on success. A SIGINT to a child shows as a null/ signalled
// status — treat that as "stop", not a hard failure.
function run(label, cmd, argv, cwd = webDir) {
  if (stopping) return false;
  console.log(`\n--- ${label} — ${hms()} ---`);
  // APOS_CHILD tells the child tools they're orchestrated: they use a SIGINT-only
  // graceful stop instead of grabbing the TTY's raw mode, so the loop's own Ctrl-C
  // (stop after this cycle) keeps working — see scripts/stop.mjs.
  const r = spawnSync(cmd, argv, { stdio: 'inherit', cwd, env: { ...process.env, APOS_CHILD: '1' } });
  if (r.signal) { stopping = true; return false; }
  // Windows delivers console Ctrl-C to the whole process group; the child then exits
  // with STATUS_CONTROL_C_EXIT (0xC000013A) instead of a signal — an interrupt, not a crash.
  if (r.status === 0xC000013A) { stopping = true; log(`${label} interrupted (Ctrl-C); stopping loop.`); return false; }
  if (r.status !== 0) { log(`${label} FAILED (exit ${r.status}); stopping loop.`); return false; }
  return true;
}

// Update the engine-strength ledger incrementally: only the engines it doesn't yet rank
// (e.g. a just-promoted champion) play a matchup against the stable hc anchor; the rest
// are reused. --no-scan keeps it fast (record counts aren't needed to drive the refresh).
// Maintenance, like the refreshes — a failure logs but doesn't stop the loop.
function runRank(label) {
  if (!cfg.rank) return;
  run(label, process.execPath,
    [rankScript, '--incremental', '--no-scan', '--no-save-games',
      `--depth=${cfg.rankDepth}`, `--games=${cfg.rankGames}`, `--out=${ledgerFile}`, ...jobArg]);
}

// Build refresh-v args. With ranking on AND a ledger present, target the WEAKEST cohort
// via the ledger (relabeling the worst `v` first) and recompute with the current champion
// — passed explicitly so a briefly-stale ledger can never pick a weaker engine. Otherwise
// the classic random-fraction refresh with the champion.
function refreshArgs(frac, depth) {
  const a = [refreshScript, `--frac=${frac}`, `--depth=${depth}`, `--seed=${Date.now()}`, ...jobArg];
  return (cfg.rank && existsSync(ledgerFile))
    ? [...a, `--ledger=${ledgerFile}`, '--eval=nn', `--weights=${champion}`]
    : [...a, '--refresh', `--weights=${champion}`];
}
const refreshMode = () => (cfg.rank && existsSync(ledgerFile)) ? 'weakest-first' : 'random';

function upsertManifest(arch) {
  let man = { default: 'balanced-64', nets: [] };
  try { man = JSON.parse(readFileSync(manifestFile, 'utf8')); } catch { /* new manifest */ }
  man.nets = (man.nets || []).filter((n) => n.name !== 'loop-champion');
  man.nets.push({ name: 'loop-champion', file: 'loop-champion.json', arch, note: 'Latest gated train:loop champion.' });
  man.nets.sort((a, b) => a.name.localeCompare(b.name));
  writeFileSync(manifestFile, JSON.stringify(man, null, 2) + '\n');
}

if (!existsSync(champion)) {
  console.error(`No champion at ${champion}. Train a net first (e.g. npm run train:fit).`);
  process.exit(1);
}
// Archive the starting champion too — it labels the data generated before the first
// promotion, so its v-contributors must stay reconstructable like every later champion.
archiveChampion(champion);
if (cfg.fresh && existsSync(rawFile)) { rmSync(rawFile); log('Cleared dataset (--fresh).'); }

const hidden = championHidden();
// A leftover lineage only makes sense for the warm path and the current candidate
// shape; otherwise start the lineage over from the champion.
if (existsSync(lineage)) {
  let drop = cfg.cold ? 'cold run' : null;
  if (!drop) {
    try {
      const a = JSON.parse(readFileSync(lineage, 'utf8')).arch;
      if (!Array.isArray(a) || a.slice(1, -1).join(',') !== hidden) drop = 'shape mismatch';
    } catch { drop = 'unreadable'; }
  }
  if (drop) { rmSync(lineage); log(`Discarded stale lineage (${drop}).`); }
}
log(`train:loop start — batch ${cfg.batch} @ depth ${cfg.depth} | gate ${cfg.gateGames}g @ depth ${cfg.gateDepth} `
  + `SPRT(0,${cfg.elo1}) | candidate hidden=[${hidden}] λ=${cfg.lam} ${cfg.cold ? 'cold' : 'warm'} start`
  + `${existsSync(lineage) ? ' (resuming lineage)' : ''} | `
  + `refresh/cycle ${cfg.refreshCycle > 0 ? `${(cfg.refreshCycle * 100).toFixed(1)}% @ depth ${cfg.refreshCycleDepth}` : 'off'} | `
  + `refresh on promotion ${cfg.refreshFrac > 0 ? `${(cfg.refreshFrac * 100).toFixed(0)}% @ depth ${cfg.refreshDepth}` : 'off'} | `
  + `rank ${cfg.rank ? `on (hc anchor, depth ${cfg.rankDepth}, ${cfg.rankGames}g/new engine)` : 'off'} | `
  + `cycles ${cfg.cycles === Infinity ? '∞' : cfg.cycles}`);

const jobArg = cfg.jobs !== undefined ? [`--jobs=${cfg.jobs}`] : [];

// No startup ranking: the ledger is built lazily on the first promotion (see runRank in
// the promote branch). Until a champion is crowned there's nothing new to rank, so the
// early cycles simply refresh with the classic random fraction; weakest-first kicks in
// once the first promotion has seeded the ledger.

const loopT0 = Date.now();
let promotions = 0;
for (let c = 1; c <= cfg.cycles && !stopping; c++) {
  const cycleT0 = Date.now();
  const dataset = existsSync(rawFile) ? ` — dataset ${fmtMB(statSync(rawFile).size)}` : '';
  log(`===== cycle ${c}${cfg.cycles === Infinity ? '' : `/${cfg.cycles}`}${dataset} =====`);

  // 1. Generate games with the champion (deeper search than the eval sees).
  //    --skip-gen: on the first cycle only, gate the games an interrupted earlier
  //    run already flushed to the dataset instead of generating a new batch.
  if (c === 1 && cfg.skipGen) {
    log('Skipping generation (--skip-gen): gating the existing dataset.');
  } else if (!run('Generate (champion self-play)', process.execPath,
    [genScript, `--games=${cfg.batch}`, `--depth=${cfg.depth}`, '--eval=nn', ...jobArg])) break;

  // 1b. Per-cycle value refresh: re-label a small random slice of the dataset with the
  //     current champion. Most records carry `v` from older champions or shallower
  //     searches, so this upgrades targets even between promotions. The seed varies per
  //     cycle so coverage spreads across the set instead of re-picking the same slice.
  if (cfg.refreshCycle > 0) {
    if (!run(`Refresh v (${(cfg.refreshCycle * 100).toFixed(1)}% ${refreshMode()} @ depth ${cfg.refreshCycleDepth})`,
      process.execPath, refreshArgs(cfg.refreshCycle, cfg.refreshCycleDepth))) break;
  }

  // 2. Featurize the raw positions for the current feature set. (After a refresh this
  //    is a full pass — the in-place rewrite invalidates the incremental prefix.)
  if (!run('Featurize', process.execPath, [featurizeScript])) break;

  // 3. Train a candidate (same shape as the champion) to a side file. --lambda blends
  //    the champion's search value into the target (TD/bootstrap) when < 1. Unless
  //    --cold, the candidate warm-starts from the lineage (a previous rejected-but-
  //    positive candidate) when one exists, else from the champion — so sub-threshold
  //    gains accumulate instead of being re-derived and discarded every cycle
  //    (train.py ignores --init gracefully if the shapes don't match).
  const initFile = !cfg.cold && existsSync(lineage) ? lineage : champion;
  if (!run(`Train candidate${initFile === lineage ? ' (warm-start from lineage)' : ''}`, python,
    [trainPy, `--hidden=${hidden}`, `--out=${candidate}`, `--lambda=${cfg.lam}`,
      ...(cfg.cold ? [] : [`--init=${initFile}`])])) break;

  // 4. Gate: candidate (A) vs champion (B), SPRT(0, elo1). Unless --no-harvest,
  //    the gate's games are appended to the dataset (they're already paid for;
  //    `v` survives only from the side the gate proves stronger) and the next
  //    cycle's incremental featurize folds them in.
  if (existsSync(resultFile)) rmSync(resultFile);
  if (!run('Gate: candidate vs champion', process.execPath,
    [matchScript, '--eval-a=nn', `--weights-a=${candidate}`, '--eval-b=nn', `--weights-b=${champion}`,
      `--depth=${cfg.gateDepth}`, '--sprt', '--elo0=0', `--elo1=${cfg.elo1}`,
      `--games=${cfg.gateGames}`, `--result-file=${resultFile}`,
      ...(cfg.harvest ? [`--save-games=${rawFile}`, `--seed=${Date.now()}`] : []), ...jobArg])) break;

  // 5. Promote only on a significant win (SPRT accepted H1). Never regress.
  let res;
  try { res = JSON.parse(readFileSync(resultFile, 'utf8')); } catch { log('No match result; keeping champion.'); continue; }
  const pct = (res.score * 100).toFixed(1);
  if (res.sprt === 'H1') {
    const arch = JSON.parse(readFileSync(candidate, 'utf8')).arch;
    copyFileSync(champion, prevChampion);   // backup for safety
    copyFileSync(candidate, champion);      // candidate becomes champion
    const champHash = archiveChampion(champion); // keep it reconstructable by its vs version
    if (existsSync(lineage)) rmSync(lineage); // lineage cleared the gate; next start = new champion
    upsertManifest(arch); copyFileSync(candidate, loopChampPub); // make it playable
    promotions++;
    log(`cycle ${c}: PROMOTED ✓  candidate ${pct}% / Elo +${res.elo.toFixed(0)} over champion `
      + `(${res.games} games, cycle took ${fmtDur((Date.now() - cycleT0) / 1000)}). `
      + `New champion published as 'loop-champion' (archived ${champHash}.json). Total promotions: ${promotions}.`);
    // Add the new champion to the strength ledger so the refresh below — and every later
    // cycle — recognizes it as the new best and can relabel the now-second-best champion's
    // `v` weakest-first. Incremental: normally just one matchup vs the anchor; the FIRST
    // promotion (no ledger yet) plays the full small gauntlet to seed it.
    runRank(`Rank engines (add champion ${champHash})`);
    // The champion (hence the `v` target) just changed: value-iterate by recomputing
    // `v` on a fraction of the dataset with the NEW champion. Optional maintenance —
    // a failure shouldn't kill the loop, so we don't gate on its result (a Ctrl-C
    // still propagates via the `stopping` flag and ends the run after this cycle).
    // Seeded per run so successive refreshes cover different slices of the set.
    if (cfg.refreshFrac > 0) {
      run(`Refresh v (${(cfg.refreshFrac * 100).toFixed(0)}% ${refreshMode()} @ depth ${cfg.refreshDepth}, new champion)`,
        process.execPath, refreshArgs(cfg.refreshFrac, cfg.refreshDepth));
    }
  } else if (!cfg.cold && res.sprt !== 'H0' && res.score >= 0.5) {
    // Inconclusive but not losing: keep the candidate as the lineage so the next
    // cycle builds on its (sub-threshold) gain instead of rederiving it from the
    // champion. The champion itself is untouched — the gate still protects it.
    copyFileSync(candidate, lineage);
    log(`cycle ${c}: kept champion — candidate ${pct}% / Elo ${res.elo.toFixed(0)} `
      + `(SPRT ${res.sprt}, ${res.games} games, cycle took ${fmtDur((Date.now() - cycleT0) / 1000)}). `
      + 'Below the gate; candidate kept as lineage for the next cycle.');
  } else {
    const hadLineage = existsSync(lineage);
    if (hadLineage) rmSync(lineage);
    log(`cycle ${c}: kept champion — candidate ${pct}% / Elo ${res.elo.toFixed(0)} `
      + `(SPRT ${res.sprt}, ${res.games} games, cycle took ${fmtDur((Date.now() - cycleT0) / 1000)}). `
      + `Not a gain.${hadLineage ? ' Lineage reset to the champion.' : ''}`);
  }
}

log(`train:loop stopped after ${promotions} promotion(s) in ${fmtDur((Date.now() - loopT0) / 1000)}. `
  + `Champion: web/src/nn-weights.json${promotions ? " (also catalog 'loop-champion')" : ''}.`);
if (promotions) console.log('Run `npm run build` to ship the new champion in the production bundle.');
