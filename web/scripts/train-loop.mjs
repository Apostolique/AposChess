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
// Node-tools default). On each promotion it's also published to the web catalog under the
// next free human name (Ada, Boris, …) and flagged the current champion, so you can play it
// in the app under a real name from the moment it's promoted (rebuild for the production
// bundle; `npm run dev` serves it live).
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
//   --depth=D       search depth while generating (default 8 — deeper = better labels)
//   --openings=K    forwarded to gen: starting plies to vary (default: gen's 8)
//   --opening-topk=N  forwarded to gen: 0 (default) = uniform-random openings; N>=1
//                   samples among the engine's N best opening moves (sound but varied).
//                   Off by default, so the loop's data is unchanged unless you set it.
//   --cycles=N      stop after N cycles (default: run forever until Ctrl-C)
//   --gate-games=N  max games in the candidate-vs-champion match (default 2000 — mature
//                   gains are small, and small edges need many games to clear the SPRT:
//                   a true +20 candidate clears an 800g gate only ~1/3 of the time but
//                   ~80% at 2000g, with the false-reject rate pinned at beta throughout)
//   --gate-depth=D  search depth for the gating match (default 6)
//   --elo1=E        SPRT H1 promotion threshold in Elo (default 20; elo0 is 0). This
//                   is the SMALLEST gain worth promoting; it must be wide enough that
//                   the SPRT can actually decide within --gate-games. A too-small band
//                   (e.g. [0,5] over 400 games) needs a candidate ~+170 Elo to fire,
//                   so real improvements get rejected — keep elo1 vs gate-games sane.
//   --lambda=L      TD/bootstrap target mix for training the candidate (default 1 =
//                   pure game result; <1 leans on the champion's own search value,
//                   an unbiased bootstrap — recorded because generation uses the net)
//   --hidden=H      candidate architecture (default: same shape as the champion)
//   --cold          train the FIRST cycle's candidate from random init instead of
//                   warm-starting from the champion (warm start fine-tunes in a few
//                   epochs and starts at champion strength; a cold start occasionally
//                   explores a different basin but relearns everything). Only the first
//                   cycle is cold — every later cycle warm-starts from the PREVIOUS
//                   cycle's candidate, so the run bootstraps a fresh net once and then
//                   keeps refining that same net. This is how a candidate whose --hidden
//                   shape differs from the champion's evolves: the champion can't seed it
//                   (wrong architecture), so the chain runs through the candidate itself.
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
//                   in. They're played at --gate-depth (default 6), a notch below the
//                   generation --depth (8); refresh-v walks these labels up over cycles.
//   --jobs=N        parallel workers for gen + match
//   --quiet-only    featurize only QUIET positions (drop side-to-move-in-check and
//                   positions with a winning capture available). NNUE is a static eval
//                   called only at quiescence-search leaves, so loud positions mismatch
//                   that distribution and add label noise. Off by default — gate it
//                   head-to-head before adopting. Toggling forces a full re-featurize.
//   --filter-weak=DELTA  featurize-time WEAK-GAMES filter: drop whole games whose weaker
//                   player rates more than DELTA Elo below the current champion on the
//                   rank ledger (featurize --min-elo). refresh-v can repair a stale `v`,
//                   but never who PLAYED: a weak engine's trajectories are off-distribution
//                   positions and its blunder-decided result is label noise on every
//                   position of the game. SELF-ADJUSTING: the absolute cutoff is recomputed
//                   from the champion's ledger Elo each cycle (quantized to 50 Elo so it
//                   moves — and forces a full re-featurize — only when the champion has
//                   actually climbed), so improving champions retire old weak cohorts
//                   automatically. STRICT: a player that can't be positively rated (ledger,
//                   ephemeral tag, or the game's ephemeral `vs` evidence) counts as weakest
//                   and the game is dropped. Inactive until the ledger rates the champion
//                   (cycle 2 on a fresh clone). 0 = off (default). A recipe knob — keys its
//                   own experiment track.
//   --drop-conflicts=CP  featurize-time SEARCH-VS-RESULT filter: drop positions whose
//                   recorded `v` is confident (|v| >= CP centipawns) but contradicts the
//                   game result — there the result label is lying about the position (a
//                   later blunder decided the game), which is exactly the noise a pure-
//                   result target (lambda=1) trains on. Gives refresh-v a second job:
//                   better `v` labels also mean better noise detection. 0 = off (default).
//                   A recipe knob — keys its own experiment track.
//   --fresh         clear the dataset before the first cycle (clean deep-search start)
//   --refresh-frac=P  after each PROMOTION, recompute `v` on a random fraction P of the
//                   dataset with the new champion (value iteration; 0 = off, default).
//                   Only runs on promotion — between promotions the champion (hence v)
//                   is unchanged, so a refresh would just recompute identical values.
//                   Cost scales with P × depth; e.g. P=0.2 touches the whole set every
//                   ~5 promotions. Re-featurize happens next cycle, so it flows in.
//   --refresh-depth=D  search depth for the refresh (default 8 — matches generation;
//                   a depth-8 refresh of a big fraction is hours, lower it for speed)
//   --refresh-cycle=P  EVERY cycle (between generation and featurize), recompute `v` with
//                   the current champion (default 1 = the whole weakest cohort; 0 = off).
//                   Unlike --refresh-frac this helps between promotions too: most records
//                   carry `v` from OLDER champions (or shallower gate-harvest/backfill
//                   searches), so re-labeling them with the current champion steadily
//                   upgrades the TD target even while the champion is unchanged. In
//                   ledger mode (the loop's default once a champion exists) refresh-v
//                   already restricts itself to the single WEAKEST cohort and is capped
//                   at a 10-minute wall-clock budget, so P throttles only how much of
//                   that cohort is eligible — P=1 simply spends the whole budget draining
//                   the weakest labels first (and, unthrottled, reaches the "nothing to
//                   refresh" steady state sooner, after which the refresh is a cheap
//                   read-only scan that no longer forces a full re-featurize). A small P
//                   wastes budget; only lower it if you want shorter cycles than 10m.
//   --refresh-cycle-depth=D  search depth for the per-cycle refresh (default: --depth,
//                   so the re-labels match generation's deep-label quality)
//   --no-refresh    skip ALL value refreshing (both --refresh-cycle and --refresh-frac),
//                   regardless of their values — shorthand for --refresh-cycle=0 with no
//                   promotion refresh, for when you want the fastest possible cycles and
//                   accept the staler `v` targets
//   --float / --no-quant  train a NON-quantized (float) candidate instead of the default
//                   quantized one. Quant is a recipe knob, so this forks a distinct track.
//   --scale=S / --lr=L / --wd=W  forwarded to train.py (else its own defaults). Each is part
//                   of the recipe when set, so it keys a distinct experiment track.
//   --epochs=N / --patience=P  forwarded to train.py (else its defaults: 200 / 8; patience 0
//                   disables early stopping). NOT recipe keys — a training-length tuning knob,
//                   so changing them refines the same track's net rather than forking a track.
//   --recipe-extra=k=v,k2=v2  free-form namespace to fork a separate track for a training
//                   experiment the loop has no first-class flag for yet. Labels/keys the track
//                   and rides along in its resume command; NOT forwarded to train.py.
//
// Experiment tracks (persistent, non-destructive): the training RECIPE — architecture
// (--hidden), TD mix (--lambda), --quiet-only, quant (--float), and --scale/--lr/--wd/
// --recipe-extra — keys a per-recipe TRACK under loop/experiments/<id>/ (see
// experiment-registry.mjs). Each track keeps its OWN warm-start lineage, its strongest net
// ever (best.json, by estimated absolute Elo), and a per-cycle history. So trying a different
// architecture (or quiet-games, or any recipe knob) no longer clobbers the previous recipe's
// accumulated progress: run another recipe in between, come back, and the SAME recipe resumes
// its lineage/best automatically. The champion stays SHARED and best-wins — any recipe's
// candidate gates against it, and whichever wins promotes. Browse/suggest with
// `npm run train:experiments`. (--quiet-only also gets its own featurized file, so alternating
// quiet/all-positions recipes don't force a re-featurize each switch.)
//
// Candidate lineage (automatic): when the gate is inconclusive but
// the candidate scored >= 50%, the candidate is KEPT (this recipe's track lineage) and the next
// cycle's candidate warm-starts from IT instead of the champion — so sub-threshold
// gains (+10-ish Elo, real but below the SPRT's resolution) accumulate across cycles
// until the lineage clears the gate, instead of being re-derived and discarded every
// cycle. The champion is still protected by the gate; a candidate scoring < 50% (or a
// decided H0) resets the lineage, and the next warm-start falls back to this recipe's best net.

import { spawnSync } from 'node:child_process';
import {
  existsSync, rmSync, mkdirSync, copyFileSync, readFileSync, writeFileSync, appendFileSync, statSync,
  openSync, readSync, closeSync,
} from 'node:fs';
import { dirname, resolve, join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';
import { cpus } from 'node:os';

import { fmtDur, fmtMB } from './fmt.mjs';
import { weightsHash, ephemeralVersion } from './vtag.mjs';
import { STOP_EXIT_CODE } from './stop.mjs';
import { isGameRecord, vsAt, setVsAt, normalizeVs, serializeGameRecord } from './gameRecord.mjs';
import {
  buildRecipe, parseRecipeExtra, ensureTrack, beginRun, recordCycle, recipeLabel,
} from './experiment-registry.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const webDir = resolve(here, '..');
const repoDir = resolve(webDir, '..');
const dataDir = resolve(repoDir, 'training', 'data');
const loopDir = join(dataDir, 'loop');
mkdirSync(loopDir, { recursive: true });

const featurizeScript = resolve(here, 'featurize.mjs');
const refreshScript = resolve(here, 'refresh-v.mjs');
const rankScript = resolve(here, 'depth-ladder.mjs');
const trainPy = resolve(repoDir, 'training', 'train.py');

// Generation and the gate run on the native Zig engine: apos-gen for self-play, apos-match
// for the candidate-vs-champion gate (with --save-games harvest). Built once at startup
// from web/engine; the binaries run with cwd = web/ so their relative paths resolve there.
const engineDir = resolve(webDir, 'engine');
const isWin = process.platform === 'win32';
const genBin = resolve(engineDir, 'zig-out', 'bin', isWin ? 'apos-gen.exe' : 'apos-gen');
const matchBin = resolve(engineDir, 'zig-out', 'bin', isWin ? 'apos-match.exe' : 'apos-match');
function buildEngine() {
  // String form (not args-array) with shell:true so Windows resolves `zig` on PATH
  // without the DEP0190 arg-concatenation warning.
  const r = spawnSync('zig build -Doptimize=ReleaseFast', { cwd: engineDir, stdio: 'inherit', shell: true });
  if (r.status !== 0) {
    console.error('zig build failed (is Zig 0.16 on PATH?). The loop needs the native engine for gen + gate.');
    process.exit(1);
  }
}

// Each loop step shells out to a standalone script that also has its own npm entry
// point — so anyone watching the loop can reproduce/resume a single step by hand.
// Map the script we spawn back to the command you'd type to run it yourself.
const scriptCmd = new Map([
  [genBin, 'npm run train:gen'],
  [featurizeScript, 'npm run train:featurize'],
  [matchBin, 'npm run match'],
  [rankScript, 'npm run rank:pool'],
  [refreshScript, 'node scripts/refresh-v.mjs'], // no npm alias
  [trainPy, 'npm run train:fit'],                // train:fit forwards args to train.py
]);
// Each spawned script's own per-flag defaults, keyed by the command above. A flag the
// loop passes that just restates the script's default is dropped from the echoed command
// (it'd behave identically if omitted), so what's shown is only what actually differs.
// Values are compared as strings (argv flags are strings). Keep in sync with the
// referenced scripts' arg parsing. Flags absent here are always shown; `--seed` is
// dropped unconditionally below (the loop always seeds it from the clock).
const scriptDefaults = {
  'npm run train:gen': { games: '200', depth: '6', eval: 'handcrafted', openings: '8', 'opening-topk': '0', maxmoves: '200' },
  'npm run match': { games: '100', movetime: '50', 'eval-a': 'handcrafted', 'eval-b': 'handcrafted', openings: '6', maxmoves: '200', elo0: '0', elo1: '15', alpha: '0.05', beta: '0.05' },
  'npm run rank:pool': { 'anchor-depth': '6', games: '10', openings: '6', maxmoves: '200', prior: '1' },
  'node scripts/refresh-v.mjs': { frac: '1', depth: '6', minutes: '10' },
  'npm run train:fit': { hidden: '128', lambda: '1' }, // train:fit forwards to train.py
};
// Rewrite an absolute-path argument (e.g. --out=C:\…\candidate.json) to a path relative
// to web/, where these commands are meant to run — shorter and still copy-pasteable.
function relArg(tok) {
  const eq = tok.indexOf('=');
  if (eq < 0) return tok;
  const key = tok.slice(0, eq), val = tok.slice(eq + 1);
  if (!/^([A-Za-z]:[\\/]|[\\/])/.test(val)) return tok; // not an absolute path
  return `${key}=${relative(webDir, val).replace(/\\/g, '/')}`;
}
// The hand-runnable form of a spawned step: `npm run … -- <flags>` (npm needs the `--`
// to forward flags), with redundant defaults and the clock-based seed stripped so only
// the meaningful overrides show. Returns null for anything not in the map.
function friendlyCmd(cmd, argv) {
  // The spawned program is either `cmd` itself (the native binaries apos-gen / apos-match,
  // whose flags are ALL of argv) or, when cmd is the node/python interpreter, the script in
  // argv[0] (whose flags are argv.slice(1)). Resolve whichever the scriptCmd map knows, and
  // take the flag list from the matching position — otherwise the native steps (Generate,
  // Gate) would look up a --flag as the program and silently echo nothing.
  let base = scriptCmd.get(cmd), flagArgs;
  if (base) { flagArgs = argv; }
  else { base = scriptCmd.get(argv[0]); flagArgs = argv.slice(1); }
  if (!base) return null;
  const def = scriptDefaults[base] || {};
  const flags = [];
  for (const tok of flagArgs) {
    const eq = tok.indexOf('=');
    const key = (eq < 0 ? tok : tok.slice(0, eq)).replace(/^--/, '');
    const val = eq < 0 ? null : tok.slice(eq + 1);
    if (key === 'seed') continue;            // clock-seeded each run, never reproducible
    if (val !== null && def[key] === val) continue; // restates the script's own default
    flags.push(relArg(tok));
  }
  if (!flags.length) return base;
  const sep = base.startsWith('npm run') ? ' --' : '';
  return `${base}${sep} ${flags.join(' ')}`;
}

const rawFile = join(dataDir, 'selfplay.jsonl');
// Featurized output is keyed by the featurize-affecting recipe knobs (--quiet-only,
// --filter-weak, --drop-conflicts), so switching between filter configs no longer forces a
// full re-featurize each time — each config keeps its own incrementally-maintained file +
// meta sidecar. The unfiltered file keeps the bare default name (backward-compatible with
// the existing incremental state and the non-loop tools). --filter-weak is keyed by its
// DELTA (stable), not the per-cycle absolute cutoff — when the champion climbs a 50-Elo
// step the same file is rebuilt in place (the meta sidecar detects the cutoff change).
function featurizeFile() {
  const parts = [];
  if (cfg.quietOnly) parts.push('quiet');
  if (cfg.filterWeak > 0) parts.push(`w${cfg.filterWeak}`);
  if (cfg.dropConflicts > 0) parts.push(`c${cfg.dropConflicts}`);
  return join(dataDir, `selfplay.features${parts.length ? `.${parts.join('.')}` : ''}.jsonl`);
}
const champion = resolve(webDir, 'src', 'nn-weights.json');
const candidate = join(loopDir, 'candidate.json');
// The recipe's warm-start lineage + persistent best now live in this recipe's TRACK directory
// (see experiment-registry.mjs), assigned once the recipe is resolved at startup — NOT a single
// global slot, so switching recipes between runs is non-destructive. `lineage`/`trackBest` are
// the resolved per-track paths; `track` is the track handle.
let track = null, lineage = null, trackBest = null, runNo = 0;
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
// The gate's --save-games harvest is written HERE (a temp), not straight into the dataset,
// so the loop can rewrite a non-promoted candidate's provenance before folding it in (see
// foldGateHarvest). Cleared before each gate and deleted after folding.
const gateHarvest = join(loopDir, 'gate-harvest.jsonl');
// Bradley-Terry pool ledger (npm run rank:pool) + its persisted pairwise-results store. The
// store accumulates games across cycles; the ledger is the fitted Elo the refreshes consume.
const ledgerFile = join(loopDir, 'engine-elo.ladder.json');
const ladderStore = join(loopDir, 'ladder-pool.json');
// rank:pool's harvested games (its --save-games default). A persistent, accumulating archive of
// every game the strength pool has played — kept in lockstep with ladderStore. Each cycle we
// append only the games it ADDED this cycle into the dataset (like the gate harvest), so the
// next incremental featurize trains on them; their players are all rankable engines (champion /
// hc / archived / material), so no provenance rewrite is needed. --corpus subtracts the store,
// so re-reading them from the dataset never double-counts in the ratings.
const ladderGames = join(loopDir, 'ladder-games.jsonl');
// Persistent high-water mark: bytes of ladderGames already folded into the dataset. Persisting
// it (vs a per-cycle snapshot) means the fold also catches games a STANDALONE `rank:pool` in
// another terminal appended between cycles — not just the loop's own rank step. Missing/0 ⇒ fold
// the whole archive (first run on this clone, or after --fresh clears the dataset but keeps the
// archive); merge-data dedups by game id any game an earlier fold already added.
const ladderFoldMark = join(loopDir, 'ladder-fold.json');
const logFile = join(loopDir, 'loop.log');
// PID of this loop, so `npm run train:pause`/`train:resume` (scripts/loop-ctl.mjs) can find
// and freeze/thaw the loop's whole process tree from another terminal — a long run pegs every
// core, so pausing hands the machine back without losing the in-flight gate/generation.
const pidFile = join(loopDir, 'loop.pid');
const pauseFlag = join(loopDir, 'PAUSED'); // marker loop-ctl writes while suspended
const publicNN = resolve(webDir, 'public', 'nn');
const manifestFile = join(publicNN, 'manifest.json');

const args = Object.fromEntries(
  process.argv.slice(2).map((a) => {
    const m = a.replace(/^--/, '').split('=');
    return [m[0], m.length > 1 ? m[1] : true];
  }),
);
const num = (v, d) => (v === undefined ? d : Number(v));
// One full parallel wave of the match runner: apos-match dispatches games one at a time
// across --jobs workers, so a matchup smaller than the worker count leaves cores idle from
// the start (not just at the tail). Rounded up to even — games come in color-reversed pairs.
const effJobs = args.jobs !== undefined ? Number(args.jobs) : cpus().length;
const defaultRankGames = Math.max(2, Math.ceil(effJobs / 2) * 2);
const cfg = {
  batch: num(args.batch, 200),
  // Self-play GENERATION depth — the deep label anchor (each generated position's value
  // target is its search value at this depth). Raised 6 -> 8 (2026-06-27): deeper labels
  // are the NN's real lever (first-layer width is a dead end — docs/first-layer-strategy.md),
  // and the NNUE accumulator (~1.5x) offsets part of the cost. Gen is only `batch` games, so
  // this ~doubles cycle time (gen ≈ the gate's cost) rather than exploding it; the gate stays
  // at gateDepth and refresh-v walks the dataset up to this depth over cycles. --depth=6 to
  // revert, --depth=7 for a gentler cost.
  depth: num(args.depth, 8),
  openings: args.openings !== undefined ? Number(args.openings) : null, // null = gen default (8)
  openingTopk: num(args['opening-topk'], 0), // 0 = uniform-random opening (gen default)
  cycles: args.cycles !== undefined ? Number(args.cycles) : Infinity,
  gateGames: num(args['gate-games'], 2000),
  gateDepth: num(args['gate-depth'], 6),
  elo1: num(args.elo1, 20), // wide enough that SPRT can decide within --gate-games
  lam: num(args.lambda, 1), // TD target mix passed to train.py (1 = pure result)
  // Drop tactically loud positions (in check / winning capture available) at featurize time
  // so the static net trains on the quiet-position distribution it's actually queried on at
  // qsearch leaves. Off by default (gate it head-to-head before adopting). Toggling it forces
  // the next featurize to be a full pass (the meta sidecar records the filter state).
  quietOnly: !!args['quiet-only'],
  // Featurize-time dataset filters (recipe knobs — each keys its own experiment track):
  // --filter-weak=DELTA drops games whose weaker player is > DELTA Elo below the current
  // champion (cutoff recomputed per cycle from the ledger, so it tracks the champion);
  // --drop-conflicts=CP drops positions whose |v| >= CP contradicts the game result.
  filterWeak: num(args['filter-weak'], 0),
  dropConflicts: num(args['drop-conflicts'], 0),
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
  // Matches the gen depth (8) so promotion refreshes relabel at the full value-accuracy of
  // the deep anchor. A depth-8 refresh of a big fraction is many hours — lower it (or the
  // fraction) to trade accuracy for speed. Off by default (refreshFrac 0).
  refreshDepth: num(args['refresh-depth'], 8),
  // Per-cycle refresh: a small slice of the dataset re-labeled with the current champion
  // every cycle. Helps between promotions too — most `v` in the set came from older
  // champions or shallower searches, so "unchanged champion" does NOT mean "nothing to
  // refresh"; only records the current champion already labeled at this depth are no-ops.
  refreshCycle: args['no-refresh'] ? 0 : num(args['refresh-cycle'], 1),
  refreshCycleDepth: num(args['refresh-cycle-depth'], num(args.depth, 8)),
  // Engine ranking for smart weakest-first v refresh. On by default, now driven by the
  // self-relative Bradley-Terry POOL (rank:pool / depth-ladder.mjs), not the old anchor
  // gauntlet. EVERY cycle the loop refits the pool: it folds the whole dataset's harvested
  // games into the fit (--corpus — so the new champion is rated automatically from its gate
  // matches, no dedicated gauntlet) and plays a short --rank-minutes budget of the most-
  // ambiguous matchups to tighten ratings. The refreshes below read the resulting parallel
  // ledger (engine-elo.ladder.json) to relabel the WEAKEST engine's `v` first. --no-rank reverts.
  rank: !args['no-rank'],
  // hc pin depth for the pool (Elo 0) — every rating lands on this stable scale.
  rankDepth: num(args['rank-depth'], 6),
  // Wall-clock the pool plays per cycle, on top of the (free) corpus fold. Short, because the
  // corpus already rates the champion from its gate games and the store accumulates across
  // cycles — each cycle just tightens the most-ambiguous orderings.
  rankMinutes: num(args['rank-minutes'], 5),
  // Games per scheduled pool matchup. Defaults to ONE PARALLEL WAVE (the --jobs count, rounded
  // up to even): fewer games than workers leaves cores idle for the whole matchup, while a big
  // batch would blow far past --rank-minutes (the budget is only checked BETWEEN matchups — a
  // matchup always plays to completion, and depth-8 matchups are slow). One wave keeps every
  // core busy exactly once per matchup; the store accumulates the games across cycles regardless.
  rankGames: num(args['rank-games'], defaultRankGames),
  // On each promotion the NEW champion is published into the playable net catalog
  // (web/public/nn) under the next free human name and flagged the current champion, so it's
  // pickable in the app under a real name from the moment it's promoted (past champions stay
  // too). Only the most recent --keep-champions retired nets are kept (the current champion is
  // always kept); older ones are pruned (weights file + manifest entry) to bound the deployed
  // bundle (~0.5 MB each). 0 = off.
  keepChampions: num(args['keep-champions'], 12),
  // Strong-engine ladder play as the generator. With no dedicated generation (--batch=0), the
  // per-cycle rank step restricts --play to the strongest nn engines (current champion + recent
  // champions) at --play-depth, so its harvested games — already folded into the dataset
  // — are deep, strong-play training data: the ranked pool IS the generator. On by default when
  // --batch=0; --play-strong / --no-play-strong force it; --rank-play=SPEC pins the play set by
  // hand (goes stale on promotion — prefer the auto set for an unattended loop). See docs.
  playStrong: args['no-play-strong'] ? false
    : (args['play-strong'] !== undefined ? !!args['play-strong'] : num(args.batch, 200) === 0),
  playDepth: num(args['play-depth'], num(args.depth, 8)),
  playTop: num(args['play-top'], 8), // cap the strong set so the round-robin stays fast
  rankPlay: typeof args['rank-play'] === 'string' ? args['rank-play'] : null,
  // --- Training-recipe knobs (define the experiment TRACK; see experiment-registry.mjs) ---
  // These are what turn the shared dataset into THIS candidate net, so each distinct
  // combination is its own persistent, warm-startable track. quant is on by default
  // (the loop has always trained quantized); --float / --no-quant makes a float track.
  // scale/lr/wd default to train.py's own defaults unless set (undefined => omitted from
  // the recipe id, so they never fragment a track while unused).
  quant: !(args['no-quant'] || args.float),
  scale: args.scale !== undefined ? Number(args.scale) : undefined,
  lr: args.lr !== undefined ? Number(args.lr) : undefined,
  wd: args.wd !== undefined ? Number(args.wd) : undefined,
  // --epochs/--patience forward to train.py too, but are NOT recipe keys: tuning how long a
  // net trains refines the SAME recipe's net rather than forking a track (unlike scale/lr/wd,
  // which change the trained weights' identity). Undefined => train.py's own defaults
  // (epochs 200, patience 8; patience 0 disables early stopping).
  epochs: args.epochs !== undefined ? Number(args.epochs) : undefined,
  patience: args.patience !== undefined ? Number(args.patience) : undefined,
  // Free-form namespace for future training-affecting systems: --recipe-extra=key=val,key2=val2.
  // Purely a track KEY/label (it distinguishes tracks and rides along in the resume command);
  // it isn't forwarded to train.py. Use it to fork a separate track for an experiment the loop
  // doesn't yet have a first-class flag for.
  recipeExtra: parseRecipeExtra(args['recipe-extra']),
};
// The loop's rank-games default is machine-dependent (one parallel wave), so sync the
// command-echo suppression map to it — the echoed `npm run rank:pool` then shows --games
// only when an explicit --rank-games differs from this machine's computed default.
scriptDefaults['npm run rank:pool'].games = String(defaultRankGames);

// hash -> human name (champions from the web catalog manifest), so loop output shows 'Leo'
// next to 9e31ca wherever a hash appears. Read fresh per call (cheap, ~once a cycle) because
// a promotion renames the current champion mid-run.
function nnNames() {
  const m = new Map();
  try { for (const n of (JSON.parse(readFileSync(manifestFile, 'utf8')).nets || [])) if (n.hash && n.name) m.set(n.hash, n.name); } catch { /* no manifest yet */ }
  return m;
}

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

// A loud, multi-line boxed banner for cycle headers so they're trivial to spot when
// scrolling back through a long run (a lone `===== … =====` line blends in). The console
// gets bold-cyan; the log file gets the same box in plain text (no escape codes), both
// preceded by a blank line for extra separation.
function banner(title) {
  const w = 54;
  const t = title.length > w - 2 ? `${title.slice(0, w - 5)}…` : title;
  const pad = w - t.length, left = pad >> 1;
  const top = `╔${'═'.repeat(w)}╗`;
  const mid = `║${' '.repeat(left)}${t}${' '.repeat(pad - left)}║`;
  const bot = `╚${'═'.repeat(w)}╝`;
  console.log(`\n\x1b[1;36m[${hms()}] ${top}\n${' '.repeat(11)}${mid}\n${' '.repeat(11)}${bot}\x1b[0m`);
  appendFileSync(logFile, `\n[${stamp()}] ${top}\n[${stamp()}] ${mid}\n[${stamp()}] ${bot}\n`);
}

let stopping = false;
process.on('SIGINT', () => { stopping = true; console.log('\n  Ctrl-C: stopping after this cycle…'); });

// Elo from a win rate (same logarithmic curve the match runner / rank ledger use, so the
// numbers line up with the ledger Elos we add the gate edge to).
const eloFromScore = (p) => (p <= 0 ? -800 : p >= 1 ? 800 : -400 * Math.log10(1 / p - 1));

// The current champion's Elo (vs the rank ledger's stable hc anchor) per search depth, plus
// its best across depths. Returns null unless the ledger exists and actually ranks this
// champion — without it we can't place an ephemeral candidate on the hc scale, so the gate
// harvest is folded in unchanged (its candidate-hash labels stay −∞ "unrecoverable", as before).
function championLedgerElo() {
  if (!existsSync(ledgerFile)) return null;
  let ledger;
  try { ledger = JSON.parse(readFileSync(ledgerFile, 'utf8')); } catch { return null; }
  const champHash = weightsHash(champion);
  const byDepth = new Map();
  let best = -Infinity;
  for (const e of ledger.ranking || []) {
    if (e.eng !== 'nn' || e.version !== champHash || e.elo == null) continue;
    byDepth.set(String(e.depth), e.elo);
    best = Math.max(best, e.elo);
  }
  return byDepth.size ? { byDepth, best } : null;
}

// Fold the gate's harvested games (in the temp file) into the dataset. The match runner
// stamps every position with the MOVER's vs tag (the engine that searched it). The
// candidate's own plies thus carry its content hash — an engine we never archive or rank,
// so refresh-v/merge would read it as −∞ "unrecoverable" and relabel it on sight (the
// champion's plies already carry a ranked tag and pass through). When the candidate WASN'T
// promoted (the common lineage / sub-threshold case), we rewrite its lines' vs to a
// self-describing ephemeral tag "nn<d>@elo<E>", where E is the candidate's strength vs the hc
// anchor: the champion's ledger Elo at that depth plus the LOWER bound of the gate's measured
// edge (so a short or early-stopped gate is treated cautiously — weaker, hence refreshed
// sooner — rather than over-credited on thin evidence). A promoted candidate (its hash is
// archived + ranked) or a champion-won gate is already tagged with a ranked engine, so it
// passes through untouched.
function foldGateHarvest(promoted, res) {
  if (!existsSync(gateHarvest)) return;
  const champElo = (!promoted && res && res.games > 0) ? championLedgerElo() : null;
  // Lower bound of the candidate-vs-champion Elo edge from this gate's score + game count.
  let gateEloLo = null;
  if (champElo) {
    const p = res.score, se = Math.sqrt(Math.max(p * (1 - p), 0) / res.games);
    const pLo = Math.min(Math.max(p - 1.96 * se, 1e-9), 1 - 1e-9);
    gateEloLo = eloFromScore(pLo);
  }
  const candHash = weightsHash(candidate);
  const out = [];
  let folded = 0, relabeled = 0;
  for (const line of readFileSync(gateHarvest, 'utf8').split('\n')) {
    if (!line) continue;
    folded++;
    // Each harvested GAME interleaves both players' positions in its per-position `vs`
    // array; only the candidate's own positions (tagged with its content hash) need the
    // ephemeral rewrite, since a non-promoted candidate is never archived/rankable. The
    // champion's positions stay as-is (it IS rankable). Cheap pre-check: a game without the
    // candidate's hash anywhere needs no work. (When promoted, champElo is null — the
    // candidate became the champion and is recoverable, so nothing is relabeled.)
    if (!champElo || !line.includes(candHash)) { out.push(line); continue; }
    let rec; try { rec = JSON.parse(line); } catch { out.push(line); continue; }
    if (!isGameRecord(rec)) { out.push(line); continue; }
    const n = rec.v ? rec.v.length : rec.moves.length + 1;
    let changed = false;
    for (let i = 0; i < n; i++) {
      const m = /^nn(\d+|t)@([0-9a-f]+)$/.exec(vsAt(rec, i) || '');
      if (!m || m[2] !== candHash) continue;
      const depth = m[1];
      const base = champElo.byDepth.has(depth) ? champElo.byDepth.get(depth) : champElo.best;
      setVsAt(rec, i, `nn${depth}@${ephemeralVersion(base + gateEloLo)}`);
      relabeled++; changed = true;
    }
    if (changed) normalizeVs(rec);
    // Relabel the candidate's `players` entry the same way: it carries the same unrankable
    // content hash, and downstream consumers judge a game by who PLAYED it (featurize
    // --min-elo drops games with unrateable players; merge --drop-unlabeled counts them
    // unlabeled). The BT corpus fit is indifferent — an ephemeral tag is a non-pool id and
    // is skipped exactly like the raw hash was.
    for (const side of ['w', 'b']) {
      const pm = /^nn(\d+|t)@([0-9a-f]+)$/.exec((rec.players && rec.players[side]) || '');
      if (!pm || pm[2] !== candHash) continue;
      const base = champElo.byDepth.has(pm[1]) ? champElo.byDepth.get(pm[1]) : champElo.best;
      rec.players[side] = `nn${pm[1]}@${ephemeralVersion(base + gateEloLo)}`;
    }
    out.push(serializeGameRecord(rec));
  }
  if (out.length) appendFileSync(rawFile, out.join('\n') + '\n');
  rmSync(gateHarvest, { force: true });
  if (relabeled) {
    log(`  Folded ${folded} harvested position(s); relabeled ${relabeled} non-promoted-candidate `
      + `label(s) as ephemeral (gate edge lower-bound ${gateEloLo >= 0 ? '+' : ''}${gateEloLo.toFixed(0)} Elo vs champion).`);
  }
}

// Run a step; return true on success. A SIGINT to a child shows as a null/ signalled
// status — treat that as "stop", not a hard failure.
function run(label, cmd, argv, cwd = webDir) {
  if (stopping) return false;
  console.log(`\n--- ${label} — ${hms()} ---`);
  // Echo the equivalent stand-alone command so the step can be reproduced/resumed by
  // hand (run from web/). Logged too, so the persisted log records exactly what ran.
  const friendly = friendlyCmd(cmd, argv);
  if (friendly) log(`  $ ${friendly}`);
  // APOS_CHILD tells the child tools they're orchestrated: they use a SIGINT-only
  // graceful stop instead of grabbing the TTY's raw mode, so the loop's own Ctrl-C
  // (stop after this cycle) keeps working — see scripts/stop.mjs.
  const r = spawnSync(cmd, argv, { stdio: 'inherit', cwd, env: { ...process.env, APOS_CHILD: '1' } });
  if (r.signal) { stopping = true; return false; }
  // A child that caught Ctrl-C, drained its in-flight work cleanly, and exited reports
  // STOP_EXIT_CODE — a clean early finish, not a crash. Without this it looked like a
  // success (when the child exited 0) and the loop rolled into the next cycle. End here.
  if (r.status === STOP_EXIT_CODE) { stopping = true; log(`${label} stopped early (Ctrl-C); ending loop after a clean drain.`); return false; }
  // Windows delivers console Ctrl-C to the whole process group; the child then exits
  // with STATUS_CONTROL_C_EXIT (0xC000013A) instead of a signal — an interrupt, not a crash.
  if (r.status === 0xC000013A) { stopping = true; log(`${label} interrupted (Ctrl-C); stopping loop.`); return false; }
  if (r.status !== 0) { log(`${label} FAILED (exit ${r.status}); stopping loop.`); return false; }
  return true;
}

// The loop rates the SAME full pool as a standalone `npm run rank:pool` — every engine across
// depth-ladder's default depth spectrum (1-8), not a narrowed slice — so its ledger is the one
// unified pool, not a loop-specific variant. hc<rankDepth> stays the pinned Elo-0 node (via
// --anchor-depth below), so all ratings land on the same stable scale. (The two depths positions
// are LABELED at, nn6@/nn8@, are just a subset of that spectrum — foldGateHarvest still finds
// them in the ledger.)

// Refit the Bradley-Terry strength pool (rank:pool / depth-ladder.mjs). Runs EVERY cycle:
//   --corpus folds the whole dataset's harvested games (each record's players + result) into
//   the fit, so the current champion is rated automatically from its gate matches — no
//   dedicated gauntlet, and a just-promoted champion already has games against the engine it
//   dethroned. On top of that the pool plays a short --rank-minutes budget of the matchups
//   whose ORDERING is currently most ambiguous (naturally including the new champion), and
//   persists every game into the store so ratings tighten cumulatively across cycles.
// The fitted ledger (engine-elo.ladder.json) is what the weakest-first refreshes read.
// Maintenance, like the refreshes — a failure logs but doesn't stop the loop.
function runRankPool(label) {
  if (!cfg.rank) return;
  // When the ranked pool is the generator (cfg.playStrong), restrict --play to the strongest nn
  // engines at cfg.playDepth so this step's harvested games are deep, strong training data (the
  // whole pool is still RATED from the corpus + store — --play only bounds which nodes play NEW
  // games). null = schedule unrestricted (too few strong engines yet, or --no-play-strong).
  const play = cfg.playStrong ? strongPlaySpec() : null;
  // Show the play set with human names — the raw spec is hashes only (it must stay a valid
  // --play argument), so name the engines here where a reader actually sees the plan.
  if (play) {
    const names = nnNames();
    log(`  Strong play set: ${play.split(',').map((s) => {
      const m = /@([0-9a-f]+)$/.exec(s); const n = m && names.get(m[1]);
      return n ? `${s} (${n})` : s;
    }).join(', ')}`);
  }
  // With harvesting on (default), rank:pool appends its games to its archive (ladderGames); we
  // then fold everything past the persistent mark into the dataset — the loop's own rank games
  // AND any a standalone rank:pool added between cycles. With --no-harvest, suppress the archive
  // too (--no-save-games), consistent with the gate.
  run(label, process.execPath,
    [rankScript, '--corpus', `--minutes=${cfg.rankMinutes}`,
      `--anchor-depth=${cfg.rankDepth}`,
      ...(play ? [`--play=${play}`] : []),
      `--games=${cfg.rankGames}`, `--store=${ladderStore}`, `--ledger=${ledgerFile}`,
      ...(cfg.harvest ? [] : ['--no-save-games']),
      '--no-scan', `--seed=${Date.now()}`, ...jobArg]);
  if (cfg.harvest) foldNewLadderGames();
}

// Last byte of ladderGames already folded into the dataset (0 if never / after --fresh).
const readLadderMark = () => { try { return JSON.parse(readFileSync(ladderFoldMark, 'utf8')).offset || 0; } catch { return 0; } };
const writeLadderMark = (offset) => { try { writeFileSync(ladderFoldMark, JSON.stringify({ offset }) + '\n'); } catch { /* best-effort: a missed write only re-folds dedup-able games next run */ } };

// Fold every ladder game past the persistent mark into the dataset, so the next incremental
// featurize trains on it. The mark persists across cycles (and across runs), so this catches not
// just the loop's own rank games but any a STANDALONE rank:pool appended to the archive between
// cycles — the whole point of persisting it. On a clone with no mark yet it folds the entire
// archive once; games an earlier fold already placed are collapsed by merge-data's game-id dedup.
// Players are all rankable engines, so labels are kept as-is (no ephemeral rewrite).
function foldNewLadderGames() {
  if (!existsSync(ladderGames)) return;
  const size = statSync(ladderGames).size;
  const mark = readLadderMark();
  const from = mark > size ? 0 : mark; // archive shrank (deleted + recreated) ⇒ re-fold from 0
  if (size <= from) return; // nothing new (rank played no games, or harvesting was off)
  const fd = openSync(ladderGames, 'r');
  try {
    const buf = Buffer.alloc(size - from);
    readSync(fd, buf, 0, buf.length, from);
    let chunk = buf.toString('utf8');
    if (!chunk.endsWith('\n')) chunk += '\n'; // each game is a full line; guard a torn tail
    appendFileSync(rawFile, chunk);
    const added = chunk.split('\n').filter(Boolean).length;
    log(`  Folded ${added} new ladder game(s) into the dataset (next featurize trains on them).`);
  } finally { closeSync(fd); }
  writeLadderMark(size); // advance only after a successful append
}

// Featurize args for this recipe's dataset filters. --drop-conflicts forwards as-is.
// --filter-weak resolves to an ABSOLUTE ledger-scale cutoff each cycle: the champion's
// current ledger Elo minus the delta, quantized to 50-Elo steps so the cutoff (recorded in
// the featurize meta sidecar) moves — and forces a full re-featurize — only when the
// champion has actually climbed a step, not on every ledger-refit jitter. Until the ledger
// rates the champion (cycle 1 on a fresh clone) the weak filter is inactive for the cycle.
function filterArgs() {
  const a = [];
  if (cfg.dropConflicts > 0) a.push(`--drop-conflicts=${cfg.dropConflicts}`);
  if (cfg.filterWeak > 0) {
    const champElo = championLedgerElo();
    if (champElo) {
      const cutoff = Math.round((champElo.best - cfg.filterWeak) / 50) * 50;
      a.push(`--ledger=${ledgerFile}`, `--min-elo=${cutoff}`);
    } else {
      log(`  --filter-weak=${cfg.filterWeak}: the ledger doesn't rate the champion yet — weak-games filter inactive this cycle.`);
    }
  }
  return a;
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

// Human names for champions, handed out in order (the first eight — Ada..Hugo — were the
// initial hand-published lineage). A name freed by pruning becomes reusable.
const CHAMPION_NAMES = ['Ada', 'Boris', 'Clara', 'Dexter', 'Elena', 'Felix', 'Greta', 'Hugo',
  'Ivy', 'Jack', 'Kara', 'Leo', 'Mona', 'Nash', 'Olga', 'Pia', 'Quinn', 'Rosa', 'Sven',
  'Tara', 'Uma', 'Victor', 'Wren', 'Xena', 'Yuri', 'Zara'];

// Publish the just-promoted champion `file` into the net catalog under the next free human
// name and flag it the current champion (named at PROMOTION, not when dethroned) — so it's
// pickable in the app under a real name from the moment it's promoted, and the app default +
// analysis eval bar resolve to it via its `current` flag. Clears the previous current flag,
// then prunes to the most recent cfg.keepChampions retired champions (deleting their weights +
// manifest entries; the current one is always kept). Idempotent by content hash, so re-running
// a promotion is a no-op. Returns the assigned name. Loads + writes the manifest itself.
function publishChampion(file, arch) {
  let man = { default: null, nets: [] };
  try { man = JSON.parse(readFileSync(manifestFile, 'utf8')); } catch { /* new manifest */ }
  // Drop any legacy generic 'loop-champion' alias entry (superseded by named champions).
  man.nets = (man.nets || []).filter((n) => n.name !== 'loop-champion');
  const champs = () => man.nets.filter((n) => n.loopChampion);
  const hash = weightsHash(file);
  let entry = hash !== '?' ? champs().find((n) => n.hash === hash) : null;
  if (!entry) {
    const used = new Set(man.nets.map((n) => n.name));
    const name = CHAMPION_NAMES.find((n) => !used.has(n)) || `champ-${hash}`;
    const out = `${name.toLowerCase()}.json`;
    const gen = Math.max(0, ...champs().map((n) => n.gen || 0)) + 1;
    copyFileSync(file, join(publicNN, out));
    entry = { name, file: out, arch, loopChampion: true, current: true, hash, gen,
      note: `train:loop champion ${name} (gen ${gen}, ${hash}, ${new Date().toISOString().slice(0, 10)}).` };
    man.nets.push(entry);
  }
  // Exactly one current champion; it's also the catalog default (so the UI shows its name).
  for (const n of champs()) delete n.current;
  entry.current = true;
  man.default = entry.name;
  // Keep only the most recent cfg.keepChampions retired champions; never prune the current one.
  if (cfg.keepChampions > 0) {
    const byAge = champs().filter((n) => !n.current).sort((a, b) => (a.gen || 0) - (b.gen || 0));
    for (const e of byAge.slice(0, Math.max(0, byAge.length - cfg.keepChampions))) {
      const p = join(publicNN, e.file);
      if (existsSync(p)) rmSync(p);
      man.nets = man.nets.filter((n) => n !== e);
    }
  }
  man.nets.sort((a, b) => a.name.localeCompare(b.name));
  writeFileSync(manifestFile, JSON.stringify(man, null, 2) + '\n');
  return entry.name;
}

// The --play spec for the per-cycle rank step when the ranked pool is the generator (cfg.playStrong):
// the strongest nn engines — current champion and recent champions — each at cfg.playDepth, capped
// to cfg.playTop. All are archived by hash (existence-checked, so every spec resolves in
// depth-ladder) and near-champion strength, so the pool plays deep, high-quality games among them
// and harvests them as training data. Returns "nn8@h1,nn8@h2,…", an explicit --rank-play override,
// or null when < 2 strong engines exist yet (the rank step then schedules unrestricted).
function strongPlaySpec() {
  if (cfg.rankPlay) return cfg.rankPlay;
  const seen = new Set();
  const hashes = [];
  const add = (h) => {
    if (h && h !== '?' && !seen.has(h) && existsSync(join(championsDir, `${h}.json`))) { seen.add(h); hashes.push(h); }
  };
  add(weightsHash(champion)); // current champion (strongest)
  try {
    const man = JSON.parse(readFileSync(manifestFile, 'utf8'));
    const champs = (man.nets || []).filter((n) => n.loopChampion && n.hash).sort((a, b) => (b.gen || 0) - (a.gen || 0));
    for (const n of champs) add(n.hash); // recent champions, newest (strongest) first
  } catch { /* no manifest yet */ }
  const picked = hashes.slice(0, Math.max(2, cfg.playTop));
  return picked.length >= 2 ? picked.map((h) => `nn${cfg.playDepth}@${h}`).join(',') : null;
}

if (!existsSync(champion)) {
  console.error(`No champion at ${champion}. Train a net first (e.g. npm run train:fit).`);
  process.exit(1);
}
// Build the native engine up front (cached — near-instant if unchanged) so the gen and
// gate steps spawn the fast Zig binaries.
buildEngine();
// Archive the starting champion too — it labels the data generated before the first
// promotion, so its v-contributors must stay reconstructable like every later champion.
archiveChampion(champion);
// Publish this loop's PID for the pause/resume control, and drop any stale PAUSED marker
// from a previous run (a fresh loop starts running). The pidfile is removed on exit.
writeFileSync(pidFile, `${process.pid}\n`);
rmSync(pauseFlag, { force: true });
process.on('exit', () => { try { rmSync(pidFile, { force: true }); } catch { /* best effort */ } });
if (cfg.fresh && existsSync(rawFile)) { rmSync(rawFile); log('Cleared dataset (--fresh).'); }
// --fresh empties the dataset but keeps the ladder archive (games between static engines), so
// reset the fold mark too — otherwise the kept archive wouldn't re-enter the fresh dataset.
if (cfg.fresh && existsSync(ladderFoldMark)) rmSync(ladderFoldMark);

// Omitting --hidden pins the candidate to the champion's CURRENT shape (resolved here), so
// the recipe id is a concrete architecture, not a moving target.
const hidden = championHidden();
// Resolve this run's TRAINING RECIPE and its persistent track. The recipe (architecture + TD
// mix + quiet filter + quant + trainer knobs + any --recipe-extra) is keyed to a directory
// under loop/experiments/, so its lineage/best/history survive being switched away from and
// resume automatically when the same recipe runs again — even after other recipes in between.
const recipe = buildRecipe({
  hidden, lambda: cfg.lam, quietOnly: cfg.quietOnly, quant: cfg.quant,
  scale: cfg.scale, lr: cfg.lr, wd: cfg.wd,
  filterWeak: cfg.filterWeak > 0 ? cfg.filterWeak : undefined,
  dropConflicts: cfg.dropConflicts > 0 ? cfg.dropConflicts : undefined,
  extra: cfg.recipeExtra,
});
track = ensureTrack(loopDir, recipe, stamp());
lineage = track.paths.lineage;   // this recipe's accumulated sub-threshold warm-start net
trackBest = track.paths.best;    // this recipe's strongest net ever (by estimated abs Elo)
runNo = beginRun(track.dir, stamp());
// No shape-mismatch discard anymore: the track is keyed by the exact recipe (hidden included),
// so its lineage always matches its own shape and lives in its own directory — a different
// recipe can't clobber it. A --cold run simply ignores the lineage (it chains from a fresh
// net), leaving the prior warm progress on disk intact for a later warm resume.
log(`Recipe ${track.slug} [${track.id}] — ${recipeLabel(recipe)}`
  + ` (track run #${runNo}${track.isNew ? ', new track' : ''}).`);
// Featurized file for this recipe (quiet-only gets its own; see featurizeFile).
const featFile = featurizeFile();
// Whether the global champion's architecture matches this recipe's — i.e. whether the champion
// is a usable warm-start seed for the candidate. False when --hidden differs from the champion's
// shape (a brand-new architecture track that must bootstrap from its own lineage/best or cold).
function championArchMatches() {
  try {
    const a = JSON.parse(readFileSync(champion, 'utf8')).arch;
    return Array.isArray(a) && a.length >= 3 && a.slice(1, -1).join(',') === hidden;
  } catch { return false; }
}
log(`train:loop start — ${cfg.batch === 0
    ? `no gen (data from gate harvest${cfg.playStrong ? ` + strong --play @ depth ${cfg.playDepth}` : ' + pool'})`
    : `batch ${cfg.batch} @ depth ${cfg.depth}`} | gate ${cfg.gateGames}g @ depth ${cfg.gateDepth} `
  + `SPRT(0,${cfg.elo1}) | candidate hidden=[${hidden}] λ=${cfg.lam} ${cfg.cold ? 'cold first cycle, warm after' : 'warm'} start`
  + `${existsSync(lineage) ? ' (resuming lineage)' : ''} | `
  + `refresh/cycle ${cfg.refreshCycle > 0 ? `${(cfg.refreshCycle * 100).toFixed(1)}% @ depth ${cfg.refreshCycleDepth}` : 'off'} | `
  + `refresh on promotion ${cfg.refreshFrac > 0 ? `${(cfg.refreshFrac * 100).toFixed(0)}% @ depth ${cfg.refreshDepth}` : 'off'} | `
  + `rank ${cfg.rank ? `full pool every cycle (hc${cfg.rankDepth} pin, all depths, corpus + ${cfg.rankMinutes}m play)` : 'off'} | `
  + `cycles ${cfg.cycles === Infinity ? '∞' : cfg.cycles}`);
log('Pause/resume from another terminal: `npm run train:pause` / `npm run train:resume` (frees all CPU, no work lost).');

const jobArg = cfg.jobs !== undefined ? [`--jobs=${cfg.jobs}`] : [];

// No startup ranking: the pool is refit near the END of each cycle (runRankPool), after that
// cycle's gate games have been harvested into the dataset, and the per-cycle v-refresh runs
// right after it — so every cycle (including the first) refreshes against a ledger that was
// just refit, going weakest-first as soon as the pool rates anything.

const loopT0 = Date.now();
let promotions = 0;
for (let c = 1; c <= cfg.cycles && !stopping; c++) {
  const cycleT0 = Date.now();
  const dataset = existsSync(rawFile) ? ` — dataset ${fmtMB(statSync(rawFile).size)}` : '';
  banner(`CYCLE ${c}${cfg.cycles === Infinity ? '' : `/${cfg.cycles}`}${dataset}`);
  // --cold trains from random init on the FIRST cycle only: bootstrap a fresh net once,
  // then keep refining THAT net by warm-starting every later cycle from the previous
  // cycle's candidate (see the init resolution below) instead of relearning from scratch.
  const cold = cfg.cold && c === 1;

  // 1. Generate games with the champion (deeper search than the eval sees).
  //    --skip-gen: on the first cycle only, gate the games an interrupted earlier
  //    run already flushed to the dataset instead of generating a new batch.
  if (c === 1 && cfg.skipGen) {
    log('Skipping generation (--skip-gen): gating the existing dataset.');
  } else if (cfg.batch === 0) {
    // No dedicated self-play generation: the ranked pool produces training data instead (the
    // gate harvest + strong-engine --play games below). Announced once, then silent per cycle.
    if (c === 1) log(`No dedicated generation (--batch=0): fresh data comes from the gate harvest`
      + (cfg.playStrong ? ` + strong-engine ladder --play (depth ${cfg.playDepth}).` : ' + ranked-pool play.'));
  } else if (!run('Generate (champion self-play)', genBin,
    [`--games=${cfg.batch}`, `--depth=${cfg.depth}`, '--eval=nn',
      ...(cfg.openings !== null ? [`--openings=${cfg.openings}`] : []),
      ...(cfg.openingTopk > 0 ? [`--opening-topk=${cfg.openingTopk}`] : []),
      `--seed=${Date.now()}`, ...jobArg])) break;

  // 2. Featurize the raw positions for the current feature set, into THIS recipe's featurized
  //    file (each filter config keeps its own, so alternating recipes don't re-featurize each
  //    switch), applying the recipe's dataset filters (--quiet-only / --filter-weak /
  //    --drop-conflicts). (After a refresh this is a full pass — the in-place rewrite
  //    invalidates the prefix.)
  if (!run('Featurize', process.execPath,
    [featurizeScript, `--out=${featFile}`, ...(cfg.quietOnly ? ['--quiet-only'] : []),
      ...filterArgs()])) break;

  // 3. Train a candidate to a side file. --lambda blends the champion's search value into
  //    the target (TD/bootstrap) when < 1. Warm-start source for this cycle's candidate:
  //      --cold: nothing on cycle 1 (random init), then the PREVIOUS cycle's candidate
  //              every cycle after — so the run bootstraps a fresh net once and then keeps
  //              refining THAT net. (candidate.json persists across cycles holding last
  //              cycle's output; init==out is safe — train.py reads --init fully before it
  //              writes --out. This is the source that lets a fresh --hidden shape evolve:
  //              the champion is often a different architecture and so unusable as init.)
  //      otherwise: this recipe's track — its accumulated lineage if present, else its saved
  //              best net (the strongest it ever produced — the safe resume point after a gap),
  //              else the global champion but ONLY if the champion shares this recipe's shape
  //              (a foreign-arch champion can't seed it — train.py would fall back to random),
  //              else cold. So sub-threshold gains accumulate per-recipe and a brand-new
  //              architecture track bootstraps itself instead of silently starting from scratch.
  const initFile = cfg.cold
    ? (cold ? null : candidate)
    : (existsSync(lineage) ? lineage
      : existsSync(trackBest) ? trackBest
      : championArchMatches() ? champion
      : null);
  const warm = !!initFile && existsSync(initFile);
  const initLabel = !warm ? ' (cold start)'
    : cfg.cold ? ' (warm-start from previous candidate)'
    : initFile === lineage ? ' (warm-start from lineage)'
    : initFile === trackBest ? ' (warm-start from track best)'
    : ' (warm-start from champion)';
  // --quant (recipe knob, on by default): export the candidate as a quantized integer net, so
  // every champion keeps the incremental-accumulator speedup (~1.5× nodes/sec) in the gate,
  // generation, and the app. Quantization is bit-exact JS/Zig and faithful to the float net
  // (~1cp); warm_start dequantizes an int --init so the float fine-tune is unaffected. --float
  // forks a non-quantized track. --scale/--lr/--wd are passed only when set (else train.py's
  // defaults). --data points the trainer at this recipe's featurized file.
  if (!run(`Train candidate${initLabel}`, python,
    [trainPy, `--hidden=${hidden}`, `--data=${featFile}`, `--out=${candidate}`, `--lambda=${cfg.lam}`,
      ...(cfg.quant ? ['--quant'] : []),
      ...(cfg.scale !== undefined ? [`--scale=${cfg.scale}`] : []),
      ...(cfg.lr !== undefined ? [`--lr=${cfg.lr}`] : []),
      ...(cfg.wd !== undefined ? [`--wd=${cfg.wd}`] : []),
      ...(cfg.epochs !== undefined ? [`--epochs=${cfg.epochs}`] : []),
      ...(cfg.patience !== undefined ? [`--patience=${cfg.patience}`] : []),
      ...(warm ? [`--init=${initFile}`] : [])])) break;

  // 4. Gate: candidate (A) vs champion (B), SPRT(0, elo1). Unless --no-harvest,
  //    the gate's games are appended to the dataset (they're already paid for;
  //    every position gets the value from the engine that searched it — the mover's
  //    own direct depth-d search, tagged with its engine×depth provenance) and the
  //    next cycle's incremental featurize folds them in.
  if (existsSync(resultFile)) rmSync(resultFile);
  if (cfg.harvest && existsSync(gateHarvest)) rmSync(gateHarvest); // no stale harvest from a prior cycle
  if (!run('Gate: candidate vs champion', matchBin,
    ['--eval-a=nn', `--weights-a=${candidate}`, '--eval-b=nn', `--weights-b=${champion}`,
      `--depth=${cfg.gateDepth}`, '--sprt', '--elo0=0', `--elo1=${cfg.elo1}`,
      `--games=${cfg.gateGames}`, `--result-file=${resultFile}`,
      ...(cfg.harvest ? [`--save-games=${gateHarvest}`, `--seed=${Date.now()}`] : []), ...jobArg])) {
    // Ctrl-C / failure mid-gate: the runner still drained its played games to the harvest
    // temp. Fold them in (relabeling if a partial result is readable, else unchanged) so the
    // already-played games aren't lost, then end the loop.
    if (cfg.harvest) {
      let r = null; try { r = JSON.parse(readFileSync(resultFile, 'utf8')); } catch { /* no usable result */ }
      foldGateHarvest(r ? r.sprt === 'H1' : false, r);
    }
    break;
  }

  // 5. Promote only on a significant win (SPRT accepted H1). Never regress.
  let res;
  try { res = JSON.parse(readFileSync(resultFile, 'utf8')); }
  catch {
    log('No match result; keeping champion.');
    if (cfg.harvest) foldGateHarvest(false, null); // fold the played games in unchanged (no edge to relabel with)
    continue;
  }
  const pct = (res.score * 100).toFixed(1);
  // Eval-divergence between candidate and champion (only present when both sides are nn,
  // which the gate always is): how differently the two nets judge midgame positions —
  // context for reading a near-50% result (a corr-1.00 candidate is a clone of the champion).
  const divNote = res.div
    ? ` | divergence ${(res.div.confidentRate * 100).toFixed(1)}% conf-disagree, ${res.div.meanCp.toFixed(0)}cp mean, corr ${res.div.corr.toFixed(2)} (n=${res.div.positions})`
    : '';
  // Snapshot the values the track record needs BEFORE the promote branch overwrites the
  // champion file: the candidate's estimated ABSOLUTE Elo (the champion's current ledger Elo
  // + this gate's edge over it) stays comparable across cycles as the champion strengthens,
  // unlike a raw gate score, so it's what "track best" is ranked by. Null until the ledger
  // rates the champion (from cycle 2 on).
  const gatedVsChampHash = weightsHash(champion);
  const champLedgerNow = championLedgerElo();
  const candAbsElo = champLedgerNow ? champLedgerNow.best + res.elo : null;
  const candHashForTrack = weightsHash(candidate);
  // Fold the gate's harvested games into the dataset, relabeling a non-promoted gate-winning
  // candidate's provenance to a self-describing ephemeral Elo first (foldGateHarvest). Done
  // here, before the promote branch copies the candidate over the champion, so weightsHash
  // still identifies the candidate that actually played.
  if (cfg.harvest) foldGateHarvest(res.sprt === 'H1', res);
  if (res.sprt === 'H1') {
    const arch = JSON.parse(readFileSync(candidate, 'utf8')).arch;
    copyFileSync(champion, prevChampion);   // backup for safety
    copyFileSync(candidate, champion);      // candidate becomes champion
    const champHash = archiveChampion(champion); // keep it reconstructable by its vs version
    if (existsSync(lineage)) rmSync(lineage); // lineage cleared the gate; next start = new champion
    // Publish the new champion into the catalog under its own human name right away, flagged
    // the current champion (so the app shows a real name during its reign, not a generic id).
    const champName = publishChampion(champion, arch);
    promotions++;
    log(`cycle ${c}: PROMOTED ✓  candidate ${pct}% / Elo +${res.elo.toFixed(0)} over champion `
      + `(${res.games} games, cycle took ${fmtDur((Date.now() - cycleT0) / 1000)}). `
      + `New champion named '${champName}' in the catalog (archived ${champHash}.json). Total promotions: ${promotions}.`
      + divNote);
    // (The strength pool is refit at the end of every cycle — runRankPool below — so the
    // just-promoted champion is rated automatically from its harvested gate games via
    // --corpus; no per-promotion ranking step is needed here.)
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
    // champion. (Not for --cold runs: those already chain from the previous candidate
    // unconditionally, so the lineage plays no part.) The champion itself is untouched —
    // the gate still protects it.
    copyFileSync(candidate, lineage);
    log(`cycle ${c}: kept champion — candidate ${pct}% / Elo ${res.elo.toFixed(0)} `
      + `(SPRT ${res.sprt}, ${res.games} games, cycle took ${fmtDur((Date.now() - cycleT0) / 1000)}). `
      + 'Below the gate; candidate kept as lineage for the next cycle.'
      + divNote);
  } else {
    const hadLineage = existsSync(lineage);
    if (hadLineage) rmSync(lineage);
    log(`cycle ${c}: kept champion — candidate ${pct}% / Elo ${res.elo.toFixed(0)} `
      + `(SPRT ${res.sprt}, ${res.games} games, cycle took ${fmtDur((Date.now() - cycleT0) / 1000)}). `
      + `Not a gain.${hadLineage ? ' Lineage reset (next warm-start falls back to this recipe\'s best net).' : ''}`
      + divNote);
  }

  // Record this cycle into the recipe's persistent TRACK (history line + rollup in state.json),
  // and keep the track's `best.json` = the strongest net this recipe ever produced, ranked by
  // estimated absolute Elo (candAbsElo). This is what a later warm resume of the same recipe
  // seeds from, and what train:experiments reads to suggest reviving a promising-but-stalled
  // recipe. Best-effort maintenance, so a write failure logs but never aborts the loop.
  try {
    const rc = recordCycle(track.dir, {
      run: runNo, cycle: c, ts: stamp(),
      score: res.score, edgeElo: res.elo, absElo: candAbsElo,
      sprt: res.sprt, promoted: res.sprt === 'H1',
      div: res.div ? { corr: res.div.corr, meanCp: res.div.meanCp } : null,
      championHash: gatedVsChampHash,
      datasetBytes: existsSync(rawFile) ? statSync(rawFile).size : 0,
      hash: candHashForTrack,
    });
    if (rc.isBest) copyFileSync(candidate, trackBest);
  } catch (e) { log(`  (track record skipped: ${e.message})`); }

  // Refit the strength pool now that this cycle's gate games are harvested into the dataset:
  // --corpus folds them into the Bradley-Terry fit (rating the current champion from its own
  // gate matches) plus a short play budget tightens the most-ambiguous orderings. Runs every
  // cycle so this cycle's own weakest-first refresh (the last step, just below) reads a
  // current ledger. Maintenance: a failure logs but doesn't abort the run (Ctrl-C still ends
  // it via the `stopping` flag).
  if (!stopping) runRankPool('Rank pool (Bradley-Terry, corpus + scheduled play)');

  // Per-cycle value refresh — the LAST step of the cycle. Re-label a small slice of the
  // dataset with the current champion (most records carry `v` from older champions or
  // shallower searches, so this upgrades targets even between promotions). Deliberately
  // placed here rather than before training so a fresh loop start goes STRAIGHT into
  // featurize→train (no upfront depth-8 refresh to sit through) — the point being to iterate
  // quickly on training knobs. Running it after runRankPool also means it reads THIS cycle's
  // freshly-refit ledger (weakest-first targets the current worst cohort) and uses the
  // post-gate champion. The in-place `v` rewrite invalidates the featurize prefix, so the
  // NEXT cycle's featurize is a full pass (same total cost as before, shifted half a cycle).
  // Seeded per cycle so coverage spreads across the set. Maintenance: don't start a long
  // refresh if we're already stopping.
  if (cfg.refreshCycle > 0 && !stopping) {
    run(`Refresh v (${(cfg.refreshCycle * 100).toFixed(1)}% ${refreshMode()} @ depth ${cfg.refreshCycleDepth})`,
      process.execPath, refreshArgs(cfg.refreshCycle, cfg.refreshCycleDepth));
  }
}

log(`train:loop stopped after ${promotions} promotion(s) in ${fmtDur((Date.now() - loopT0) / 1000)}. `
  + `Champion: web/src/nn-weights.json${promotions ? ' (also published in the net catalog under its name)' : ''}.`);
if (promotions) console.log('Run `npm run build` to ship the new champion in the production bundle.');
