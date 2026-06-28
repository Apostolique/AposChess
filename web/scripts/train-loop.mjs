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
//
// Candidate lineage (automatic): when the gate is inconclusive but
// the candidate scored >= 50%, the candidate is KEPT (loop/lineage.json) and the next
// cycle's candidate warm-starts from IT instead of the champion — so sub-threshold
// gains (+10-ish Elo, real but below the SPRT's resolution) accumulate across cycles
// until the lineage clears the gate, instead of being re-derived and discarded every
// cycle. The champion is still protected by the gate; a candidate scoring < 50% (or a
// decided H0) resets the lineage back to the champion.

import { spawnSync } from 'node:child_process';
import {
  existsSync, rmSync, mkdirSync, copyFileSync, readFileSync, writeFileSync, appendFileSync, statSync,
  openSync, readSync, closeSync,
} from 'node:fs';
import { dirname, resolve, join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';

import { fmtDur, fmtMB } from './fmt.mjs';
import { weightsHash, ephemeralVersion } from './vtag.mjs';
import { STOP_EXIT_CODE } from './stop.mjs';
import { isGameRecord, vsAt, setVsAt, normalizeVs, serializeGameRecord } from './gameRecord.mjs';

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
  'npm run rank:pool': { depths: '6,8', 'anchor-depth': '6', games: '10', openings: '6', maxmoves: '200', prior: '1' },
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
function friendlyCmd(argv) {
  const base = scriptCmd.get(argv[0]);
  if (!base) return null;
  const def = scriptDefaults[base] || {};
  const flags = [];
  for (const tok of argv.slice(1)) {
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
const logFile = join(loopDir, 'loop.log');
// PID of this loop, so `npm run train:pause`/`train:resume` (scripts/loop-ctl.mjs) can find
// and freeze/thaw the loop's whole process tree from another terminal — a long run pegs every
// core, so pausing hands the machine back without losing the in-flight gate/generation.
const pidFile = join(loopDir, 'loop.pid');
const pauseFlag = join(loopDir, 'PAUSED'); // marker loop-ctl writes while suspended
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
  // Games per scheduled pool matchup. Kept SMALL on purpose: the pool's --minutes budget is
  // only checked BETWEEN matchups (a matchup always plays to completion), and a depth-8 matchup
  // is slow — so a big batch would blow far past --rank-minutes. Small batches let the budget
  // bound the step to ~rank-minutes; the store accumulates the games across cycles regardless.
  rankGames: num(args['rank-games'], 10),
  // On each promotion the OUTGOING champion is retired into the playable net catalog
  // (web/public/nn) under the next free human name, so past champions stay pickable in the
  // app. Only the most recent --keep-champions retired nets are kept; older ones are pruned
  // (weights file + manifest entry) to bound the deployed bundle (~0.5 MB each). 0 = off.
  keepChampions: num(args['keep-champions'], 12),
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
  const friendly = friendlyCmd(argv);
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

// Depths the pool rates every engine at: the gate depth (cfg.rankDepth, also the hc pin)
// and the generation depth (cfg.depth) — the two depths positions are labeled at (nn6@/nn8@).
// hc<rankDepth> is the pinned Elo-0 node, so all ratings land on one stable scale.
const poolDepths = [...new Set([cfg.rankDepth, cfg.depth])].filter((d) => d >= 1);

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
  // With harvesting on (default), rank:pool appends its games to its archive (ladderGames); we
  // then fold this cycle's additions into the dataset. With --no-harvest, suppress the archive
  // too (--no-save-games), consistent with the gate. Capture the archive size up front so we can
  // append only the games this run added.
  const before = (cfg.harvest && existsSync(ladderGames)) ? statSync(ladderGames).size : 0;
  run(label, process.execPath,
    [rankScript, '--corpus', `--minutes=${cfg.rankMinutes}`,
      `--depths=${poolDepths.join(',')}`, `--anchor-depth=${cfg.rankDepth}`,
      `--games=${cfg.rankGames}`, `--store=${ladderStore}`, `--ledger=${ledgerFile}`,
      ...(cfg.harvest ? [] : ['--no-save-games']),
      '--no-scan', `--seed=${Date.now()}`, ...jobArg]);
  if (cfg.harvest) foldNewLadderGames(before);
}

// Append the games rank:pool added to its archive THIS cycle (everything past `beforeSize`) into
// the dataset, so the next incremental featurize trains on them. The games are fresh (unique
// game ids), so no dedup is needed here; merge-data later collapses any overlap idempotently.
// Their players are all rankable engines, so labels are kept as-is (no ephemeral rewrite).
function foldNewLadderGames(beforeSize) {
  if (!existsSync(ladderGames)) return;
  const size = statSync(ladderGames).size;
  if (size <= beforeSize) return; // nothing new (rank played no games, or harvesting was off)
  const fd = openSync(ladderGames, 'r');
  try {
    const buf = Buffer.alloc(size - beforeSize);
    readSync(fd, buf, 0, buf.length, beforeSize);
    let chunk = buf.toString('utf8');
    if (!chunk.endsWith('\n')) chunk += '\n'; // each game is a full line; guard a torn tail
    appendFileSync(rawFile, chunk);
    const added = chunk.split('\n').filter(Boolean).length;
    log(`  Folded ${added} new ladder game(s) into the dataset (next featurize trains on them).`);
  } finally { closeSync(fd); }
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

// Human names for retired champions, handed out in order (the first eight — Ada..Hugo —
// were the initial hand-published lineage). A name freed by pruning becomes reusable.
const CHAMPION_NAMES = ['Ada', 'Boris', 'Clara', 'Dexter', 'Elena', 'Felix', 'Greta', 'Hugo',
  'Ivy', 'Jack', 'Kara', 'Leo', 'Mona', 'Nash', 'Olga', 'Pia', 'Quinn', 'Rosa', 'Sven',
  'Tara', 'Uma', 'Victor', 'Wren', 'Xena', 'Yuri', 'Zara'];

function upsertManifest(arch, retireFile) {
  let man = { default: 'balanced-64', nets: [] };
  try { man = JSON.parse(readFileSync(manifestFile, 'utf8')); } catch { /* new manifest */ }
  man.nets = (man.nets || []).filter((n) => n.name !== 'loop-champion');
  man.nets.push({ name: 'loop-champion', file: 'loop-champion.json', arch, note: 'Latest gated train:loop champion (current).' });
  if (retireFile) retireChampion(man, retireFile);
  man.nets.sort((a, b) => a.name.localeCompare(b.name));
  writeFileSync(manifestFile, JSON.stringify(man, null, 2) + '\n');
}

// Publish the just-dethroned champion `file` into the net catalog under the next free human
// name so it stays pickable in the app, then prune the catalog to the most recent
// cfg.keepChampions retired champions (deleting their weights + manifest entries). Mutates
// `man.nets`; the caller writes the manifest. Idempotent by content hash, so re-running a
// promotion (or starting a loop whose champion is already retired) is a no-op.
function retireChampion(man, file) {
  if (cfg.keepChampions <= 0) return;
  const hash = weightsHash(file);
  if (hash === '?') return;
  const retired = () => man.nets.filter((n) => n.loopChampion);
  if (retired().some((n) => n.hash === hash)) return; // already published
  const arch = JSON.parse(readFileSync(file, 'utf8')).arch;
  const used = new Set(man.nets.map((n) => n.name));
  const name = CHAMPION_NAMES.find((n) => !used.has(n)) || `champ-${hash}`;
  const out = `${name.toLowerCase()}.json`;
  const gen = Math.max(0, ...retired().map((n) => n.gen || 0)) + 1;
  copyFileSync(file, join(publicNN, out));
  man.nets.push({ name, file: out, arch, loopChampion: true, hash, gen,
    note: `Retired train:loop champion (${hash}, ${new Date().toISOString().slice(0, 10)}).` });
  // Keep only the most recent cfg.keepChampions; prune the oldest by generation.
  const byAge = retired().sort((a, b) => (a.gen || 0) - (b.gen || 0));
  for (const e of byAge.slice(0, Math.max(0, byAge.length - cfg.keepChampions))) {
    const p = join(publicNN, e.file);
    if (existsSync(p)) rmSync(p);
    man.nets = man.nets.filter((n) => n !== e);
  }
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
  + `SPRT(0,${cfg.elo1}) | candidate hidden=[${hidden}] λ=${cfg.lam} ${cfg.cold ? 'cold first cycle, warm after' : 'warm'} start`
  + `${existsSync(lineage) ? ' (resuming lineage)' : ''} | `
  + `refresh/cycle ${cfg.refreshCycle > 0 ? `${(cfg.refreshCycle * 100).toFixed(1)}% @ depth ${cfg.refreshCycleDepth}` : 'off'} | `
  + `refresh on promotion ${cfg.refreshFrac > 0 ? `${(cfg.refreshFrac * 100).toFixed(0)}% @ depth ${cfg.refreshDepth}` : 'off'} | `
  + `rank ${cfg.rank ? `pool every cycle (hc${cfg.rankDepth} pin, depths ${poolDepths.join('+')}, corpus + ${cfg.rankMinutes}m play)` : 'off'} | `
  + `cycles ${cfg.cycles === Infinity ? '∞' : cfg.cycles}`);
log('Pause/resume from another terminal: `npm run train:pause` / `npm run train:resume` (frees all CPU, no work lost).');

const jobArg = cfg.jobs !== undefined ? [`--jobs=${cfg.jobs}`] : [];

// No startup ranking: the pool is refit at the END of each cycle (runRankPool), after that
// cycle's gate games have been harvested into the dataset. So cycle 1 still refreshes with
// the classic random fraction (no ledger yet); from cycle 2 on the ledger exists and the
// refreshes go weakest-first.

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
  } else if (!run('Generate (champion self-play)', genBin,
    [`--games=${cfg.batch}`, `--depth=${cfg.depth}`, '--eval=nn',
      ...(cfg.openings !== null ? [`--openings=${cfg.openings}`] : []),
      ...(cfg.openingTopk > 0 ? [`--opening-topk=${cfg.openingTopk}`] : []),
      `--seed=${Date.now()}`, ...jobArg])) break;

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
  if (!run('Featurize', process.execPath,
    [featurizeScript, ...(cfg.quietOnly ? ['--quiet-only'] : [])])) break;

  // 3. Train a candidate to a side file. --lambda blends the champion's search value into
  //    the target (TD/bootstrap) when < 1. Warm-start source for this cycle's candidate:
  //      --cold: nothing on cycle 1 (random init), then the PREVIOUS cycle's candidate
  //              every cycle after — so the run bootstraps a fresh net once and then keeps
  //              refining THAT net. (candidate.json persists across cycles holding last
  //              cycle's output; init==out is safe — train.py reads --init fully before it
  //              writes --out. This is the source that lets a fresh --hidden shape evolve:
  //              the champion is often a different architecture and so unusable as init.)
  //      otherwise: the kept lineage (a previous rejected-but-positive candidate) when one
  //              exists, else the champion — so sub-threshold gains accumulate instead of
  //              being re-derived every cycle (train.py ignores --init if the shapes differ).
  const initFile = cfg.cold
    ? (cold ? null : candidate)
    : (existsSync(lineage) ? lineage : champion);
  const warm = !!initFile && existsSync(initFile);
  const initLabel = !warm ? ' (cold start)'
    : cfg.cold ? ' (warm-start from previous candidate)'
    : initFile === lineage ? ' (warm-start from lineage)' : '';
  // --quant: export the candidate as a quantized integer net, so every champion keeps the
  // incremental-accumulator speedup (~1.5× nodes/sec) in the gate, generation, and the app.
  // Quantization is bit-exact JS/Zig and faithful to the float net (~1cp); warm_start
  // dequantizes an int --init so the float fine-tune is unaffected.
  if (!run(`Train candidate${initLabel}`, python,
    [trainPy, `--hidden=${hidden}`, `--out=${candidate}`, `--lambda=${cfg.lam}`, '--quant',
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
    // prevChampion now holds the dethroned champion (backed up just above): retire it into
    // the catalog under a name, and publish the new one as the live 'loop-champion'.
    upsertManifest(arch, prevChampion); copyFileSync(candidate, loopChampPub);
    promotions++;
    log(`cycle ${c}: PROMOTED ✓  candidate ${pct}% / Elo +${res.elo.toFixed(0)} over champion `
      + `(${res.games} games, cycle took ${fmtDur((Date.now() - cycleT0) / 1000)}). `
      + `New champion published as 'loop-champion' (archived ${champHash}.json). Total promotions: ${promotions}.`);
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
      + 'Below the gate; candidate kept as lineage for the next cycle.');
  } else {
    const hadLineage = existsSync(lineage);
    if (hadLineage) rmSync(lineage);
    log(`cycle ${c}: kept champion — candidate ${pct}% / Elo ${res.elo.toFixed(0)} `
      + `(SPRT ${res.sprt}, ${res.games} games, cycle took ${fmtDur((Date.now() - cycleT0) / 1000)}). `
      + `Not a gain.${hadLineage ? ' Lineage reset to the champion.' : ''}`);
  }

  // Refit the strength pool now that this cycle's gate games are harvested into the dataset:
  // --corpus folds them into the Bradley-Terry fit (rating the current champion from its own
  // gate matches) plus a short play budget tightens the most-ambiguous orderings. Runs every
  // cycle so the next cycle's weakest-first refresh reads a current ledger. Maintenance: a
  // failure logs but doesn't abort the run (Ctrl-C still ends it via the `stopping` flag).
  if (!stopping) runRankPool('Rank pool (Bradley-Terry, corpus + scheduled play)');
}

log(`train:loop stopped after ${promotions} promotion(s) in ${fmtDur((Date.now() - loopT0) / 1000)}. `
  + `Champion: web/src/nn-weights.json${promotions ? " (also catalog 'loop-champion')" : ''}.`);
if (promotions) console.log('Run `npm run build` to ship the new champion in the production bundle.');
