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
//   --batch=N       dedicated champion self-play games per cycle (default 0 = none; the
//                   ranked pool's strong --play games are the generator). Set N>0 to add a
//                   deep champion self-play batch on top.
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
//   --gate-futility=G  SPRT futility stop (default 0.05; 0 = off), forwarded to the
//                   match runner as --sprt-futility. SPRT decides fast at the extremes
//                   but burns the whole --gate-games cap when the candidate is roughly
//                   EVEN with the champion (true edge between 0 and elo1 — exactly where
//                   warm-started near-clones live), on a verdict ("inconclusive") that
//                   was knowable at halfway. From 30% of the cap on, the runner stops
//                   once even an optimistic read of the score leaves < G chance of
//                   reaching the promotion bound in the games left. Monte-Carlo'd
//                   (2026-07-14): ~20-25% fewer games on even candidates, < 2 points of
//                   promotion probability lost on a true +20 (and a futility-stopped
//                   gainer survives as lineage and re-gates next cycle). The verdict
//                   stays "inconclusive"; the log line notes the early stop.
//   --no-screen     turn OFF the shadow-mode low-depth screen, which is ON by default.
//                   Before the gate, it plays the candidate vs the champion at --screen-depth
//                   for a fixed --screen-games, converts that edge to a gate-depth-equivalent
//                   Elo (divide by --screen-ratio), and logs what a screen WOULD have decided
//                   — then runs the real gate anyway and records both. It never promotes,
//                   never rejects, and its games never reach the trainer; it exists to measure
//                   whether a cheap screen could replace most of the gate.
//                   It is ON by default because a bare `npm run train:loop` is how this loop
//                   gets run, and an opt-in measurement is one that never happens. It cannot
//                   change a verdict — the cost is ~4-5 min/cycle (~2% of a full-gate cycle)
//                   plus ~40 MB/cycle of kept games (--no-screen-save drops that half).
//                   Why: a full 2000-game gate at depth 6 is ~3.5-4h, most of a cycle, and
//                   the futility stop only helps the CLEARLY bad candidates — the expensive
//                   cycles are the ambiguous 51-52% ones. Depth 1 is ~414x cheaper per game
//                   (0.155s vs 64s), so 20k screen games cost ~5 min and pin the score far
//                   tighter than the gate does. Across this engine's 21 champions, depth-1
//                   Elo ranks depth-8 Elo at Spearman 0.974, and the cumulative d1/d6 gain
//                   ratio has held at 0.62 +/- 0.035 over the last ten champions.
//                   The open question, and the ONLY reason this is shadow mode rather than a
//                   cascade: that ratio is a POPULATION slope over well-separated champions.
//                   What a screen needs is the PER-CANDIDATE residual on near-clones, and the
//                   ladder cannot measure it (its own +/-31 and +/-35 Elo margins at d1/d6
//                   already explain more scatter than the observed residual). Rejected
//                   candidates are never archived, so the only way to get paired data on the
//                   population a screen would actually face is to measure it in-loop. Run
//                   ~10 cycles, then `npm run screen:report` (it refuses to conclude below 8).
//   --screen-depth=D  screen search depth (default 1)
//   --screen-games=N  screen games, played to a fixed N with NO SPRT — the screen estimates
//                   an edge rather than testing a hypothesis (default 20000, ~5 min at d1)
//   --screen-ratio=R  assumed screen-depth:gate-depth Elo ratio (default 0.62). Affects only
//                   the logged prediction; screen:report re-fits it from the recorded pairs.
//   --no-screen-save  discard the screen's games instead of keeping them. By default they go
//                   to loop/screen-games.jsonl — a SEPARATE dataset, never selfplay.jsonl.
//                   Two reasons to keep them. (1) The ledger can rate from them: the loop
//                   passes the file to rank:pool as --corpus-extra, so a PROMOTED candidate's
//                   20k direct games against the champion it dethroned become the densest
//                   single-pair evidence the pool has (its own per-cycle play budget buys ~28
//                   games, and 62 rank-adjacent pairs have never met at all). A non-promoted
//                   candidate isn't archived, so its games get the same ephemeral
//                   "nn<d>@elo<E>" tag the gate harvest uses — keyed off the SCREEN's edge at
//                   the SCREEN's depth, since that tag is an absolute Elo at that depth.
//                   (2) They're real games with real terminal results, so if a use for
//                   low-depth play is ever found the data is already there.
//                   What they must NOT do is reach the trainer: depth-1 `v` labels on
//                   depth-1-play positions are precisely the weak off-distribution cohort
//                   --filter-weak and refresh-v exist to drain, and at ~20k games/cycle they
//                   would outnumber the cycle's real data ~10x. Hence the separate file.
//                   Disk: ~2 KB per game (measured — depth-1 games run ~78 recorded plies vs
//                   the gate's ~48, since weaker play shuffles longer), so ~40 MB per cycle at
//                   the default 20k, in git-ignored training/data/loop/.
//   --lambda=L      TD/bootstrap target mix for training the candidate (default 1 =
//                   pure game result; <1 leans on the champion's own search value,
//                   an unbiased bootstrap — recorded because generation uses the net)
//   --hidden=H      candidate architecture. OMIT IT and the loop picks one itself — see
//                   "Choosing the recipe" below, along with the knobs it can also choose.
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
//                   the current champion. DEFAULT 0 = OFF (2026-07-26); P=1 means "the
//                   whole weakest cohort". It re-labels older/shallower `v` upward, which
//                   is real work at λ<1 — but it no longer scales. Measured on the 21.0M-
//                   position set: refresh-v does ~5.6k positions per minute of budget, so
//                   the 10-minute budget relabels ~0.25% of the set per cycle (~350+ cycles
//                   to walk it once), while the in-place rewrite forces a FULL re-featurize
//                   (7m22s per track on a 1.9 GB features file, vs seconds incremental) —
//                   ~15% of a 2-hour cycle for a quarter-percent of the labels. New data
//                   already arrives labeled by top-band engines, so the fresh share grows
//                   on its own; the cheap lever for the old bulk is featurize --filter-weak
//                   (it drops weak-labeled games outright and is a gateable recipe knob),
//                   not this. Set P>0 to turn it back on — worthwhile in bursts, e.g. to
//                   drain the unrecoverable/untagged (-Inf) cohort, which nothing else fixes.
//   --refresh-cycle-depth=D  search depth for the per-cycle refresh (default: --depth,
//                   so the re-labels match generation's deep-label quality)
//   --no-refresh    skip ALL value refreshing (both --refresh-cycle and --refresh-frac),
//                   regardless of their values. Both default to 0 now, so this only matters
//                   as a hard override on a command line that also passes a fraction.
//   --rotate=auto|N|off  AUTO-ROTATE the architecture: when the current track has stalled, ask
//                   the experiment registry what to try next (suggestRecipes) and switch to it,
//                   creating/resuming that recipe's track. **`auto` is the default.**
//                   `auto` decides from the track's absElo TRAJECTORY, not a raw cycle count,
//                   because the count alone cannot see the one case that matters: a new or
//                   grafted architecture routinely loses the gate for many cycles while its
//                   absolute Elo climbs toward overtaking the champion (the pattern that
//                   produced Mona), and a fixed counter rotates away from it exactly when it is
//                   working. So `auto` never rotates before 3 cycles on the track, always
//                   rotates by 8 (the registry's own advice is that a track which hasn't
//                   promoted by ~8-10 cycles won't), and in between rotates only when the
//                   re-anchored absElo trend is flat or falling (< 2 Elo/cycle). Re-anchored
//                   because a stored absElo drifts as the pool re-fits, so comparing raw
//                   snapshots would read drift as trend.
//                   `--rotate=N` keeps the old fixed rule (N cycles without a promotion) and
//                   `--rotate=off` (or `=0`) disables rotation entirely, running one recipe
//                   until Ctrl-C. This is what makes an
//                   unattended multi-day run useful: measured over 120 cycles, a NEW track's first
//                   cycle promoted ~29% of the time versus ~1.8% for a continuation cycle, so a
//                   loop that keeps grinding one shape after it stalls is spending ~2h a cycle to
//                   re-test a net that correlates 0.99 with the champion. A PROMOTION resets the
//                   counter — a shape that just won has earned more cycles.
//                   The registry's suggestions are a space-filling design over
//                   architecture space until it has enough tracks to rank on predicted Elo
//                   (`npm run train:experiments` shows which mode it's in).
//                   The counter starts at 0 on every launch, so relaunching does NOT immediately
//                   re-rotate a track that was already stale — the recipe you launch with always
//                   gets its full window.
//                   A rotation adopts the suggested recipe WHOLE — architecture and knobs — even
//                   the parts you pinned on the command line. That's the point of asking for it:
//                   flags decide where the run STARTS, --rotate is you telling the loop to move
//                   on from there, and a rotation that honoured --hidden could never change the
//                   architecture. Leave --rotate off to stay on the recipe you launched with.
//   --no-link-pass  turn OFF the unrestricted LINK PASS, which is on by default (see runLinkPass).
//                   `--play-strong` pins the routine rank step's --play to the ~8 strongest nn
//                   engines at one depth, so it can only ever play games INSIDE that set — while
//                   the ladder's convergence deficit lives across the whole 184-node pool. That
//                   is a reachability problem, not a budget one, and it is why the ledger could
//                   sit permanently un-converged no matter how large --rank-minutes got: measured
//                   2026-08-06, 62 rank-adjacent pairs had never met and `adjacentUnderLinked-
//                   Schedulable` was 0, so not one of them could be scheduled. Closing them meant
//                   stopping the loop and running rank:pool by hand (no --play restriction, which
//                   is exactly why that worked). The link pass does it automatically: when the
//                   ledger reports unlinked adjacent pairs AND none are schedulable, it runs a
//                   short unrestricted rank:pool. Its games go to loop/ladder-link-games.jsonl,
//                   NOT the dataset — it plays whatever pairs the graph is missing, weak nodes
//                   included, which is precisely what --play-strong keeps out of training — and
//                   come back as --corpus-extra so the evidence still accumulates across cycles.
//   --link-minutes=M  play budget for that pass (default 10, capped at 30).
//   --rank-cycle=auto|N  refit the Bradley-Terry pool every N cycles instead of EVERY cycle.
//                   **`auto` is the default**: refit every cycle while the ladder is still short
//                   of convergence (the ratings are genuinely moving then), dropping to every 3rd
//                   once it converges, and always on a promotion. The corpus fold is
//                   free, but the --rank-minutes play budget is not — measured 2026-07-26 it
//                   was ~35 min of a ~2 h cycle (25-35%) for 28 new ladder games. Between
//                   promotions that mostly re-confirms a ledger that isn't moving: the
//                   champion's margin holds steady (the adaptive note printed an identical
//                   `±42 → 1.06×` every cycle) and --filter-weak's cutoff is quantized to
//                   50-Elo steps, so it only shifts when the champion actually climbs. N=3
//                   buys back ~25% of a cycle for a slightly staler ledger — a one-cycle lag
//                   is benign, it just means the weakest-first refresh targets a marginally
//                   out-of-date cohort. A PROMOTION always forces a refit regardless of N,
//                   so a fresh champion is rated and depth-calibrated immediately.
//   --calibrate-minutes=M  after a PROMOTION, extend that cycle's rank pass by M minutes
//                   (default 10) with the new champion schedulable at EVERY depth, so the pool's
//                   onboard floor rates its 0-game depth bands instead of leaving them at the
//                   placeholder floor (the "best across depths" absElo the loop steers by then
//                   rests on real games at each depth, not just the gate/play depths).
//   --no-calibrate  disable that post-promotion depth calibration (M=0).
//   --no-adaptive   disable the adaptive maintenance budget (ON by default): revert to the FIXED
//                   --rank-minutes / --refresh-cycle every cycle. On, each is bounded-scaled per
//                   cycle from a signal — rank-minutes ↑ when the champion's ledger margin (the CI
//                   on its absElo) is wide, refresh ↑ in the cycles right after a promotion (stalest
//                   `v`) — anchored so a neutral signal == today's fixed values (can't starve a step).
//                   With --refresh-cycle at its 0 default only the rank-minutes half does anything.
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
// Choosing the recipe (what you don't pass, the loop picks): the registry designs experiments
// over two families — the ARCHITECTURE (--hidden) and the training KNOBS (--lambda,
// --quiet-only, --filter-weak, --drop-conflicts) — and fills in whichever you left off the
// command line, so `npm run train:loop -- --rotate=8` is a complete instruction:
//   * omit --hidden and it picks a shape;
//   * omit ALL FOUR knobs and it picks those too. All-or-nothing on purpose: omitting a flag
//     that defaults to off is the ordinary way to write a command, so pin any one knob and the
//     rest keep their documented defaults — every command line that used to work still means
//     exactly what it meant.
// Suggestions are one-axis variations of the REIGNING recipe (the most-promoted track), so a
// result reads as an ablation of what's working rather than a jump to an unrelated corner, and
// the family the registry has sampled less goes first — the first 9 tracks covered 8 distinct
// shapes but only 3 knob settings, with λ never once off 0.5, so the knob axes lead until they
// catch up. On a fresh clone with no tracks, the loop falls back to the champion's shape and
// your flags as given: the suggester bounds its design region relative to an existing track, so
// with none its "most novel" pick is the widest, deepest net in the grid. Startup logs what it
// chose, why, and the exact command that pins it. --rotate then keeps this going unattended.
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
} from 'node:fs';
import { dirname, resolve, join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';
import { cpus } from 'node:os';

import { fmtDur, fmtMB } from './fmt.mjs';
import { weightsHash, ephemeralVersion } from './vtag.mjs';
import { STOP_EXIT_CODE } from './stop.mjs';
import { isGameRecord, vsAt, setVsAt, normalizeVs, serializeGameRecord } from './gameRecord.mjs';
import {
  buildRecipe, parseRecipeExtra, ensureTrack, beginRun, recordCycle, recipeLabel, readState,
  suggestRecipes, recipeToFlags, recipeId, readAllTracks,
  readHistory, reAnchoredAbsElo, ledgerBestByVersion,
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
const benchBin = resolve(engineDir, 'zig-out', 'bin', isWin ? 'apos-bench.exe' : 'apos-bench');
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
// The low-depth SCREEN's result file (--screen; see the flag doc). Separate from the gate's
// match.json so a screen can never be mistaken for the promotion verdict, and so an aborted
// screen leaves the gate's own result untouched.
const screenFile = join(loopDir, 'screen.json');
// --screen-save: the screen's games, kept in a SEPARATE dataset — never selfplay.jsonl.
// They're real games with real terminal results, so they're worth keeping, but their `v`
// labels are depth-1 opinions and their positions come from depth-1 play, which is exactly
// the off-distribution weak-label cohort --filter-weak and refresh-v exist to drain. Keeping
// them beside the training set rather than inside it means the ledger can rate from them (see
// --corpus-extra in runRankPool) without a single depth-1 label reaching the trainer.
const screenHarvest = join(loopDir, 'screen-harvest.jsonl'); // per-cycle temp, folded below
const screenArchive = join(loopDir, 'screen-games.jsonl');   // the persistent separate dataset
// The unrestricted link pass's games (runLinkPass). Rating evidence, not training data — it
// plays whatever rank-adjacent pairs have never met, which includes the weak nodes that
// --play-strong deliberately keeps out of the trainer. Read back via --corpus-extra.
const linkArchive = join(loopDir, 'ladder-link-games.jsonl');
// The gate's --save-games harvest is written HERE (a temp), not straight into the dataset,
// so the loop can rewrite a non-promoted candidate's provenance before folding it in (see
// foldGateHarvest). Cleared before each gate and deleted after folding.
const gateHarvest = join(loopDir, 'gate-harvest.jsonl');
// Bradley-Terry pool ledger (npm run rank:pool): the fitted Elo the refreshes consume. rank:pool
// harvests its games straight into the dataset and re-derives its ratings from it every run, so
// there's nothing to fold or keep in lockstep here — a standalone rank:pool between cycles is
// picked up automatically, because it wrote to the same file.
const ledgerFile = join(loopDir, 'engine-elo.ladder.json');
const logFile = join(loopDir, 'loop.log');
// PID of this loop, so `npm run train:pause`/`train:resume` (scripts/loop-ctl.mjs) can find
// and freeze/thaw the loop's whole process tree from another terminal — a long run pegs every
// core, so pausing hands the machine back without losing the in-flight gate/generation.
const pidFile = join(loopDir, 'loop.pid');
const pauseFlag = join(loopDir, 'PAUSED'); // marker loop-ctl writes while suspended
const stopFlag = join(loopDir, 'STOP');    // marker `npm run train:stop` writes (see stopRequested)
const publicNN = resolve(webDir, 'public', 'nn');
const manifestFile = join(publicNN, 'manifest.json');
// Champions pruned from the manifest keep their identity here (hash -> name for ledger labels,
// and their names are never handed out again). Append-only, written by publishChampion.
const nameHistoryFile = join(publicNN, 'name-history.json');

const args = Object.fromEntries(
  process.argv.slice(2).map((a) => {
    const m = a.replace(/^--/, '').split('=');
    return [m[0], m.length > 1 ? m[1] : true];
  }),
);
const num = (v, d) => (v === undefined ? d : Number(v));
// A boolean flag: `--x` is on, `--x=false|0|off|no` is off, absent takes the default. A bare
// `!!args.x` reads `--quiet-only=false` as ON, which matters now that OMITTING the flag hands
// the choice to the registry — turning a knob off explicitly has to be sayable.
const flag = (v, d = false) =>
  (v === undefined ? d : !(v === 'false' || v === '0' || v === 'off' || v === 'no'));
// One full parallel wave of the match runner: apos-match dispatches games one at a time
// across --jobs workers, so a matchup smaller than the worker count leaves cores idle from
// the start (not just at the tail). Rounded up to even — games come in color-reversed pairs.
const effJobs = args.jobs !== undefined ? Number(args.jobs) : cpus().length;
const defaultRankGames = Math.max(2, Math.ceil(effJobs / 2) * 2);
const cfg = {
  // Dedicated champion self-play generation per cycle. Defaults to 0: the ranked pool's strong
  // --play games (cfg.playStrong, auto-on when batch is 0) are the generator, so a bare
  // `npm run train:loop` needs no --batch. Set --batch=N to add a deep champion self-play batch
  // on top (then --play-strong to keep the pool generating too). --skip-gen still gates an
  // already-flushed dataset on cycle 1 without generating.
  batch: num(args.batch, 0),
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
  // Futility stop for the gate SPRT (0 = off). See the flag doc above; the measured trade
  // (Monte Carlo vs the exact walk) is ~20-25% fewer games on even candidates for < 2 points
  // of promotion probability on a true +elo1 — which the lineage recovers next cycle.
  gateFutility: num(args['gate-futility'], 0.05),
  // Low-depth SCREEN, shadow mode (see the --screen flag doc). ON by default: it measures the
  // candidate at a cheap depth BEFORE the gate, predicts the gate-depth edge, and logs the
  // prediction — then runs the real gate regardless. Instrumentation only: nothing here feeds
  // the promotion decision, and no screen game ever reaches the trainer's dataset
  // (--screen-save keeps them in a separate file the ledger rates from; see its flag doc).
  // Default-on because it is answering an open question and a bare `npm run train:loop` is how
  // this loop actually gets run — an opt-in measurement is a measurement that never happens.
  // It cannot change a verdict, so the whole cost is ~4-5 min/cycle. --no-screen turns it off.
  screen: flag(args.screen, true) && !args['no-screen'],
  screenDepth: num(args['screen-depth'], 1),
  screenGames: num(args['screen-games'], 20000),
  // Elo transfer ratio screen-depth : gate-depth. 0.62 is the measured d1/d6 ratio over this
  // engine's champion sequence (see the flag doc). Only affects the LOGGED prediction — the
  // paired (screen, gate) observations recorded per cycle let screen:report re-fit it.
  screenRatio: num(args['screen-ratio'], 0.62),
  // Keep the screen's games in loop/screen-games.jsonl (a separate dataset — never the
  // trainer's). On by default when --screen is on: the games are already paid for, and the
  // ledger can rate from them. --no-screen-save discards them instead (see the flag doc for
  // the disk cost, which is the only reason you'd want to).
  screenSave: flag(args['screen-save'], true) && !args['no-screen-save'],
  lam: num(args.lambda, 1), // TD target mix passed to train.py (1 = pure result)
  // Drop tactically loud positions (in check / winning capture available) at featurize time
  // so the static net trains on the quiet-position distribution it's actually queried on at
  // qsearch leaves. Off by default (gate it head-to-head before adopting). Toggling it forces
  // the next featurize to be a full pass (the meta sidecar records the filter state).
  quietOnly: flag(args['quiet-only']),
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
  // Per-cycle refresh: a slice of the dataset re-labeled with the current champion every
  // cycle. OFF by default since 2026-07-26 — at 21M positions it relabels ~0.25% per cycle
  // yet forces a full re-featurize on every track, ~15% of cycle time for a quarter-percent
  // of the labels (see the --refresh-cycle help above). --refresh-cycle=1 turns it back on.
  refreshCycle: args['no-refresh'] ? 0 : num(args['refresh-cycle'], 0),
  refreshCycleDepth: num(args['refresh-cycle-depth'], num(args.depth, 8)),
  // Engine ranking for smart weakest-first v refresh. On by default, driven by the
  // self-relative Bradley-Terry POOL (rank:pool / depth-ladder.mjs).
  // EVERY cycle the loop refits the pool: it derives the ratings from the whole dataset's games
  // (--corpus — so the new champion is rated automatically from its gate matches, no dedicated
  // gauntlet) and plays a short --rank-minutes budget of the most-ambiguous matchups, harvested
  // back into that same dataset, to tighten ratings. The refreshes below read the resulting parallel
  // ledger (engine-elo.ladder.json) to relabel the WEAKEST engine's `v` first. --no-rank reverts.
  rank: !args['no-rank'],
  // hc pin depth for the pool (Elo 1500) — every rating lands on this stable scale.
  rankDepth: num(args['rank-depth'], 6),
  // Unrestricted link-closing pass (runLinkPass). ON by default: without it the ladder can sit
  // permanently un-converged whenever --play-strong makes the deficient pairs unschedulable,
  // which is the state the pool was actually found in. --no-link-pass reverts.
  linkPass: !args['no-link-pass'],
  linkMinutes: num(args['link-minutes'], 10),
  // Wall-clock the pool plays per cycle, on top of the (free) corpus fold. Short, because the
  // corpus already rates the champion from its gate games and the store accumulates across
  // cycles — each cycle just tightens the most-ambiguous orderings.
  rankMinutes: num(args['rank-minutes'], 5),
  // Refit the pool every N cycles instead of every cycle (1 = every cycle, the old behaviour).
  // The --rank-minutes play budget is the second-largest slice of a cycle after the gate, and
  // between promotions the ledger barely moves — the champion's margin holds steady and
  // --filter-weak's cutoff is quantized to 50-Elo steps, so a refit mostly re-confirms what the
  // last one said. Keyed on the TRACK-cumulative cycle number at the call site, so the cadence
  // survives a Ctrl-C + warm relaunch instead of restarting (and re-ranking) on every launch's
  // first cycle. A promotion overrides it, so a fresh champion is never left uncalibrated.
  // Non-finite input falls back to 1 rather than propagating NaN — `c % NaN === 0` is never
  // true, which would silently disable the refit for the whole run instead of erroring.
  // 'auto' (the default) keys the cadence off the ledger's own convergence verdict instead of a
  // hand-picked N — see rankCadence. A number pins it to the old fixed every-N-cycles rule.
  rankCycle: args['rank-cycle'] === undefined || args['rank-cycle'] === 'auto' ? 'auto'
    : (Number.isFinite(Math.round(Number(args['rank-cycle'])))
      ? Math.max(1, Math.round(Number(args['rank-cycle']))) : 1),
  // Auto-rotate the architecture after this many non-promoting cycles on the current track.
  // 0 = off (run one recipe forever, the historical behaviour). See the --rotate help above for
  // why rotation beats grinding: promotions overwhelmingly land on a NEW track's early cycles.
  // 'auto' (the default) rotates on the track's absElo TREND rather than a raw cycle count —
  // see rotationDue. A number keeps the old fixed rule, 0/'off' disables rotation entirely.
  rotate: args.rotate === undefined || args.rotate === 'auto' ? 'auto'
    : (args.rotate === 'off' || Number(args.rotate) === 0 ? 'off'
      : Math.max(1, Math.round(Number(args.rotate)))),
  // After a PROMOTION, extend that cycle's rank pass by this many minutes with the NEW champion
  // schedulable at EVERY depth (a bare-hash --play spec), so the ladder's onboard floor anchors
  // its 0-game depth nodes to the scale. Without it a fresh champion is rated only at the depths
  // it actually played — the gate depth and the strong-play depth — leaving its ledger Elo at the
  // placeholder floor for every other depth, so the "best across depths" absElo the loop steers by
  // (see loop-progress) rests on one or two thin bands. 0 / --no-calibrate disables it.
  calibrateMinutes: args['no-calibrate'] ? 0 : num(args['calibrate-minutes'], 10),
  // Adaptive maintenance budget (ON by default; --no-adaptive reverts to fixed knobs). Each
  // per-cycle maintenance knob starts at its configured value and shifts only within a bounded
  // band around it, driven by a robust signal — so a neutral signal reproduces today's fixed
  // behaviour exactly and a strong signal can't starve a step (the reallocation is bounded, not
  // free). Signals: rank-minutes scales with the champion's ledger-margin (measure more when the
  // absElo the loop steers by is uncertain), refresh fraction scales with post-promotion staleness
  // (relabel more right after the champion — hence every `v` target — changed, then taper). See
  // adaptiveMaintenance().
  adaptive: !args['no-adaptive'],
  // Games per scheduled pool matchup. Defaults to ONE PARALLEL WAVE (the --jobs count, rounded
  // up to even): fewer games than workers leaves cores idle for the whole matchup, while a big
  // batch would blow far past --rank-minutes (the budget is only checked BETWEEN matchups — a
  // matchup always plays to completion, and depth-8 matchups are slow). One wave keeps every
  // core busy exactly once per matchup; the store accumulates the games across cycles regardless.
  rankGames: num(args['rank-games'], defaultRankGames),
  // Direct-link floor passed through to rank:pool (--link): games a rank-adjacent pair must have
  // played AGAINST EACH OTHER before the ordering objective gets the budget. Left unset, rank:pool
  // defaults it to half a matchup, so one of the loop's small matchups clears a pair; --rank-link=0
  // turns it off and hands the whole budget back to the ordering objective.
  rankLink: args['rank-link'] !== undefined ? Math.max(0, Number(args['rank-link'])) : null,
  // On each promotion the NEW champion is published into the playable net catalog
  // (web/public/nn) under the next free human name and flagged the current champion, so it's
  // pickable in the app under a real name from the moment it's promoted (past champions stay
  // too). Only the most recent --keep-champions retired nets are kept (the current champion is
  // always kept); older ones are pruned (weights file + manifest entry) to bound the deployed
  // bundle (~0.5 MB each), but their name+hash move to name-history.json so ledger labels
  // survive and names are never reused. 0 = off.
  keepChampions: num(args['keep-champions'], 12),
  // Strong-engine ladder play as the generator. With no dedicated generation (--batch=0), the
  // per-cycle rank step restricts --play to the strongest nn engines (current champion + recent
  // champions) at --play-depth, so its harvested games — written straight into the dataset —
  // are deep, strong-play training data: the ranked pool IS the generator. On by default when
  // --batch=0; --play-strong / --no-play-strong force it; --rank-play=SPEC pins the play set by
  // hand (goes stale on promotion — prefer the auto set for an unattended loop). See docs.
  playStrong: args['no-play-strong'] ? false
    : (args['play-strong'] !== undefined ? !!args['play-strong'] : num(args.batch, 0) === 0),
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

// The name-history entries (champions pruned from the manifest), or [] if none yet.
function nameHistory() {
  try { return JSON.parse(readFileSync(nameHistoryFile, 'utf8')).names || []; } catch { return []; }
}

// hash -> human name (champions from the web catalog manifest, plus pruned ones from the
// name history — manifest wins), so loop output shows 'Leo' next to 9e31ca wherever a hash
// appears. Read fresh per call (cheap, ~once a cycle) because a promotion renames the
// current champion mid-run.
function nnNames() {
  const m = new Map();
  for (const n of nameHistory()) if (n.hash && n.name) m.set(n.hash, n.name);
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

// The champion's CURRENT hidden shape ("128,16,16,16"), or '64' if it can't be read. The
// architecture floor: the one shape the champion seeds with no graft, and the answer when
// there's nothing in the registry to pick from.
function championHidden() {
  try {
    const a = JSON.parse(readFileSync(champion, 'utf8')).arch;
    if (Array.isArray(a) && a.length >= 3) return a.slice(1, -1).join(',');
  } catch { /* fall through */ }
  return '64';
}

// Which RECIPE knobs the command line PINNED. Everything else is the loop's to choose
// (autoRecipe below): what you passed wins, the registry fills in the rest — so
// `npm run train:loop -- --rotate=8` with no recipe flags at all is a complete instruction.
// Quantization is deliberately absent: it's free at fixed depth and worth ~1.5x the nps, so
// --float is a diagnostic fork you ask for, not an experiment worth spending rotations on.
// Same for --scale/--lr/--wd (trainer tuning) and --recipe-extra (yours by definition).
const pinned = {
  hidden: typeof args.hidden === 'string',
  // The knob set is ALL-OR-NOTHING on purpose, unlike the architecture. Omitting a flag that
  // defaults to off is the ordinary way to write a command — `--lambda=0.5 --quiet-only
  // --drop-conflicts=700` means "and no weak-game filter", not "surprise me on filter-weak" —
  // so reading a single omission as an invitation would silently fork a new track out of every
  // command line that used to work. Pin any one knob and the rest keep their documented
  // defaults; pin none and the registry designs the whole set.
  knobs: ['lambda', 'quiet-only', 'filter-weak', 'drop-conflicts'].some((f) => args[f] !== undefined),
};

// Resolve the candidate's RECIPE — architecture and knobs — from the experiment registry, for
// every knob the command line didn't pin. The suggester is the same one --rotate consults, so a
// launch and a rotation explore the same design: one-axis knob variations of the reigning recipe
// (λ, --quiet-only, --filter-weak, --drop-conflicts) and untried architectures, whichever family
// the registry has sampled less. That's what the first 9 tracks needed: 8 distinct shapes, but λ
// never once off 0.5 and --quiet-only never once off.
//
// A pinned knob always wins, which is also what keeps the pick safe: a suggestion is guaranteed
// untried as a WHOLE recipe, but pinning parts of it can collapse the merge back onto a track
// you already have — so merges that land on an existing recipe id are skipped and the next
// suggestion is tried instead.
//
// Falls back to the command line as given (champion's shape for the architecture) when the
// registry is empty. That matters on a fresh clone: the suggester bounds its design region
// RELATIVE to a reigning track (a dense-cost ceiling and a first-layer floor), so with no tracks
// its space-filling pick walks to the far corner of the grid — a 12-layer 256-wide net no
// fixed-depth gate would ever reward.
// Returns { picks, why }; `why` is null when the command line pinned everything.
function autoRecipe() {
  // The knobs exactly as the command line left them: the fallback, and the values that override
  // a suggestion wherever they were pinned.
  const current = {
    hidden: cfg.hidden || championHidden(),
    lambda: cfg.lam,
    quietOnly: cfg.quietOnly,
    filterWeak: cfg.filterWeak,
    dropConflicts: cfg.dropConflicts,
  };
  if (pinned.hidden && pinned.knobs) return { picks: current, why: null };

  // A suggestion, with everything you pinned restored to what you asked for.
  const merge = (r) => ({
    hidden: pinned.hidden ? current.hidden : r.hidden,
    lambda: pinned.knobs ? current.lambda : Number(r.lambda),
    quietOnly: pinned.knobs ? current.quietOnly : !!r.quietOnly,
    filterWeak: pinned.knobs ? current.filterWeak : (r.filterWeak ?? 0),
    dropConflicts: pinned.knobs ? current.dropConflicts : (r.dropConflicts ?? 0),
  });
  // Canonicalized the same way the call site builds the run's recipe, so the ids compare.
  const asRecipe = (p) => buildRecipe({
    hidden: p.hidden, lambda: p.lambda, quietOnly: p.quietOnly, quant: cfg.quant,
    scale: cfg.scale, lr: cfg.lr, wd: cfg.wd,
    filterWeak: p.filterWeak > 0 ? p.filterWeak : undefined,
    dropConflicts: p.dropConflicts > 0 ? p.dropConflicts : undefined,
    extra: cfg.recipeExtra,
  });

  const tracks = readAllTracks(loopDir);
  if (!tracks.length) {
    return { picks: current, why: "registry is empty — the champion's shape, and your flags as given" };
  }
  let sugg = [];
  try { sugg = suggestRecipes(loopDir, {}); }
  catch (e) {
    log(`(recipe suggester failed — ${e.message})`);
    return { picks: current, why: "suggester unavailable — the champion's shape, and your flags as given" };
  }
  const tracked = new Set(tracks.map((t) => t.id));
  // A suggestion only helps if it varies an axis that's actually FREE. Every knob trial sits on
  // the reigning shape, so with the knobs pinned it would hand back that shape and call it a
  // choice; every architecture pick carries the reigning knobs, so with the shape pinned it's
  // just as empty. Drop whichever family can't contribute here. ('resume' picks vary both and
  // carry no family, so they always survive as the fallback.)
  const contributes = (s) => (s.family === 'knob' ? !pinned.knobs
    : s.family === 'arch' ? !pinned.hidden
    : true);
  const ranked = sugg.filter(contributes);
  // 'new' before 'resume' — an untried recipe is the point; reviving a past one is the fallback.
  for (const s of [...ranked.filter((x) => x.kind === 'new'), ...ranked.filter((x) => x.kind === 'resume')]) {
    const picks = merge(s.recipe);
    const id = recipeId(asRecipe(picks));
    if (tracked.has(id)) continue; // pinning collapsed this suggestion onto an existing track
    const src = s.family === 'knob' ? 'knob trial'
      : s.kind === 'resume' ? 'reviving a past recipe'
      : 'untried architecture';
    // The suggestion survives a merge unchanged only when nothing you pinned contradicted it —
    // say so, because a 'resume' reason ("has a saved best to warm-start from") is only true of
    // the verbatim recipe, not of an adapted one.
    const verbatim = id === recipeId(s.recipe);
    return { picks, why: `${src}${verbatim ? '' : ' (adapted — your pinned flags kept)'} — ${s.reason}` };
  }
  return { picks: current, why: 'every suggestion collapses onto a track you already have — your flags as given' };
}

// A net's architecture (layer widths, e.g. [768,128,64,32,1]) as a compact string for logs,
// or null if unreadable. The input/output dims are fixed, so the hidden shape is what varies.
function archOf(file) {
  try { const a = JSON.parse(readFileSync(file, 'utf8')).arch; return Array.isArray(a) ? a.join(',') : null; }
  catch { return null; }
}

// Per-node SEARCH TIME of a net — its eval "frame time". For a given arch+quant this is a
// property of the shape, not the weights (a fixed-MAC NNUE forward + branchless clamps), so a
// quick shallow search reads it stably. Returned in nanoseconds/node (null if the bench binary
// or its output is unavailable — this is a pure readout, never fatal). The point of surfacing
// it: live browser play is fixed-TIME, so nodes/move ≈ budget_ms / (ns_per_node/1e6); a shape
// whose ns/node climbs searches shallower in the browser no matter how well it wins a
// fixed-DEPTH gate. Read it like a frame-time meter (absolute), not against another net.
function benchNsPerNode(weightsFile, depth = 6) {
  if (!existsSync(benchBin) || !existsSync(weightsFile)) return null;
  try {
    // cwd = engineDir so the binary starts where its build left it; --weights is absolute so the
    // relative default (../src/nn-weights.json) never applies. depth 6 = a sub-second reading.
    const r = spawnSync(benchBin, [`--depth=${depth}`, `--weights=${weightsFile}`],
      { cwd: engineDir, encoding: 'utf8' });
    const m = /ns\/node=([\d.]+)/.exec(`${r.stdout || ''}${r.stderr || ''}`); // print goes to stderr
    return m ? Number(m[1]) : null;
  } catch { return null; }
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

// `npm run train:stop` from another terminal drops a STOP marker here. The loop spends
// nearly all of a cycle blocked inside spawnSync, where its own SIGINT handler (a JS
// callback) cannot run — so a Ctrl-C is only ever noticed indirectly, via how the child
// happened to die, and a child that finishes normally right as you press it leaves the
// loop rolling on. A file the steps check BETWEEN them is the signal that always lands.
// It stops at the next step boundary, so a running gate still has to play itself out;
// `train:stop --now` kills the tree instead.
function stopRequested() {
  if (stopping) return true;
  if (!existsSync(stopFlag)) return false;
  stopping = true;
  log('  Stop requested (npm run train:stop) — ending after this step.');
  return true;
}

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

// The champion's rating CONFIDENCE from the ledger: the margin (±Elo CI) and game count at the
// depth that defines its best Elo (hence its absElo — the number the loop steers by). Returns null
// unless it is meaningfully rated there (enough games), so adaptive sizing falls back to the
// baseline instead of reacting to a noisy margin — a freshly promoted champion with thin games is
// handled by the calibration pass, not by inflating the routine rank budget.
function championLedgerConfidence() {
  if (!existsSync(ledgerFile)) return null;
  let ledger; try { ledger = JSON.parse(readFileSync(ledgerFile, 'utf8')); } catch { return null; }
  const champHash = weightsHash(champion);
  let best = null;
  for (const e of ledger.ranking || []) {
    if (e.eng !== 'nn' || e.version !== champHash || e.elo == null) continue;
    if (!best || e.elo > best.elo) best = { elo: e.elo, margin: e.margin ?? null, games: e.games ?? 0, depth: e.depth };
  }
  if (!best || best.margin == null || best.games < 30) return null;
  return best; // { elo, margin, games, depth }
}

// The ledger's own CONVERGENCE verdict — the signal that says whether the ladder is keeping up,
// and the one the loop ignored until 2026-08-06. depth-ladder already computes it every fit
// (docs/tools.md): how many rank-adjacent pairs are below the direct-link floor, how many have
// never met at all, how many of those THIS run's --play restriction can actually schedule, and
// the total mis-order cost. Returns null when the ledger has no convergence block (an older file).
//
// `schedulable` is the load-bearing field. --play-strong pins --play to the ~8 strongest nn
// engines at one depth, so the rank step can only ever play games inside that set — while the
// link deficit lives across the whole 184-node pool (other depths, hc nodes, retired champions).
// When `unlinked > 0` and `schedulable === 0`, the ladder is telling us plainly that no amount of
// --rank-minutes can close the gap, because every deficient pair is outside the play set. That is
// a REACHABILITY problem wearing a budget problem's clothes, and it's why a hand-run rank:pool
// (which carries no --play restriction) fixes what a bigger --rank-minutes cannot.
function ledgerConvergence() {
  if (!existsSync(ledgerFile)) return null;
  let ledger; try { ledger = JSON.parse(readFileSync(ledgerFile, 'utf8')); } catch { return null; }
  const c = ledger.convergence;
  if (!c) return null;
  return {
    converged: !!c.converged,
    linked: !!c.linked,
    underLinked: c.adjacentUnderLinked ?? 0,
    unlinked: c.adjacentUnlinked ?? 0,
    schedulable: c.adjacentUnderLinkedSchedulable ?? 0,
    misorderCost: c.misorderCost ?? null,
    pairs: c.pairs ?? 0,
    verdict: c.verdict || '',
    generated: ledger.generated || null,
  };
}

// Baseline-anchored, BOUNDED adaptive sizing of this cycle's maintenance budget. Each knob starts
// at its configured value and moves only within [floor, ceil] on a robust signal, so a neutral or
// absent signal reproduces today's fixed behaviour and a strong signal shifts — never starves —
// the step. The two downsides of a mis-read are both just "a slightly slower cycle" (more rank
// games are still harvested as training data; more refresh means fresher labels), never worse data.
//   rank-minutes ← ledger confidence: play MORE when the champion's best-depth margin (the CI on
//                  its absElo) is wide, LESS when it's already tight. (Unrated → baseline.)
//   refresh-frac ← post-promotion staleness: relabel MORE in the cycles right after a promotion
//                  (the whole dataset's `v` came from the OLD champion), tapering as the
//                  weakest-first refresh chips the staleness down with the champion unchanged.
//                  Inert unless --refresh-cycle>0, which is off by default — scaling 0 is 0.
// `rankDue` is false on a cycle the --rank-cycle cadence skips: the rank half of the budget is
// moot then, so the note says so rather than advertising minutes that never get spent.
function adaptiveMaintenance(cyclesSincePromo, rankDue = true) {
  const base = { rankMinutes: cfg.rankMinutes, refreshFrac: cfg.refreshCycle, notes: null };
  if (!cfg.adaptive) return base;
  const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));

  // rank-minutes takes the STRONGER of two signals. The champion's own margin was the original
  // one, and it turned out to be nearly constant in practice (it printed an identical
  // "±42 → 1.06×" every cycle for a week, because between promotions the champion's rating
  // simply isn't what's moving). The ladder's convergence verdict is the signal that actually
  // varies, and it's the one that matches the observed symptom of the pool falling behind.
  let rankMinutes = cfg.rankMinutes, rankWhy = 'baseline (champion not yet confidently rated)';
  const conf = championLedgerConfidence();
  if (conf) {
    // Nominal margin ≈ 40 Elo ⇒ 1× (baseline). Wider ⇒ up to 3×, tighter ⇒ down to 0.5×.
    const factor = clamp(conf.margin / 40, 0.5, 3);
    rankMinutes = Math.round(clamp(cfg.rankMinutes * factor, 2, Math.max(20, cfg.rankMinutes * 3)));
    rankWhy = `champion d${conf.depth} margin ±${Math.round(conf.margin)} → ${factor.toFixed(2)}×`;
  }
  const conv = ledgerConvergence();
  if (conv) {
    // Only SCHEDULABLE link debt justifies more minutes here: an unschedulable deficit is a
    // reachability problem that the link pass (runLinkPass) handles instead, and inflating this
    // budget for it would buy more games inside the strong-play set that cannot touch it.
    // Converged ⇒ back off hard; the pool is done and the minutes are better spent elsewhere.
    let factor = null, why = null;
    if (conv.converged) { factor = 0.5; why = 'ladder converged'; }
    else if (conv.schedulable > 0) {
      // Scale with how much of the adjacency graph this run can still fix, capped at 3x.
      factor = clamp(1 + conv.schedulable / 4, 1, 3);
      why = `${conv.schedulable} schedulable under-linked pair(s)`;
    } else if (conv.misorderCost != null && conv.pairs > 0) {
      // No link debt to close: fall back to ordering sharpness, per adjacent pair.
      const perPair = conv.misorderCost / conv.pairs;
      factor = clamp(perPair / 1.5, 0.5, 2);
      why = `${perPair.toFixed(1)} Elo mis-order risk per adjacent pair`;
    }
    if (factor != null) {
      const m = Math.round(clamp(cfg.rankMinutes * factor, 2, Math.max(20, cfg.rankMinutes * 3)));
      if (m > rankMinutes || conv.converged) { rankMinutes = m; rankWhy = `${why} → ${factor.toFixed(2)}×`; }
    }
  }

  let refreshFrac = cfg.refreshCycle, refreshWhy = 'baseline';
  if (cfg.refreshCycle > 0) {
    // 2.2× right after a promotion (cyclesSincePromo 0), tapering ~0.3×/cycle to a 0.7× floor.
    const factor = clamp(2.2 - 0.3 * cyclesSincePromo, 0.7, 2.5);
    refreshFrac = clamp(cfg.refreshCycle * factor, cfg.refreshCycle * 0.5, Math.min(0.05, cfg.refreshCycle * 3));
    refreshWhy = `${cyclesSincePromo} cycle(s) since promo → ${factor.toFixed(2)}×`;
  }
  return {
    rankMinutes, refreshFrac,
    notes: (rankDue ? `rank ${rankMinutes}m (${rankWhy})` : 'rank skipped this cycle')
      + (cfg.refreshCycle > 0 ? `; refresh ${(refreshFrac * 100).toFixed(1)}% (${refreshWhy})` : ''),
  };
}

// Fold the gate's harvested games (in the temp file) into the dataset. The match runner
// stamps every position with the MOVER's vs tag (the engine that searched it). The
// candidate's own plies thus carry its content hash — an engine we never archive or rank,
// so refresh-v/merge would read it as −∞ "unrecoverable" and relabel it on sight (the
// champion's plies already carry a ranked tag and pass through). When the candidate WASN'T
// promoted (the common lineage / sub-threshold case), we rewrite its lines' vs to a
// self-describing ephemeral tag "nn<d>@elo<E>", where E is the candidate's absolute Elo on the
// hc-anchored ledger scale: the champion's ledger Elo at that depth plus the LOWER bound of the gate's measured
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
  foldHarvest(gateHarvest, rawFile, champElo, gateEloLo, 'gate edge');
}

// The body shared by the gate harvest and the screen harvest (--screen-save). `champElo` is
// null when nothing needs relabeling (a promoted candidate is archived, hence rankable, so its
// lines pass through as-is); otherwise every line tagged with the candidate's content hash is
// rewritten to the ephemeral "nn<d>@elo<E>" form. `edgeLo` MUST be measured at the same search
// depth as the harvested games — the tag's E is an absolute Elo at that depth, so pairing a
// depth-1 harvest with a depth-6 edge would misrate every one of those games.
function foldHarvest(src, dest, champElo, edgeLo, what) {
  if (!existsSync(src)) return;
  const candHash = weightsHash(candidate);
  const out = [];
  let folded = 0, relabeled = 0;
  for (const line of readFileSync(src, 'utf8').split('\n')) {
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
      setVsAt(rec, i, `nn${depth}@${ephemeralVersion(base + edgeLo)}`);
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
      rec.players[side] = `nn${pm[1]}@${ephemeralVersion(base + edgeLo)}`;
    }
    out.push(serializeGameRecord(rec));
  }
  if (out.length) appendFileSync(dest, out.join('\n') + '\n');
  rmSync(src, { force: true });
  if (relabeled) {
    log(`  Folded ${folded} harvested game(s); relabeled ${relabeled} non-promoted-candidate `
      + `label(s) as ephemeral (${what} lower-bound ${edgeLo >= 0 ? '+' : ''}${edgeLo.toFixed(0)} Elo vs champion).`);
  }
  return { folded, relabeled };
}

// Run a step; return true on success. A SIGINT to a child shows as a null/ signalled
// status — treat that as "stop", not a hard failure.
function run(label, cmd, argv, cwd = webDir) {
  if (stopRequested()) return false;
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

// --- Low-depth screen (shadow mode) -------------------------------------------------------
// The 95% CI around an Elo edge, from the standard error of the MEAN score mapped through
// eloFromScore (above) — deliberately the same convention as the match runner's own eloWithCI
// (engine/src/main_match.zig), so a screen's numbers are directly comparable to a gate's.
// Reconstructed from the W/D/L counts, which is exact: every game scores 1, 0.5 or 0.
function eloWithCI(wins, draws, losses) {
  const n = wins + draws + losses;
  if (!n) return null;
  const p = (wins + draws * 0.5) / n;
  const varSum = wins * (1 - p) ** 2 + draws * (0.5 - p) ** 2 + losses * p ** 2;
  const se = Math.sqrt(varSum / n / n); // standard error of the mean score
  return { score: p, elo: eloFromScore(p), lo: eloFromScore(p - 1.96 * se), hi: eloFromScore(p + 1.96 * se) };
}

// Play the candidate vs the champion at a CHEAP depth and predict the gate-depth edge.
// Shadow mode: the return value is recorded and logged, and nothing else reads it — the gate
// below still makes the promotion decision on its own evidence. Three deliberate choices:
//   - fixed --games, NO --sprt: the screen ESTIMATES an edge (we want the CI, and a decision
//     rule can be simulated offline from it afterwards); an SPRT would stop early and throw
//     away exactly the precision that makes the screen worth anything.
//   - the games NEVER reach selfplay.jsonl. This is the trap in the whole idea: the gate
//     harvests ~97k positions per 2000 games, so 20k screen games would add ~970k positions
//     per cycle — 10x the dataset's normal growth, at the worst label quality in the pool (a
//     depth-1 `v`), landing straight in the cohort --filter-weak and refresh-v exist to drain.
//     With --screen-save they go to a SEPARATE dataset (screenArchive) instead, which the
//     ledger can rate from and a future experiment can mine, with no path to the trainer.
//   - failure is non-fatal. It's instrumentation; a broken screen must never cost a cycle.
// Returns the record for the track history, or null.
function runScreen() {
  if (!cfg.screen) return null;
  if (existsSync(screenFile)) rmSync(screenFile);
  if (existsSync(screenHarvest)) rmSync(screenHarvest); // no stale harvest from a prior cycle
  const t0 = Date.now();
  if (!run(`Screen (shadow): candidate vs champion @ depth ${cfg.screenDepth}`, matchBin,
    ['--eval-a=nn', `--weights-a=${candidate}`, '--eval-b=nn', `--weights-b=${champion}`,
      `--depth=${cfg.screenDepth}`, `--games=${cfg.screenGames}`,
      `--result-file=${screenFile}`, `--seed=${Date.now()}`,
      ...(cfg.screenSave ? [`--save-games=${screenHarvest}`] : []), ...jobArg])) return null;
  let r;
  try { r = JSON.parse(readFileSync(screenFile, 'utf8')); }
  catch { log('  Screen produced no readable result; continuing to the gate.'); return null; }
  const ci = eloWithCI(r.wins, r.draws, r.losses);
  if (!ci) { log('  Screen played no games; continuing to the gate.'); return null; }
  // Rescale to a gate-depth-equivalent edge. The ratio compresses low-depth Elo, so dividing
  // by it also widens the CI proportionally — the screen's precision at the gate's scale is
  // what matters, not its precision at depth 1.
  const k = 1 / cfg.screenRatio;
  const pred = { elo: ci.elo * k, lo: ci.lo * k, hi: ci.hi * k };
  // The rule a real cascade would use: escalate unless the promotion bound is outside the
  // prediction's CI. Rejecting on the UPPER bound (not the point estimate) is what keeps a
  // screen conservative — it only kills a candidate it can rule out.
  const wouldReject = pred.hi < cfg.elo1;
  const secs = (Date.now() - t0) / 1000;
  log(`  Screen: ${(ci.score * 100).toFixed(1)}% / ${ci.elo >= 0 ? '+' : ''}${ci.elo.toFixed(0)} Elo @ depth `
    + `${cfg.screenDepth} over ${r.games} games in ${fmtDur(secs)} → predicts `
    + `${pred.elo >= 0 ? '+' : ''}${pred.elo.toFixed(0)} Elo [${pred.lo.toFixed(0)}, ${pred.hi.toFixed(0)}] `
    + `at depth ${cfg.gateDepth} (ratio ${cfg.screenRatio}). `
    + `A screen WOULD ${wouldReject ? `REJECT (upper bound < +${cfg.elo1}, skipping the gate)` : 'ESCALATE'}. `
    + 'Shadow mode — running the real gate anyway.');
  return {
    depth: cfg.screenDepth, games: r.games, seconds: Math.round(secs), ratio: cfg.screenRatio,
    score: ci.score, elo: ci.elo, eloLo: ci.lo, eloHi: ci.hi,
    predElo: pred.elo, predLo: pred.lo, predHi: pred.hi, wouldReject,
  };
}

// The loop rates the SAME full pool as a standalone `npm run rank:pool` — every engine across
// depth-ladder's default depth spectrum (1-8), not a narrowed slice — so its ledger is the one
// unified pool, not a loop-specific variant. hc<rankDepth> stays the pinned Elo-1500 node (via
// --anchor-depth below), so all ratings land on the same stable scale. (The two depths positions
// are LABELED at, nn6@/nn8@, are just a subset of that spectrum — foldGateHarvest still finds
// them in the ledger.)

// Game files that are RATING evidence but not training data, passed to every fit as
// --corpus-extra. Kept out of the dataset on purpose (see each one's flag doc), but a game's
// players+result rates its pair wherever the file lives.
function corpusExtraFiles() {
  return [
    ...(cfg.screen && cfg.screenSave ? [screenArchive] : []),
    linkArchive,
  ].filter((f) => existsSync(f));
}

// Is a pool refit due this cycle? A PROMOTION always forces one (a fresh champion sits at the
// placeholder floor at every depth it hasn't played, and the depth-calibration pass only runs
// here). Otherwise `--rank-cycle=N` is the old fixed every-N rule, and the default 'auto' asks
// the ledger instead: while the ladder is still short of convergence the ratings are genuinely
// moving and every cycle earns its refit; once converged the pass mostly re-confirms a ledger
// that isn't changing, so it drops to every third cycle. The cost is a ledger up to N-1 cycles
// stale, which only means the weakest-first refresh targets a slightly out-of-date cohort.
// Keyed on the TRACK-cumulative cycle so a warm relaunch continues the cadence.
const AUTO_RANK_CYCLE_CONVERGED = 3;
function rankCadence(c, promoted) {
  if (promoted) return { due: true, why: 'promotion forces a refit' };
  if (cfg.rankCycle === 'auto') {
    const conv = ledgerConvergence();
    if (!conv || !conv.converged) return { due: true, why: 'ladder not converged — refitting every cycle' };
    const due = c % AUTO_RANK_CYCLE_CONVERGED === 0;
    return due
      ? { due: true, why: `ladder converged — every ${AUTO_RANK_CYCLE_CONVERGED} cycles` }
      : { due: false, why: `ladder converged, so refitting every ${AUTO_RANK_CYCLE_CONVERGED} cycles (next at ${c + AUTO_RANK_CYCLE_CONVERGED - (c % AUTO_RANK_CYCLE_CONVERGED)}), or sooner on a promotion` };
  }
  if (cfg.rankCycle <= 1) return { due: true, why: '--rank-cycle=1' };
  return c % cfg.rankCycle === 0
    ? { due: true, why: `--rank-cycle=${cfg.rankCycle}` }
    : { due: false, why: `--rank-cycle=${cfg.rankCycle}, next at cycle ${c + cfg.rankCycle - (c % cfg.rankCycle)}, or sooner on a promotion` };
}

// THE LINK PASS — the fix for "the ladder isn't keeping up".
//
// The routine rank step runs with --play pinned to the ~8 strongest nn engines at one depth
// (strongPlaySpec), because at --batch=0 that step is also the data generator and its games
// should be deep strong-play data. But the pool has 184 nodes, and the ladder's convergence
// deficit lives outside that set: measured 2026-08-06, 62 rank-adjacent pairs had never met,
// and `adjacentUnderLinkedSchedulable` was 0 — not one of them could be scheduled under the
// strong-play restriction. So the budget was being spent where it could not help, the ledger
// never converged, and closing those links meant stopping the loop and running rank:pool by
// hand (which carries no --play restriction, which is exactly why it worked).
//
// One budget was serving two objectives that want opposite things: data quality wants NARROW
// and deep, ladder convergence wants BROAD. So they get separate passes. This one is
// unrestricted, short, and fires only when the deficit is genuinely unreachable otherwise.
//
// Its games go to their own file, not the dataset: they're played between whatever pairs the
// adjacency graph is missing, which includes the weak nodes --play-strong deliberately keeps
// out of training. They come back as --corpus-extra, so the evidence persists and accumulates
// across cycles (a --no-save-games pass would inform only the run that played it).
function runLinkPass(conv) {
  if (!cfg.rank || !cfg.linkPass || !conv) return;
  // Only when the deficit is real AND the routine pass cannot reach it. A schedulable deficit
  // is already handled by adaptiveMaintenance giving the routine pass more minutes.
  if (conv.converged || conv.unlinked <= 0 || conv.schedulable > 0) return;
  const minutes = Math.max(2, Math.min(cfg.linkMinutes, 30));
  log(`  Ladder link deficit: ${conv.unlinked} adjacent pair(s) have never met and NONE are `
    + `schedulable under the strong-play set — more --rank-minutes cannot close them. `
    + `Running a ${minutes}m unrestricted link pass (games to ${linkArchive}, rating evidence only).`);
  run('Rank pool: unrestricted link pass', process.execPath,
    [rankScript, '--corpus', `--minutes=${minutes}`, `--anchor-depth=${cfg.rankDepth}`,
      // No --play: every node is schedulable, which is the entire point.
      // --onboard=0 keeps the budget on EDGES (the link deficit) rather than on under-played
      // nodes; the routine pass and the promotion calibration already cover node onboarding.
      '--onboard=0', `--games=${cfg.rankGames}`,
      `--data=${rawFile}`, `--ledger=${ledgerFile}`,
      ...(corpusExtraFiles().length ? [`--corpus-extra=${corpusExtraFiles().join(',')}`] : []),
      `--save-games=${linkArchive}`,
      '--no-scan', `--seed=${Date.now()}`, ...jobArg]);
}

// Refit the Bradley-Terry strength pool (rank:pool / depth-ladder.mjs). Runs EVERY cycle:
//   --corpus folds the whole dataset's harvested games (each record's players + result) into
//   the fit, so the current champion is rated automatically from its gate matches — no
//   dedicated gauntlet, and a just-promoted champion already has games against the engine it
//   dethroned. On top of that the pool plays a short --rank-minutes budget of the matchups
//   whose ORDERING is currently most ambiguous (naturally including the new champion), and
//   persists every game into the store so ratings tighten cumulatively across cycles.
// The fitted ledger (engine-elo.ladder.json) is what the weakest-first refreshes read.
// Maintenance, like the refreshes — a failure logs but doesn't stop the loop.
function runRankPool(label, opts = {}) {
  if (!cfg.rank) return;
  // When the ranked pool is the generator (cfg.playStrong), restrict --play to the strongest nn
  // engines at cfg.playDepth so this step's harvested games are deep, strong training data (the
  // whole pool is still RATED from the corpus + store — --play only bounds which nodes play NEW
  // games). null = schedule unrestricted (too few strong engines yet, or --no-play-strong).
  const play = cfg.playStrong ? strongPlaySpec() : null;
  // On the cycle that just promoted, ALSO make the new champion schedulable at every depth: a
  // bare-hash spec (no nn<d>@ prefix) means "every --depths of this engine". Combined with an
  // aggressive onboard floor the ladder fills the champion's 0-game depth nodes first, anchoring
  // each depth band to the scale instead of leaving it at the placeholder floor (see cfg.calibrateMinutes).
  const calib = opts.calibrateChamp && cfg.calibrateMinutes > 0 ? opts.calibrateChamp : null;
  const playSet = [play, calib].filter(Boolean).join(',') || null;
  // Show the play set with human names — the raw spec is hashes only (it must stay a valid
  // --play argument), so name the engines here where a reader actually sees the plan.
  if (play) {
    const names = nnNames();
    log(`  Strong play set: ${play.split(',').map((s) => {
      const m = /@([0-9a-f]+)$/.exec(s); const n = m && names.get(m[1]);
      return n ? `${s} (${n})` : s;
    }).join(', ')}`);
  }
  if (calib) log(`  Calibrating champion ${calib} across all depths (+${cfg.calibrateMinutes}m, onboard floor fills its 0-game depth nodes).`);
  // Base play budget: the adaptive per-cycle value when given (opts.minutes), else the fixed knob.
  const baseMinutes = opts.minutes ?? cfg.rankMinutes;
  // With harvesting on (default), rank:pool appends its games straight into the dataset — the same
  // file the next incremental featurize trains on and the next fit reads its ratings back from, so
  // there's no fold step. Their players are all rankable engines (champion / hc / archived /
  // material), so no provenance rewrite is needed either (unlike the gate's candidate).
  run(label, process.execPath,
    [rankScript, '--corpus', `--minutes=${baseMinutes + (calib ? cfg.calibrateMinutes : 0)}`,
      `--anchor-depth=${cfg.rankDepth}`,
      ...(playSet ? [`--play=${playSet}`] : []),
      ...(calib ? ['--onboard=1'] : []), // fill the champion's under-played depths before the ordering objective
      `--games=${cfg.rankGames}`, ...(cfg.rankLink === null ? [] : [`--link=${cfg.rankLink}`]),
      `--data=${rawFile}`, `--ledger=${ledgerFile}`,
      // Rate from the rating-only game files as well. Neither is in --data, because neither is
      // meant for the trainer: the screen's games are depth-1 labels, and the link pass plays
      // whatever pairs the adjacency graph is missing (weak nodes included, which is exactly
      // what --play-strong exists to keep out of the training set). Both are still perfectly
      // good EVIDENCE, and --save-games still points at --data, so nothing new is written here.
      ...(corpusExtraFiles().length ? [`--corpus-extra=${corpusExtraFiles().join(',')}`] : []),
      ...(cfg.harvest ? [] : ['--no-save-games']),
      '--no-scan', `--seed=${Date.now()}`, ...jobArg]);
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
// initial hand-published lineage). Names are permanent: a pruned champion's name moves to
// name-history.json and is never handed out again (a hash is a hash, a name is a net). When
// all 26 are spent the fallback is champ-<hash>.
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
    // Names in the pruned-champion history are spent too — never reassign them.
    const used = new Set([...man.nets.map((n) => n.name), ...nameHistory().map((h) => h.name)]);
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
  // A pruned champion's identity (name+hash) is appended to name-history.json so the ledger
  // keeps labeling its hash and the name stays spent; only the deployed weights go away.
  if (cfg.keepChampions > 0) {
    const byAge = champs().filter((n) => !n.current).sort((a, b) => (a.gen || 0) - (b.gen || 0));
    const pruned = byAge.slice(0, Math.max(0, byAge.length - cfg.keepChampions));
    for (const e of pruned) {
      const p = join(publicNN, e.file);
      if (existsSync(p)) rmSync(p);
      man.nets = man.nets.filter((n) => n !== e);
    }
    if (pruned.length) {
      const names = nameHistory();
      const today = new Date().toISOString().slice(0, 10);
      for (const e of pruned) {
        if (names.some((h) => h.hash === e.hash)) continue;
        names.push({ name: e.name, hash: e.hash, gen: e.gen, arch: e.arch,
          note: `${(e.note || `Retired train:loop champion (${e.hash}).`).replace(/\.$/, '')}; pruned from the catalog ${today}.` });
      }
      writeFileSync(nameHistoryFile, JSON.stringify({
        note: 'Champions pruned from manifest.json live on here: the ledger labels their hash by'
          + ' name forever, and these names are never reused for future champions. Append-only;'
          + ' maintained by train:loop (publishChampion).',
        names,
      }, null, 2) + '\n');
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
// Publish this loop's PID for the pause/resume/stop control, and drop any stale PAUSED or
// STOP marker from a previous run (a fresh loop starts running, and a STOP left behind by
// the run you just ended would otherwise stop this one on its first step). Both markers and
// the pidfile are removed on exit.
writeFileSync(pidFile, `${process.pid}\n`);
rmSync(pauseFlag, { force: true });
rmSync(stopFlag, { force: true });
process.on('exit', () => {
  for (const f of [pidFile, stopFlag]) { try { rmSync(f, { force: true }); } catch { /* best effort */ } }
});
// --fresh clears the dataset, which is also the strength pool's body of evidence: the ledger's
// ratings rebuild from whatever the fresh run plays (plus loop/legacy-pairs.json).
if (cfg.fresh && existsSync(rawFile)) { rmSync(rawFile); log('Cleared dataset (--fresh).'); }

// The recipe is resolved to concrete values HERE — whether they came from flags or the registry
// picked them (autoRecipe) — so the recipe id is fixed for the run rather than a moving target
// ("whatever the champion happens to be this cycle"). The chosen knobs are written back onto cfg
// because that's where the featurize/train call sites read them from.
const autoPick = autoRecipe();
let hidden = autoPick.picks.hidden;
cfg.lam = autoPick.picks.lambda;
cfg.quietOnly = autoPick.picks.quietOnly;
cfg.filterWeak = autoPick.picks.filterWeak;
cfg.dropConflicts = autoPick.picks.dropConflicts;
// Resolve this run's TRAINING RECIPE and its persistent track. The recipe (architecture + TD
// mix + quiet filter + quant + trainer knobs + any --recipe-extra) is keyed to a directory
// under loop/experiments/, so its lineage/best/history survive being switched away from and
// resume automatically when the same recipe runs again — even after other recipes in between.
let recipe = buildRecipe({
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

// Switch the loop onto a different recipe mid-run (--rotate). Everything downstream of the recipe
// has to move together: the architecture passed to train.py, the dataset filters (which decide
// WHICH featurized file this recipe reads — each filter config keeps its own incrementally
// maintained one), and the track paths that hold the lineage/best used as the warm-start source.
// The filter knobs live on `cfg` because the featurize/train invocations read them there, so the
// adopted recipe is written back onto cfg rather than threaded through every call site.
// Returns false (and changes nothing) if the recipe is already the active one.
function adoptRecipe(next, why) {
  if (recipeId(next) === recipeId(recipe)) return false;
  recipe = next;
  hidden = next.hidden;
  cfg.lam = next.lambda;
  cfg.quietOnly = !!next.quietOnly;
  cfg.quant = !!next.quant;
  cfg.scale = next.scale;
  cfg.lr = next.lr;
  cfg.wd = next.wd;
  cfg.filterWeak = next.filterWeak ?? 0;
  cfg.dropConflicts = next.dropConflicts ?? 0;
  cfg.recipeExtra = next.extra;
  featFile = featurizeFile();      // filters changed => a different features file
  track = ensureTrack(loopDir, recipe, stamp());
  lineage = track.paths.lineage;
  trackBest = track.paths.best;
  runNo = beginRun(track.dir, stamp());
  // Continue THIS track's numbering, not the old one's — the banner and history must agree.
  trackCycleNo = readState(track.dir)?.cycles || 0;
  cyclesSinceAdopt = 0;
  rotations++;
  log('');
  log(`  ↻ Rotating recipe (${why}) → ${recipeLabel(recipe)}`);
  log(`    track ${track.id}${trackCycleNo ? ` (resuming at cycle ${trackCycleNo + 1})` : ' (new)'}`
    + ` · features ${featFile.split(/[\\/]/).pop()}`);
  log(`    equivalent to: npm run train:loop -- ${recipeToFlags(recipe)}`);
  return true;
}

// Should the loop rotate off this track now? `--rotate=N` keeps the old fixed rule (N cycles on
// this track without a promotion). The default `--rotate=auto` adds the one thing a raw cycle
// count cannot see: WHERE THE TRACK IS HEADED.
//
// A new or grafted architecture can lose the gate for dozens of cycles while its absolute Elo
// climbs steadily toward overtaking the champion — that is the pattern that produced Mona, and a
// fixed counter rotates away from it right as it's working. Conversely a track whose absElo has
// gone flat has told you everything it's going to: the registry measured a new track's cycle 1 at
// ~29% promotion versus ~1.8% for a continuation cycle, so a flat track is worth abandoning fast.
//
// So: never rotate before AUTO_MIN cycles (every shape gets a fair look, and a 2-point trend is
// noise), always rotate by AUTO_MAX (nothing grinds forever), and in between rotate only when the
// trend is not climbing. absElo is re-anchored onto today's ledger before comparing, because a
// stored absElo drifts as the pool re-fits — comparing raw snapshots would read drift as trend.
// 3 matches the aggressive hand-set cadence this loop has actually been run at; 8 matches the
// registry's own advice ("a track that hasn't promoted by ~8-10 cycles is far less likely to
// than a fresh shape is on its first"), and is what stops a climbing-but-never-promoting track
// from grinding forever the way the 22-cycle 39ef6efc track did.
const AUTO_ROTATE_MIN = 3, AUTO_ROTATE_MAX = 8;
function rotationDue() {
  if (cfg.rotate === 'off') return { due: false };
  if (cfg.rotate !== 'auto') {
    return cyclesSinceAdopt >= cfg.rotate
      ? { due: true, why: `${cyclesSinceAdopt} cycle(s) without a promotion` }
      : { due: false };
  }
  if (!track || cyclesSinceAdopt < AUTO_ROTATE_MIN) return { due: false };
  if (cyclesSinceAdopt >= AUTO_ROTATE_MAX) {
    return { due: true, why: `${cyclesSinceAdopt} cycle(s) on this track without a promotion (auto cap)` };
  }
  const trend = trackAbsEloTrend();
  if (trend == null) {
    return { due: true, why: `${cyclesSinceAdopt} cycle(s) without a promotion, no absElo trend to justify staying` };
  }
  if (trend > 2) return { due: false }; // climbing — the Mona pattern; let it run
  return {
    due: true,
    why: `${cyclesSinceAdopt} cycle(s) without a promotion and absElo trend ${trend >= 0 ? '+' : ''}${trend.toFixed(1)} Elo/cycle (flat or falling)`,
  };
}

// Least-squares slope of this track's re-anchored absElo against cycle number, in Elo/cycle over
// the cycles since this track was adopted. null when there aren't enough rated cycles to say.
function trackAbsEloTrend() {
  try {
    const ledgerElo = ledgerBestByVersion(loopDir);
    const pts = [];
    for (const h of readHistory(track.dir)) {
      const e = reAnchoredAbsElo(h, ledgerElo);
      if (Number.isFinite(e)) pts.push(e);
    }
    const tail = pts.slice(-Math.max(AUTO_ROTATE_MIN, cyclesSinceAdopt));
    if (tail.length < 3) return null;
    const n = tail.length, mx = (n - 1) / 2, my = tail.reduce((a, b) => a + b, 0) / n;
    let sxy = 0, sxx = 0;
    for (let i = 0; i < n; i++) { sxy += (i - mx) * (tail[i] - my); sxx += (i - mx) ** 2; }
    return sxx > 0 ? sxy / sxx : null;
  } catch { return null; }
}

// Pick the next recipe to rotate onto. Prefers an untried one ('new' — the registry's
// space-filling / UCB design over architecture space), falling back to reviving a past track
// ('resume') when the candidate pool is exhausted. suggestRecipes already excludes every recipe
// that has a track, so it never hands back the shape we're rotating away from. Returns null when
// there's nothing to switch to, which leaves the loop on its current recipe.
function nextRotationRecipe() {
  try {
    const sugg = suggestRecipes(loopDir, {});
    const pick = sugg.find((s) => s.kind === 'new') || sugg.find((s) => s.kind === 'resume');
    return pick ? pick.recipe : null;
  } catch (e) {
    log(`  (rotation skipped — suggester failed: ${e.message})`);
    return null;
  }
}
// Cycle numbering CONTINUES across warm relaunches of the same recipe: the track's state
// already counts every cycle it has recorded, so a warm (re)start picks up at prior+1 instead
// of announcing "CYCLE 1" again after a Ctrl-C + relaunch (train:progress merges those
// launches into one run the same way). A --cold run chains from a fresh net, so it starts
// over at 1 — its cycles still accrue to the track for the next warm resume to continue from.
const cycleBase = cfg.cold ? 0 : (readState(track.dir)?.cycles || 0);
// Cycle number ON THE CURRENT TRACK. Mutable because --rotate can move the loop to a different
// track mid-run, and the banner/history have to follow the track they're actually writing to.
let trackCycleNo = cycleBase;
// Cycles run on the current track since the loop adopted it, reset by a promotion. This — not
// the global cyclesSincePromo — is what --rotate triggers on: it asks "has THIS shape had its
// fair shot", which is the question the rotation is answering. It starts at 0 on every launch so
// the recipe you explicitly asked for always gets its full N cycles before being rotated away.
let cyclesSinceAdopt = 0;
let rotations = 0;
// No shape-mismatch discard anymore: the track is keyed by the exact recipe (hidden included),
// so its lineage always matches its own shape and lives in its own directory — a different
// recipe can't clobber it. A --cold run simply ignores the lineage (it chains from a fresh
// net), leaving the prior warm progress on disk intact for a later warm resume.
log(`Recipe ${track.slug} [${track.id}] — ${recipeLabel(recipe)}`
  + ` (track run #${runNo}${track.isNew ? ', new track' : ''}).`);
// Part of the recipe wasn't given, so the loop chose it. Say what and why, and print the command
// that pins this exact recipe — the run is otherwise unreproducible from its own log.
if (autoPick.why) {
  log(`  ↳ recipe chosen automatically — ${autoPick.why}.`);
  log(`    to pin it: npm run train:loop -- ${recipeToFlags(recipe)}`);
}
// Featurized file for this recipe (quiet-only gets its own; see featurizeFile).
let featFile = featurizeFile();
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
  + `SPRT(0,${cfg.elo1})${cfg.gateFutility > 0 ? ` futility<${cfg.gateFutility}` : ''} | `
  + (cfg.screen
    ? `screen ${cfg.screenGames}g @ depth ${cfg.screenDepth} ratio ${cfg.screenRatio} (SHADOW — logged, never acted on`
      + `${cfg.screenSave ? `; games kept in ${screenArchive}` : '; games discarded'}) | `
    : '')
  + `candidate hidden=[${hidden}] λ=${cfg.lam} ${cfg.cold ? 'cold first cycle, warm after' : 'warm'} start`
  + `${existsSync(lineage) ? ' (resuming lineage)' : ''} | `
  + `refresh/cycle ${cfg.refreshCycle > 0 ? `${(cfg.refreshCycle * 100).toFixed(1)}% @ depth ${cfg.refreshCycleDepth}` : 'off'} | `
  + `refresh on promotion ${cfg.refreshFrac > 0 ? `${(cfg.refreshFrac * 100).toFixed(0)}% @ depth ${cfg.refreshDepth}` : 'off'} | `
  + `rank ${cfg.rank ? `full pool ${cfg.rankCycle === 'auto' ? 'auto-cadence (every cycle until the ladder converges)'
    : cfg.rankCycle > 1 ? `every ${cfg.rankCycle} cycles` : 'every cycle'} `
    + `(hc${cfg.rankDepth} pin, all depths, corpus + ${cfg.rankMinutes}m play${cfg.adaptive ? ', adaptive' : ''}, always on promotion`
    + `${cfg.linkPass ? `; link pass ${cfg.linkMinutes}m when the deficit is unreachable` : ''})` : 'off'} | `
  + `rotate ${cfg.rotate === 'auto' ? `auto (absElo trend, ${AUTO_ROTATE_MIN}-${AUTO_ROTATE_MAX} cycles)`
    : cfg.rotate === 'off' ? 'off' : `after ${cfg.rotate} cycle(s) without a promotion`} | `
  + `cycles ${cfg.cycles === Infinity ? '∞' : cfg.cycles}`);
if (cycleBase > 0) log(`Continuing cycle numbering at ${cycleBase + 1} — this recipe's track already has ${cycleBase} recorded cycle(s).`);
log('Pause/resume from another terminal: `npm run train:pause` / `npm run train:resume` (frees all CPU, no work lost).');

const jobArg = cfg.jobs !== undefined ? [`--jobs=${cfg.jobs}`] : [];

// No startup ranking: the pool is refit near the END of each cycle (runRankPool), after that
// cycle's gate games have been harvested into the dataset, and the per-cycle v-refresh runs
// right after it — so every cycle (including the first) refreshes against a ledger that was
// just refit, going weakest-first as soon as the pool rates anything.

const loopT0 = Date.now();
let promotions = 0;
// Cycles since the champion last changed (reset to 0 on promotion). Drives the adaptive refresh
// fraction — the whole dataset's `v` targets are stalest right after a promotion. Starts at 0 so
// the first cycles treat the loaded champion's inherited labels as worth refreshing.
let cyclesSincePromo = 0;
// `i` counts THIS launch's cycles (what --cycles bounds and the first-cycle behaviours key
// on); `c` is the track-cumulative cycle number shown in banners/logs and recorded in the
// track history — they differ when a warm relaunch continues an earlier run's numbering.
for (let i = 1; i <= cfg.cycles && !stopRequested(); i++) {
  // Rotate BEFORE doing any work this cycle, so a stalled shape never costs another full
  // featurize/train/gate. Rotation is best-effort maintenance: if the suggester has nothing to
  // offer (or throws), the loop simply carries on with the current recipe rather than stopping.
  const rot = rotationDue();
  if (rot.due) {
    const next = nextRotationRecipe();
    if (next && adoptRecipe(next, rot.why)) {
      // adoptRecipe reset the counters; nothing else to do.
    } else {
      log(`  (rotation due — ${rot.why} — but the registry had no untried recipe to switch to; continuing.)`);
      cyclesSinceAdopt = 0; // don't re-ask every cycle once the pool is exhausted
    }
  }
  const c = ++trackCycleNo;
  cyclesSinceAdopt++;
  const cycleT0 = Date.now();
  cyclesSincePromo++;
  const dataset = existsSync(rawFile) ? ` — dataset ${fmtMB(statSync(rawFile).size)}` : '';
  banner(`CYCLE ${c}${cfg.cycles === Infinity ? '' : (cycleBase ? ` (${i}/${cfg.cycles} this run)` : `/${cfg.cycles}`)}${dataset}`);
  // --cold trains from random init on the FIRST cycle only: bootstrap a fresh net once,
  // then keep refining THAT net by warm-starting every later cycle from the previous
  // cycle's candidate (see the init resolution below) instead of relearning from scratch.
  const cold = cfg.cold && i === 1;

  // 1. Generate games with the champion (deeper search than the eval sees).
  //    --skip-gen: on the first cycle only, gate the games an interrupted earlier
  //    run already flushed to the dataset instead of generating a new batch.
  if (i === 1 && cfg.skipGen) {
    log('Skipping generation (--skip-gen): gating the existing dataset.');
  } else if (cfg.batch === 0) {
    // No dedicated self-play generation: the ranked pool produces training data instead (the
    // gate harvest + strong-engine --play games below). Announced once, then silent per cycle.
    if (i === 1) log(`No dedicated generation (--batch=0): fresh data comes from the gate harvest`
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
  //              else the global champion. The champion seeds the candidate even when its shape
  //              DIFFERS: train.py's --init now GRAFTS a foreign-arch net into this shape
  //              (function-preserving widening when every layer grows, a lossy sub-block copy
  //              otherwise) instead of falling back to random. So a brand-new architecture track
  //              starts from the champion's learned function on its first cycle — no more 50-cycle
  //              bootstrap from scratch — then warm-starts from its own track best/lineage after.
  const initFile = cfg.cold
    ? (cold ? null : candidate)
    : (existsSync(lineage) ? lineage
      : existsSync(trackBest) ? trackBest
      : champion);
  const warm = !!initFile && existsSync(initFile);
  // A champion seed whose arch differs from this recipe's is a GRAFT, not a plain warm-start
  // (train.py logs the exact graft mode). Flagged here for the label + track provenance.
  const grafting = warm && !cfg.cold && initFile === champion && !championArchMatches();
  const initLabel = !warm ? ' (cold start)'
    : cfg.cold ? ' (warm-start from previous candidate)'
    : initFile === lineage ? ' (warm-start from lineage)'
    : initFile === trackBest ? ' (warm-start from track best)'
    : grafting ? ' (graft from champion — different arch)'
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

  // Candidate "frame time": its per-node search cost (an arch property, not a strength claim and
  // not part of the gate decision). Live browser play is fixed-TIME, so a shape whose ns/node
  // climbs reaches fewer nodes/move there even when it wins the fixed-DEPTH gate below — surface
  // the number so that trade-off is visible when growing/reshaping the arch. Absolute; no compare.
  const candNs = benchNsPerNode(candidate);
  if (candNs != null) {
    const nps = 1e9 / candNs;
    log(`  Candidate speed: ${candNs.toFixed(0)} ns/node (~${Math.round(nps / 1000)}k nps) — arch [${archOf(candidate) ?? '?'}].`);
  }

  // 3b. Shadow screen (--screen, off by default): a cheap low-depth read on the candidate,
  //     logged and recorded but never acted on. Runs BEFORE the gate so the pairs it records
  //     are (screen prediction, gate truth) on the same candidate, which is the only
  //     population a real cascade would ever face. See runScreen.
  const screen = runScreen();

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
      ...(cfg.gateFutility > 0 ? [`--sprt-futility=${cfg.gateFutility}`] : []),
      `--games=${cfg.gateGames}`, `--result-file=${resultFile}`,
      ...(cfg.harvest ? [`--save-games=${gateHarvest}`, `--seed=${Date.now()}`] : []), ...jobArg])) {
    // Ctrl-C / failure mid-gate: the runner still drained its played games to the harvest
    // temp. Fold them in (relabeling if a partial result is readable, else unchanged) so the
    // already-played games aren't lost, then end the loop.
    if (cfg.harvest) {
      let r = null; try { r = JSON.parse(readFileSync(resultFile, 'utf8')); } catch { /* no usable result */ }
      foldGateHarvest(r ? r.sprt === 'H1' : false, r);
    }
    // The screen's games were played before the gate started, so they survive the interrupt
    // too. No verdict reached, so treat the candidate as unpromoted (the ephemeral tag is the
    // conservative read — it rates the candidate rather than claiming it's a pool node).
    if (cfg.screen && cfg.screenSave && existsSync(screenHarvest)) {
      foldHarvest(screenHarvest, screenArchive, championLedgerElo(),
        screen ? screen.eloLo : 0, `depth-${cfg.screenDepth} screen edge`);
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
  // How close the gate came to promoting. The SPRT verdict alone can't tell a candidate that
  // stalled near 50% from one that died a hair short of the bound, and those call for different
  // next moves (the first says the recipe is flat; the second says it wanted more games). The
  // LLR walk crosses +llrUpper to promote and -llrLower to reject; the bounds move with
  // --alpha/--beta so they come from the result file rather than being assumed here. Omitted
  // for a promotion (it crossed, by definition) and for result files written before the bounds
  // were reported.
  const llrNote = res.llr != null && res.llrUpper
    ? ` LLR ${res.llr.toFixed(2)} of [${res.llrLower.toFixed(2)}, ${res.llrUpper.toFixed(2)}] — `
      + `${Math.max(0, Math.min(100, (res.llr / res.llrUpper) * 100)).toFixed(0)}% of the way to the promotion bound.`
    : '';
  // Snapshot the values the track record needs BEFORE the promote branch overwrites the
  // champion file: the candidate's estimated ABSOLUTE Elo (the champion's current ledger Elo
  // + this gate's edge over it) stays comparable across cycles as the champion strengthens,
  // unlike a raw gate score, so it's what "track best" is ranked by. Null until the ledger
  // rates the champion (from cycle 2 on).
  const gateCI = eloWithCI(res.wins, res.draws, res.losses);
  const gatedVsChampHash = weightsHash(champion);
  const champLedgerNow = championLedgerElo();
  const candAbsElo = champLedgerNow ? champLedgerNow.best + res.elo : null;
  const candHashForTrack = weightsHash(candidate);
  // Fold the gate's harvested games into the dataset, relabeling a non-promoted gate-winning
  // candidate's provenance to a self-describing ephemeral Elo first (foldGateHarvest). Done
  // here, before the promote branch copies the candidate over the champion, so weightsHash
  // still identifies the candidate that actually played.
  if (cfg.harvest) foldGateHarvest(res.sprt === 'H1', res);
  // Same treatment for the screen's games, into their own dataset. A PROMOTED candidate is
  // archived by hash, so its screen games become directly rateable evidence for the ledger —
  // and 20k direct games on one pair is worth far more to the fit than the ~28 the pool's
  // whole per-cycle budget buys (the ladder currently reports 62 rank-adjacent pairs that have
  // never met at all). Non-promoted candidates get the ephemeral tag, keyed off the SCREEN's
  // own edge at the SCREEN's depth — screen.eloLo, not the rescaled gate-depth prediction.
  if (cfg.screen && cfg.screenSave && existsSync(screenHarvest)) {
    const promoted = res.sprt === 'H1';
    const fold = foldHarvest(screenHarvest, screenArchive,
      promoted ? null : championLedgerElo(),
      screen ? screen.eloLo : 0, `depth-${cfg.screenDepth} screen edge`);
    // The promoted case relabels nothing (the candidate is archived, so its own hash is a real
    // ledger node) and would otherwise archive 20k games silently — say so, since that's the
    // case where the games are worth the most to the pool.
    if (fold && fold.folded) {
      log(`  Archived ${fold.folded} depth-${cfg.screenDepth} screen game(s) to ${screenArchive}`
        + `${promoted ? ' — rateable directly (candidate promoted, hash archived)' : ''}. `
        + 'Separate dataset: rated by rank:pool, never featurized.');
    }
  }
  let promotedChampHash = null; // set on promotion, so the end-of-cycle rank calibrates all its depths
  if (res.sprt === 'H1') {
    const arch = JSON.parse(readFileSync(candidate, 'utf8')).arch;
    copyFileSync(champion, prevChampion);   // backup for safety
    copyFileSync(candidate, champion);      // candidate becomes champion
    const champHash = archiveChampion(champion); // keep it reconstructable by its vs version
    promotedChampHash = champHash;               // calibrate this champion's depths at end-of-cycle rank
    cyclesSincePromo = 0;                        // champion (hence every `v` target) just changed — refresh more
    cyclesSinceAdopt = 0;                        // this shape just won the gate — it earns another --rotate window
    if (existsSync(lineage)) rmSync(lineage); // lineage cleared the gate; next start = new champion
    // Publish the new champion into the catalog under its own human name right away, flagged
    // the current champion (so the app shows a real name during its reign, not a generic id).
    const champName = publishChampion(champion, arch);
    promotions++;
    log(`cycle ${c}: PROMOTED ✓  candidate ${pct}% / Elo +${res.elo.toFixed(0)} over champion `
      + `(${res.games} games, cycle took ${fmtDur((Date.now() - cycleT0) / 1000)}). `
      + `New champion named '${champName}' in the catalog (archived ${champHash}.json). Total promotions: ${promotions}.`
      + divNote);
    // (The strength pool is refit at the end of every cycle — runRankPool below. --corpus rates
    // the just-promoted champion from its harvested gate games, but only at the gate/strong-play
    // depths it actually played; the end-of-cycle rank is passed promotedChampHash so it also
    // calibrates every OTHER depth — see runRankPool / cfg.calibrateMinutes.)
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
      + `(SPRT ${res.sprt}${res.futility ? ' by futility stop' : ''}, ${res.games} games, cycle took ${fmtDur((Date.now() - cycleT0) / 1000)}).`
      + `${llrNote} Below the gate; candidate kept as lineage for the next cycle.`
      + divNote);
  } else {
    const hadLineage = existsSync(lineage);
    if (hadLineage) rmSync(lineage);
    log(`cycle ${c}: kept champion — candidate ${pct}% / Elo ${res.elo.toFixed(0)} `
      + `(SPRT ${res.sprt}${res.futility ? ' by futility stop' : ''}, ${res.games} games, cycle took ${fmtDur((Date.now() - cycleT0) / 1000)}).`
      + `${llrNote} Not a gain.${hadLineage ? ' Lineage reset (next warm-start falls back to this recipe\'s best net).' : ''}`
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
      // The gate's OWN 95% CI. `edgeElo` alone hides how much of it is noise, and the spread
      // is huge on a futility-stopped gate (a 134-game stop carries several times the error
      // bar of a full 2000-game one). screen:report needs it to separate real screen-vs-gate
      // disagreement from the gate's own sampling error — without it a screen looks worse
      // than it is, because the "truth" it's scored against is itself noisy.
      ...(gateCI ? { edgeLo: gateCI.lo, edgeHi: gateCI.hi } : {}),
      sprt: res.sprt, futility: !!res.futility, promoted: res.sprt === 'H1',
      // Where the SPRT walk stopped, against the bound it had to cross. Turns a column of
      // identical "inconclusive" verdicts into a near-miss trend: a track creeping from 0.4 to
      // 2.5 across cycles is going somewhere, one flat at 0.2 isn't.
      ...(res.llr != null ? { llr: res.llr, llrUpper: res.llrUpper ?? null } : {}),
      div: res.div ? { corr: res.div.corr, meanCp: res.div.meanCp } : null,
      championHash: gatedVsChampHash,
      // Provenance: when this candidate was grafted from a foreign-arch champion (a new-arch
      // track's bootstrap cycle), record which net it grafted from. null on a normal warm-start.
      graftParent: grafting ? gatedVsChampHash : null,
      datasetBytes: existsSync(rawFile) ? statSync(rawFile).size : 0,
      hash: candHashForTrack,
      // Shadow-screen observation, omitted entirely unless --screen ran. Paired with this
      // same line's `edgeElo`/`score`, which are the gate's truth for the very same
      // candidate — that pairing is the point, and `npm run screen:report` reads it back out.
      ...(screen ? { screen } : {}),
    });
    if (rc.isBest) copyFileSync(candidate, trackBest);
  } catch (e) { log(`  (track record skipped: ${e.message})`); }

  // Refit the strength pool now that this cycle's gate games are harvested into the dataset:
  // --corpus folds them into the Bradley-Terry fit (rating the current champion from its own
  // gate matches) plus a short play budget tightens the most-ambiguous orderings. Runs every
  // cycle so this cycle's own weakest-first refresh (the last step, just below) reads a
  // current ledger. Maintenance: a failure logs but doesn't abort the run (Ctrl-C still ends
  // it via the `stopping` flag).
  // Size this cycle's maintenance budget (rank-minutes + refresh fraction) from the current
  // signals, bounded around the configured values (see adaptiveMaintenance). Computed once here so
  // the rank pass and the refresh below use one consistent, logged decision.
  // Refit cadence (--rank-cycle=N): every N-th cycle, keyed on the TRACK-cumulative `c` so a warm
  // relaunch continues the cadence instead of re-ranking on every launch's first cycle. A
  // PROMOTION always forces a refit — a fresh champion sits at the placeholder-floor Elo at every
  // depth it hasn't played, and the depth-calibration pass that fixes it only runs here, so
  // deferring it would leave the absElo the loop steers by resting on one or two thin bands.
  const cadence = rankCadence(c, Boolean(promotedChampHash));
  const budget = adaptiveMaintenance(cyclesSincePromo, cadence.due);
  if (budget.notes) log(`  Adaptive maintenance: ${budget.notes}.`);
  if (!cadence.due) log(`  Pool refit skipped — ${cadence.why}. The last fitted ledger still drives the filters/refresh.`);
  if (!stopping && cadence.due) {
    runRankPool('Rank pool (Bradley-Terry, corpus + scheduled play)',
      { calibrateChamp: promotedChampHash, minutes: budget.rankMinutes });
    // Re-read AFTER the refit: the pass just rewrote the convergence block, so this sees the
    // deficit as it stands now rather than last cycle's. The link pass no-ops unless the
    // remaining deficit is genuinely unreachable from the strong-play set.
    if (!stopping) runLinkPass(ledgerConvergence());
  }

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
    run(`Refresh v (${(budget.refreshFrac * 100).toFixed(1)}% ${refreshMode()} @ depth ${cfg.refreshCycleDepth})`,
      process.execPath, refreshArgs(budget.refreshFrac, cfg.refreshCycleDepth));
  }
}

log(`train:loop stopped after ${promotions} promotion(s)${rotations ? `, ${rotations} rotation(s)` : ''} in ${fmtDur((Date.now() - loopT0) / 1000)}. `
  + `Champion: web/src/nn-weights.json${promotions ? ' (also published in the net catalog under its name)' : ''}.`);
if (promotions) console.log('Run `npm run build` to ship the new champion in the production bundle.');
