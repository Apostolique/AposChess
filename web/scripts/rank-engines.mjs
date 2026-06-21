// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2019-2026 Jean-David Moisan
//
// Engine strength ranking — the prerequisite for *smart weakest-first* `v` refresh.
//
// The raw dataset's `v` values are labelled by many evolving evals over time, each
// stamped with a `vs` provenance tag (scripts/vtag.mjs): `nn6@94c88e`, `hc6@2`, ...
// A full relabel is infeasible (~days at depth 6 over millions of positions), so the
// refresh budget should target the LOWEST-QUALITY labels first — i.e. the labels
// produced by the WEAKEST contributing engine. To know which engine is weakest we
// have to rank them, and the promotion chain alone doesn't (it never pits hc vs nn,
// nor non-adjacent champions). So we make the contributors PLAY each other.
//
// Efficiently: each engine plays a common ANCHOR — O(N) gauntlet matches, not O(N^2)
// round-robin — and its score vs the anchor converts to an Elo offset (anchor := 0).
// The anchor MUST be a STABLE engine or the baseline drifts run-to-run, so it defaults
// to the handcrafted eval (it changes only on a deliberate HC_VERSION bump, never when
// the loop promotes a champion) — NOT the champion, which the loop overwrites. The
// anchor searches at its OWN fixed depth (default 6), independent of the contenders'
// gauntlet depth: it's the deep, stable reference yardstick while the contenders can
// search cheaper (they don't need to match it — their Elo is still measured against the
// same fixed anchor). The result is an Elo ledger `version -> elo` written to
// training/data/loop/engine-elo.json that refresh-v reads to relabel weakest-first.
//
// A contributor is rankable only if it can be re-INSTANTIATED:
//   hc@<HC_VERSION>   the current handcrafted eval (older hc versions need old code).
//   nn@<hash>         an archived champion at training/data/loop/champions/<hash>.json
//                     (the loop archives every champion by its content hash = its `vs`
//                     version; the current src/nn-weights.json is always available).
//   nn@?              the nn material fallback (weights missing). Ranked by default as the
//                     immutable FLOOR reference, but it's an internal stat only: it's
//                     never used to recompute v, so it's excluded from "weakest to relabel".
// A `vs` tag in the dataset whose engine can't be instantiated (an overwritten old
// champion, an old hc) is UNRECOVERABLE: it has no Elo and is treated as "weakest,
// refresh on sight".
//
// NOTE: the `vs` tagging + champion archival are new; the existing dataset is mostly
// LEGACY (no `vs`). This tool doesn't need tags to run — it ranks whatever engines are
// instantiable today (current champion + handcrafted [+ archived champions as the loop
// produces them]); the dataset scan is only a cross-reference (record counts per tag,
// and which tags are unrecoverable). So it's useful immediately and gets richer as the
// loop archives champions and stamps `vs`.
//
// Usage (run from web/):
//   node scripts/rank-engines.mjs [options]
//   npm run rank -- [options]
//
// Options:
//   --depth=D       fixed search depth for the CONTENDERS in the gauntlet (default 6).
//                   The anchor searches separately (--anchor-depth); contenders don't
//                   need to match it.
//   --movetime=MS   give the contenders a time budget instead of a fixed depth.
//   --anchor-depth=D     fixed search depth for the anchor (default 6) — the deep,
//                   stable yardstick, decoupled from the contenders' --depth.
//   --anchor-movetime=MS give the anchor a time budget instead of a fixed depth.
//   --games=N       games per matchup (even; default 200). Each is anchor-vs-engine.
//                   More games = a tighter Elo; closely-spaced engines (e.g. successive
//                   champions, ~elo1 apart) need a match-runner-sized count to separate.
//   --anchor=WHICH  the common opponent, whose Elo is fixed at 0. MUST be a stable
//                   engine or the baseline drifts run-to-run: 'hc' / 'handcrafted'
//                   (default — changes only on a manual HC_VERSION bump, never when the
//                   loop promotes), 'champion' (= src/nn-weights.json — NOT recommended,
//                   the loop overwrites it every promotion), or a champion content hash
//                   (e.g. 94c88e) to pin a specific frozen archived champion.
//   --no-material   skip ranking the nn material fallback (nn@?). By default it IS
//                   ranked: it's the immutable FLOOR reference (pure material eval), the
//                   Elo that records with no usable provenance inherit. It's never used
//                   to recompute v, so it's excluded from "weakest engine to relabel".
//   --jobs=N        parallel game workers per matchup (default: CPU cores).
//   --openings=K    random opening plies per game (default 6).
//   --maxmoves=N    draw adjudication ply cap (default 200).
//   --seed=S        base seed (default 1).
//   --incremental   reuse Elos already in the output ledger and only play engines it
//                   doesn't yet rank AT THIS DEPTH (engine strength vs a fixed anchor
//                   doesn't drift). A promotion then costs one matchup. A different --depth
//                   just ADDS that depth's entries beside the existing ones. Falls back to a
//                   full gauntlet only if no ledger exists or its ANCHOR differs from this
//                   run's (a different anchor makes the old Elos incomparable).
//   --only=SPEC[,…] (re)rank ONLY these contenders against the anchor at this run's
//                   --depth, splice the result into the existing ledger, and reuse every
//                   other engine's entries unchanged — measure one engine without re-running
//                   the whole gauntlet. Each engine×depth is its OWN ledger entry, so
//                   running --only at a NEW depth ADDS an entry (e.g. nn6@a14d52) beside the
//                   engine's existing one (nn4@a14d52) rather than replacing it — a deeper
//                   search is a stronger player and earns its own, higher, Elo. Every Elo is
//                   relative to the SAME anchor, so the depths stay comparable; consumers
//                   (merge-data, refresh-v) look the exact engine×depth tag up first and
//                   fall back to the engine's per-version Elo for a depth never ranked. SPEC
//                   = a content hash / prefix (e.g. a14d52), a champion filename, 'hc', or
//                   'champion'. Requires an existing ledger built against the same anchor.
//                   Re-running --only at the SAME depth ACCUMULATES: it plays fresh games
//                   (auto-advanced seed) and POOLS them with that entry for a tighter Elo,
//                   instead of discarding it — so run it again to refine. Use --fresh to
//                   discard the prior games and re-measure that entry from scratch.
//   --fresh         with --only, do NOT pool with the prior entry's games — re-measure it
//                   from scratch at the original seed (reproduces the original games).
//   --no-scan       skip the dataset cross-reference (record counts / unrecoverable tags).
//   --data=FILE     dataset to scan (default ../training/data/selfplay.jsonl).
//   --out=FILE      ledger output (default ../training/data/loop/engine-elo.json).
//   --save-games=F  harvest the ranking games as training data (default
//                   ../training/data/rank-games.jsonl). --no-save-games to disable.
//
// Each matchup reuses the tested match runner (scripts/selfplay.mjs) as a subprocess
// with --result-file, so the Elo here is measured exactly as everywhere else. The
// games aren't wasted: --save-games harvests each matchup's positions as training data
// (the stronger engine's `v`, tagged with its vtag — exactly the harvesting the gate
// does). It writes a SEPARATE file, not the live selfplay.jsonl, so it can't race a
// concurrent in-place `v` refresh (refresh-v's atomic rename would drop appends made to
// the old file); fold it in afterwards with `npm run train:merge`.

import { spawnSync } from 'node:child_process';
import { createReadStream, createWriteStream, existsSync, readFileSync, readdirSync, mkdirSync, rmSync } from 'node:fs';
import { createInterface } from 'node:readline';
import { resolve, dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { cpus } from 'node:os';

import { fmtDur, fmtNum, liveStatus, everyMs } from './fmt.mjs';
import { weightsHash } from './vtag.mjs';
import { installStop, printStopHint } from './stop.mjs';
import { HC_VERSION } from '../src/ai.js';

const here = dirname(fileURLToPath(import.meta.url));
const webDir = resolve(here, '..');
const repoDir = resolve(webDir, '..');
const dataDir = resolve(repoDir, 'training', 'data');
const loopDir = join(dataDir, 'loop');
const championsDir = join(loopDir, 'champions');
const champion = resolve(webDir, 'src', 'nn-weights.json');

// Each matchup runs on the native Zig match runner (apos-match), a drop-in for
// selfplay.mjs (per-side depth via --depth-b, SPRT, --result-file, --save-games harvest)
// that's 2-3x faster. Built once up front from web/engine; spawned with cwd = web/ so its
// relative paths resolve like the JS tools'.
const engineDir = resolve(webDir, 'engine');
const matchBin = resolve(engineDir, 'zig-out', 'bin',
  process.platform === 'win32' ? 'apos-match.exe' : 'apos-match');
function buildEngine() {
  // String form (not args-array) with shell:true so Windows resolves `zig` on PATH
  // without the DEP0190 arg-concatenation warning.
  const r = spawnSync('zig build -Doptimize=ReleaseFast', { cwd: engineDir, stdio: 'inherit', shell: true });
  if (r.status !== 0) {
    console.error('zig build failed (is Zig 0.16 on PATH?). rank needs the native match runner.');
    process.exit(1);
  }
}

const args = Object.fromEntries(process.argv.slice(2).map((a) => {
  const m = a.replace(/^--/, '').split('='); return [m[0], m.length > 1 ? m[1] : true];
}));
const num = (v, d) => (v === undefined ? d : Number(v));

const cfg = {
  // Contenders' gauntlet depth/movetime.
  depth: args.movetime !== undefined ? null : (args.depth !== undefined ? Number(args.depth) : 6),
  movetime: args.movetime !== undefined ? Number(args.movetime) : null,
  // The anchor searches at its OWN fixed depth (default 6), independent of the
  // contenders above — it's the deep, stable reference yardstick. --anchor-movetime
  // gives it a time budget instead.
  anchorDepth: args['anchor-movetime'] !== undefined ? null : (args['anchor-depth'] !== undefined ? Number(args['anchor-depth']) : 6),
  anchorMovetime: args['anchor-movetime'] !== undefined ? Number(args['anchor-movetime']) : null,
  games: Math.max(2, Math.round(num(args.games, 200) / 2) * 2),
  // The anchor (Elo := 0) must be a STABLE engine, or the whole ledger's baseline drifts
  // run-to-run. The champion is the WRONG choice — the loop overwrites it on every
  // promotion. Default to the handcrafted eval: it changes only on a deliberate
  // HC_VERSION bump (recorded in the anchor's tag), never automatically.
  anchor: typeof args.anchor === 'string' ? args.anchor : 'hc',
  material: !args['no-material'],
  jobs: args.jobs !== undefined ? Number(args.jobs) : cpus().length,
  openings: num(args.openings, 6),
  maxmoves: num(args.maxmoves, 200),
  seed: num(args.seed, 1),
  scan: !args['no-scan'],
  // Incremental: reuse Elos already in the output ledger (engine strength vs a fixed
  // anchor doesn't drift), and only play engines the ledger doesn't yet rank — so a
  // champion promotion costs one matchup (new champion vs anchor), not the whole gauntlet.
  incremental: !!args.incremental,
  // --only=SPEC[,SPEC]: force-(re)rank just these contenders at this run's depth, reusing
  // every other engine's Elo from the existing ledger (see header). null = rank everyone.
  only: typeof args.only === 'string' ? args.only.split(',').map((s) => s.trim()).filter(Boolean) : null,
  // By default an --only re-run at the SAME depth POOLS its new games with the prior result
  // (more games -> tighter Elo) and advances the seed so they're fresh. --fresh discards the
  // prior games and re-measures from scratch (the original seed -> the original games).
  fresh: !!args.fresh,
  data: typeof args.data === 'string' ? resolve(process.cwd(), args.data) : join(dataDir, 'selfplay.jsonl'),
  out: typeof args.out === 'string' ? resolve(process.cwd(), args.out) : join(loopDir, 'engine-elo.json'),
  // Harvest the ranking games into a side file (NOT the live dataset — see header) that
  // train:merge folds in later. --no-save-games disables harvesting entirely.
  saveGames: args['no-save-games'] ? null
    : (typeof args['save-games'] === 'string' ? resolve(process.cwd(), args['save-games'])
      : join(dataDir, 'rank-games.jsonl')),
};

// Depth marker used in the canonical `vs` tag we report for each engine (matches
// vtag.mjs: a number, or 't' for a time-based search). Contenders and the anchor search
// at different depths, so each gets its own marker (e.g. contenders nn4@…, anchor hc6@2).
const depthMark = cfg.depth != null ? cfg.depth : 't';
const anchorDepthMark = cfg.anchorDepth != null ? cfg.anchorDepth : 't';

// --- Elo helpers (mirror selfplay.mjs, so the numbers line up) -----------------
const eloFromScore = (p) => (p <= 0 ? -800 : p >= 1 ? 800 : -400 * Math.log10(1 / p - 1));
function eloMargin(score, n) {
  // 95% CI half-width on the Elo, from the score's standard error — same as the
  // match runner's error bar, recomputed here since the result file stores only score.
  // A clean sweep (score 0 or 1) has no spread to report, so the margin is 0.
  const v = score * (1 - score);
  if (v <= 0 || n <= 0) return 0;
  const se = Math.sqrt(v / n);
  return (eloFromScore(score + 1.96 * se) - eloFromScore(score - 1.96 * se)) / 2;
}

// --- enumerate the instantiable engines (the players) --------------------------
// Each player: { version, eng, evalName, weights|null, tag }. Keyed by `${eng}@${version}`
// (dedup the current champion against its own archived copy).
const players = new Map();
function addPlayer(eng, version, evalName, weights) {
  const key = `${eng}@${version}`;
  if (players.has(key)) return players.get(key);
  const p = { key, version, eng, evalName, weights, tag: `${eng}${depthMark}@${version}` };
  players.set(key, p);
  return p;
}

// Handcrafted (always available, at the current HC_VERSION).
addPlayer('hc', String(HC_VERSION), 'handcrafted', null);

// The current champion (always available).
if (existsSync(champion)) addPlayer('nn', weightsHash(champion), 'nn', champion);
else console.warn(`(no champion at ${champion} — skipping it)`);

// Archived champions, keyed by their content hash (= their `vs` version).
if (existsSync(championsDir)) {
  for (const f of readdirSync(championsDir).filter((f) => f.endsWith('.json'))) {
    const file = join(championsDir, f);
    addPlayer('nn', weightsHash(file), 'nn', file);
  }
}

// The nn material fallback (weights missing) — optional; it's the floor.
if (cfg.material) addPlayer('nn', '?', 'nn', null); // null weights -> worker uses material

// --- resolve the anchor (one of the players) -----------------------------------
function resolveAnchor(which) {
  if (which === 'hc' || which === 'handcrafted') return players.get(`hc@${HC_VERSION}`);
  if (which === 'champion' || which === undefined) {
    return existsSync(champion) ? players.get(`nn@${weightsHash(champion)}`) : null;
  }
  // Otherwise treat it as an nn version (content hash).
  return players.get(`nn@${which}`) || null;
}
const anchor = resolveAnchor(cfg.anchor);
if (!anchor) {
  console.error(`Anchor '${cfg.anchor}' is not an instantiable engine. ` +
    `Available: ${[...players.keys()].join(', ')}`);
  process.exit(1);
}
// The anchor plays at its own (deeper) depth, so its tag carries the anchor depth mark
// (e.g. hc6@2), not the contenders' gauntlet depth mark.
anchor.tag = `${anchor.eng}${anchorDepthMark}@${anchor.version}`;

const contenders = [...players.values()].filter((p) => p.key !== anchor.key);
if (!contenders.length) {
  console.error('Nothing to rank: only the anchor is instantiable. ' +
    'Run the train:loop so it archives champions, or add --material.');
  process.exit(1);
}

// --only=SPEC[,…]: resolve each spec to exactly one contender to (re)play; every other
// engine's Elo is reused from the existing ledger. Versions to force-replay go in onlyVersions.
let onlyVersions = null;
if (cfg.only) {
  if (!existsSync(cfg.out)) {
    console.error(`--only updates an existing ledger, but none at ${cfg.out}. Run a full 'npm run rank' first.`);
    process.exit(1);
  }
  const resolveSpec = (spec) => {
    if (spec === 'hc' || spec === 'handcrafted') { const p = players.get(`hc@${HC_VERSION}`); return p ? [p] : []; }
    if (spec === 'champion') { const h = existsSync(champion) ? weightsHash(champion) : null; const p = h && players.get(`nn@${h}`); return p ? [p] : []; }
    if (players.has(`nn@${spec}`)) return [players.get(`nn@${spec}`)];
    if (players.has(`hc@${spec}`)) return [players.get(`hc@${spec}`)];
    const base = spec.replace(/\.json$/i, '');
    return [...players.values()].filter((p) => {
      if (p.version === base || p.version.startsWith(base)) return true;
      const wb = p.weights ? p.weights.split(/[\\/]/).pop().replace(/\.json$/i, '') : null;
      return !!wb && (wb === base || wb.startsWith(base));
    });
  };
  onlyVersions = new Set();
  for (const spec of cfg.only) {
    const hits = resolveSpec(spec);
    if (hits.length === 0) {
      console.error(`--only '${spec}' matched no instantiable engine. Available: ${[...players.keys()].join(', ')}`);
      process.exit(1);
    }
    if (hits.length > 1) {
      console.error(`--only '${spec}' is ambiguous: ${hits.map((p) => p.key).join(', ')}. Use a longer prefix.`);
      process.exit(1);
    }
    if (hits[0].key === anchor.key) {
      console.error(`--only '${spec}' is the anchor (${anchor.tag}); its Elo is fixed at 0. Pick a contender, or change --anchor.`);
      process.exit(1);
    }
    onlyVersions.add(hits[0].version);
  }
}

console.log(cfg.only
  ? `Re-ranking ${cfg.only.length} engine(s) (--only ${cfg.only.join(', ')}) vs anchor ${anchor.tag}, reusing the rest of ${cfg.out}`
  : `Ranking ${players.size} engine(s) vs anchor ${anchor.tag}`);
console.log(`  contenders ${cfg.depth != null ? `depth ${cfg.depth}` : `${cfg.movetime}ms/move`} | ` +
  `anchor ${cfg.anchorDepth != null ? `depth ${cfg.anchorDepth}` : `${cfg.anchorMovetime}ms/move`} | ` +
  `${cfg.games} games/matchup | jobs ${cfg.jobs} | openings ${cfg.openings} | seed ${cfg.seed}`);
console.log(`  contenders: ${contenders.map((p) => p.tag).join(', ')}`);
buildEngine(); // ensure the native match runner is built (cached — near-instant if unchanged)
printStopHint();

// --- graceful early stop ----------------------------------------------------------
// A key/Ctrl-C stops before the NEXT matchup (the current spawned match blocks the
// event loop, so it finishes first), then we still write the ledger. To keep that
// ledger VALID we preserve the previous ledger's Elo for any contender we didn't get
// to replay — an early stop must never downgrade a known engine to a null Elo, which
// refresh-v would then read as "weakest, refresh on sight".
let stopped = false;
const stopper = installStop(() => {
  if (stopped) return;
  stopped = true;
  console.log('\n  Stopping early — writing the ledger with the engines ranked so far…');
});
let priorByTag = new Map();
try {
  if (existsSync(cfg.out)) {
    const prev = JSON.parse(readFileSync(cfg.out, 'utf8')).ranking || [];
    priorByTag = new Map(prev.filter((e) => e.elo != null && !e.anchor).map((e) => [e.tag, e]));
  }
} catch { /* no usable prior ledger */ }

// --- play each contender against the anchor (the gauntlet) ---------------------
mkdirSync(loopDir, { recursive: true });
const t0 = Date.now();
// tag (eng+depth+version) -> { score, elo, margin, games } relative to the anchor. Keyed by
// the FULL tag, not the version, so ONE engine can hold several entries — one per search
// depth it was measured at (a deeper search is a stronger player, so it earns its own,
// usually higher, Elo). The consumers (merge-data, refresh-v) look an exact tag up first
// and fall back to the engine's per-version Elo for a depth that was never ranked.
const results = new Map([[anchor.tag, { score: 0.5, elo: 0, margin: 0, games: 0, anchor: true }]]);

// Reuse the existing ledger (--incremental / --only). Every Elo is relative to the anchor,
// so the ONLY thing that must match is the anchor tag (its depth included); a different
// contender --depth doesn't invalidate anything — it just ADDS new per-depth entries beside
// the ones already there. A mismatched anchor makes the old numbers incomparable:
// --incremental re-ranks from scratch; --only refuses (it splices into the existing ledger).
// --only also remembers each target's PRIOR entry AT THIS RUN'S DEPTH, so a same-depth
// re-run POOLS its fresh games with it for a tighter Elo (other depths are left untouched).
const poolPrior = new Map(); // this-depth tag -> prior entry, for --only pooling
let prevLedger = null;
if ((cfg.incremental || cfg.only) && existsSync(cfg.out)) {
  try { prevLedger = JSON.parse(readFileSync(cfg.out, 'utf8')); } catch (e) { console.warn(`(could not read existing ledger: ${e.message})`); }
  if (prevLedger) {
    if (prevLedger.anchor !== anchor.tag) {
      if (cfg.only) {
        console.error(`--only needs the existing ledger to share this run's anchor: ledger anchor ${prevLedger.anchor}, ` +
          `this run ${anchor.tag}. Re-run with a matching --anchor/--anchor-depth, or do a full re-rank.`);
        process.exit(1);
      }
      console.warn(`(existing ledger is anchored on ${prevLedger.anchor}, this run on ${anchor.tag} — the Elos aren't ` +
        `comparable, so re-ranking everything from scratch.)`);
      prevLedger = null;
    } else {
      const thisDepthTag = (e) => `${e.eng}${depthMark}@${e.version}`;
      for (const e of prevLedger.ranking || []) {
        if (e.anchor || e.elo == null) continue;
        if (!players.has(`${e.eng}@${e.version}`)) continue; // no longer instantiable -> drop (becomes unrecoverable)
        // --only: the target's entry at THIS run's depth is the one we replay (and pool with);
        // keep the target's OTHER-depth entries, and every non-target entry, as reused.
        if (cfg.only && onlyVersions.has(e.version) && e.tag === thisDepthTag(e)) { poolPrior.set(e.tag, e); continue; }
        results.set(e.tag, { score: e.score, elo: e.elo, margin: e.margin, games: e.games, reused: true, tag: e.tag });
      }
    }
  } else if (cfg.only) {
    console.error(`--only updates an existing ledger at ${cfg.out}, but it couldn't be read. Run a full 'npm run rank' first.`);
    process.exit(1);
  }
}

function runMatch(p, idx, seedOffset = 0) {
  // Index-keyed temp name: the version can contain filesystem-unsafe chars (e.g. the
  // material fallback's '?'), and matchups run sequentially anyway.
  const tmp = join(loopDir, `rank-match-${idx}.json`);
  const argv = [
    `--games=${cfg.games}`,
    `--jobs=${cfg.jobs}`,
    `--openings=${cfg.openings}`,
    `--maxmoves=${cfg.maxmoves}`,
    // Decorrelate matchups so they aren't all the same openings, yet stay reproducible.
    // seedOffset advances the seed for a pooled --only re-run so its games are FRESH (the
    // match seeds every opening from the base seed, so the same seed = the same games).
    `--seed=${cfg.seed + idx + seedOffset}`,
    `--eval-a=${p.evalName}`,
    `--eval-b=${anchor.evalName}`,
    `--result-file=${tmp}`,
  ];
  // Contenders are engine A, the anchor is engine B — give each its own search budget
  // (selfplay's --depth-b/--movetime-b override side B independently of side A).
  if (cfg.depth != null) argv.push(`--depth=${cfg.depth}`);
  else argv.push(`--movetime=${cfg.movetime}`);
  if (cfg.anchorDepth != null) argv.push(`--depth-b=${cfg.anchorDepth}`);
  else argv.push(`--movetime-b=${cfg.anchorMovetime}`);
  if (p.evalName === 'nn' && p.weights) argv.push(`--weights-a=${p.weights}`);
  if (anchor.evalName === 'nn' && anchor.weights) argv.push(`--weights-b=${anchor.weights}`);
  // Harvest these games into the side dataset file (the winner's `v`, vtag-stamped).
  if (cfg.saveGames) argv.push(`--save-games=${cfg.saveGames}`);

  console.log(`\n=== ${p.tag}  vs  ${anchor.tag}  (${idx + 1}/${contenders.length}) ===`);
  // Mark the match as orchestrated so it doesn't fight us for the TTY's raw mode
  // (it falls back to a SIGINT-only stop — see scripts/stop.mjs).
  const r = spawnSync(matchBin, argv, { stdio: 'inherit', cwd: webDir, env: { ...process.env, APOS_CHILD: '1' } });
  if (r.status !== 0) { console.error(`  match failed (exit ${r.status}); skipping ${p.tag}.`); return null; }
  try {
    const res = JSON.parse(readFileSync(tmp, 'utf8'));
    rmSync(tmp, { force: true });
    return res;
  } catch (e) { console.error(`  could not read result for ${p.tag}: ${e.message}`); return null; }
}

// --only plays only its targets; otherwise play every contender whose entry AT THIS DEPTH
// (p.tag) isn't already reused from the ledger.
const toPlay = contenders.filter((p) => (cfg.only ? onlyVersions.has(p.version) : !results.has(p.tag)));
if (cfg.incremental || cfg.only) {
  const reused = [...results.values()].filter((r) => r.reused).length;
  console.log(`  ${cfg.only ? 'only (--only)' : 'incremental'}: reuse ${reused} ledger entr${reused === 1 ? 'y' : 'ies'}, ` +
    `play ${toPlay.length}${toPlay.length ? ` (${toPlay.map((p) => p.tag).join(', ')})` : ''}`);
}

contenders.forEach((p, idx) => {
  if (stopped) return; // early stop: leave the rest unplayed (prior Elo preserved below)
  if (cfg.only && !onlyVersions.has(p.version)) return; // --only: non-targets stay reused
  if (results.has(p.tag)) { // already have this engine AT THIS DEPTH (reused); don't replay
    const r = results.get(p.tag);
    if (r.reused) console.log(`= ${p.tag}: reusing ledger Elo ${r.elo >= 0 ? '+' : ''}${r.elo.toFixed(0)} (${idx + 1}/${contenders.length})`);
    return;
  }
  // --only same-depth re-run: pool fresh games with the prior entry at this exact tag (more
  // games -> tighter Elo). A different depth isn't a pool — it's a different tag, so it
  // simply becomes its own new entry. --fresh discards the prior games and re-measures.
  const prior = cfg.fresh ? null : poolPrior.get(p.tag);
  const pool = !!prior && prior.games > 0;
  // Advance the seed past the prior games so this run's games don't repeat them.
  const res = runMatch(p, idx, pool ? prior.games : 0);
  if (!res) { results.set(p.tag, { score: null, elo: null, margin: null, games: 0 }); return; }
  let score = res.score, games = res.games;
  if (pool) {
    games = prior.games + res.games;
    score = (prior.score * prior.games + res.score * res.games) / games;
    console.log(`  + pooled with ${prior.games} prior game(s): ${prior.games}+${res.games} = ${games} total`);
  }
  results.set(p.tag, {
    score, elo: eloFromScore(score), margin: eloMargin(score, games), games,
  });
});

// --- optional dataset cross-reference (record counts + unrecoverable tags) ------
// version -> records, and the set of distinct `vs` tags seen (with their depths).
const tagCounts = new Map();      // full `vs` tag -> record count
let legacyNoTag = 0, noV = 0, totalLines = 0;
async function scanDataset() {
  if (!cfg.scan || !existsSync(cfg.data)) {
    if (cfg.scan) console.warn(`\n(no dataset at ${cfg.data} — skipping cross-reference)`);
    return;
  }
  console.log(`\nScanning ${cfg.data} for vs tags...`);
  const status = liveStatus();
  const tick = everyMs(500);
  // Read-only and tolerant: regex over raw lines (no JSON.parse), so a half-written
  // line from a concurrent in-place refresh just fails to match rather than throwing.
  // If the read itself errors (e.g. the file is atomically renamed mid-read on another
  // machine), treat the cross-reference as best-effort and carry on.
  try {
    const rl = createInterface({ input: createReadStream(cfg.data), crlfDelay: Infinity });
    for await (const line of rl) {
      if (!line) continue;
      totalLines++;
      if (!line.includes('"v":')) { noV++; continue; }
      const m = line.match(/"vs":"([^"]+)"/);
      if (!m) { legacyNoTag++; continue; }
      tagCounts.set(m[1], (tagCounts.get(m[1]) || 0) + 1);
      if (tick()) status.update(`  ${fmtNum(totalLines)} lines...`);
    }
  } catch (e) {
    status.clear();
    console.warn(`  dataset scan interrupted (${e.message}); cross-reference is partial.`);
  }
  status.clear();
}
await scanDataset();

// Parse a `vs` tag into { eng, depth, version }. Returns null if malformed.
function parseTag(tag) {
  const m = /^(nn|hc)(\d+|t)@(.+)$/.exec(tag);
  return m ? { eng: m[1], depth: m[2], version: m[3] } : null;
}

// --- assemble the ranking ------------------------------------------------------
// Record count attributed to each instantiable engine = sum over the dataset tags
// that share its eng+version (any depth). When --no-scan, carry the counts from the
// previous ledger (incremental) so a fast promotion re-rank doesn't drop them.
const recordsByVersion = new Map();
if (cfg.scan) {
  for (const [tag, n] of tagCounts) {
    const t = parseTag(tag);
    if (!t) continue;
    const key = `${t.eng}@${t.version}`;
    recordsByVersion.set(key, (recordsByVersion.get(key) || 0) + n);
  }
} else if (prevLedger) {
  for (const e of prevLedger.ranking || []) recordsByVersion.set(`${e.eng}@${e.version}`, e.records || 0);
}

// Early stop: preserve every prior entry we didn't get to replay, so an interrupted run
// never drops or downgrades a known engine×depth (refresh-v reads a missing Elo as weakest).
if (stopped) {
  for (const [tag, e] of priorByTag) {
    if (results.has(tag)) continue;
    const t = parseTag(tag);
    if (t && players.has(`${t.eng}@${t.version}`)) {
      results.set(tag, { score: e.score, elo: e.elo, margin: e.margin, games: e.games, reused: true, tag });
    }
  }
}

// One ranking entry per engine×depth (every tag in `results`), not per engine — that's what
// lets a single engine carry several depth-specific Elos. `records` is per-version, so an
// engine's depth entries share its dataset record count.
const ranking = [...results.entries()].map(([tag, r]) => {
  const t = parseTag(tag) || {};
  const key = `${t.eng}@${t.version}`;
  const p = players.get(key);
  return {
    tag,
    eng: t.eng,
    version: t.version,
    depth: t.depth,
    anchor: tag === anchor.tag,
    elo: r.elo,
    score: r.score,
    margin: r.margin,
    games: r.games,
    records: recordsByVersion.get(key) || 0,
    recoverable: true,
    file: p ? p.weights : null,
  };
}).sort((a, b) => {
  // Weakest first; an unrankable engine (no Elo) sorts to the very bottom (= weakest).
  if (a.elo == null) return -1;
  if (b.elo == null) return 1;
  return a.elo - b.elo;
});

// Dataset `vs` tags whose engine we could NOT instantiate -> unrecoverable contributors.
// They have no Elo and are "weakest, refresh on sight". Carried from the previous ledger
// when --no-scan (incremental), since we didn't re-scan the dataset to rebuild it.
let unrecoverable = [];
if (cfg.scan) {
  for (const [tag, n] of tagCounts) {
    const t = parseTag(tag);
    if (!t) { unrecoverable.push({ tag, records: n, reason: 'malformed tag' }); continue; }
    if (players.has(`${t.eng}@${t.version}`)) continue; // covered by a ranked engine
    let reason;
    if (t.eng === 'hc') reason = `old handcrafted (HC_VERSION now ${HC_VERSION})`;
    else if (t.version === '?') reason = 'nn material fallback (add --material to rank it)';
    else reason = 'champion not archived (overwritten)';
    unrecoverable.push({ tag, records: n, reason });
  }
  unrecoverable.sort((a, b) => b.records - a.records);
} else if (prevLedger) {
  // Drop any now-instantiable engines (e.g. a champion that's since been archived).
  unrecoverable = (prevLedger.unrecoverable || []).filter((u) => {
    const t = parseTag(u.tag);
    return !(t && players.has(`${t.eng}@${t.version}`));
  });
}

// --- write the ledger ----------------------------------------------------------
// The ledger now holds one entry per engine×depth, so a single global depth can't describe
// it. These fields just record THIS run's contender/anchor search budget (each entry's real
// depth lives in its own tag) — they're informational; reuse keys off the anchor tag alone.
const ledger = {
  generated: new Date().toISOString(),
  anchor: anchor.tag,
  depth: cfg.depth,
  movetime: cfg.movetime,
  anchorDepth: cfg.anchorDepth,
  anchorMovetime: cfg.anchorMovetime,
  games: cfg.games,
  seed: cfg.seed,
  dataset: cfg.scan ? { file: cfg.data, totalLines, withV: totalLines - noV, legacyNoTag } : (prevLedger ? prevLedger.dataset : null),
  ranking,
  unrecoverable,
};
mkdirSync(dirname(cfg.out), { recursive: true });
createWriteStream(cfg.out).end(JSON.stringify(ledger, null, 2) + '\n');
stopper.dispose();

// --- report --------------------------------------------------------------------
const pad = (s, n) => String(s).padStart(n);
console.log(`\n===== Engine Elo ranking (anchor ${anchor.tag} := 0) — weakest first =====`);
console.log(`  ${'engine'.padEnd(16)} ${'Elo'.padStart(8)} ${'±'.padStart(5)}  ${'score'.padStart(6)}  ${'records'.padStart(11)}`);
for (const e of ranking) {
  const elo = e.elo == null ? '  n/a' : `${e.elo >= 0 ? '+' : ''}${e.elo.toFixed(0)}`;
  const mar = e.margin == null ? '' : e.margin.toFixed(0);
  const sc = e.score == null ? '' : `${(100 * e.score).toFixed(1)}%`;
  console.log(`  ${e.tag.padEnd(16)} ${pad(elo, 8)} ${pad(mar, 5)}  ${pad(sc, 6)}  ${pad(fmtNum(e.records), 11)}` +
    `${e.anchor ? '   (anchor)' : ''}`);
}
if (unrecoverable.length) {
  console.log(`\n  Unrecoverable contributors (no Elo -> treat as weakest, refresh on sight):`);
  for (const u of unrecoverable) console.log(`    ${u.tag.padEnd(16)} ${pad(fmtNum(u.records), 11)} records  — ${u.reason}`);
}

// The weakest engine to actually relabel from — skip the anchor and the material floor
// ('?'), which is only a reference point and is never used to recompute v.
const weakest = ranking.find((e) => !e.anchor && e.version !== '?');
console.log(`\nLedger -> ${cfg.out}  (in ${fmtDur((Date.now() - t0) / 1000)})`);
if (weakest) {
  console.log(`Weakest ranked engine: ${weakest.tag}` +
    `${weakest.elo != null ? ` (Elo ${weakest.elo.toFixed(0)} vs anchor)` : ''}.`);
  console.log('Next: refresh ascending by Elo — recompute v on the weakest engine\'s positions first.');
}
if (cfg.saveGames) {
  console.log(`Ranking games harvested -> ${cfg.saveGames}. ` +
    'Fold into the dataset with `npm run train:merge` once any in-place v refresh has finished.');
}
