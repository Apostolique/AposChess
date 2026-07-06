// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2019-2026 Jean-David Moisan
//
// Self-relative engine ranking — ONE Bradley-Terry / BayesElo rating pool (no modes).
//
// The old anchor-gauntlet approach (since retired) measured every engine against ONE fixed reference.
// That floors the far end of a wide range: a contender >~400 Elo from the anchor scores
// ~0%/100%, which carries no information no matter how many games you play (and the champions
// are now far above the hc anchor, so the gauntlet saturates). The fix: stop measuring against
// a fixed point — let engines play EACH OTHER near their own strength, then recover all ratings
// with one joint fit. Every node only ever plays near-strength opponents (scores stay in the
// informative 30-70% band), and the whole range is stitched onto one scale by transitivity. As
// games accumulate, EVERY rating tightens.
//
// A node is an (engine, depth) pair: apos-match plays one weights file against another (or
// itself) via --depth/--depth-b. hc<anchor-depth> (hc6) is ALWAYS a node and is the pin
// (Elo := 0), so every pool — all engines, or one net's depth sweep — lands on the SAME stable
// scale. So there is no separate "depth curve" mode: a curve is just the ledger filtered to one
// net's nodes. The store persists pairwise results, so re-running ACCUMULATES games.
//
// Output is the ledger (engine-elo.*.json), the schema refresh-v/merge already read, so
// refresh-v/merge can read it. The ACTIVE scheduler targets the RANKING objective — each step
// it plays the matchup whose ORDERING is currently most ambiguous (P(mis-order) from the full
// Bradley-Terry covariance), with periodic rigidity cross-links. Runs until stopped (q/Ctrl-C),
// --minutes, or --matchups; persists every matchup (lossless overnight stop/start).
//
// Method: Bradley-Terry, fit by the MM algorithm (Hunter 2004) — monotone, no matrix inversion.
// Draws score as half a point each side (consistent with the codebase's eloFromScore logistic
// model). A mild prior (virtual draws vs an even phantom) regularizes sparse pools. 95% CIs +
// the scheduler's contrast variances come from the full Fisher-information covariance.
//
// Usage (run from web/):  node scripts/depth-ladder.mjs [options]  |  npm run rank:pool -- [options]
// Zero-arg `npm run rank:pool` ranks all engines across the whole depth spectrum (1-8), pinned
// hc6, until you stop it.
//
// Options:
//   --engines=SPEC  which engines: 'all' (default = hc + champion + archived champions
//                   [+ material]) or a comma list of specs (content hash/prefix, archived
//                   filename/path, 'champion', 'hc', 'material'). hc<anchor-depth> is added
//                   regardless. --net=X is shorthand for --engines=X (one net's depth sweep).
//   --depths=LIST   depths to rate each engine at: range (1-8) or list (6,8). Default 1-8
//                   (the whole spectrum). Narrow it (e.g. --depths=6,8) for a quick run.
//   --anchor-depth=D  the hc depth that is the pin / Elo 0 (default 6). Always present as a node.
//   --play=SPEC     restrict NEW scheduled games to matchups among these specs only (comma list).
//                   Each is EITHER a bare engine spec (same forms as --engines) — every --depths
//                   of it is schedulable — OR a depth-qualified NODE id `<eng><depth>@<spec>` (the
//                   exact string the ladder prints, e.g. nn8@08df7b, hc6@2, hc6@?) — only that one
//                   (engine, depth) node, so you can target an asymmetric cross-depth matchup like
//                   --play=nn8@08df7b,nn6@22577c. A named depth outside --depths is force-added.
//                   The pin and the rest of the pool are STILL rated from the games already in the
//                   store + --corpus; they just don't play any new games. Use this to pile games
//                   onto a specific head-to-head while keeping everyone on the stable hc6 scale.
//   --no-material   skip the hc<d>@? material fallback node (otherwise included as the floor).
//   --minutes=M     play for M minutes, then finalize. Omit to run until you stop it (q/Ctrl-C).
//   --matchups=N    stop after N matchups (default unlimited).
//   --rounds=0      play nothing; just merge/refit the existing store and emit (offline).
//   --merge=F[,F]   fold pool stores from other machines in before fitting (pairwise counts are
//                   additive). Requires the SAME node set and DISTINCT --seed per machine.
//   --games=N       games per matchup (even; default 100). Re-running accumulates.
//   --onboard=F     no-one-left-behind floor (default 0.5): while any schedulable node has
//                   fewer games than F × the schedulable pool's AVERAGE games, the scheduler
//                   plays the least-played such node (vs the nearest-Elo established node)
//                   before the ordering objective — so fresh champions get anchored to the
//                   scale right away instead of waiting to be "ambiguous enough". Relative,
//                   so a fresh store (everyone at 0) onboards no one. 0 disables.
//   --prior=P       virtual draws vs an even phantom, per node (regularizer). Default 1.
//   --jobs=N        parallel game workers (default: CPU cores).
//   --openings=K    random opening plies per game (default 6).
//   --maxmoves=N    draw adjudication ply cap (default 200).
//   --seed=S        base seed (default 1). The store advances a seed cursor so accumulated
//                   re-runs never replay identical games.
//   --store=FILE    pairwise-results store (default loop/ladder-pool.json — one shared pool).
//   --ledger=FILE   ledger output (default loop/engine-elo.ladder.json — the live strength
//                   ledger train:loop / refresh-v / merge consume for weakest-first v refresh).
//   --data=FILE     dataset to scan for record counts (default selfplay.jsonl). --no-scan skips.
//   --corpus        fold the WHOLE dataset's game results into the fit: every game record's
//                   players + result becomes a pairwise W/D/L among the ranked nodes, so the
//                   entire self-play corpus informs the ratings (not just dedicated matchups).
//                   ON by default; pass --no-corpus for the pool's dedicated matchups only.
//                   Recomputed each run (never persisted), so it can't double-count itself;
//                   self-play games (same engine both sides) are skipped. The pool store's own
//                   games are SUBTRACTED per pair before folding (corpus contributes only the
//                   games beyond what the store already counts), so harvested ladder games that
//                   later land in the dataset can't be counted twice. Pairs with `node.js
//                   --rounds=0` for a pure corpus-only refit.
//   --csv=FILE      also write a flat engine,version,depth,elo,ci95,games CSV (for plotting a
//                   depth curve — filter to one version). Off unless given.
//   --save-games[=F]  harvest the played games to F (default ON -> loop/ladder-games.jsonl):
//                   the self-contained body of every game this ladder has played, appended
//                   across runs. These are games between STATIC engines (archived champions +
//                   hc), so they're permanent training data — PRESERVED across --fresh (delete
//                   the file by hand to clear it). It lives under loop/ so merge-data (top-level
//                   scan) won't auto-fold it into selfplay; merge it in explicitly for training.
//   --no-save-games  don't harvest games (ranking only).
//   --fresh         discard the existing store (ratings) and start over. The harvested games
//                   archive is PRESERVED (permanent games between static engines).

import { spawn, spawnSync } from 'node:child_process';
import { existsSync, readFileSync, writeFileSync, mkdirSync, rmSync, readdirSync, createReadStream, statSync } from 'node:fs';
import { createInterface } from 'node:readline';
import { resolve, dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { cpus } from 'node:os';

import { weightsHash } from './vtag.mjs';
import { installStop, printStopHint } from './stop.mjs';
import { HC_VERSION } from '../src/ai.js';
import { isGameRecord, tallyVs } from './gameRecord.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const webDir = resolve(here, '..');
const repoDir = resolve(webDir, '..');
const dataDir = resolve(repoDir, 'training', 'data');
const loopDir = join(dataDir, 'loop');
const championsDir = join(loopDir, 'champions');
const champion = resolve(webDir, 'src', 'nn-weights.json');
const engineDir = resolve(webDir, 'engine');
const matchBin = resolve(engineDir, 'zig-out', 'bin', process.platform === 'win32' ? 'apos-match.exe' : 'apos-match');

// Nice display names, keyed by the same 6-char content hash weightsHash() produces (= a node's
// version), from the web UI net catalog (public/nn/manifest.json — champions + hand nets).
// Best-effort: a version not in the catalog (an unarchived/quantized net, hc, material) simply
// has no name.
const nnNames = (() => {
  const m = new Map();
  try {
    const mani = JSON.parse(readFileSync(resolve(webDir, 'public', 'nn', 'manifest.json'), 'utf8'));
    for (const n of mani.nets || []) if (n.hash && n.name) m.set(n.hash, n.name);
  } catch { /* no manifest -> no nice names */ }
  return m;
})();
const niceName = (version) => nnNames.get(version) || null;
// A node id with its human name attached — `nn8@9e31ca (Leo)` — so every line that names a
// competitor reads without a hash→name lookup. Nodes without a catalog name (hc, material,
// unarchived nets) print as their bare id.
const nodeLabel = (c) => `${c.id}${niceName(c.version) ? ` (${niceName(c.version)})` : ''}`;
const fmtSigned = (x) => `${x >= 0 ? '+' : ''}${x.toFixed(0)}`;

const args = Object.fromEntries(process.argv.slice(2).map((a) => {
  const m = a.replace(/^--/, '').split('='); return [m[0], m.length > 1 ? m[1] : true];
}));
const num = (v, d) => (v === undefined ? d : Number(v));
function parseDepths(spec, dflt) {
  if (spec === undefined) return dflt;
  const s = String(spec);
  const r = /^(\d+)-(\d+)$/.exec(s);
  if (r) { const out = []; for (let d = +r[1]; d <= +r[2]; d++) out.push(d); return out; }
  return s.split(',').map((x) => Number(x.trim())).filter((x) => Number.isInteger(x) && x > 0);
}

// ONE rating pool — no modes. Nodes are (engine, depth) pairs; hc<anchor-depth> (hc6 by
// default) is ALWAYS a node and is the pin (Elo := 0), so every pool — all engines, or one
// net's depth sweep — lands on the same stable scale. Pick the engines with --engines (default
// 'all') or --net=X (one net); --depths sets the depths. The ledger is the single artifact: a
// "depth curve" is just the ledger filtered to one net, so there's no separate mode to maintain.
const cfg = {
  // 'all' = hc + current champion + every archived champion [+ material]; or a comma list of
  // specs (a content hash/prefix, an archived filename/path, 'champion', 'hc', 'material').
  // --net=X is shorthand for --engines=X.
  engines: typeof args.net === 'string' ? args.net : (typeof args.engines === 'string' ? args.engines : 'all'),
  depths: parseDepths(args.depths, [1, 2, 3, 4, 5, 6, 7, 8]),
  anchorDepth: num(args['anchor-depth'], 6), // the hc depth that is the pin (Elo := 0)
  // --play: restrict NEW scheduled games to matchups among these engines (comma list of specs).
  // The pin + the rest of the pool are still rated from already-played store/--corpus games;
  // they just don't play new games. null = schedule across the whole pool (default).
  play: typeof args.play === 'string' ? args.play.split(',').map((s) => s.trim()).filter(Boolean) : null,
  material: !args['no-material'],
  // Pool stores from other machines to fold in before fitting (pairwise counts are additive).
  // Each MUST come from a distinct --seed (else identical games would be double-counted).
  merge: typeof args.merge === 'string' ? args.merge.split(',').map((s) => resolve(process.cwd(), s.trim())).filter(Boolean) : [],
  rounds: Math.max(0, num(args.rounds, 1)), // 0 = don't play; just merge/refit and emit (offline)
  // ACTIVE scheduler: play until --minutes elapse / --matchups are played / the user stops
  // (q or Ctrl-C). Defaults run until stopped — built for overnight.
  minutes: args.minutes !== undefined ? Number(args.minutes) : null,
  matchups: args.matchups !== undefined ? Number(args.matchups) : Infinity,
  games: Math.max(2, Math.round(num(args.games, 100) / 2) * 2),
  // Onboarding floor, as a FRACTION of the schedulable pool's average game count (relative on
  // purpose — no magic absolute number: the floor scales with how much the pool has actually
  // played, and a fresh store where everyone sits at 0 onboards no one). A node below the
  // floor is "under-played" and gets scheduled first (least-played, vs the nearest-Elo
  // established node). Without it, fresh nodes sit at the prior (-35 ±600s) where the ordering
  // objective pairs them with EACH OTHER — two unknowns playing each other stay disconnected
  // from the scale — while the well-played cluster keeps winning the ambiguity contest. 0 off.
  onboard: Math.max(0, num(args.onboard, 0.5)),
  prior: num(args.prior, 1),
  jobs: args.jobs !== undefined ? Number(args.jobs) : cpus().length,
  openings: num(args.openings, 6),
  maxmoves: num(args.maxmoves, 200),
  seed: num(args.seed, 1),
  scan: !args['no-scan'],
  // Fold the WHOLE self-play corpus into the fit: every stored game records who played
  // (players.w/.b) and its result, so the entire dataset becomes a pairwise W/D/L matrix
  // among the ranked nodes — not just the pool's own dedicated matchups. The corpus
  // contribution is recomputed each run (not persisted into the pool store), so it never
  // double-counts. Self-play games (same engine both sides) are uninformative and skipped.
  corpus: !args['no-corpus'],
  data: typeof args.data === 'string' ? resolve(process.cwd(), args.data) : join(dataDir, 'selfplay.jsonl'),
  store: typeof args.store === 'string' ? resolve(process.cwd(), args.store) : join(loopDir, 'ladder-pool.json'),
  ledger: typeof args.ledger === 'string' ? resolve(process.cwd(), args.ledger) : join(loopDir, 'engine-elo.ladder.json'),
  csv: typeof args.csv === 'string' ? resolve(process.cwd(), args.csv) : null, // opt-in flat dump for plotting
  // Harvest the played games (default ON). An absolute path string when enabled, else false.
  // Default target is loop/ladder-games.jsonl — the self-contained body of every game the
  // ladder has played (appended across runs; PRESERVED across --fresh, since they're permanent
  // games between static engines). It's under loop/, so merge-data's top-level scan won't
  // auto-fold it into selfplay (no corpus double-count); fold it in explicitly for training.
  saveGames: args['no-save-games'] ? false
    : (typeof args['save-games'] === 'string' ? resolve(process.cwd(), args['save-games']) : join(loopDir, 'ladder-games.jsonl')),
  fresh: !!args.fresh,
};

// --- resolve an engine spec to { eng, eval, weights, version } ------------------
function makeEngine(spec) {
  if (spec === 'hc' || spec === 'handcrafted') return { eng: 'hc', eval: 'handcrafted', weights: null, version: String(HC_VERSION) };
  // The material baseline: the engine's bare piece-count eval (EvalKind.material), NOT an
  // nn eval with a missing net. (It was the latter, which silently made apos-match fall back
  // to its default --weights = the champion — so the "material" node was really the champion
  // in disguise and ranked absurdly high.) id is hc<d>@? — an hc-family engine (no net) whose
  // '?' version marks it the material floor, distinct from real handcrafted hc<d>@<HC_VERSION>.
  if (spec === 'material') return { eng: 'hc', eval: 'material', weights: null, version: '?' };
  if (spec === 'champion') {
    if (!existsSync(champion)) { console.error(`No champion at ${champion}.`); process.exit(1); }
    return { eng: 'nn', eval: 'nn', weights: champion, version: weightsHash(champion) };
  }
  const cand = [join(championsDir, spec.endsWith('.json') ? spec : `${spec}.json`), resolve(process.cwd(), spec)].find(existsSync);
  if (!cand) { console.error(`Could not find net '${spec}' (champion archive or path).`); process.exit(1); }
  return { eng: 'nn', eval: 'nn', weights: cand, version: weightsHash(cand) };
}
function allEngines() {
  const list = [makeEngine('hc')];
  if (existsSync(champion)) list.push(makeEngine('champion'));
  else console.warn(`(no champion at ${champion} — skipping it)`);
  if (existsSync(championsDir)) for (const f of readdirSync(championsDir).filter((f) => f.endsWith('.json'))) list.push(makeEngine(f));
  if (cfg.material) list.push(makeEngine('material'));
  return list;
}
// Parse one --play token into { engine, depth }. A bare engine spec (08df7b, champion, hc,
// material, a path) has depth = null: EVERY --depths of that engine is schedulable. A
// depth-qualified NODE id — the exact `<eng><depth>@<spec>` string the ladder prints, e.g.
// nn8@08df7b / hc6@2 / hc6@? — pins the schedulable set to that one (engine, depth) node, so
// you can target an asymmetric cross-depth matchup. The part after @ is resolved like any spec
// (hash/'champion'/path), with the ladder's own hc/material node ids handled specially so a
// pasted id round-trips.
function parsePlaySpec(spec) {
  const m = /^(nn|hc)(\d+)@(.+)$/.exec(spec);
  if (!m) return { engine: makeEngine(spec), depth: null }; // bare engine spec -> all depths
  const [, eng, d, ver] = m;
  // '?' version -> the material floor (hc-family, no net); else hc<ver> = real handcrafted, nn<ver> = a net.
  const engine = ver === '?' ? makeEngine('material') : eng === 'hc' ? makeEngine('hc') : makeEngine(ver);
  if (engine.eng !== eng) { console.error(`--play: '${spec}' names a ${eng} node but '${ver}' resolves to a ${engine.eng} engine.`); process.exit(1); }
  return { engine, depth: Number(d) };
}

// Selected engines + the ALWAYS-included handcrafted, deduped by eng@version. hc is a full
// participant at EVERY --depth (near-strength anchor points connecting every depth band to the
// scale), and hc<anchor-depth> (hc6) is the pin. So even a single-net sweep is tied to the
// stable scale at each depth, not just reaching down from hc6.
let engines = cfg.engines === 'all' ? allEngines()
  : cfg.engines.split(',').map((s) => s.trim()).filter(Boolean).map(makeEngine);
// --play engines must be in the rated pool too (so they're competitors that can play AND be
// rated). Parse each spec (bare = all depths, or a depth-qualified node id — see parsePlaySpec)
// and fold any engine that --engines didn't already include into the list.
const playSpecs = cfg.play ? cfg.play.map(parsePlaySpec) : null;
if (playSpecs) for (const { engine } of playSpecs) if (!engines.some((e) => `${e.eng}@${e.version}` === `${engine.eng}@${engine.version}`)) engines.push(engine);
if (!engines.some((e) => `${e.eng}@${e.version}` === `hc@${HC_VERSION}`)) engines.push(makeEngine('hc'));
{ const seen = new Set(); engines = engines.filter((e) => { const k = `${e.eng}@${e.version}`; if (seen.has(k)) return false; seen.add(k); return true; }); }

// --- competitors (nodes of the rating pool) ------------------------------------
// Each node is an (engine, depth) pair; id is its `vs`-tag (stable across runs, the store
// key AND the ledger tag): nn<d>@<ver> / hc<d>@<HC_VERSION> / hc<d>@? (material).
const node = (e, d) => ({ id: `${e.eng}${d}@${e.version}`, eng: e.eng, eval: e.eval, weights: e.weights, version: e.version, depth: d });
const competitors = [];
for (const e of engines) for (const d of cfg.depths) competitors.push(node(e, d));
// The pin node hc<anchor-depth> is ALWAYS present, even if anchor-depth ∉ --depths.
const pinId = `hc${cfg.anchorDepth}@${HC_VERSION}`;
if (!competitors.some((c) => c.id === pinId)) competitors.push(node(makeEngine('hc'), cfg.anchorDepth));
// A depth-qualified --play node may name a depth outside --depths; force that exact node in
// (like the pin) so it can play and be rated.
if (playSpecs) for (const { engine, depth } of playSpecs) {
  if (depth == null) continue;
  const id = `${engine.eng}${depth}@${engine.version}`;
  if (!competitors.some((c) => c.id === id)) competitors.push(node(engine, depth));
}
if (competitors.length < 2) { console.error('Need at least 2 nodes (engines × depths).'); process.exit(1); }
const byId = new Map(competitors.map((c) => [c.id, c]));

// --- schedulable subset (--play): which nodes may play NEW games ---------------
// fit() always rates ALL competitors (the pin + the whole pool) from the persisted store +
// --corpus. --play just restricts the ACTIVE SCHEDULER to matchups among the named specs, so
// only that head-to-head gains new games while everyone else keeps their from-existing-data
// estimate. A bare engine spec (depth == null) matches every --depths of that engine; a
// depth-qualified node id matches only that exact (engine, depth) node.
const playMatch = playSpecs
  ? (c) => playSpecs.some(({ engine, depth }) => c.eng === engine.eng && c.version === engine.version && (depth == null || c.depth === depth))
  : null;
const schedulable = playMatch ? competitors.filter(playMatch) : competitors;
if (cfg.rounds !== 0 && schedulable.length < 2) {
  console.error(`--play needs ≥2 schedulable nodes to play (got ${schedulable.length}); widen --play or --depths.`);
  process.exit(1);
}

// --- persistent pairwise store -------------------------------------------------
// pairs: { "idA|idB" (sorted): { games, sumA } } where sumA = points scored by idA. One shared
// store ('rank') accumulates ALL pool games — any run fits only its own node subset, so a
// single-net depth sweep and the full ledger pool feed the same body of knowledge.
mkdirSync(loopDir, { recursive: true });
const poolId = 'rank';
// --fresh resets the RATINGS (the store) only; the harvested games archive is PRESERVED. Those
// are real games between STATIC engines (archived champions + hc, which never change), so they
// stay valid training data and a valid body of results forever — discarding them would throw
// away genuine games just because the BT fit is being recomputed. The store going empty is fine:
// --corpus subtracts the store, so when the store is 0 the archive's games (if in the dataset)
// simply count once in the refit. Delete the archive by hand if you ever truly want it gone.
let store = { net: poolId, seedCursor: cfg.seed, pairs: {} };
if (!cfg.fresh && existsSync(cfg.store)) {
  try {
    const s = JSON.parse(readFileSync(cfg.store, 'utf8'));
    if (s.net === poolId) store = { net: poolId, seedCursor: s.seedCursor ?? cfg.seed, pairs: s.pairs || {} };
    else console.warn(`(existing store is for pool ${s.net}, this run ${poolId} — starting fresh)`);
  } catch (e) { console.warn(`(could not read store: ${e.message}; starting fresh)`); }
}
// Fold in pool stores from other machines (--merge). Pairwise {games, sumA} are additive, so
// a second machine's games simply sum in — provided it used a DISTINCT --seed (same seed =
// identical openings = double-counting) and the SAME node set (same champion + archive, so
// the pair keys align). seedCursor takes the max so continued play won't reuse a seed window.
for (const mf of cfg.merge) {
  let s;
  try { s = JSON.parse(readFileSync(mf, 'utf8')); }
  catch (e) { console.error(`--merge: could not read ${mf}: ${e.message}`); process.exit(1); }
  if (s.net !== poolId) { console.error(`--merge: ${mf} is pool '${s.net}', this run is '${poolId}' — incompatible node sets.`); process.exit(1); }
  let folded = 0;
  for (const [key, v] of Object.entries(s.pairs || {})) {
    const e = store.pairs[key] || { games: 0, sumA: 0 };
    e.games += v.games; e.sumA += v.sumA; store.pairs[key] = e; folded += v.games;
  }
  store.seedCursor = Math.max(store.seedCursor, s.seedCursor ?? 0);
  console.log(`merged ${folded} games from ${mf.replace(/^.*[\\/]/, '')}`);
}
const pairKey = (a, b) => (a < b ? `${a}|${b}` : `${b}|${a}`);

// Corpus-derived pairwise results (from --corpus): recomputed each run, NOT persisted into
// the pool store, so re-runs never double-count. Same {games, sumA} shape (sumA = points for
// the lexically-first id). fit() sums these on top of the persisted store.
const corpusPairs = new Map();
let corpusGames = 0;
// Combined view of persisted + corpus pairs. The persisted store is authoritative for the
// pool's own matchups, so the corpus contributes only the games BEYOND what the store already
// counts (corpus − store per pair, clamped at 0). That way, if harvested ladder games ever
// land in the scanned dataset, --corpus re-reads them but they aren't double-counted on top of
// the store; pairs the corpus knows but the store doesn't (organic self-play) pass through in
// full. Used by fit() so the corpus contributes to every rating without touching disk.
function combinedPairs() {
  const m = new Map();
  for (const [k, v] of Object.entries(store.pairs)) m.set(k, { games: v.games, sumA: v.sumA });
  for (const [k, v] of corpusPairs) {
    const s = store.pairs[k] || { games: 0, sumA: 0 };
    const extraGames = v.games - s.games;
    if (extraGames <= 0) continue; // corpus fully overlaps the pool store — don't double-count
    const extraSumA = Math.min(Math.max(v.sumA - s.sumA, 0), extraGames);
    const e = m.get(k) || { games: 0, sumA: 0 };
    e.games += extraGames; e.sumA += extraSumA; m.set(k, e);
  }
  return m;
}

function record(idA, idB, gamesPlayed, scoreA) {
  // scoreA = A's average score (wins+0.5 draws)/games, from apos-match.
  const key = pairKey(idA, idB);
  const lowFirst = idA < idB;
  const sumForA = scoreA * gamesPlayed; // points A scored
  const e = store.pairs[key] || { games: 0, sumA: 0 };
  e.games += gamesPlayed;
  e.sumA += lowFirst ? sumForA : (gamesPlayed - sumForA); // sumA tracks the lexically-first id
  store.pairs[key] = e;
}

// --- Bradley-Terry fit (MM algorithm) ------------------------------------------
function fit() {
  const ids = competitors.map((c) => c.id);
  const gamma = new Map(ids.map((id) => [id, 1]));
  const W = new Map(ids.map((id) => [id, cfg.prior * 0.5])); // points, incl. prior half-draws vs phantom
  const adj = new Map(ids.map((id) => [id, []]));            // id -> [{opp, N}]
  const pairs = combinedPairs();                             // persisted store + --corpus games
  for (const [key, v] of pairs) {
    const [i, j] = key.split('|');
    if (!W.has(i) || !W.has(j)) continue;
    W.set(i, W.get(i) + v.sumA);
    W.set(j, W.get(j) + (v.games - v.sumA));
    adj.get(i).push({ opp: j, N: v.games });
    adj.get(j).push({ opp: i, N: v.games });
  }
  for (let iter = 0; iter < 10000; iter++) {
    let maxd = 0;
    for (const id of ids) {
      let denom = cfg.prior / (gamma.get(id) + 1); // phantom opponent at gamma=1
      for (const { opp, N } of adj.get(id)) denom += N / (gamma.get(id) + gamma.get(opp));
      if (denom <= 0) continue;
      const g = W.get(id) / denom;
      maxd = Math.max(maxd, Math.abs(Math.log(g) - Math.log(gamma.get(id))));
      gamma.set(id, g);
    }
    const gm = Math.exp([...gamma.values()].reduce((s, g) => s + Math.log(g), 0) / ids.length);
    for (const id of ids) gamma.set(id, gamma.get(id) / gm); // anchor scale each iter
    if (maxd < 1e-10) break;
  }
  const elo = new Map(ids.map((id) => [id, 400 * Math.log10(gamma.get(id))]));
  if (pinId) { const off = elo.get(pinId); for (const id of ids) elo.set(id, elo.get(id) - off); }
  // Full covariance from the Fisher information matrix H (beta = ln gamma): H_ii = phantom
  // self-anchor + Σ_j N_ij·p_ij(1−p_ij), H_ij = −N_ij·p_ij(1−p_ij). The prior on the diagonal
  // makes H positive-definite (invertible) even on a sparse/tree graph, so we can read off the
  // variance of ANY rating contrast — exactly what the active scheduler needs to find the
  // least-resolved ordering. (The old diagonal-only approximation ignored correlations.)
  const n = ids.length;
  const pos = new Map(ids.map((id, k) => [id, k]));
  const H = Array.from({ length: n }, () => new Float64Array(n));
  for (let k = 0; k < n; k++) H[k][k] = cfg.prior * 0.25; // phantom self-anchor (p(1−p)=0.25 at parity)
  for (const [key, v] of pairs) {
    const [i, j] = key.split('|');
    if (!pos.has(i) || !pos.has(j)) continue;
    const gi = gamma.get(i), gj = gamma.get(j);
    const w = v.games * (gi * gj) / ((gi + gj) * (gi + gj)); // N·p(1−p)
    const a = pos.get(i), b = pos.get(j);
    H[a][a] += w; H[b][b] += w; H[a][b] -= w; H[b][a] -= w;
  }
  const S = invSym(H, n);
  const C = (400 / Math.LN10) ** 2; // beta-variance -> Elo-variance
  const cov = (a, b) => S[pos.get(a)][pos.get(b)];
  // Var of the Elo difference between two nodes (Elo²) — the scheduler's currency.
  const varDiff = (a, b) => (a === b ? 0 : C * (cov(a, a) + cov(b, b) - 2 * cov(a, b)));
  const ci = new Map(ids.map((id) => [id, 1.96 * Math.sqrt(Math.max(0, pinId ? varDiff(id, pinId) : C * cov(id, id)))]));
  return { elo, ci, varDiff, gamesOf: (id) => adj.get(id).reduce((s, e) => s + e.N, 0) };
}

// Gauss-Jordan inverse of a symmetric positive-definite matrix (n small: ~node count).
function invSym(A, n) {
  const M = A.map((r) => Float64Array.from(r));
  const I = Array.from({ length: n }, (_, k) => { const r = new Float64Array(n); r[k] = 1; return r; });
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
    if (Math.abs(M[piv][col]) < 1e-12) continue; // ~singular (prior should prevent); skip
    if (piv !== col) { const t = M[piv]; M[piv] = M[col]; M[col] = t; const u = I[piv]; I[piv] = I[col]; I[col] = u; }
    const d = M[col][col];
    for (let c = 0; c < n; c++) { M[col][c] /= d; I[col][c] /= d; }
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = M[r][col]; if (f === 0) continue;
      for (let c = 0; c < n; c++) { M[r][c] -= f * M[col][c]; I[r][c] -= f * I[col][c]; }
    }
  }
  return I;
}

// Standard normal CDF (Abramowitz-Stegun 7.1.26 erf) — P(mis-ordered) for a standardized gap.
function erf(x) {
  const s = x < 0 ? -1 : 1; x = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * x);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t) * Math.exp(-x * x);
  return s * y;
}
const ncdf = (x) => 0.5 * (1 + erf(x / Math.SQRT2));

// Active scheduler for the RANKING objective: pick the matchup that most reduces ordering
// uncertainty. Score each rank-adjacent pair by P(mis-ordered) = Φ(−|Δ|/σ_Δ); play the worst.
// Every 6th pick (or once all orderings are sharp) play the least-constrained near-strength
// contrast instead, to keep the graph rigid and catch transitivity errors.
// Scheduling is restricted to `schedulable` (the --play subset; the whole pool by default) — the
// elo/varDiff still come from the full fit, so only the chosen engines play new games.
//
// ONBOARDING (no one left behind): before the ordering objective, any schedulable node whose
// game count is below --onboard × the schedulable pool's average is played first — least-played
// node vs the nearest-Elo ESTABLISHED node (one at/above the floor; nearest keeps p(1−p)
// informative, and a bad initial estimate self-corrects as the first matchups move it). Without
// this, a fresh node sits at the prior where the ordering objective pairs it with the OTHER
// fresh nodes — two unknowns playing each other stay disconnected from the scale — while the
// well-played cluster keeps winning the ambiguity contest. Least-played-first cycles through
// every under-played node, so a new champion (and each of its depths, in a full-pool run) is
// anchored within a run or two; the floor being relative means everyone graduates as the pool's
// average rises, and a fresh store (all zeros) onboards no one.
function pickMatchup(elo, varDiff, iter, gamesOf) {
  if (cfg.onboard > 0 && schedulable.length > 1) {
    const avg = schedulable.reduce((s, c) => s + gamesOf(c.id), 0) / schedulable.length;
    const floor = cfg.onboard * avg;
    const under = schedulable.filter((c) => gamesOf(c.id) < floor);
    if (under.length) {
      under.sort((a, b) => gamesOf(a.id) - gamesOf(b.id));
      const nov = under[0];
      const established = schedulable.filter((c) => c !== nov && gamesOf(c.id) >= floor);
      const pool = established.length ? established : schedulable.filter((c) => c !== nov);
      let opp = null, best = Infinity;
      for (const c of pool) {
        const d = Math.abs(elo.get(c.id) - elo.get(nov.id));
        if (d < best) { best = d; opp = c; }
      }
      if (opp) return { pair: [nov, opp], reason: 'onboard', metric: gamesOf(nov.id), floor };
    }
  }
  const sorted = [...schedulable].sort((a, b) => elo.get(a.id) - elo.get(b.id));
  let best = null, bestAmb = -1;
  for (let k = 0; k < sorted.length - 1; k++) {
    const a = sorted[k], b = sorted[k + 1];
    const d = Math.abs(elo.get(a.id) - elo.get(b.id));
    const sd = Math.sqrt(Math.max(varDiff(a.id, b.id), 1e-9));
    const amb = ncdf(-d / sd);
    if (amb > bestAmb) { bestAmb = amb; best = [a, b]; }
  }
  if (iter % 6 === 5 || bestAmb < 0.02) {
    let rb = null, rv = -1;
    for (let i = 0; i < schedulable.length; i++) for (let j = i + 1; j < schedulable.length; j++) {
      const a = schedulable[i], b = schedulable[j];
      if (Math.abs(elo.get(a.id) - elo.get(b.id)) > 300) continue; // keep p(1−p) meaningful
      const v = varDiff(a.id, b.id);
      if (v > rv) { rv = v; rb = [a, b]; }
    }
    if (rb) return { pair: rb, reason: 'rigidity', metric: Math.sqrt(Math.max(0, rv)) };
  }
  return { pair: best, reason: 'ordering', metric: bestAmb };
}

// --- one matchup on apos-match -------------------------------------------------
function buildEngine() {
  const r = spawnSync('zig build -Doptimize=ReleaseFast', { cwd: engineDir, stdio: 'inherit', shell: true });
  if (r.status !== 0) { console.error('zig build failed (is Zig 0.16 on PATH?).'); process.exit(1); }
}
// Stop request via a stop-file (loop/ladder-stop) that apos-match polls ~1×/s. On seeing it,
// the match stops RIGHT AWAY: it writes its result + harvest from the games already COMPLETED
// and exits 0, abandoning the games still in flight. So the finished games are RECORDED (no
// more round-multiple-of-100 counts) while the stop is near-instant. The old behavior killed
// the child outright, discarding the whole matchup. The file is cleared at startup and on exit.
const stopFile = join(loopDir, 'ladder-stop');
function requestMatchStop() { try { writeFileSync(stopFile, ''); } catch { /* best effort */ } }

// Async spawn (not spawnSync) so Node's event loop stays free to handle q/Ctrl-C while a
// matchup runs — a synchronous spawn would block the stop handler for the whole matchup.
// stdin is 'ignore' so the parent keeps sole ownership of the TTY for keypress detection.
function playPair(a, b) {
  const tmp = join(loopDir, 'ladder-match.json');
  const seed = store.seedCursor;
  store.seedCursor += cfg.games; // advance so accumulated re-runs use fresh openings
  const argv = [
    `--games=${cfg.games}`, `--jobs=${cfg.jobs}`, `--openings=${cfg.openings}`, `--maxmoves=${cfg.maxmoves}`,
    `--seed=${seed}`,
    `--eval-a=${a.eval}`, `--eval-b=${b.eval}`, `--depth=${a.depth}`, `--depth-b=${b.depth}`,
    `--result-file=${tmp}`, `--stop-file=${stopFile}`,
  ];
  if (a.eval === 'nn' && a.weights) argv.push(`--weights-a=${a.weights}`);
  if (b.eval === 'nn' && b.weights) argv.push(`--weights-b=${b.weights}`);
  // apos-match only honours --save-games=<path> (a bare flag is silently ignored), so cfg.saveGames
  // is always an absolute path (default loop/ladder-games.jsonl) or false.
  if (typeof cfg.saveGames === 'string') argv.push(`--save-games=${cfg.saveGames}`);
  console.log(`\n=== ${nodeLabel(a)}  vs  ${nodeLabel(b)}  (${cfg.games} games) ===`);
  return new Promise((done) => {
    const child = spawn(matchBin, argv, { stdio: ['ignore', 'inherit', 'inherit'], cwd: webDir, env: { ...process.env, APOS_CHILD: '1' } });
    child.on('error', (e) => { console.error(`  could not run match: ${e.message}; skipping.`); done(); });
    child.on('exit', (code, signal) => {
      // A stop-file stop still exits 0 with the completed-games result, so it's recorded like any other.
      if (code !== 0) { console.error(`  match failed (exit ${code ?? signal}); skipping.`); done(); return; }
      try { const res = JSON.parse(readFileSync(tmp, 'utf8')); rmSync(tmp, { force: true }); record(a.id, b.id, res.games, res.score); }
      catch (e) { console.error(`  could not read result: ${e.message}`); }
      done();
    });
  });
}


// --- dataset cross-reference (--rank): record counts + unrecoverable tags ------
// Read-only regex scan over the dataset's vs tags, so the emitted ledger carries the same
// `records` / `unrecoverable` fields the report and downstream tooling expect.
const tagCounts = new Map();
let totalLines = 0, noV = 0, legacyNoTag = 0;
const scanCacheFile = join(loopDir, 'ladder-scan-cache.json');
async function scanDataset() {
  if (!cfg.scan || !existsSync(cfg.data)) { if (cfg.scan) console.warn(`(no dataset at ${cfg.data} — skipping cross-reference)`); return; }
  // The dataset doesn't change during a ranking run, so cache the scan by file size+mtime —
  // a no-arg overnight stop/start then doesn't re-read 1 GB on every restart.
  const st = statSync(cfg.data);
  if (existsSync(scanCacheFile)) {
    try {
      const c = JSON.parse(readFileSync(scanCacheFile, 'utf8'));
      if (c.file === cfg.data && c.size === st.size && c.mtimeMs === st.mtimeMs) {
        for (const [t, n] of Object.entries(c.tagCounts)) tagCounts.set(t, n);
        ({ totalLines, noV, legacyNoTag } = c);
        console.log(`Using cached dataset scan (${totalLines} lines, ${tagCounts.size} tags).`);
        return;
      }
    } catch { /* stale/unreadable cache -> rescan */ }
  }
  console.log(`\nScanning ${cfg.data} for vs tags...`);
  try {
    const rl = createInterface({ input: createReadStream(cfg.data), crlfDelay: Infinity });
    for await (const line of rl) {
      if (!line) continue;
      if (line.includes('"moves":')) {
        let rec = null; try { rec = JSON.parse(line); } catch { /* half-written */ }
        if (rec && Array.isArray(rec.moves)) { const { positions, missing } = tallyVs(rec, tagCounts); totalLines += positions; noV += missing; continue; }
      }
      totalLines++;
      if (!line.includes('"v":')) { noV++; continue; }
      const m = line.match(/"vs":"([^"]+)"/);
      if (!m) { legacyNoTag++; continue; }
      tagCounts.set(m[1], (tagCounts.get(m[1]) || 0) + 1);
    }
    writeFileSync(scanCacheFile, JSON.stringify({ file: cfg.data, size: st.size, mtimeMs: st.mtimeMs, totalLines, noV, legacyNoTag, tagCounts: Object.fromEntries(tagCounts) }));
  } catch (e) { console.warn(`  dataset scan interrupted (${e.message}); cross-reference is partial.`); }
}
// --- corpus cross-reference (--corpus): the whole dataset as a pairwise W/D/L matrix ----
// Every game record stores who played (players.w/.b) and its White-view result, so the entire
// self-play corpus is a body of games among the ranked nodes. We fold those into the fit
// (corpusPairs, summed on top of the persisted pool store) so ranking uses far more evidence
// than the pool's own dedicated matchups. Self-play games (same engine both sides) carry no
// relative-ranking signal and are skipped; legacy games without players are ignored.
const corpusCacheFile = join(loopDir, 'ladder-corpus-cache.json');
async function scanCorpus() {
  if (!cfg.corpus) return;
  if (!existsSync(cfg.data)) { console.warn(`(no dataset at ${cfg.data} — --corpus has nothing to fold in)`); return; }
  const st = statSync(cfg.data);
  if (existsSync(corpusCacheFile)) {
    try {
      const c = JSON.parse(readFileSync(corpusCacheFile, 'utf8'));
      if (c.file === cfg.data && c.size === st.size && c.mtimeMs === st.mtimeMs) {
        for (const [k, v] of Object.entries(c.pairs)) corpusPairs.set(k, v);
        corpusGames = c.corpusGames;
        console.log(`Using cached corpus scan (${corpusGames} games, ${corpusPairs.size} pairs).`);
        return;
      }
    } catch { /* stale/unreadable -> rescan */ }
  }
  console.log(`\nScanning ${cfg.data} for game results (--corpus)...`);
  let skippedSelf = 0, noPlayers = 0;
  try {
    const rl = createInterface({ input: createReadStream(cfg.data), crlfDelay: Infinity });
    for await (const line of rl) {
      if (!line) continue;
      let rec; try { rec = JSON.parse(line); } catch { continue; }
      if (!isGameRecord(rec) || !rec.players) { noPlayers++; continue; }
      const a = rec.players.w, b = rec.players.b;
      if (!a || !b || a === '?' || b === '?') { noPlayers++; continue; }
      if (a === b) { skippedSelf++; continue; } // self-play: uninformative for relative ranking
      const pointsW = rec.r > 0 ? 1 : (rec.r < 0 ? 0 : 0.5);
      const key = pairKey(a, b);
      const e = corpusPairs.get(key) || { games: 0, sumA: 0 };
      e.games += 1;
      e.sumA += (a < b) ? pointsW : (1 - pointsW); // sumA tracks the lexically-first id
      corpusPairs.set(key, e);
      corpusGames++;
    }
    writeFileSync(corpusCacheFile, JSON.stringify({ file: cfg.data, size: st.size, mtimeMs: st.mtimeMs, corpusGames, pairs: Object.fromEntries(corpusPairs) }));
  } catch (e) { console.warn(`  corpus scan interrupted (${e.message}); partial.`); }
  console.log(`  corpus: ${corpusGames} mixed-engine games -> ${corpusPairs.size} pair(s) folded into the fit `
    + `(${skippedSelf} self-play skipped).`);
}
const parseTag = (tag) => { const m = /^(nn|hc)(\d+|t)@(.+)$/.exec(tag); return m ? { eng: m[1], depth: m[2], version: m[3] } : null; };

const median = (xs) => { if (!xs.length) return null; const s = [...xs].sort((a, b) => a - b); const m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };

// Is the pool actually trustworthy yet? Two orthogonal signals that "100 games each" misses:
//  1) RESOLUTION — median ±95 margin vs the median gap between rank-adjacent engines. While the
//     bootstrap pairs leave the graph under-connected, margins (±600) dwarf the gaps they must
//     distinguish, so the whole order sits inside the noise. Converged ⇒ ratio drops toward ~1.
//  2) DEPTH ORDER — deeper search MUST be monotonically stronger, so each engine's depth curve is
//     a free ground truth. Strict inversions (a deeper node estimated weaker) are the visible
//     non-monotonicity; a "confident" inversion (gap exceeds the pair's combined CI) is a real
//     transitivity/bug alarm that should never survive convergence.
function convergenceReport(elo, ci) {
  const sorted = [...competitors].sort((a, b) => elo.get(a.id) - elo.get(b.id));
  const gaps = [];
  for (let i = 1; i < sorted.length; i++) gaps.push(elo.get(sorted[i].id) - elo.get(sorted[i - 1].id));
  const margins = competitors.map((c) => ci.get(c.id)).filter((m) => m != null && m > 0 && isFinite(m));
  const medMargin = median(margins), medGap = median(gaps);
  const ratio = (medMargin != null && medGap > 0) ? medMargin / medGap : null;

  // Per-version depth curves (need ≥2 depths to say anything about monotonicity).
  const byVersion = new Map();
  for (const c of competitors) { const k = `${c.eng}@${c.version}`; (byVersion.get(k) || byVersion.set(k, []).get(k)).push(c); }
  const curves = [];
  for (const [k, nodes] of byVersion) {
    if (nodes.length < 2) continue;
    nodes.sort((a, b) => a.depth - b.depth);
    let inv = 0, confInv = 0, worst = 0;
    for (let i = 0; i < nodes.length; i++) for (let j = i + 1; j < nodes.length; j++) {
      const drop = elo.get(nodes[i].id) - elo.get(nodes[j].id); // shallower minus deeper; >0 = inverted
      if (drop > 1e-9) {
        inv++; worst = Math.max(worst, drop);
        const comb = Math.hypot(ci.get(nodes[i].id) || 0, ci.get(nodes[j].id) || 0);
        if (drop > comb) confInv++;
      }
    }
    curves.push({ k, nodes, inv, confInv, worst });
  }
  const monotonic = curves.filter((c) => c.inv === 0).length;
  const confInvTotal = curves.reduce((s, c) => s + c.confInv, 0);
  const nonMono = curves.filter((c) => c.inv > 0).sort((a, b) => b.worst - a.worst);

  const resolved = ratio != null && ratio < 1.5;
  const ordered = confInvTotal === 0;
  let verdict;
  if (!resolved) verdict = `NOT converged — under-resolved (ratio ${ratio == null ? 'n/a' : ratio.toFixed(1)} ≥ 1.5). Keep running.`;
  else if (!ordered) verdict = `RESOLVED but ${confInvTotal} confident depth inversion(s) — possible non-transitivity/bug, inspect.`;
  else verdict = `converged ✓ — margins resolved and every depth curve monotonic.`;

  return {
    summary: { medMargin, medGap, ratio, versionsMonotonic: monotonic, versionsWithDepthCurve: curves.length, confidentInversions: confInvTotal, resolved, ordered, converged: resolved && ordered, verdict },
    nonMono, elo,
  };
}

function printConvergence(rep) {
  const s = rep.summary;
  console.log(`\n===== Convergence check =====`);
  console.log(`  resolution:  median ±95 = ${s.medMargin == null ? 'n/a' : s.medMargin.toFixed(0)}  |  median neighbor gap = ${s.medGap == null ? 'n/a' : s.medGap.toFixed(0)}  |  ratio ${s.ratio == null ? 'n/a' : s.ratio.toFixed(1)}  (want < ~1.5)`);
  console.log(`  depth order: ${s.versionsMonotonic}/${s.versionsWithDepthCurve} versions monotonic  |  ${s.confidentInversions} confident inversion(s)`);
  for (const c of rep.nonMono.slice(0, 4)) {
    const seq = c.nodes.map((n) => `d${n.depth}=${rep.elo.get(n.id) >= 0 ? '+' : ''}${rep.elo.get(n.id).toFixed(0)}`).join(' ');
    const nm = niceName(c.k.split('@')[1]);
    console.log(`    ${(nm ? `${c.k} (${nm})` : c.k).padEnd(20)} ${seq}   (${c.inv} inv${c.confInv ? `, ${c.confInv} confident` : ''})`);
  }
  console.log(`  verdict: ${s.verdict}`);
}

// Build + write the ledger (the engine-elo.*.json refresh-v/merge consume). Callable
// periodically (verbose=false) during a long run and once at the end (verbose=true), so the
// ledger is always reasonably fresh even if the run is killed hard.
function writeRankLedger(verbose) {
  const { elo, ci, gamesOf } = fit();
  const ranked = [...competitors].sort((a, b) => elo.get(a.id) - elo.get(b.id));
  const recordsByVersion = new Map();
  for (const [tag, n] of tagCounts) { const t = parseTag(tag); if (!t) continue; const k = `${t.eng}@${t.version}`; recordsByVersion.set(k, (recordsByVersion.get(k) || 0) + n); }
  const ranking = ranked.map((c) => ({
    tag: c.id, eng: c.eng, version: c.version, name: niceName(c.version), depth: String(c.depth),
    anchor: c.id === pinId, elo: elo.get(c.id), score: null,
    margin: ci.get(c.id) ?? null, games: gamesOf(c.id),
    records: recordsByVersion.get(`${c.eng}@${c.version}`) || 0, recoverable: true, file: c.weights,
  }));
  const unrecoverable = [];
  if (cfg.scan) {
    for (const [tag, n] of tagCounts) {
      const t = parseTag(tag);
      if (!t) { unrecoverable.push({ tag, records: n, reason: 'malformed tag' }); continue; }
      if (/^elo-?\d+$/.test(t.version)) continue;       // ephemeral candidate tag — self-describing
      if (byId.has(tag)) continue;                       // ranked node at this exact depth
      if (competitors.some((c) => c.eng === t.eng && c.version === t.version)) continue; // engine ranked (other depth)
      let reason;
      if (t.version === '?') reason = 'material fallback (drop --no-material to rank it)';
      else if (t.eng === 'hc') reason = `old handcrafted (HC_VERSION now ${HC_VERSION})`;
      else reason = 'champion not archived (overwritten)';
      unrecoverable.push({ tag, records: n, reason });
    }
    unrecoverable.sort((a, b) => b.records - a.records);
  }
  const conv = convergenceReport(elo, ci);
  const ledger = {
    generated: new Date().toISOString(), anchor: pinId, method: 'bradley-terry-pool',
    depths: cfg.depths, games: cfg.games, seed: cfg.seed,
    dataset: cfg.scan ? { file: cfg.data, totalLines, withV: totalLines - noV, legacyNoTag } : null,
    convergence: conv.summary,
    ranking, unrecoverable,
  };
  mkdirSync(dirname(cfg.ledger), { recursive: true });
  writeFileSync(cfg.ledger, JSON.stringify(ledger, null, 2) + '\n');
  if (!verbose) return;
  console.log(`\n===== Elo ladder (${pinId} := 0) — weakest first =====`);
  console.log(`  ${'#'.padStart(3)} ${'engine'.padEnd(16)} ${'name'.padEnd(12)} ${'Elo'.padStart(8)} ${'±95'.padStart(6)} ${'games'.padStart(7)}`);
  for (const [i, c] of ranked.entries()) {
    const rank = ranked.length - i; // ranked is weakest-first, so #1 = strongest
    console.log(`  ${String(rank).padStart(3)} ${c.id.padEnd(16)} ${(niceName(c.version) || '').padEnd(12)} ${(`${elo.get(c.id) >= 0 ? '+' : ''}${elo.get(c.id).toFixed(0)}`).padStart(8)} ${(ci.get(c.id) == null ? '' : ci.get(c.id).toFixed(0)).padStart(6)} ${String(gamesOf(c.id)).padStart(7)}${c.id === pinId ? '  (pin)' : ''}`);
  }
  if (unrecoverable.length) {
    console.log(`\n  Unrecoverable contributors (no Elo -> weakest, refresh on sight):`);
    for (const u of unrecoverable.slice(0, 12)) console.log(`    ${u.tag.padEnd(16)} ${String(u.records).padStart(10)} records  — ${u.reason}`);
  }
  printConvergence(conv);
  // Optional flat CSV (engine,version,depth,elo,ci95,games) for plotting — a derived view of
  // the ledger (e.g. filter to one version for that net's depth curve), not a separate artifact.
  if (cfg.csv) {
    const rows = ['engine,version,depth,elo,ci95,games'];
    for (const c of [...competitors].sort((a, b) => a.eng.localeCompare(b.eng) || a.version.localeCompare(b.version) || a.depth - b.depth))
      rows.push([`${c.eng}${c.depth}`, c.version, c.depth, elo.get(c.id).toFixed(1), (ci.get(c.id) ?? '').toString(), gamesOf(c.id)].join(','));
    writeFileSync(cfg.csv, rows.join('\n') + '\n');
    console.log(`\nCSV -> ${cfg.csv}`);
  }
  console.log(`\nLedger -> ${cfg.ledger}  (store ${cfg.store}; re-run to add games and tighten all ratings)`);
  console.log('This is the strength ledger refresh-v/merge read for weakest-first v refresh (they default to engine-elo.ladder.json).');
}

console.log(`Engine ranking pool (active scheduler)`);
console.log(`  ${competitors.length} node(s): ${competitors.map((c) => `${c.id.split('@')[0]}@${c.version.slice(0, 6)}${niceName(c.version) ? ` (${niceName(c.version)})` : ''}`).join(', ')}`);
console.log(`  pin ${pinId} | ${cfg.games} games/matchup | onboard ${cfg.onboard ? `${cfg.onboard}×avg` : 'off'} | ${cfg.jobs} parallel job(s) | store ${cfg.store}`);
if (playMatch) console.log(`  --play: new games only among ${schedulable.map(nodeLabel).join(', ')} (rest rated from existing data)`);
console.log(`  games -> ${cfg.saveGames || '(not harvested; --no-save-games)'}`);
if (cfg.rounds !== 0) await scanDataset(); // once up front, so the periodic ledger emit has record counts
await scanCorpus(); // fold the dataset's game results into the fit (no-op unless --corpus)

if (cfg.rounds === 0) {
  // Offline: no matches — just persist the (possibly merged) store, refit, and emit.
  if (cfg.merge.length) writeFileSync(cfg.store, JSON.stringify(store, null, 2) + '\n');
  await scanDataset();
  console.log('rounds=0: merge/refit only, no games played.');
} else {
  // Active scheduler: refit -> pick the most ordering-ambiguous matchup -> play -> persist ->
  // repeat, until --minutes / --matchups / the user stops it. The store is written after every
  // matchup (overnight stop/start is lossless — a restart resumes from it), and the ledger is
  // re-emitted every few matchups so it's never far behind even on a hard kill.
  const EMIT_EVERY = 5;
  buildEngine();
  rmSync(stopFile, { force: true }); // clear any stale stop-file from a prior run
  printStopHint();
  let stopped = false;
  // Stop right away while keeping finished games: drop the stop-file so the in-flight matchup
  // saves its completed games and exits immediately — pressing q/Ctrl-C no longer waits out the
  // whole matchup, nor throws away the games already played.
  const stopper = installStop(() => {
    if (!stopped) { stopped = true; console.log('\n  Stopping — saving completed games, abandoning those in flight…'); }
    requestMatchStop();
  });
  const t0 = Date.now();
  const deadline = cfg.minutes ? t0 + cfg.minutes * 60000 : Infinity;
  let played = 0;
  while (!stopped && Date.now() < deadline && played < cfg.matchups) {
    const { elo, ci, varDiff, gamesOf } = fit();
    const { pair, reason, metric, floor } = pickMatchup(elo, varDiff, played, gamesOf);
    if (!pair) break;
    const [a, b] = pair;
    const tag = `[${played + 1}${Number.isFinite(cfg.matchups) ? `/${cfg.matchups}` : ''}]`;
    const why = reason === 'ordering' ? `ordering: P(mis-order) ${(metric * 100).toFixed(0)}%`
      : reason === 'onboard' ? `onboard: ${metric} game(s), below floor ${floor.toFixed(0)} (${cfg.onboard}×pool avg)`
      : `rigidity: ±${metric.toFixed(0)} Elo`;
    // Announce with each node's CURRENT fitted Elo ±95 (the ledger's real estimate), so the
    // matchup reads on the stable hc scale up front — the match runner's own live Elo is only
    // this matchup's games and always starts at 0 with a huge margin.
    const lbl = (c) => `${nodeLabel(c)} ${fmtSigned(elo.get(c.id))} ±${(ci.get(c.id) ?? 0).toFixed(0)}`;
    const pA = 1 / (1 + 10 ** ((elo.get(b.id) - elo.get(a.id)) / 400));
    console.log(`\n${tag} ${why} -> ${lbl(a)}  vs  ${lbl(b)}  (ledger expects A ${(pA * 100).toFixed(0)}%)`);
    await playPair(a, b);
    // playPair recorded this matchup — a full run, or the completed games if a stop landed
    // mid-matchup. Persist the store either way so the games already played are never lost.
    writeFileSync(cfg.store, JSON.stringify(store, null, 2) + '\n');
    { // refit so the pair's post-matchup ratings show right away (cheap next to the games)
      const { elo: e2, ci: c2 } = fit();
      const upd = (c) => `${nodeLabel(c)} ${fmtSigned(e2.get(c.id))} ±${(c2.get(c.id) ?? 0).toFixed(0)}`;
      console.log(`  ledger now: ${upd(a)}  |  ${upd(b)}`);
    }
    played++;
    if (played % EMIT_EVERY === 0) writeRankLedger(false);
    if (stopped) break; // stop requested: the in-flight matchup drained + was saved; finish up.
  }
  stopper.dispose();
  rmSync(stopFile, { force: true }); // don't leave a stop-file that would halt the next run
  console.log(`\nActive ranking: ${played} matchup(s) in ${((Date.now() - t0) / 60000).toFixed(1)} min` +
    `${stopped ? ' (stopped)' : cfg.minutes && Date.now() >= deadline ? ' (time budget reached)' : ''}.`);
}

writeRankLedger(true);
