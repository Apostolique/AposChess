// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2019-2026 Jean-David Moisan
//
// Self-relative Elo for an Elo-vs-search-depth curve (Bradley-Terry / BayesElo-style).
//
// The anchor approach (rank-engines.mjs / depth-sweep.mjs) measures every engine against
// ONE fixed reference. That floors the far end of a wide range: a contender >~400 Elo from
// the anchor scores ~0%/100%, which carries no information no matter how many games you
// play. The fix is to stop measuring against a fixed point and instead let competitors
// play EACH OTHER near their own strength, then recover all ratings with one joint fit.
// Every node only ever plays near-strength opponents (scores stay in the informative
// 30-70% band), and the whole range is stitched onto one scale by transitivity. As games
// accumulate, EVERY rating tightens — not just one matchup. The anchor, if present at all,
// is just another competitor; with none, the scale is mean-centered.
//
// Competitors here are the SAME net at several depths (apos-match plays one weights file
// against itself via --depth/--depth-b), optionally plus stable hc@D references. The store
// persists pairwise results, so re-running ACCUMULATES games and re-fits everything.
//
// Method: Bradley-Terry, fit by the MM algorithm (Hunter 2004) — monotone, no matrix
// inversion. Draws score as half a point each side (consistent with the codebase's
// eloFromScore logistic model). A mild prior (virtual draws vs an even phantom) keeps
// early/sparse rounds from diverging. Per-node 95% CIs come from the Fisher information.
//
// Usage (run from web/):
//   node scripts/depth-ladder.mjs [options]
//   npm run rank:ladder -- [options]
//
// Two modes:
//   (default) DEPTH CURVE — one net at several depths -> depth,elo CSV.
//   --rank    ENGINE RANKING — all instantiable engines (hc + champion + archived
//             champions [+ material]) at --depths, pinned to hc<anchor-depth>, emitting the
//             same ledger schema as rank-engines.mjs (engine-elo.json) so refresh-v/merge
//             can read it. The pool replacement for the anchor gauntlet: champions play
//             near-strength neighbors instead of a now-saturating single anchor.
//
// Options:
//   --rank          engine-ranking mode (see above). Default off (depth-curve mode).
//   --net=SPEC      (depth-curve) net to curve: 'champion' (default = src/nn-weights.json),
//                   a champion content hash, or an archived champion filename/path.
//   --depths=LIST   depths to rate: range (2-10) or list (2,4,6,8). Default 2-10 (curve), 6,8 (--rank).
//   --hc=LIST       (depth-curve) also include handcrafted references at these depths.
//   --anchor-depth=D (--rank) hc depth pinned to Elo 0 (default 6) — must be in --depths.
//   --no-material   (--rank) skip the nn material fallback node (it's the floor reference).
//   --pin=ID        fix this competitor at Elo 0 (e.g. hc6). Default: hc<anchor-depth> in
//                   --rank, else the deepest hc node, else mean-centered.
//   --rounds=N      scheduling rounds per invocation (re-fit + re-pair between each). Default 1.
//                   0 = play nothing; just merge/refit the existing store and emit (offline).
//   --merge=F[,F]   fold pool stores from other machines into this one before fitting (pairwise
//                   counts are additive). Requires the SAME node set and DISTINCT --seed per
//                   machine. Combine with --rounds=0 for a pure offline merge+refit.
//   --games=N       games per matchup per round (even; default 100). Re-running accumulates.
//   --prior=P       virtual draws vs an even phantom, per node (regularizer). Default 1.
//   --jobs=N        parallel game workers (default: CPU cores).
//   --openings=K    random opening plies per game (default 6).
//   --maxmoves=N    draw adjudication ply cap (default 200).
//   --seed=S        base seed (default 1). The store advances a seed cursor so accumulated
//                   re-runs never replay identical games.
//   --store=FILE    pairwise-results store (default loop/depth-ladder.json | loop/ladder-pool.json).
//   --ledger=FILE   (--rank) ledger output (default loop/engine-elo.ladder.json — a PARALLEL
//                   ledger, NOT the live engine-elo.json, so it can be A/B'd safely).
//   --data=FILE     (--rank) dataset to scan for record counts (default selfplay.jsonl). --no-scan skips.
//   --csv=FILE      (depth-curve) CSV output (default loop/depth-ladder.csv).
//   --save-games[=F]  harvest the games as training data (default OFF).
//   --fresh         discard any existing store and start the pool from scratch.

import { spawnSync } from 'node:child_process';
import { existsSync, readFileSync, writeFileSync, mkdirSync, rmSync, readdirSync, createReadStream } from 'node:fs';
import { createInterface } from 'node:readline';
import { resolve, dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { cpus } from 'node:os';

import { weightsHash } from './vtag.mjs';
import { HC_VERSION } from '../src/ai.js';

const here = dirname(fileURLToPath(import.meta.url));
const webDir = resolve(here, '..');
const repoDir = resolve(webDir, '..');
const dataDir = resolve(repoDir, 'training', 'data');
const loopDir = join(dataDir, 'loop');
const championsDir = join(loopDir, 'champions');
const champion = resolve(webDir, 'src', 'nn-weights.json');
const engineDir = resolve(webDir, 'engine');
const matchBin = resolve(engineDir, 'zig-out', 'bin', process.platform === 'win32' ? 'apos-match.exe' : 'apos-match');

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

// Two modes share one pool/fit core:
//   default (depth-curve): nodes = ONE net at several --depths.
//   --rank: nodes = ALL instantiable engines (hc + champion + archived champions
//           [+ material]) at --depths — the ledger-emitting replacement for the anchor
//           gauntlet (rank-engines.mjs), pinned to hc<anchor-depth> for a stable scale.
const RANK_MODE = !!args.rank;
const cfg = {
  net: typeof args.net === 'string' ? args.net : 'champion',
  depths: parseDepths(args.depths, RANK_MODE ? [6, 8] : [2, 3, 4, 5, 6, 7, 8, 9, 10]),
  hc: parseDepths(args.hc, []),
  pin: typeof args.pin === 'string' ? args.pin : null,
  anchorDepth: num(args['anchor-depth'], 6), // hc depth pinned to 0 in --rank mode
  material: RANK_MODE && !args['no-material'],
  // Pool stores from other machines to fold in before fitting (pairwise counts are additive).
  // Each MUST come from a distinct --seed (else identical games would be double-counted).
  merge: typeof args.merge === 'string' ? args.merge.split(',').map((s) => resolve(process.cwd(), s.trim())).filter(Boolean) : [],
  rounds: Math.max(0, num(args.rounds, 1)), // 0 = don't play; just merge/refit and emit (offline)

  games: Math.max(2, Math.round(num(args.games, 100) / 2) * 2),
  prior: num(args.prior, 1),
  jobs: args.jobs !== undefined ? Number(args.jobs) : cpus().length,
  openings: num(args.openings, 6),
  maxmoves: num(args.maxmoves, 200),
  seed: num(args.seed, 1),
  scan: RANK_MODE && !args['no-scan'],
  data: typeof args.data === 'string' ? resolve(process.cwd(), args.data) : join(dataDir, 'selfplay.jsonl'),
  store: typeof args.store === 'string' ? resolve(process.cwd(), args.store)
    : join(loopDir, RANK_MODE ? 'ladder-pool.json' : 'depth-ladder.json'),
  ledger: typeof args.ledger === 'string' ? resolve(process.cwd(), args.ledger) : join(loopDir, 'engine-elo.ladder.json'),
  csv: typeof args.csv === 'string' ? resolve(process.cwd(), args.csv) : join(loopDir, 'depth-ladder.csv'),
  saveGames: args['save-games'],
  fresh: !!args.fresh,
};

// --- resolve the net to a weights file + version hash (depth-curve mode) --------
function resolveNet(spec) {
  if (spec === 'champion') {
    if (!existsSync(champion)) { console.error(`No champion at ${champion}.`); process.exit(1); }
    return { file: champion, version: weightsHash(champion) };
  }
  const cand = [
    join(championsDir, spec.endsWith('.json') ? spec : `${spec}.json`),
    resolve(process.cwd(), spec),
  ].find(existsSync);
  if (!cand) { console.error(`Could not find net '${spec}' (champion archive or path).`); process.exit(1); }
  return { file: cand, version: weightsHash(cand) };
}

// --- enumerate the instantiable engines (--rank mode) --------------------------
// Mirrors rank-engines.mjs: handcrafted, the current champion, every archived champion,
// and (optionally) the nn material fallback. Deduped by eng@version.
function instantiableEngines() {
  const list = [{ eng: 'hc', eval: 'handcrafted', weights: null, version: String(HC_VERSION) }];
  if (existsSync(champion)) list.push({ eng: 'nn', eval: 'nn', weights: champion, version: weightsHash(champion) });
  else console.warn(`(no champion at ${champion} — skipping it)`);
  if (existsSync(championsDir)) {
    for (const f of readdirSync(championsDir).filter((f) => f.endsWith('.json'))) {
      const file = join(championsDir, f);
      list.push({ eng: 'nn', eval: 'nn', weights: file, version: weightsHash(file) });
    }
  }
  if (cfg.material) list.push({ eng: 'nn', eval: 'nn', weights: null, version: '?' }); // material fallback (floor)
  const seen = new Set();
  return list.filter((e) => { const k = `${e.eng}@${e.version}`; if (seen.has(k)) return false; seen.add(k); return true; });
}

let netFile, version;
if (!RANK_MODE) ({ file: netFile, version } = resolveNet(cfg.net));

// --- competitors (nodes of the rating pool) ------------------------------------
// Each node is an (engine, depth) pair; id is its `vs`-tag (stable across runs, the store
// key AND the ledger tag): nn<d>@<ver> / hc<d>@<HC_VERSION> / nn<d>@? (material).
const competitors = [];
if (RANK_MODE) {
  for (const e of instantiableEngines())
    for (const d of cfg.depths) competitors.push({ id: `${e.eng}${d}@${e.version}`, eng: e.eng, eval: e.eval, weights: e.weights, version: e.version, depth: d });
} else {
  for (const d of cfg.depths) competitors.push({ id: `nn${d}@${version}`, eng: 'nn', eval: 'nn', weights: netFile, version, depth: d });
  for (const d of cfg.hc) competitors.push({ id: `hc${d}@${HC_VERSION}`, eng: 'hc', eval: 'handcrafted', weights: null, version: String(HC_VERSION), depth: d });
}
if (competitors.length < 2) { console.error('Need at least 2 competitors.'); process.exit(1); }
const byId = new Map(competitors.map((c) => [c.id, c]));

// pin target (Elo := 0). --rank: hc at --anchor-depth (the stable yardstick, reproducing
// the gauntlet's hc<anchorDepth>@<HC_VERSION> anchor so the scale stays comparable). Else
// explicit --pin, else the deepest hc node, else null (mean-centered).
let pinId = cfg.pin;
if (pinId && !byId.has(pinId)) { console.error(`--pin '${pinId}' is not a competitor. Have: ${[...byId.keys()].join(', ')}`); process.exit(1); }
if (!pinId && RANK_MODE) {
  pinId = `hc${cfg.anchorDepth}@${HC_VERSION}`;
  if (!byId.has(pinId)) { console.error(`--rank pins ${pinId} but it isn't a node; include depth ${cfg.anchorDepth} in --depths (or set --anchor-depth/--pin).`); process.exit(1); }
}
if (!pinId) { const hcNodes = competitors.filter((c) => c.eng === 'hc'); if (hcNodes.length) pinId = hcNodes.sort((a, b) => b.depth - a.depth)[0].id; }

// --- persistent pairwise store -------------------------------------------------
// pairs: { "idA|idB" (sorted): { games, sumA } } where sumA = points scored by idA.
// `pool` identifies the node set so a config change doesn't silently mix incompatible
// pools; in --rank mode the pool spans many engines (keyed 'rank'), not one net.
mkdirSync(loopDir, { recursive: true });
const poolId = RANK_MODE ? 'rank' : version;
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
  for (const [key, v] of Object.entries(store.pairs)) {
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
  // 95% CI from Fisher information in beta = ln(gamma): Var(beta_i) ~ 1/I_i (diagonal approx).
  const ci = new Map();
  for (const id of ids) {
    let I = cfg.prior * 0.25; // phantom contributes p0(1-p0)=0.25 at parity
    for (const { opp, N } of adj.get(id)) { const p = gamma.get(id) / (gamma.get(id) + gamma.get(opp)); I += N * p * (1 - p); }
    ci.set(id, I > 0 ? 1.96 * (400 / Math.LN10) / Math.sqrt(I) : null);
  }
  return { elo, ci, gamesOf: (id) => adj.get(id).reduce((s, e) => s + e.N, 0) };
}

// --- one matchup on apos-match -------------------------------------------------
function buildEngine() {
  const r = spawnSync('zig build -Doptimize=ReleaseFast', { cwd: engineDir, stdio: 'inherit', shell: true });
  if (r.status !== 0) { console.error('zig build failed (is Zig 0.16 on PATH?).'); process.exit(1); }
}
function playPair(a, b) {
  const tmp = join(loopDir, 'ladder-match.json');
  const seed = store.seedCursor;
  store.seedCursor += cfg.games; // advance so accumulated re-runs use fresh openings
  const argv = [
    `--games=${cfg.games}`, `--jobs=${cfg.jobs}`, `--openings=${cfg.openings}`, `--maxmoves=${cfg.maxmoves}`,
    `--seed=${seed}`,
    `--eval-a=${a.eval}`, `--eval-b=${b.eval}`, `--depth=${a.depth}`, `--depth-b=${b.depth}`,
    `--result-file=${tmp}`,
  ];
  if (a.eval === 'nn' && a.weights) argv.push(`--weights-a=${a.weights}`);
  if (b.eval === 'nn' && b.weights) argv.push(`--weights-b=${b.weights}`);
  if (cfg.saveGames === true) argv.push('--save-games');
  else if (typeof cfg.saveGames === 'string') argv.push(`--save-games=${cfg.saveGames}`);
  console.log(`\n=== ${a.id}  vs  ${b.id}  (${cfg.games} games) ===`);
  const r = spawnSync(matchBin, argv, { stdio: 'inherit', cwd: webDir, env: { ...process.env, APOS_CHILD: '1' } });
  if (r.status !== 0) { console.error(`  match failed (exit ${r.status}); skipping.`); return; }
  try { const res = JSON.parse(readFileSync(tmp, 'utf8')); rmSync(tmp, { force: true }); record(a.id, b.id, res.games, res.score); }
  catch (e) { console.error(`  could not read result: ${e.message}`); }
}

// --- round scheduling: pair adjacent-in-current-rating + sparse skip links -----
function roundPairs(elo) {
  const sorted = [...competitors].sort((a, b) => (elo.get(a.id) ?? a.depth * 100) - (elo.get(b.id) ?? b.depth * 100));
  const pairs = [];
  for (let k = 0; k < sorted.length - 1; k++) pairs.push([sorted[k], sorted[k + 1]]); // near-strength (high info)
  for (let k = 0; k + 2 < sorted.length; k += 2) pairs.push([sorted[k], sorted[k + 2]]); // rigidity cross-links
  return pairs;
}

// --- dataset cross-reference (--rank): record counts + unrecoverable tags ------
// Same read-only regex scan as rank-engines.mjs, so the emitted ledger carries the same
// `records` / `unrecoverable` fields the report and downstream tooling expect.
const tagCounts = new Map();
let totalLines = 0, noV = 0, legacyNoTag = 0;
async function scanDataset() {
  if (!cfg.scan || !existsSync(cfg.data)) { if (cfg.scan) console.warn(`(no dataset at ${cfg.data} — skipping cross-reference)`); return; }
  console.log(`\nScanning ${cfg.data} for vs tags...`);
  try {
    const rl = createInterface({ input: createReadStream(cfg.data), crlfDelay: Infinity });
    for await (const line of rl) {
      if (!line) continue;
      totalLines++;
      if (!line.includes('"v":')) { noV++; continue; }
      const m = line.match(/"vs":"([^"]+)"/);
      if (!m) { legacyNoTag++; continue; }
      tagCounts.set(m[1], (tagCounts.get(m[1]) || 0) + 1);
    }
  } catch (e) { console.warn(`  dataset scan interrupted (${e.message}); cross-reference is partial.`); }
}
const parseTag = (tag) => { const m = /^(nn|hc)(\d+|t)@(.+)$/.exec(tag); return m ? { eng: m[1], depth: m[2], version: m[3] } : null; };

console.log(`${RANK_MODE ? 'Engine ranking pool' : `Depth ladder: net nn@${version}`}`);
console.log(`  ${competitors.length} node(s): ${competitors.map((c) => c.id.split('@')[0] + (RANK_MODE ? `@${c.version.slice(0, 6)}` : '')).join(', ')}`);
console.log(`  pin ${pinId || '(mean-centered)'} | ${cfg.rounds} round(s) x ${cfg.games} games/matchup | store ${cfg.store}`);

if (cfg.rounds === 0) {
  // Offline: no matches — just persist the (possibly merged) store, refit, and emit.
  if (cfg.merge.length) writeFileSync(cfg.store, JSON.stringify(store, null, 2) + '\n');
  console.log('rounds=0: merge/refit only, no games played.');
} else {
  buildEngine();
  for (let round = 1; round <= cfg.rounds; round++) {
    const { elo } = fit();
    const pairs = roundPairs(elo);
    console.log(`\n----- round ${round}/${cfg.rounds}: ${pairs.length} matchups -----`);
    for (const [a, b] of pairs) playPair(a, b);
    writeFileSync(cfg.store, JSON.stringify(store, null, 2) + '\n'); // persist after each round
  }
}
await scanDataset();

// --- final fit + report --------------------------------------------------------
const { elo, ci, gamesOf } = fit();
const ranked = [...competitors].sort((a, b) => elo.get(a.id) - elo.get(b.id));
console.log(`\n===== Elo ladder (${pinId ? `${pinId} := 0` : 'mean-centered'}) — weakest first =====`);
console.log(`  ${'engine'.padEnd(16)} ${'Elo'.padStart(8)} ${'±95'.padStart(6)} ${'games'.padStart(7)}`);
for (const c of ranked) {
  const e = elo.get(c.id), m = ci.get(c.id);
  const label = RANK_MODE ? c.id : c.id.split('@')[0];
  console.log(`  ${label.padEnd(16)} ${(`${e >= 0 ? '+' : ''}${e.toFixed(0)}`).padStart(8)} ${(m == null ? '' : m.toFixed(0)).padStart(6)} ${String(gamesOf(c.id)).padStart(7)}${c.id === pinId ? '  (pin)' : ''}`);
}

if (RANK_MODE) {
  // --- emit the ledger schema (drop-in for rank-engines.mjs's engine-elo.json) --
  // recordsByVersion: dataset records attributed to each eng@version (any depth).
  const recordsByVersion = new Map();
  for (const [tag, n] of tagCounts) { const t = parseTag(tag); if (!t) continue; const k = `${t.eng}@${t.version}`; recordsByVersion.set(k, (recordsByVersion.get(k) || 0) + n); }
  const ranking = ranked.map((c) => ({
    tag: c.id, eng: c.eng, version: c.version, depth: String(c.depth),
    anchor: c.id === pinId, elo: elo.get(c.id), score: null,
    margin: ci.get(c.id) ?? null, games: gamesOf(c.id),
    records: recordsByVersion.get(`${c.eng}@${c.version}`) || 0, recoverable: true, file: c.weights,
  }));
  // Dataset `vs` tags whose engine we couldn't instantiate -> unrecoverable (weakest).
  const unrecoverable = [];
  if (cfg.scan) {
    for (const [tag, n] of tagCounts) {
      const t = parseTag(tag);
      if (!t) { unrecoverable.push({ tag, records: n, reason: 'malformed tag' }); continue; }
      if (/^elo-?\d+$/.test(t.version)) continue;       // ephemeral candidate tag — self-describing
      if (byId.has(tag)) continue;                       // ranked node at this exact depth
      if (competitors.some((c) => c.eng === t.eng && c.version === t.version)) continue; // engine ranked (other depth)
      let reason;
      if (t.eng === 'hc') reason = `old handcrafted (HC_VERSION now ${HC_VERSION})`;
      else if (t.version === '?') reason = 'nn material fallback (drop --no-material to rank it)';
      else reason = 'champion not archived (overwritten)';
      unrecoverable.push({ tag, records: n, reason });
    }
    unrecoverable.sort((a, b) => b.records - a.records);
  }
  const ledger = {
    generated: new Date().toISOString(), anchor: pinId, method: 'bradley-terry-pool',
    depths: cfg.depths, games: cfg.games, seed: cfg.seed,
    dataset: cfg.scan ? { file: cfg.data, totalLines, withV: totalLines - noV, legacyNoTag } : null,
    ranking, unrecoverable,
  };
  mkdirSync(dirname(cfg.ledger), { recursive: true });
  writeFileSync(cfg.ledger, JSON.stringify(ledger, null, 2) + '\n');
  if (unrecoverable.length) {
    console.log(`\n  Unrecoverable contributors (no Elo -> weakest, refresh on sight):`);
    for (const u of unrecoverable.slice(0, 12)) console.log(`    ${u.tag.padEnd(16)} ${String(u.records).padStart(10)} records  — ${u.reason}`);
  }
  console.log(`\nLedger -> ${cfg.ledger}  (store ${cfg.store}; re-run to add games and tighten all ratings)`);
  console.log('This is a parallel ledger — point refresh-v/merge at it with --ledger=engine-elo.ladder.json to A/B it against `npm run rank`.');
} else {
  // --- depth-curve CSV + Δ/ply (one net, sorted by depth) ----------------------
  const nnNodes = competitors.filter((c) => c.eng === 'nn').sort((a, b) => a.depth - b.depth);
  const rows = ['depth,elo,ci95,games'];
  for (const c of nnNodes) rows.push([c.depth, elo.get(c.id).toFixed(1), (ci.get(c.id) ?? '').toString(), gamesOf(c.id)].join(','));
  writeFileSync(cfg.csv, rows.join('\n') + '\n');
  console.log(`\n  depth -> Elo (Δ/ply):`);
  let prev = null;
  for (const c of nnNodes) {
    const e = elo.get(c.id);
    const dpp = prev ? `  (${e - prev.e >= 0 ? '+' : ''}${((e - prev.e) / (c.depth - prev.d)).toFixed(0)}/ply)` : '';
    console.log(`    d${c.depth}: ${e >= 0 ? '+' : ''}${e.toFixed(0)}${dpp}`);
    prev = { e, d: c.depth };
  }
  console.log(`\nCSV -> ${cfg.csv}  (store ${cfg.store}; re-run to add games and tighten all ratings)`);
}
