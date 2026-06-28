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
// Zero-arg `npm run rank:pool` ranks all engines at depths 6,8, pinned hc6, until you stop it.
//
// Options:
//   --engines=SPEC  which engines: 'all' (default = hc + champion + archived champions
//                   [+ material]) or a comma list of specs (content hash/prefix, archived
//                   filename/path, 'champion', 'hc', 'material'). hc<anchor-depth> is added
//                   regardless. --net=X is shorthand for --engines=X (one net's depth sweep).
//   --depths=LIST   depths to rate each engine at: range (1-8) or list (6,8). Default 6,8.
//   --anchor-depth=D  the hc depth that is the pin / Elo 0 (default 6). Always present as a node.
//   --no-material   skip the nn material fallback node (otherwise included as the floor).
//   --minutes=M     play for M minutes, then finalize. Omit to run until you stop it (q/Ctrl-C).
//   --matchups=N    stop after N matchups (default unlimited).
//   --rounds=0      play nothing; just merge/refit the existing store and emit (offline).
//   --merge=F[,F]   fold pool stores from other machines in before fitting (pairwise counts are
//                   additive). Requires the SAME node set and DISTINCT --seed per machine.
//   --games=N       games per matchup (even; default 100). Re-running accumulates.
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
//                   Recomputed each run (never persisted), so it can't double-count; self-play
//                   games (same engine both sides) are skipped. Pairs with `node.js --rounds=0`
//                   for a pure corpus-only refit.
//   --csv=FILE      also write a flat engine,version,depth,elo,ci95,games CSV (for plotting a
//                   depth curve — filter to one version). Off unless given.
//   --save-games[=F]  harvest the games as training data (default OFF).
//   --fresh         discard any existing store and start the pool from scratch.

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
  depths: parseDepths(args.depths, [6, 8]),
  anchorDepth: num(args['anchor-depth'], 6), // the hc depth that is the pin (Elo := 0)
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
  corpus: !!args.corpus,
  data: typeof args.data === 'string' ? resolve(process.cwd(), args.data) : join(dataDir, 'selfplay.jsonl'),
  store: typeof args.store === 'string' ? resolve(process.cwd(), args.store) : join(loopDir, 'ladder-pool.json'),
  ledger: typeof args.ledger === 'string' ? resolve(process.cwd(), args.ledger) : join(loopDir, 'engine-elo.ladder.json'),
  csv: typeof args.csv === 'string' ? resolve(process.cwd(), args.csv) : null, // opt-in flat dump for plotting
  saveGames: args['save-games'],
  fresh: !!args.fresh,
};

// --- resolve an engine spec to { eng, eval, weights, version } ------------------
function makeEngine(spec) {
  if (spec === 'hc' || spec === 'handcrafted') return { eng: 'hc', eval: 'handcrafted', weights: null, version: String(HC_VERSION) };
  if (spec === 'material') return { eng: 'nn', eval: 'nn', weights: null, version: '?' };
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

// Selected engines + the ALWAYS-included handcrafted, deduped by eng@version. hc is a full
// participant at EVERY --depth (near-strength anchor points connecting every depth band to the
// scale), and hc<anchor-depth> (hc6) is the pin. So even a single-net sweep is tied to the
// stable scale at each depth, not just reaching down from hc6.
let engines = cfg.engines === 'all' ? allEngines()
  : cfg.engines.split(',').map((s) => s.trim()).filter(Boolean).map(makeEngine);
if (!engines.some((e) => `${e.eng}@${e.version}` === `hc@${HC_VERSION}`)) engines.push(makeEngine('hc'));
{ const seen = new Set(); engines = engines.filter((e) => { const k = `${e.eng}@${e.version}`; if (seen.has(k)) return false; seen.add(k); return true; }); }

// --- competitors (nodes of the rating pool) ------------------------------------
// Each node is an (engine, depth) pair; id is its `vs`-tag (stable across runs, the store
// key AND the ledger tag): nn<d>@<ver> / hc<d>@<HC_VERSION> / nn<d>@? (material).
const node = (e, d) => ({ id: `${e.eng}${d}@${e.version}`, eng: e.eng, eval: e.eval, weights: e.weights, version: e.version, depth: d });
const competitors = [];
for (const e of engines) for (const d of cfg.depths) competitors.push(node(e, d));
// The pin node hc<anchor-depth> is ALWAYS present, even if anchor-depth ∉ --depths.
const pinId = `hc${cfg.anchorDepth}@${HC_VERSION}`;
if (!competitors.some((c) => c.id === pinId)) competitors.push(node(makeEngine('hc'), cfg.anchorDepth));
if (competitors.length < 2) { console.error('Need at least 2 nodes (engines × depths).'); process.exit(1); }
const byId = new Map(competitors.map((c) => [c.id, c]));

// --- persistent pairwise store -------------------------------------------------
// pairs: { "idA|idB" (sorted): { games, sumA } } where sumA = points scored by idA. One shared
// store ('rank') accumulates ALL pool games — any run fits only its own node subset, so a
// single-net depth sweep and the full ledger pool feed the same body of knowledge.
mkdirSync(loopDir, { recursive: true });
const poolId = 'rank';
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
// Combined view of persisted + corpus pairs, summed by key (different bodies of games, so
// additive). Used by fit() so the corpus contributes to every rating without touching disk.
function combinedPairs() {
  const m = new Map();
  for (const [k, v] of Object.entries(store.pairs)) m.set(k, { games: v.games, sumA: v.sumA });
  for (const [k, v] of corpusPairs) { const e = m.get(k) || { games: 0, sumA: 0 }; e.games += v.games; e.sumA += v.sumA; m.set(k, e); }
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
function pickMatchup(elo, varDiff, iter) {
  const sorted = [...competitors].sort((a, b) => elo.get(a.id) - elo.get(b.id));
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
    for (let i = 0; i < competitors.length; i++) for (let j = i + 1; j < competitors.length; j++) {
      const a = competitors[i], b = competitors[j];
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
// The currently-running apos-match child, so a stop request can kill it immediately
// instead of waiting the whole matchup out. null when no match is in flight.
let activeChild = null;
let killedByStop = false;
function killActive() { if (activeChild) { killedByStop = true; try { activeChild.kill(); } catch { /* already gone */ } } }

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
    `--result-file=${tmp}`,
  ];
  if (a.eval === 'nn' && a.weights) argv.push(`--weights-a=${a.weights}`);
  if (b.eval === 'nn' && b.weights) argv.push(`--weights-b=${b.weights}`);
  if (cfg.saveGames === true) argv.push('--save-games');
  else if (typeof cfg.saveGames === 'string') argv.push(`--save-games=${cfg.saveGames}`);
  console.log(`\n=== ${a.id}  vs  ${b.id}  (${cfg.games} games) ===`);
  return new Promise((done) => {
    const child = spawn(matchBin, argv, { stdio: ['ignore', 'inherit', 'inherit'], cwd: webDir, env: { ...process.env, APOS_CHILD: '1' } });
    activeChild = child;
    child.on('error', (e) => { activeChild = null; console.error(`  could not run match: ${e.message}; skipping.`); done(); });
    child.on('exit', (code, signal) => {
      activeChild = null;
      if (killedByStop) { done(); return; } // we killed it on a stop request — not a failure
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

// Build + write the ledger (the engine-elo.*.json refresh-v/merge consume). Callable
// periodically (verbose=false) during a long run and once at the end (verbose=true), so the
// ledger is always reasonably fresh even if the run is killed hard.
function writeRankLedger(verbose) {
  const { elo, ci, gamesOf } = fit();
  const ranked = [...competitors].sort((a, b) => elo.get(a.id) - elo.get(b.id));
  const recordsByVersion = new Map();
  for (const [tag, n] of tagCounts) { const t = parseTag(tag); if (!t) continue; const k = `${t.eng}@${t.version}`; recordsByVersion.set(k, (recordsByVersion.get(k) || 0) + n); }
  const ranking = ranked.map((c) => ({
    tag: c.id, eng: c.eng, version: c.version, depth: String(c.depth),
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
  if (!verbose) return;
  console.log(`\n===== Elo ladder (${pinId} := 0) — weakest first =====`);
  console.log(`  ${'engine'.padEnd(16)} ${'Elo'.padStart(8)} ${'±95'.padStart(6)} ${'games'.padStart(7)}`);
  for (const c of ranked) {
    console.log(`  ${c.id.padEnd(16)} ${(`${elo.get(c.id) >= 0 ? '+' : ''}${elo.get(c.id).toFixed(0)}`).padStart(8)} ${(ci.get(c.id) == null ? '' : ci.get(c.id).toFixed(0)).padStart(6)} ${String(gamesOf(c.id)).padStart(7)}${c.id === pinId ? '  (pin)' : ''}`);
  }
  if (unrecoverable.length) {
    console.log(`\n  Unrecoverable contributors (no Elo -> weakest, refresh on sight):`);
    for (const u of unrecoverable.slice(0, 12)) console.log(`    ${u.tag.padEnd(16)} ${String(u.records).padStart(10)} records  — ${u.reason}`);
  }
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
console.log(`  ${competitors.length} node(s): ${competitors.map((c) => `${c.id.split('@')[0]}@${c.version.slice(0, 6)}`).join(', ')}`);
console.log(`  pin ${pinId} | ${cfg.games} games/matchup | ${cfg.jobs} parallel job(s) | store ${cfg.store}`);
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
  printStopHint();
  let stopped = false;
  const stopper = installStop(() => { if (!stopped) { stopped = true; console.log('\n  Stopping…'); } killActive(); });
  const t0 = Date.now();
  const deadline = cfg.minutes ? t0 + cfg.minutes * 60000 : Infinity;
  let played = 0;
  while (!stopped && Date.now() < deadline && played < cfg.matchups) {
    const { elo, varDiff } = fit();
    const { pair, reason, metric } = pickMatchup(elo, varDiff, played);
    if (!pair) break;
    const [a, b] = pair;
    const tag = `[${played + 1}${Number.isFinite(cfg.matchups) ? `/${cfg.matchups}` : ''}]`;
    const why = reason === 'ordering' ? `ordering: P(mis-order) ${(metric * 100).toFixed(0)}%` : `rigidity: ±${metric.toFixed(0)} Elo`;
    console.log(`\n${tag} ${why} -> ${a.id} vs ${b.id}`);
    await playPair(a, b);
    if (stopped) break; // a stop killed the match mid-way — don't record/loop a partial result
    writeFileSync(cfg.store, JSON.stringify(store, null, 2) + '\n');
    played++;
    if (played % EMIT_EVERY === 0) writeRankLedger(false);
  }
  stopper.dispose();
  console.log(`\nActive ranking: ${played} matchup(s) in ${((Date.now() - t0) / 60000).toFixed(1)} min` +
    `${stopped ? ' (stopped)' : cfg.minutes && Date.now() >= deadline ? ' (time budget reached)' : ''}.`);
}

writeRankLedger(true);
