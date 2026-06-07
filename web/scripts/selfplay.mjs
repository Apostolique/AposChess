// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2019-2026 Jean-David Moisan
//
// Self-play match runner — the statistically honest way to tell whether an engine
// change actually gained strength. A correct-in-principle feature can still lose
// Elo if it costs node speed, and a 20-game "it won more" check is pure noise; only
// a many-game match with error bars (or SPRT) settles it. See the blog post that
// motivated this tool: changes must be *measured*, not assumed.
//
// Usage (run from web/):
//   node scripts/selfplay.mjs [options]
//   npm run match -- [options]
//
// Options:
//   --games=N         total games to play (rounded to an even number; default 100)
//   --movetime=MS     ms per move for both engines (default 50)
//   --movetime-b=MS   override think time for engine B only
//   --depth=D         fixed-depth search instead of movetime (overrides --movetime)
//   --depth-b=D       override fixed depth for engine B only
//   --a=FILE          module for engine A, path relative to web/src (default ./ai.js,
//                     i.e. the working tree). Lets you pit any two snapshots.
//   --baseline=FILE   module for engine B, path relative to web/src (default ./ai.js).
//                     To compare against an earlier commit:
//                       git show HEAD:web/src/ai.js > web/src/ai.baseline.js
//                       node scripts/selfplay.mjs --baseline=ai.baseline.js
//                       (delete web/src/ai.baseline.js afterwards)
//   --eval-a=NAME     evaluation for engine A: 'handcrafted' (default) or 'nn'
//   --eval-b=NAME     evaluation for engine B (default 'handcrafted'). Lets you pit
//                     the neural-net eval against the handcrafted one in one file.
//   --no-tt           disable the transposition table for both engines
//   --openings=K      random legal plies played to diversify each opening (default 6)
//   --maxmoves=N      adjudicate as a draw after N plies (default 200)
//   --seed=S          base RNG seed for reproducibility (default 1)
//   --sprt            stop early once the result is statistically decided
//   --elo0=E          SPRT H0 bound in Elo (default 0)
//   --elo1=E          SPRT H1 bound in Elo (default 15)
//   --alpha=A         SPRT type-I error (default 0.05)
//   --beta=B          SPRT type-II error (default 0.05)
//
// Each opening is played twice with colors reversed (a "pair"), so neither engine
// is helped by always getting White. Engine A is the code under test; results are
// always reported from A's perspective. A and B are imported as *separate* module
// instances (distinct query strings) so each keeps its own transposition table,
// exactly as two real engines would.

import { newGameState, opponent } from '../src/board.js';
import { legalMoves, applyMove, gameStatus } from '../src/engine.js';

// --- args --------------------------------------------------------------------
const args = Object.fromEntries(
  process.argv.slice(2).map((a) => {
    const m = a.replace(/^--/, '').split('=');
    return [m[0], m.length > 1 ? m[1] : true];
  }),
);
const num = (v, d) => (v === undefined ? d : Number(v));

const cfg = {
  games: Math.max(2, Math.round(num(args.games, 100) / 2) * 2),
  movetime: num(args.movetime, 50),
  movetimeB: num(args['movetime-b'], num(args.movetime, 50)),
  depth: args.depth !== undefined ? Number(args.depth) : null,
  depthB: args['depth-b'] !== undefined ? Number(args['depth-b']) : (args.depth !== undefined ? Number(args.depth) : null),
  engineA: typeof args.a === 'string' ? args.a : './ai.js',
  baseline: typeof args.baseline === 'string' ? args.baseline : './ai.js',
  evalA: typeof args['eval-a'] === 'string' ? args['eval-a'] : 'handcrafted',
  evalB: typeof args['eval-b'] === 'string' ? args['eval-b'] : 'handcrafted',
  useTT: !args['no-tt'],
  openings: num(args.openings, 6),
  maxmoves: num(args.maxmoves, 200),
  seed: num(args.seed, 1),
  sprt: !!args.sprt,
  elo0: num(args.elo0, 0),
  elo1: num(args.elo1, 15),
  alpha: num(args.alpha, 0.05),
  beta: num(args.beta, 0.05),
};

// --- helpers -----------------------------------------------------------------
// Mulberry32: small deterministic PRNG so a given --seed reproduces the match.
function makeRng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// A player is a configured engine: how to pick a move + its own persistent table.
function makePlayer(mod, { depth, movetime, evalName }) {
  return {
    move: (state, rng, prevHashes) =>
      mod.chooseMove(state, depth ?? 99, rng, depth != null ? Infinity : movetime, cfg.useTT, prevHashes, evalName),
    hashOf: mod._internal.hashOf,
    resetTT: () => mod._internal.resetTT(),
  };
}

// Play one game; `whiteIsA` decides which engine has White. Returns A's score:
// 1 win, 0.5 draw, 0 loss.
function playGame(openingState, A, B, whiteIsA, rng) {
  A.resetTT();
  B.resetTT();
  let st = openingState;
  const seen = []; // positions played this game, so each engine can detect repetition
  for (let ply = 0; ply < cfg.maxmoves; ply++) {
    const status = gameStatus(st);
    if (status.over) {
      if (status.result !== 'checkmate') return 0.5; // stalemate / fifty-move
      const winnerIsA = (status.winner === 'white') === whiteIsA;
      return winnerIsA ? 1 : 0;
    }
    const aToMove = (st.turn === 'white') === whiteIsA;
    const player = aToMove ? A : B;
    seen.push(st);
    // Only positions since the last irreversible move (the last `halfmove` plies) can
    // recur, so that's all the engine needs for repetition detection.
    const window = seen.slice(-(st.halfmove + 1));
    const mv = player.move(st, rng, window.map(player.hashOf));
    if (!mv) return 0.5;
    st = applyMove(st, mv);
  }
  return 0.5; // adjudicated draw at the move cap
}

// Random but legal opening so games are diverse; both colors play it once.
function makeOpening(rng) {
  let st = newGameState();
  for (let i = 0; i < cfg.openings; i++) {
    const moves = legalMoves(st);
    if (!moves.length) break;
    st = applyMove(st, moves[Math.floor(rng() * moves.length)]);
    if (gameStatus(st).over) return newGameState(); // ran into a terminal opening; restart
  }
  return st;
}

const eloFromScore = (p) => (p <= 0 ? -800 : p >= 1 ? 800 : -400 * Math.log10(1 / p - 1));
const scoreFromElo = (e) => 1 / (1 + Math.pow(10, -e / 400));

// Format a duration in seconds as "Mm SSs" (or "SSs" under a minute).
function fmt(secs) {
  secs = Math.round(secs);
  return secs < 60 ? `${secs}s` : `${Math.floor(secs / 60)}m ${String(secs % 60).padStart(2, '0')}s`;
}

// Generalized SPRT (normal approximation): log-likelihood ratio that the true mean
// score corresponds to elo1 rather than elo0, using the observed score variance.
function llr(scores, elo0, elo1) {
  const n = scores.length;
  if (n < 2) return 0;
  const mu0 = scoreFromElo(elo0), mu1 = scoreFromElo(elo1);
  const S = scores.reduce((a, b) => a + b, 0);
  const mean = S / n;
  let varSum = 0;
  for (const s of scores) varSum += (s - mean) * (s - mean);
  // Floor the variance: with only a handful of identical early results the sample
  // variance collapses toward 0 and the ratio explodes. 1e-3 keeps the degenerate
  // case finite while staying far below a realistic per-game variance (~0.2).
  const variance = Math.max(varSum / n, 1e-3);
  return ((mu1 - mu0) / variance) * (S - (n * (mu0 + mu1)) / 2);
}

function report(scores, done) {
  const n = scores.length;
  const S = scores.reduce((a, b) => a + b, 0);
  const wins = scores.filter((s) => s === 1).length;
  const draws = scores.filter((s) => s === 0.5).length;
  const losses = scores.filter((s) => s === 0).length;
  const p = S / n;
  const mean = p;
  let varSum = 0;
  for (const s of scores) varSum += (s - mean) * (s - mean);
  const se = Math.sqrt(varSum / n / n); // standard error of the mean score
  const elo = eloFromScore(p);
  const eloLo = eloFromScore(p - 1.96 * se);
  const eloHi = eloFromScore(p + 1.96 * se);
  const pct = (100 * p).toFixed(1);
  let line = `${done ? 'FINAL' : 'after'} ${n} games  A: +${wins} =${draws} -${losses}  ` +
    `score ${pct}%  Elo ${elo >= 0 ? '+' : ''}${elo.toFixed(0)} [${eloLo.toFixed(0)}, ${eloHi.toFixed(0)}]`;
  if (cfg.sprt) {
    const L = llr(scores, cfg.elo0, cfg.elo1);
    const lower = Math.log(cfg.beta / (1 - cfg.alpha));
    const upper = Math.log((1 - cfg.beta) / cfg.alpha);
    line += `  LLR ${L.toFixed(2)} [${lower.toFixed(2)}, ${upper.toFixed(2)}]`;
  }
  return line;
}

// --- run ---------------------------------------------------------------------
const engineARel = cfg.engineA.replace(/^\.\//, '');
const baselineRel = cfg.baseline.replace(/^\.\//, '');
const modA = await import(new URL(`../src/${engineARel}?a`, import.meta.url));
const modB = await import(new URL(`../src/${baselineRel}?b`, import.meta.url));

const A = makePlayer(modA, { depth: cfg.depth, movetime: cfg.movetime, evalName: cfg.evalA });
const B = makePlayer(modB, { depth: cfg.depthB, movetime: cfg.movetimeB, evalName: cfg.evalB });

const tag = (file, ev) => (ev === 'handcrafted' ? file : `${file} [${ev}]`);
console.log(`A = ${tag(cfg.engineA, cfg.evalA)}  vs  B = ${tag(cfg.baseline, cfg.evalB)}`);
console.log(
  `${cfg.depth != null ? `depth ${cfg.depth}` : `${cfg.movetime}ms/move`} | ` +
  `openings ${cfg.openings} plies | TT ${cfg.useTT ? 'on' : 'off'} | ` +
  `${cfg.sprt ? `SPRT[${cfg.elo0},${cfg.elo1}] α=${cfg.alpha} β=${cfg.beta}` : `${cfg.games} games`}`,
);

const rng = makeRng(cfg.seed);
const scores = [];
const sprtLower = Math.log(cfg.beta / (1 - cfg.alpha));
const sprtUpper = Math.log((1 - cfg.beta) / cfg.alpha);
const t0 = Date.now();
let decided = null;

// A single in-place status line (carriage-return) refreshed after every game, so a
// long match never looks frozen between the milestone reports. Lightweight on
// purpose (no Elo/CI — that's the milestone report's job); just live progress.
let liveLen = 0;
function live() {
  const n = scores.length;
  const S = scores.reduce((a, b) => a + b, 0);
  const wins = scores.filter((s) => s === 1).length;
  const draws = scores.filter((s) => s === 0.5).length;
  const losses = scores.filter((s) => s === 0).length;
  const elapsed = (Date.now() - t0) / 1000;
  let s = `  game ${n}/${cfg.games} | A +${wins} =${draws} -${losses} | `
    + `${(100 * S / n).toFixed(1)}% | ${fmt(elapsed)} elapsed`;
  // ETA assumes the full game budget; with SPRT the match may stop sooner, so it's
  // an upper bound. Drop it once we're done / decided.
  if (!decided && n < cfg.games) s += ` | ETA ${fmt((elapsed / n) * (cfg.games - n))}`;
  if (cfg.sprt) s += ` | LLR ${llr(scores, cfg.elo0, cfg.elo1).toFixed(2)} `
    + `[${sprtLower.toFixed(2)}, ${sprtUpper.toFixed(2)}]`;
  process.stdout.write('\r' + s.padEnd(liveLen));
  liveLen = s.length;
}
const clearLive = () => process.stdout.write('\r' + ' '.repeat(liveLen) + '\r');

for (let pair = 0; pair < cfg.games / 2 && !decided; pair++) {
  const opening = makeOpening(rng);
  scores.push(playGame(opening, A, B, true, rng));  // A as White
  live();
  scores.push(playGame(opening, A, B, false, rng)); // A as Black, same opening
  live();

  // Only let SPRT stop after a short warmup; before that the variance estimate is
  // too unstable to trust (a couple of identical results would end the match early).
  if (cfg.sprt && scores.length >= 16) {
    const L = llr(scores, cfg.elo0, cfg.elo1);
    if (L >= sprtUpper) decided = 'H1';      // change is an improvement
    else if (L <= sprtLower) decided = 'H0'; // change is not an improvement
  }
  // Commit a permanent milestone snapshot (full Elo/CI) without losing the live
  // line: overwrite it, then the next live() redraws on the fresh line below.
  if ((pair + 1) % 5 === 0 || decided) {
    clearLive();
    console.log('  ' + report(scores, false));
    liveLen = 0;
  }
}

clearLive();
console.log(report(scores, true));
console.log(`elapsed ${((Date.now() - t0) / 1000).toFixed(1)}s`);
if (cfg.sprt) {
  if (decided === 'H1') console.log(`SPRT: accept H1 — A is stronger than B by ≳${cfg.elo0}..${cfg.elo1} Elo.`);
  else if (decided === 'H0') console.log(`SPRT: accept H0 — A is NOT a ${cfg.elo1}-Elo improvement over B.`);
  else console.log('SPRT: inconclusive within the game budget — raise --games or widen [elo0, elo1].');
}
