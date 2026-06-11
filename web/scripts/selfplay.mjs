// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2019-2026 Jean-David Moisan
//
// Self-play match runner — the statistically honest way to tell whether an engine
// change actually gained strength. A correct-in-principle feature can still lose
// Elo if it costs node speed, and a 20-game "it won more" check is pure noise; only
// a many-game match with error bars (or SPRT) settles it. See the blog post that
// motivated this tool: changes must be *measured*, not assumed.
//
// Games run in parallel across worker threads (scripts/matchWorker.mjs) — self-play
// is embarrassingly parallel, so throughput scales ~linearly with cores. This main
// thread only dispatches pair indices and aggregates results; all game-playing (and
// the engine instances) live in the workers. Each pair is seeded purely from --seed
// and its index, so the aggregate result is reproducible from --seed regardless of
// --jobs. (With --sprt the exact stopping point can vary slightly run-to-run, since
// it depends on the order parallel results arrive; the LLR itself is order-free.)
//
// Usage (run from web/):
//   node scripts/selfplay.mjs [options]
//   npm run match -- [options]
//
// Options:
//   --games=N         total games to play (rounded to an even number; default 100)
//   --jobs=N          parallel worker threads (default: CPU core count)
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
//   --weights-a=FILE  nn weights for engine A (when --eval-a=nn); default the
//                     shipped src/nn-weights.json. Use with --weights-b for a direct
//                     net-vs-net match: --eval-a=nn --eval-b=nn --weights-a=X --weights-b=Y
//   --weights-b=FILE  nn weights for engine B (when --eval-b=nn)
//   --result-file=F   write a JSON summary {games,wins,draws,losses,score,elo,llr,
//                     sprt} at the end (for orchestration, e.g. the gated train:loop)
//   --save-games=F    harvest the match's games as training data: append one JSONL
//                     line per searched position ({fen, r, g, v?} — the generator's
//                     raw format) to F when the match ends. The search value `v` is
//                     kept only on positions where the engine the match proved
//                     STRONGER (by final score; tie -> B, the baseline) was to
//                     move — the weaker engine's opinion is a worse target, so its
//                     positions carry just the outcome. Games are buffered until
//                     the verdict, since the winner isn't known until the end.
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

import { Worker } from 'node:worker_threads';
import { cpus } from 'node:os';
import { resolve, dirname } from 'node:path';
import { writeFileSync, appendFileSync, mkdirSync } from 'node:fs';

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
  // Per-side nn weights (only used when the matching eval is 'nn'). Each side loads
  // its file into its own slot, so two nets play head-to-head; omit to use the
  // shipped src/nn-weights.json. Resolved against the current dir for intuitive paths.
  weightsA: typeof args['weights-a'] === 'string' ? resolve(process.cwd(), args['weights-a']) : null,
  weightsB: typeof args['weights-b'] === 'string' ? resolve(process.cwd(), args['weights-b']) : null,
  // Optional machine-readable result (for orchestration, e.g. the gated train:loop).
  resultFile: typeof args['result-file'] === 'string' ? resolve(process.cwd(), args['result-file']) : null,
  // Optional game harvesting: append the played positions as raw training data.
  saveGames: typeof args['save-games'] === 'string' ? resolve(process.cwd(), args['save-games']) : null,
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

const totalPairs = cfg.games / 2;
const jobs = Math.max(1, Math.min(num(args.jobs, cpus().length), totalPairs));

// --- reporting helpers -------------------------------------------------------
const eloFromScore = (p) => (p <= 0 ? -800 : p >= 1 ? 800 : -400 * Math.log10(1 / p - 1));
const scoreFromElo = (e) => 1 / (1 + Math.pow(10, -e / 400));

// Confidence interval for the reported Elo: z-multiplier and its matching label,
// kept together so the bracket and the "N% CI" text can't drift apart.
const CI = { z: 1.96, label: '95%' };

// Format a duration in seconds as "Mm SSs" (or "SSs" under a minute).
function fmt(secs) {
  secs = Math.round(secs);
  return secs < 60 ? `${secs}s` : `${Math.floor(secs / 60)}m ${String(secs % 60).padStart(2, '0')}s`;
}

// Generalized SPRT (normal approximation): log-likelihood ratio that the true mean
// score corresponds to elo1 rather than elo0, using the observed score variance.
// Order-free (a function of the score multiset), so it's valid even though parallel
// results arrive out of order.
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
  const eloLo = eloFromScore(p - CI.z * se);
  const eloHi = eloFromScore(p + CI.z * se);
  const margin = (eloHi - eloLo) / 2; // half-width of the CI: the ± error bar
  const pct = (100 * p).toFixed(1);
  let line = `${done ? 'FINAL' : 'after'} ${n} games  A: +${wins} =${draws} -${losses}  ` +
    `score ${pct}%  Elo ${elo >= 0 ? '+' : ''}${elo.toFixed(0)} ± ${margin.toFixed(0)}  ` +
    `${CI.label} CI [${eloLo.toFixed(0)}, ${eloHi.toFixed(0)}]`;
  if (cfg.sprt) {
    const L = llr(scores, cfg.elo0, cfg.elo1);
    const lower = Math.log(cfg.beta / (1 - cfg.alpha));
    const upper = Math.log((1 - cfg.beta) / cfg.alpha);
    line += `  LLR ${L.toFixed(2)} [${lower.toFixed(2)}, ${upper.toFixed(2)}]`;
  }
  return line;
}

// --- run ---------------------------------------------------------------------
const basename = (p) => (p ? p.replace(/^.*[\\/]/, '') : 'nn-weights.json');
const tag = (file, ev, w) => (ev === 'handcrafted' ? file : `${file} [${ev}:${basename(w)}]`);
console.log(`A = ${tag(cfg.engineA, cfg.evalA, cfg.weightsA)}  vs  B = ${tag(cfg.baseline, cfg.evalB, cfg.weightsB)}`);
console.log(
  `${cfg.depth != null ? `depth ${cfg.depth}` : `${cfg.movetime}ms/move`} | ` +
  `openings ${cfg.openings} plies | TT ${cfg.useTT ? 'on' : 'off'} | jobs ${jobs} | ` +
  `${cfg.sprt ? `SPRT[${cfg.elo0},${cfg.elo1}] α=${cfg.alpha} β=${cfg.beta}` : `${cfg.games} games`}`,
);

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

// Record a finished pair's two scores and refresh the display / SPRT verdict.
function record(pairScores) {
  scores.push(pairScores[0], pairScores[1]);
  live();
  // Only let SPRT stop after a short warmup; before that the variance estimate is
  // too unstable to trust (a couple of identical results would end the match early).
  if (cfg.sprt && !decided && scores.length >= 16) {
    const L = llr(scores, cfg.elo0, cfg.elo1);
    if (L >= sprtUpper) decided = 'H1';      // change is an improvement
    else if (L <= sprtLower) decided = 'H0'; // change is not an improvement
  }
  // Commit a permanent milestone snapshot (full Elo/CI) without losing the live
  // line: overwrite it, then the next live() redraws on the fresh line below.
  if ((scores.length / 2) % 5 === 0 || decided) {
    clearLive();
    console.log('  ' + report(scores, false));
    liveLen = 0;
  }
}

// Game harvesting buffer (--save-games): completed pairs' position records, held
// until the end of the match — only then is the stronger engine known, which
// decides whose search values survive. Pairs in flight when SPRT stops are lost,
// which is fine (their scores aren't counted either).
const harvested = [];

// Worker pool: pull-model dispatch (hand each idle worker the next pair index) so
// load stays balanced regardless of per-game time, and SPRT can stop promptly.
let nextPair = 0;
await new Promise((resolve) => {
  const pool = [];
  let live_ = jobs;
  const retire = () => { if (--live_ === 0) resolve(); };

  const dispatch = (w) => {
    if (decided || nextPair >= totalPairs) { w.terminate(); return; } // exit -> retire
    w.postMessage({ type: 'play', pair: nextPair++ });
  };

  for (let i = 0; i < jobs; i++) {
    const w = new Worker(new URL('./matchWorker.mjs', import.meta.url), { workerData: { cfg } });
    pool.push(w);
    w.on('message', (msg) => {
      if (msg.type === 'ready') { dispatch(w); return; }
      if (msg.type === 'result') {
        if (msg.recs) harvested.push({ pair: msg.pair, recs: msg.recs });
        record(msg.scores);
        if (decided) pool.forEach((p) => p.terminate()); // stop everyone at once
        else dispatch(w);
      }
    });
    w.on('error', (err) => { console.error('\nworker error:', err); w.terminate(); });
    w.on('exit', retire);
  }
});

clearLive();
console.log(report(scores, true));
console.log(`elapsed ${((Date.now() - t0) / 1000).toFixed(1)}s`);
if (cfg.sprt) {
  if (decided === 'H1') console.log(`SPRT: accept H1 — A is stronger than B by ≳${cfg.elo0}..${cfg.elo1} Elo.`);
  else if (decided === 'H0') console.log(`SPRT: accept H0 — A is NOT a ${cfg.elo1}-Elo improvement over B.`);
  else console.log('SPRT: inconclusive within the game budget — raise --games or widen [elo0, elo1].');
}

// Harvest (--save-games): append the buffered games as raw training data, in the
// generator's exact format ({fen, r, g, v?}). `v` survives only on positions where
// the engine the match proved stronger was to move (final score; tie -> B, the
// established baseline) — the weaker engine's positions keep just the outcome,
// which train.py already handles (no-`v` rows fall back to the pure result).
if (cfg.saveGames && harvested.length) {
  const p = scores.reduce((a, b) => a + b, 0) / scores.length;
  const winner = p > 0.5 ? 'a' : 'b';
  let lines = '';
  let nPos = 0;
  let nKept = 0;
  for (const { pair, recs } of harvested) {
    for (let gi = 0; gi < recs.length; gi++) {
      // Unique per run + pair + color, same role as the generator's game id ("g"
      // groups a game for the trainer's by-game train/val split).
      const gid = `m${cfg.seed.toString(36)}-${pair}${gi === 0 ? 'w' : 'b'}`;
      for (const rec of recs[gi]) {
        const o = { fen: rec.fen, r: rec.r, g: gid };
        if (rec.mover === winner && rec.v != null) { o.v = rec.v; nKept++; }
        lines += JSON.stringify(o) + '\n';
        nPos++;
      }
    }
  }
  mkdirSync(dirname(cfg.saveGames), { recursive: true });
  appendFileSync(cfg.saveGames, lines);
  console.log(`Saved ${nPos} positions from ${harvested.length * 2} games to ${cfg.saveGames} `
    + `(v kept from the stronger engine ${winner.toUpperCase()} on ${nKept}).`);
}

// Machine-readable summary for orchestration (train:loop reads this to gate).
if (cfg.resultFile) {
  const n = scores.length, S = scores.reduce((a, b) => a + b, 0), p = S / n;
  writeFileSync(cfg.resultFile, JSON.stringify({
    games: n,
    wins: scores.filter((s) => s === 1).length,
    draws: scores.filter((s) => s === 0.5).length,
    losses: scores.filter((s) => s === 0).length,
    score: p,
    elo: eloFromScore(p),
    llr: cfg.sprt ? llr(scores, cfg.elo0, cfg.elo1) : null,
    sprt: cfg.sprt ? (decided || 'inconclusive') : null, // 'H1' = A better, 'H0' = not
  }, null, 2));
}
