// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2019-2026 Jean-David Moisan
//
// Worker thread for the parallel self-play match runner (scripts/selfplay.mjs).
// Self-play is embarrassingly parallel — every game pair is independent — so the
// main thread hands each worker pair indices to play and aggregates the results.
//
// Each worker owns its own A/B engine module instances (separate transposition
// tables, exactly as in the single-threaded version) and loads the nn weights once
// at startup. A pair's RNG is seeded purely from the base seed and the pair index,
// so a pair plays out identically no matter which worker runs it (or how many
// workers there are) — the aggregate result is reproducible from --seed alone,
// independent of --jobs.

import { parentPort, workerData } from 'node:worker_threads';
import { readFileSync } from 'node:fs';

import { newGameState } from '../src/board.js';
import { legalMoves, applyMove, gameStatus } from '../src/engine.js';
import { loadWeights } from '../src/nn.js';

const cfg = workerData.cfg;

// Load nn weights per side into its own slot ('a'/'b') so two different nets can
// play head-to-head; both ai.js instances share one nn.js module, but each reads its
// own slot. An omitted --weights-* falls back to the shipped src/nn-weights.json; a
// missing/placeholder file leaves that side on nn.js's material fallback. The eval
// name passed to the search is 'nn:a' / 'nn:b' so each instance picks its slot.
const defaultWeights = new URL('../src/nn-weights.json', import.meta.url);
function loadSlot(path, slot) {
  try {
    loadWeights(JSON.parse(readFileSync(path ?? defaultWeights, 'utf8')), slot);
  } catch { /* fall back to material eval for this slot */ }
}
if (cfg.evalA === 'nn') loadSlot(cfg.weightsA, 'a');
if (cfg.evalB === 'nn') loadSlot(cfg.weightsB, 'b');

const engineARel = cfg.engineA.replace(/^\.\//, '');
const baselineRel = cfg.baseline.replace(/^\.\//, '');
const modA = await import(new URL(`../src/${engineARel}?a`, import.meta.url));
const modB = await import(new URL(`../src/${baselineRel}?b`, import.meta.url));

const evalNameA = cfg.evalA === 'nn' ? 'nn:a' : cfg.evalA;
const evalNameB = cfg.evalB === 'nn' ? 'nn:b' : cfg.evalB;

// Mulberry32: small deterministic PRNG.
function makeRng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function makePlayer(mod, { depth, movetime, evalName }) {
  return {
    move: (state, rng, prevHashes) =>
      mod.chooseMove(state, depth ?? 99, rng, depth != null ? Infinity : movetime, cfg.useTT, prevHashes, evalName),
    hashOf: mod._internal.hashOf,
    resetTT: () => mod._internal.resetTT(),
  };
}

const A = makePlayer(modA, { depth: cfg.depth, movetime: cfg.movetime, evalName: evalNameA });
const B = makePlayer(modB, { depth: cfg.depthB, movetime: cfg.movetimeB, evalName: evalNameB });

// Play one game; `whiteIsA` decides which engine has White. Returns A's score:
// 1 win, 0.5 draw, 0 loss.
function playGame(openingState, whiteIsA, rng) {
  A.resetTT();
  B.resetTT();
  let st = openingState;
  const seen = [];
  for (let ply = 0; ply < cfg.maxmoves; ply++) {
    const status = gameStatus(st);
    if (status.over) {
      if (status.result !== 'checkmate') return 0.5;
      const winnerIsA = (status.winner === 'white') === whiteIsA;
      return winnerIsA ? 1 : 0;
    }
    const aToMove = (st.turn === 'white') === whiteIsA;
    const player = aToMove ? A : B;
    seen.push(st);
    const window = seen.slice(-(st.halfmove + 1));
    const mv = player.move(st, rng, window.map(player.hashOf));
    if (!mv) return 0.5;
    st = applyMove(st, mv);
  }
  return 0.5;
}

function makeOpening(rng) {
  let st = newGameState();
  for (let i = 0; i < cfg.openings; i++) {
    const moves = legalMoves(st);
    if (!moves.length) break;
    st = applyMove(st, moves[Math.floor(rng() * moves.length)]);
    if (gameStatus(st).over) return newGameState();
  }
  return st;
}

// Per-pair seed: decorrelate the base seed with the pair index so each pair is
// independently reproducible regardless of worker assignment / job count.
function pairSeed(pair) {
  return (((cfg.seed >>> 0) ^ Math.imul(pair + 1, 0x9e3779b1)) >>> 0);
}

// Play the pair (same opening, colors reversed) and report A's two scores.
parentPort.on('message', (msg) => {
  if (msg.type !== 'play') return;
  const rng = makeRng(pairSeed(msg.pair));
  const opening = makeOpening(rng);
  const sWhite = playGame(opening, true, rng);  // A as White
  const sBlack = playGame(opening, false, rng); // A as Black, same opening
  parentPort.postMessage({ type: 'result', pair: msg.pair, scores: [sWhite, sBlack] });
});

parentPort.postMessage({ type: 'ready' });
