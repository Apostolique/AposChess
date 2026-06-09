// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2019-2026 Jean-David Moisan
//
// Worker thread for the parallel self-play data generator (scripts/gen-selfplay.mjs).
// Each worker plays whole games (handed out by index) and returns the JSONL text for
// that game; the main thread is the single writer to the output file, so parallel
// games never interleave mid-line. A game's RNG is seeded purely from the base seed
// and the game index, so it plays out identically regardless of worker / job count.

import { parentPort, workerData } from 'node:worker_threads';
import { readFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

import { newGameState, toFen } from '../src/board.js';
import { legalMoves, applyMove, gameStatus } from '../src/engine.js';
import { chooseMove, _internal } from '../src/ai.js';
import { featureIndices, loadWeights } from '../src/nn.js';

const here = dirname(fileURLToPath(import.meta.url));
const cfg = workerData.cfg;

if (cfg.evalName === 'nn') {
  try {
    loadWeights(JSON.parse(readFileSync(resolve(here, '../src/nn-weights.json'), 'utf8')));
  } catch { /* fall back to material eval */ }
}

function makeRng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Play one game; return { positions: [{board,turn}], result } with result from
// White's perspective (+1 / 0 / -1). The opening plies are random for variety.
function playGame(rng) {
  const pick = (arr) => arr[Math.floor(rng() * arr.length)];
  let state = newGameState();
  const positions = [];
  const seen = [];
  let result = 0;

  for (let ply = 0; ply < cfg.maxmoves; ply++) {
    const status = gameStatus(state);
    if (status.over) {
      result = status.result === 'checkmate' ? (status.winner === 'white' ? 1 : -1) : 0;
      break;
    }
    // Record real (post-opening) positions only; random opening moves aren't the
    // engine's choices, but the positions themselves are still fine training data.
    // Keep the whole state so we can serialize a FEN (applyMove returns fresh
    // states, so the reference stays valid as the game continues).
    positions.push(state);

    let move;
    if (ply < cfg.openings) {
      move = pick(status.legal);
    } else {
      const prev = seen.map((s) => _internal.hashOf(s));
      move = chooseMove(state, cfg.depth ?? 99, rng,
        cfg.depth != null ? Infinity : cfg.movetime, true, prev, cfg.evalName);
    }
    seen.push(state);
    state = applyMove(state, move);
  }
  return { positions, result };
}

// Per-game seed: decorrelate the base seed with the game index so each game is
// independently reproducible regardless of worker assignment / job count.
const gameSeed = (g) => (((cfg.seed >>> 0) ^ Math.imul(g + 1, 0x9e3779b1)) >>> 0);

parentPort.on('message', (msg) => {
  if (msg.type !== 'play') return;
  const g = msg.g;
  const { positions, result } = playGame(makeRng(gameSeed(g)));
  const gid = `${cfg.seed.toString(36)}-${g}`; // unique per run; groups one game
  let lines = '';
  for (const st of positions) {
    const { board, turn } = st;
    // Canonical features are side-to-move-relative, so the label must be too:
    // flip the White-view game result for Black-to-move positions.
    const r = turn === 'white' ? result : -result;
    // `fen` lets scripts/refeaturize.mjs recompute `f` after a feature-set change
    // without regenerating self-play; the trainer ignores it and reads `f` directly.
    lines += JSON.stringify({ f: featureIndices(board, turn), r, g: gid, fen: toFen(st) }) + '\n';
  }
  parentPort.postMessage({ type: 'result', g, lines, nPositions: positions.length });
});

parentPort.postMessage({ type: 'ready' });
