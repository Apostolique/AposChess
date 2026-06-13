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
import { chooseMoveDetailed, _internal } from '../src/ai.js';
import { loadWeights } from '../src/nn.js'; // only to set the nn teacher's weights (--eval=nn)
import { vtag as computeVtag } from './vtag.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const cfg = workerData.cfg;
const NN_WEIGHTS = resolve(here, '../src/nn-weights.json'); // the champion (--eval=nn)

if (cfg.evalName === 'nn') {
  try {
    loadWeights(JSON.parse(readFileSync(NN_WEIGHTS, 'utf8')));
  } catch { /* fall back to material eval */ }
}

// Provenance tag for every v this run writes (eval+depth+version) — see vtag.mjs.
const vtag = computeVtag(cfg.evalName, cfg.depth, NN_WEIGHTS);

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
  const scores = []; // parallel to positions: the search value (cp, stm-relative) or null
  const seenHashes = []; // Zobrist hash of every earlier position, maintained incrementally
  let result = 0;

  for (let ply = 0; ply < cfg.maxmoves; ply++) {
    const status = gameStatus(state);
    if (status.over) {
      result = status.result === 'checkmate' ? (status.winner === 'white' ? 1 : -1) : 0;
      break;
    }
    // Record every position (openings included). The random opening moves aren't the
    // engine's choices, but the positions themselves are fine training data. Keep the
    // whole state so we can serialize a FEN (applyMove returns fresh states, so the
    // reference stays valid as the game continues).
    positions.push(state);

    // Search every position to record its value `v` (cp, stm-relative) — a uniform
    // TD/bootstrap target across the whole dataset, including the openings. Opening
    // plies still PLAY a random move for variety, but `v` is the net's value of the
    // position itself, independent of the (random) move chosen next.
    // Only positions since the last irreversible move (the last `halfmove` plies)
    // can ever recur, so that window is all the repetition detection needs — and
    // each position is hashed once per game, not once per remaining ply.
    const prev = seenHashes.slice(-(state.halfmove + 1));
    const r = chooseMoveDetailed(state, cfg.depth ?? 99, rng,
      cfg.depth != null ? Infinity : cfg.movetime, true, prev, cfg.evalName);
    const move = ply < cfg.openings ? pick(status.legal) : r.move;
    scores.push(r.score);
    seenHashes.push(_internal.hashOf(state));
    state = applyMove(state, move);
  }
  return { positions, scores, result };
}

// Per-game seed: decorrelate the base seed with the game index so each game is
// independently reproducible regardless of worker assignment / job count.
const gameSeed = (g) => (((cfg.seed >>> 0) ^ Math.imul(g + 1, 0x9e3779b1)) >>> 0);

parentPort.on('message', (msg) => {
  if (msg.type !== 'play') return;
  const g = msg.g;
  const { positions, scores, result } = playGame(makeRng(gameSeed(g)));
  const gid = `${cfg.seed.toString(36)}-${g}`; // unique per run; groups one game
  let lines = '';
  for (let i = 0; i < positions.length; i++) {
    const st = positions[i];
    // Store the raw position + outcome — net-agnostic. Features for a specific net are
    // derived later by scripts/featurize.mjs. `r` is the result from the SIDE-TO-MOVE's
    // view (matching the canonical features); the FEN carries correct castling.
    const r = st.turn === 'white' ? result : -result;
    const rec = { fen: toFen(st), r, g: gid };
    // `v` = the search's value of this position (cp, side-to-move-relative) for TD /
    // bootstrap targets; omitted for random opening plies. With --eval=nn it's the
    // net's own deeper-search value — an unbiased bootstrap signal (train.py --lambda).
    if (scores[i] != null) { rec.v = scores[i]; rec.vs = vtag; }
    lines += JSON.stringify(rec) + '\n';
  }
  parentPort.postMessage({ type: 'result', g, lines, nPositions: positions.length });
});

parentPort.postMessage({ type: 'ready' });
