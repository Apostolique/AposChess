// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2019-2026 Jean-David Moisan
//
// Runs the AI search off the main thread so deeper lookahead never freezes the
// UI. Two request kinds, both tagged with a `seq` so a reply for a superseded
// position is discarded by the page:
//   { type: 'search', seq, state, depth, maxMs, engine } → { type: 'search', seq, move, ponder }
//       a real move to play; `ponder` is the predicted opponent reply { from, to }.
//   { type: 'ponder', seq, state, depth, maxMs, engine } → { type: 'ponder', seq, reached }
//       thinking on the opponent's turn; the move is irrelevant, the point is the
//       warmed transposition table. `reached` is the deepest completed iteration.
// `engine` picks the evaluation ('handcrafted' | 'nn'); see chooseMoveDetailed.
//
// The table lives in the search module and is NOT cleared between messages, so a
// real search reuses what pondering found (and vice versa). Hard resets (new
// game / stop) recreate the worker, which gives a fresh table for free.
//
// Game state is plain data (board array + flags), so it survives structured
// cloning across the worker boundary unchanged.

import { chooseMoveDetailed, _internal } from './ai.js';
import { parseFen } from './board.js';
import { loadWeights } from './nn.js';
import nnWeights from './nn-weights.json';

// The trained net weights are a build-time asset, bundled in statically (a static
// import keeps the worker a single chunk; a dynamic import would force a
// code-splitting build the IIFE worker format rejects). A placeholder file (no
// `arch`) leaves nn.js on its material fallback. Retrain -> rebuild to update.
loadWeights(nnWeights);

self.onmessage = ({ data }) => {
  const { type, seq, state, depth, maxMs, posHistory, engine } = data;
  // posHistory is the live game's prior positions as FENs (compact to clone); the
  // search wants Zobrist hashes, so convert here, on the worker thread.
  const prevHashes = posHistory ? posHistory.map((f) => _internal.hashOf(parseFen(f))) : [];
  if (type === 'ponder') {
    const { depth: reached } = chooseMoveDetailed(state, depth, Math.random, maxMs, true, prevHashes, engine);
    self.postMessage({ type: 'ponder', seq, reached });
    return;
  }
  const { move, ponder } = chooseMoveDetailed(state, depth, Math.random, maxMs, true, prevHashes, engine);
  self.postMessage({ type: 'search', seq, move, ponder });
};
