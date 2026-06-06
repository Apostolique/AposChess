// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2019-2026 Jean-David Moisan
//
// Runs the AI search off the main thread so deeper lookahead never freezes the
// UI. Two request kinds, both tagged with a `seq` so a reply for a superseded
// position is discarded by the page:
//   { type: 'search', seq, state, depth, maxMs } → { type: 'search', seq, move, ponder }
//       a real move to play; `ponder` is the predicted opponent reply { from, to }.
//   { type: 'ponder', seq, state, depth, maxMs } → { type: 'ponder', seq, reached }
//       thinking on the opponent's turn; the move is irrelevant, the point is the
//       warmed transposition table. `reached` is the deepest completed iteration.
//
// The table lives in the search module and is NOT cleared between messages, so a
// real search reuses what pondering found (and vice versa). Hard resets (new
// game / stop) recreate the worker, which gives a fresh table for free.
//
// Game state is plain data (board array + flags), so it survives structured
// cloning across the worker boundary unchanged.

import { chooseMoveDetailed, _internal } from './ai.js';
import { parseFen } from './board.js';

self.onmessage = ({ data }) => {
  const { type, seq, state, depth, maxMs, posHistory } = data;
  // posHistory is the live game's prior positions as FENs (compact to clone); the
  // search wants Zobrist hashes, so convert here, on the worker thread.
  const prevHashes = posHistory ? posHistory.map((f) => _internal.hashOf(parseFen(f))) : [];
  if (type === 'ponder') {
    const { depth: reached } = chooseMoveDetailed(state, depth, Math.random, maxMs, true, prevHashes);
    self.postMessage({ type: 'ponder', seq, reached });
    return;
  }
  const { move, ponder } = chooseMoveDetailed(state, depth, Math.random, maxMs, true, prevHashes);
  self.postMessage({ type: 'search', seq, move, ponder });
};
