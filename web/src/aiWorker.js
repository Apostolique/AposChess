// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2019-2026 Jean-David Moisan
//
// Runs the AI search off the main thread so deeper lookahead never freezes the
// UI. Two request kinds, both tagged with a `seq` so a reply for a superseded
// position is discarded by the page:
//   { type: 'search', seq, state, depth, maxMs, engine, net } → { type: 'search', seq, move, ponder, score }
//       a real move to play; `ponder` is the predicted opponent reply { from, to };
//       `score` is the root value (centipawns, side-to-move-relative — the eval bar
//       reads it; the play path ignores it). While the search runs it also streams
//       { type: 'progress', seq, score, depth } after each completed iterative-
//       deepening depth, so the page can update the eval bar in real time instead
//       of only when the move is played.
//   { type: 'ponder', seq, state, depth, maxMs, engine, net } → { type: 'ponder', seq, reached, score }
//       thinking on the opponent's turn; the move is irrelevant, the point is the
//       warmed transposition table. `reached` is the deepest completed iteration;
//       `score` is the root value of the pondered position (so the idle engine's
//       eval bar keeps refining during the opponent's turn).
// `engine` picks the evaluation ('handcrafted' | 'nn'); see chooseMoveDetailed. For
// 'nn', `net` is the full URL of the selected weights file to load (see below).
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

// Net weights are fetched at runtime from the public/ catalog (web/public/nn/) so
// the user can pick among named nets without rebuilding — the worker no longer
// carries a fixed weights blob. The page passes the chosen net's full URL in each
// 'nn' request (it knows the document base; a worker's own relative URLs would
// resolve against the worker script under assets/). fetch is not a dynamic import,
// so the single-chunk IIFE worker build the format requires is unaffected. Until a
// net loads, the nn eval falls back to material (nn.js).
const netCache = new Map(); // url -> Promise<weights json>
let currentNet = null;

function fetchNet(url) {
  let p = netCache.get(url);
  if (!p) { p = fetch(url).then((r) => r.json()); netCache.set(url, p); }
  return p;
}

// Make `url`'s weights the active net (default slot). Resets the TT on a real switch
// so the table can't serve a different net's scores. A no-op once the net is current.
async function ensureNet(url) {
  if (!url || currentNet === url) return;
  let w;
  try { w = await fetchNet(url); }
  catch { return; } // keep the current/material eval if the fetch fails
  if (currentNet === url) return; // another message switched while we awaited
  loadWeights(w);
  _internal.resetTT();
  currentNet = url;
}

self.onmessage = async ({ data }) => {
  const { type, seq, state, depth, maxMs, posHistory, engine, net, exclude } = data;
  if (engine === 'nn') await ensureNet(net); // load/switch the selected net first
  // posHistory is the live game's prior positions as FENs (compact to clone); the
  // search wants Zobrist hashes, so convert here, on the worker thread.
  const prevHashes = posHistory ? posHistory.map((f) => _internal.hashOf(parseFen(f))) : [];
  if (type === 'ponder') {
    const { depth: reached, score } = chooseMoveDetailed(state, depth, Math.random, maxMs, true, prevHashes, engine);
    self.postMessage({ type: 'ponder', seq, reached, score });
    return;
  }
  // `exclude` (move keys to skip at the root) drives the opening-variety option, so a
  // fresh AI-vs-AI game doesn't replay a recent opening; only on the game's first move.
  const excludeKeys = exclude && exclude.length ? new Set(exclude) : null;
  // Stream the best score after each completed depth so the eval bar climbs live.
  const onProgress = (score, depth) => self.postMessage({ type: 'progress', seq, score, depth });
  const { move, ponder, score } = chooseMoveDetailed(state, depth, Math.random, maxMs, true, prevHashes, engine, excludeKeys, onProgress);
  self.postMessage({ type: 'search', seq, move, ponder, score });
};
