// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2019-2026 Jean-David Moisan
//
// Worker for scripts/refresh-v.mjs. Recomputes the search value `v` (cp, side-to-move
// relative) for a batch of positions using the loaded champion weights, at a fixed
// depth. Position-grained (every record is already a {fen,...}), so no game logic.

import { parentPort, workerData } from 'node:worker_threads';
import { readFileSync } from 'node:fs';

import { parseFen } from '../src/board.js';
import { chooseMoveDetailed } from '../src/ai.js';
import { loadWeights } from '../src/nn.js';

const { weights, depth, evalName = 'nn' } = workerData;
// The handcrafted eval needs no weights; only load a net for the 'nn' eval.
if (evalName === 'nn') { try { loadWeights(JSON.parse(readFileSync(weights, 'utf8'))); } catch { /* material fallback */ } }

parentPort.on('message', (msg) => {
  if (msg.type !== 'batch') return;
  const vs = msg.items.map(({ idx, fen }) => ({
    idx,
    v: Math.round(chooseMoveDetailed(parseFen(fen), depth, Math.random, Infinity, true, [], evalName).score),
  }));
  parentPort.postMessage({ type: 'done', vs });
});
parentPort.postMessage({ type: 'ready' });
