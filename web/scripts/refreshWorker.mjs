// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2019-2026 Jean-David Moisan
//
// Worker for scripts/refresh-v.mjs. Recomputes the search value `v` (cp, side-to-move
// relative) for a batch of positions using the loaded champion weights, at a fixed
// depth. Position-grained (every record is already a {fen,...}), so no game logic.

import { parentPort, workerData } from 'node:worker_threads';

import { makeEngine } from './wasmEngine.mjs';

const { weights, depth, evalName = 'nn', stopFlag = null } = workerData;
// Score via the native Zig engine (wasm) — ~3x faster than ai.js and bit-consistent with
// the native generator/gate (same Zig eval). The handcrafted eval needs no weights.
const eng = makeEngine(evalName, evalName === 'nn' ? weights : null);

parentPort.on('message', (msg) => {
  if (msg.type !== 'batch') return;
  // Check the shared stop flag between positions: once an early stop fires, abandon the
  // rest of the batch (reported as `skipped`) instead of grinding through every search.
  // The worst case is one in-flight search per worker — the one already running here.
  const vs = [], skipped = [];
  for (const { idx, fen } of msg.items) {
    if (stopFlag && Atomics.load(stopFlag, 0)) { skipped.push(idx); continue; }
    vs.push({ idx, v: eng.score(fen, depth) });
  }
  parentPort.postMessage({ type: 'done', vs, skipped });
});
parentPort.postMessage({ type: 'ready' });
