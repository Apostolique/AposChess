// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2019-2026 Jean-David Moisan
//
// Shared Node helper that runs the search in the Zig engine compiled to wasm
// (web/engine → apos.wasm), bit-consistent with the native generator/gate (same Zig eval).
// The offline scoring tools (refresh-v, backfill-v, fill-missing-v) and the puzzle miner
// use it, so a position value is computed the same everywhere. Each instance owns its own
// transposition table, so it's safe to create one per worker thread.
//
// The JS engine (board.js/engine.js) is still used by the callers for board manipulation
// and to reconstruct the full variant move object (castle/jump/promotion flags) from the
// wasm's from/to/promo — the wasm only decides which move; JS rebuilds it via legalMoves.

import { readFileSync, existsSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

import { toFen } from '../src/board.js';
import { legalMoves } from '../src/engine.js';

const here = dirname(fileURLToPath(import.meta.url));
const engineDir = resolve(here, '..', 'engine');
const wasmPath = resolve(engineDir, 'zig-out', 'bin', 'apos.wasm');

// Zig board.Role enum order; the wasm's promo index (1-based) maps back to the JS letter.
const ROLE_CHARS = ['p', 'n', 'b', 'r', 'q', 'k'];

// Build apos.wasm if it's missing. Call once from the MAIN thread before spawning workers
// (so several workers don't race to build it). Cached — near-instant if already built.
export function ensureWasm() {
  if (existsSync(wasmPath)) return;
  const r = spawnSync('zig build wasm -Doptimize=ReleaseSmall', { cwd: engineDir, stdio: 'inherit', shell: true });
  if (r.status !== 0) throw new Error('zig build wasm failed (is Zig 0.16 on PATH?).');
}

let _module = null;
function wasmModule() {
  if (!_module) {
    ensureWasm();
    _module = new WebAssembly.Module(readFileSync(wasmPath));
  }
  return _module;
}

// Create an engine instance with its own TT. evalName 'handcrafted' | 'nn'; for 'nn' a
// weights file path is REQUIRED (there is no material fallback — nn with no net loaded would
// fault on the first eval; want a bare material baseline? select EvalKind.material instead).
export function makeEngine(evalName = 'nn', weightsPath = null) {
  const env = { aposNowMs: () => performance.now(), aposProgress: () => {} };
  const x = new WebAssembly.Instance(wasmModule(), { env }).exports;
  const enc = new TextEncoder();
  const fenPtr = x.allocBytes(128);
  const exPtr = x.allocBytes(256 * 4);
  const hPtr = x.allocBytes(8); // unused (no repetition window in these offline scorers)

  x.setEval(evalName === 'nn' ? 1 : 0);
  if (evalName === 'nn' && weightsPath) {
    const buf = readFileSync(weightsPath);
    const p = x.allocBytes(buf.length);
    new Uint8Array(x.memory.buffer, p, buf.length).set(buf);
    x.loadWeights(p, buf.length);
  }

  const writeFen = (fen) => {
    const b = enc.encode(fen);
    new Uint8Array(x.memory.buffer, fenPtr, b.length).set(b);
    return b.length;
  };

  return {
    // The search value (cp, side-to-move-relative) of a position given as a FEN string.
    score(fen, depth) {
      const n = writeFen(fen);
      x.search(fenPtr, n, depth, 0, hPtr, 0, exPtr, 0);
      return x.lastScore();
    },
    // Full search of a JS game-state: returns { move, score }. `move` is the full variant
    // move object (reconstructed via legalMoves), or null if there's none. `excludeKeys` is
    // an iterable of from*64+to keys to skip at the root (for runner-up searches).
    searchMove(state, depth, excludeKeys) {
      const n = writeFen(toFen(state));
      let exCount = 0;
      if (excludeKeys) {
        const keys = [...excludeKeys].slice(0, 256);
        const dv = new DataView(x.memory.buffer, exPtr, keys.length * 4);
        keys.forEach((k, i) => dv.setInt32(i * 4, k, true));
        exCount = keys.length;
      }
      const packed = x.search(fenPtr, n, depth, 0, hPtr, 0, exPtr, exCount);
      const score = x.lastScore();
      if (packed === 0xffff) return { move: null, score };
      const from = (packed >> 8) & 0xff;
      const to = packed & 0xff;
      const promoIdx = x.lastPromo();
      const promo = promoIdx ? ROLE_CHARS[promoIdx - 1] : null;
      const legal = legalMoves(state);
      const move =
        legal.find((m) => m.from === from && m.to === to && (promo ? m.promotion === promo : !m.promotion)) ||
        legal.find((m) => m.from === from && m.to === to) ||
        null;
      return { move, score };
    },
    // Invalidate the TT (the puzzle miner resets between positions).
    resetTT() {
      x.resetTT();
    },
  };
}
