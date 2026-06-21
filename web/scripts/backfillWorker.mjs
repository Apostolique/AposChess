// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2019-2026 Jean-David Moisan
//
// Worker for scripts/backfill-v.mjs. Given one game's legacy {f,r} records (in ply
// order), it reconstructs the TRUE position sequence and writes position-primary
// {fen,r,g,v} records with a freshly-searched value `v`.
//
// Reconstruction: the legacy `f` is canonical (side-to-move) features, so it loses
// the actual colour-to-move and castling. We recover both from the per-game ply
// sequence: ply 0 is the start (White to move) and plies alternate, so we
// un-canonicalize each ply by parity (odd ply = Black to move -> swap colours and
// flip squares i^56) to get the ACTUAL board, then REPLAY the game through the
// engine (applyMove) — which maintains castling / halfmove / fullmove exactly. Each
// ply is matched to the unique legal move that reproduces the next board (the same
// "legal-move connectivity" the original dataset migration used; validated to match
// 100% of transitions). If a transition ever fails to match, we stop trusting the
// engine chain and fall back to a per-ply board with heuristic castling so no data
// is dropped (counted and reported).
//
// `v` is the working-tree engine's fixed-depth nn search value (cp, side-to-move
// relative) — a TD/bootstrap target consistent with train.py --lambda. It is a soft
// label; the per-worker transposition table persists across positions (correctness-
// safe: keyed by full hash + eval namespace) for speed.

import { parentPort, workerData } from 'node:worker_threads';

import { newGameState, toFen } from '../src/board.js';
import { legalMoves, applyMove } from '../src/engine.js';
import { makeEngine } from './wasmEngine.mjs';

const { weights, depth } = workerData;
// Score via the native Zig engine (wasm); board reconstruction still uses the JS engine.
const eng = makeEngine('nn', weights);

const ROLES = ['p', 'n', 'b', 'r', 'q', 'k'];

// Decode the canonical (us=white) board from feature indices.
function boardFromFeatures(f) {
  const b = new Array(64).fill(null);
  for (const idx of f) {
    if (idx >= 768) continue;
    const sq = idx % 64;
    const s = (idx - sq) / 64;
    const side = s % 2;
    b[sq] = { role: ROLES[(s - side) / 2], color: side === 0 ? 'white' : 'black' };
  }
  return b;
}

// Actual board at ply i: even ply = White to move (canonical is already actual);
// odd ply = Black to move, so undo the canonical flip (swap colours + square i^56).
function actualBoard(f, i) {
  const c = boardFromFeatures(f);
  if (i % 2 === 0) return c;
  const a = new Array(64).fill(null);
  for (let s = 0; s < 64; s++) {
    const p = c[s];
    if (p) a[s ^ 56] = { role: p.role, color: p.color === 'white' ? 'black' : 'white' };
  }
  return a;
}

function boardsEqual(a, b) {
  for (let i = 0; i < 64; i++) {
    const x = a[i], y = b[i];
    if (!x !== !y) return false;
    if (x && (x.role !== y.role || x.color !== y.color)) return false;
  }
  return true;
}

// Castling fallback when the engine chain breaks: a right exists iff the king and the
// relevant rook are both on their home squares (best effort without move history).
function heuristicCastle(b) {
  const wk = b[4] && b[4].role === 'k' && b[4].color === 'white';
  const bk = b[60] && b[60].role === 'k' && b[60].color === 'black';
  const wr = (sq) => b[sq] && b[sq].role === 'r' && b[sq].color === 'white';
  const br = (sq) => b[sq] && b[sq].role === 'r' && b[sq].color === 'black';
  return { K: !!(wk && wr(7)), Q: !!(wk && wr(0)), k: !!(bk && br(63)), q: !!(bk && br(56)) };
}

function processGame(g, recs) {
  const boards = recs.map((r, i) => actualBoard(r.f, i));
  let state = newGameState();
  let chained = boardsEqual(state.board, boards[0]); // engine replay still valid
  let fallbacks = 0;
  let lines = '';
  for (let i = 0; i < recs.length; i++) {
    let st;
    if (chained) {
      st = state;
    } else {
      st = { board: boards[i], turn: i % 2 === 0 ? 'white' : 'black',
             castling: heuristicCastle(boards[i]), halfmove: 0, fullmove: (i >> 1) + 1 };
      fallbacks++;
    }
    const fen = toFen(st);
    const v = eng.score(fen, depth);
    lines += JSON.stringify({ fen, r: recs[i].r, g, v }) + '\n';
    if (chained && i < recs.length - 1) {
      let found = null;
      for (const m of legalMoves(state)) {
        if (boardsEqual(applyMove(state, m).board, boards[i + 1])) { found = m; break; }
      }
      if (found) state = applyMove(state, found); else chained = false;
    }
  }
  return { lines, fallbacks };
}

parentPort.on('message', (msg) => {
  if (msg.type !== 'game') return;
  const { lines, fallbacks } = processGame(msg.g, msg.recs);
  parentPort.postMessage({ type: 'done', u: msg.u, lines, n: msg.recs.length, fallbacks });
});
parentPort.postMessage({ type: 'ready' });
