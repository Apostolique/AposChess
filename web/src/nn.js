// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2019-2026 Jean-David Moisan
//
// Neural-network evaluation for the "Neural net" engine option. This is the JS
// side: feature extraction + a tiny MLP forward pass. The matching trainer lives
// in ../../training/ (Python/PyTorch) and writes the weights to nn-weights.json.
//
// The feature definition is deliberately kept HERE, in one place, and reused by
// both inference and the self-play data generator (scripts/gen-selfplay.mjs). The
// generator writes the active feature indices for each position, so the Python
// trainer never needs any chess logic — it only sees integer vectors and targets.
//
// Network (v1): a sparse input layer (only a board's worth of features are ever
// active, so the first layer is a handful of column adds — cheap even recomputed
// from scratch at every leaf), one ReLU hidden layer, and a scalar output. The
// raw output is squashed with tanh and scaled to centipawns, already from the
// SIDE-TO-MOVE's perspective (the search convention, matching evalStm) — no final
// flip needed.

// Feature layout (canonical / side-to-move orientation): the board is presented
// from the mover's point of view, so a position and its colour-mirror collapse to
// the SAME input and the net never has to learn the colour symmetry separately.
// 12 piece kinds (6 roles x 2 SIDES) x 64 squares; index = (role*2 + side)*64 +
// square, where side 0 = the side to move ("us"), side 1 = the opponent ("them").
// When Black is to move the square is vertically flipped (i ^ 56, the same rank
// mirror the PSTs use) so the mover's back rank is always rank 0. Side-to-move is
// encoded by the orientation itself, so there is no separate STM bit.
const PIECE_INDEX = { p: 0, n: 1, b: 2, r: 3, q: 4, k: 5 };
export const NUM_FEATURES = 12 * 64; // 768

// Active (set-to-1) input indices for a position. Sparse — typically ~17-32 of 768.
export function featureIndices(board, turn) {
  const idx = [];
  const flip = turn === 'black';
  for (let i = 0; i < 64; i++) {
    const p = board[i];
    if (!p) continue;
    const side = p.color === turn ? 0 : 1; // 0 = us (side to move), 1 = them
    const sq = flip ? i ^ 56 : i;
    idx.push((PIECE_INDEX[p.role] * 2 + side) * 64 + sq);
  }
  return idx;
}

// --- weights -----------------------------------------------------------------
// Shape written by the trainer (see training/train.py):
//   { arch: [in, hidden, 1], scale, w0, b0, w1, b1 }
//   w0: Float length in*hidden, input-major  -> w0[feature*hidden + h]
//   b0: Float length hidden
//   w1: Float length hidden  (hidden -> scalar)
//   b1: Float length 1
// A placeholder file (no arch) means "not trained yet" -> material fallback.
let W = null;

export function loadWeights(obj) {
  W = obj && obj.arch ? obj : null;
}
export function hasWeights() { return W !== null; }

const VALUE = { p: 100, n: 300, b: 330, r: 500, q: 900, k: 0 };

// Material-only fallback (white-relative), used until weights are trained so the
// "Neural net" engine still plays a legal, if weak, game out of the box.
function materialWhite(board) {
  let s = 0;
  for (let i = 0; i < 64; i++) {
    const p = board[i];
    if (p) s += p.color === 'white' ? VALUE[p.role] : -VALUE[p.role];
  }
  return s;
}

// Centipawn evaluation from the side-to-move's perspective.
export function evaluate(board, turn) {
  if (!W) {
    const s = materialWhite(board);
    return turn === 'white' ? s : -s;
  }
  const H = W.arch[1];
  const acc = W.b0.slice(); // hidden pre-activations, seeded with the bias
  const feats = featureIndices(board, turn);
  for (const f of feats) {
    const off = f * H;
    for (let h = 0; h < H; h++) acc[h] += W.w0[off + h];
  }
  let out = W.b1[0];
  for (let h = 0; h < H; h++) {
    const a = acc[h] > 0 ? acc[h] : 0; // ReLU
    out += a * W.w1[h];
  }
  // Canonical features make the output already side-to-move-relative — no flip.
  return Math.round(Math.tanh(out) * W.scale);
}
