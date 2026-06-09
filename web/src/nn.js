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
// One piece-square block: 12 piece-kinds (6 roles × 2 sides) × 64 squares, in
// canonical side-to-move orientation. Exported so tools (refeaturize) can identify
// the plain block when reconstructing a board from feature indices, and so a future
// multi-block layout (e.g. king-relative buckets) can be sized off it.
// (King-relative buckets were tried 2026-06-09 and lost Elo — the net is
// signal/data-limited, not capacity-limited; see the nn-engine-plan memory.)
export const PIECE_SQUARE_FEATURES = 12 * 64; // 768
export const NUM_FEATURES = PIECE_SQUARE_FEATURES; // 768

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
// Generic feed-forward MLP, any number of hidden layers (see training/train.py):
//   { arch: [in, h1, ..., 1], scale, layers: [{w,b}, ...] }
//   layers[0]    is the sparse input layer (in -> h1)
//   layers[1..]  are dense layers; the last one is the scalar head
//   each layer's w is input-major and flattened: w[i*outDim + o] maps input i to
//   output o (so EmbeddingBag.weight[feature] for layer 0, and W^T for the rest).
//   b is length outDim. A placeholder file (no arch) means "not trained yet".
//
// Older weights files used a flat {w0,b0,w1,b1} (single hidden layer); normalize
// migrates those into the `layers` form so a not-yet-retrained file still loads and
// `evaluate` has a single code path. Returns null for a placeholder (no arch).
function normalize(obj) {
  if (!obj || !obj.arch) return null;
  if (obj.layers) return obj;
  if (obj.w0) {
    return {
      arch: obj.arch,
      scale: obj.scale,
      layers: [{ w: obj.w0, b: obj.b0 }, { w: obj.w1, b: obj.b1 }],
    };
  }
  return null;
}

// Weights are held per named SLOT, so one process can keep several nets loaded at
// once: the match runner pits two nets head-to-head (slots 'a'/'b'), and the app can
// offer a choice of nets. The vast majority of callers use the single 'default'
// slot, which behaves exactly as a single global did. Loading a placeholder (no
// arch) clears the slot so evaluate() falls back to material for it.
const WEIGHTS = new Map();

export function loadWeights(obj, slot = 'default') {
  const W = normalize(obj);
  if (W) WEIGHTS.set(slot, W); else WEIGHTS.delete(slot);
}
export function hasWeights(slot = 'default') { return WEIGHTS.has(slot); }

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

// Centipawn evaluation from the side-to-move's perspective, using the net in `slot`.
export function evaluate(board, turn, slot = 'default') {
  const W = WEIGHTS.get(slot);
  if (!W) {
    const s = materialWhite(board);
    return turn === 'white' ? s : -s;
  }
  const layers = W.layers;

  // Layer 0: sparse input -> first hidden. Only a board's worth of features are
  // active, so this is a handful of column adds even recomputed from scratch.
  const L0 = layers[0];
  const H0 = L0.b.length;
  let act = L0.b.slice(); // pre-activations seeded with the bias
  for (const f of featureIndices(board, turn)) {
    const off = f * H0;
    for (let h = 0; h < H0; h++) act[h] += L0.w[off + h];
  }
  for (let h = 0; h < H0; h++) if (act[h] < 0) act[h] = 0; // ReLU

  // Dense hidden layers (everything between the first and the scalar head), ReLU.
  for (let li = 1; li < layers.length - 1; li++) {
    const L = layers[li];
    const outDim = L.b.length;
    const next = L.b.slice();
    for (let i = 0; i < act.length; i++) {
      const a = act[i];
      if (a === 0) continue; // post-ReLU inputs are mostly zero
      const off = i * outDim;
      for (let o = 0; o < outDim; o++) next[o] += a * L.w[off + o];
    }
    for (let o = 0; o < outDim; o++) if (next[o] < 0) next[o] = 0; // ReLU
    act = next;
  }

  // Scalar head (final layer): no activation, just the linear combination.
  const Lf = layers[layers.length - 1];
  let out = Lf.b[0];
  for (let i = 0; i < act.length; i++) out += act[i] * Lf.w[i];

  // Canonical features make the output already side-to-move-relative — no flip.
  return Math.round(Math.tanh(out) * W.scale);
}
