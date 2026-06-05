// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2019-2026 Jean-David Moisan
//
// Iterative-deepening alpha-beta search with several refinements that let it
// look deeper without examining every position:
//   - Transposition table  — Zobrist-hash each position; reuse a prior result
//                            (cutoff) when it was searched at least as deep, and
//                            seed move ordering with its best move.
//   - Quiescence search    — at a leaf, keep resolving captures/jumps/promotions
//                            so the evaluation is never taken mid-trade.
//   - PVS                  — search non-first moves with a zero-width window,
//                            re-searching only when one beats it.
//   - Null-move pruning    — if passing the move still fails high, prune (guarded
//                            against check and pawn-only "zugzwang" positions).
//   - Late move reductions — search late quiet moves shallower, re-searching on
//                            a surprise. Jumps/captures/promotions are NEVER
//                            reduced, so the variant's tactics aren't missed.
//   - Killer + history     — order quiet moves that previously caused cutoffs
//                            first, which makes the pruning above far more
//                            effective.
// Legality is guaranteed regardless: every move comes from legalMoves(), so
// pruning only changes which legal move is chosen, never whether it is legal.

import { legalMoves, applyMove, kingAttacked } from './engine.js';
import { opponent } from './board.js';

const VALUE = { p: 100, n: 300, b: 330, r: 500, q: 900, k: 0 };
const MATE = 1_000_000;
const MATE_THRESH = MATE - 1000; // scores beyond this magnitude encode a forced mate
const MAX_PLY = 64;
const QDEPTH = 6; // quiescence depth cap
const now = () => Date.now();

let killers; // killers[ply] = [moveKey, moveKey]
let history; // Int32Array[from*64+to] of cutoff counts

const keyOf = (m) => m.from * 64 + m.to;

// --- Zobrist hashing ---------------------------------------------------------
// A deterministic PRNG seeds fixed 64-bit (BigInt) keys, so the same position
// always hashes the same way within and across searches.
function mulberry32(a) {
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return (t ^ (t >>> 14)) >>> 0;
  };
}
const _rnd = mulberry32(0x1a2b3c4d);
const rand64 = () => (BigInt(_rnd()) << 32n) | BigInt(_rnd());

const ROLE_IDX = { p: 0, n: 1, b: 2, r: 3, q: 4, k: 5 };
const PIECE_KEYS = Array.from({ length: 12 * 64 }, rand64);
const SIDE_KEY = rand64(); // XORed in when Black is to move
const CASTLE_KEYS = { K: rand64(), Q: rand64(), k: rand64(), q: rand64() };
const pieceKey = (role, color, sq) =>
  PIECE_KEYS[(ROLE_IDX[role] * 2 + (color === 'white' ? 0 : 1)) * 64 + sq];

function hashOf(state) {
  let h = 0n;
  for (let i = 0; i < 64; i++) {
    const p = state.board[i];
    if (p) h ^= pieceKey(p.role, p.color, i);
  }
  if (state.turn === 'black') h ^= SIDE_KEY;
  const c = state.castling;
  if (c.K) h ^= CASTLE_KEYS.K;
  if (c.Q) h ^= CASTLE_KEYS.Q;
  if (c.k) h ^= CASTLE_KEYS.k;
  if (c.q) h ^= CASTLE_KEYS.q;
  return h;
}

// Incrementally derive the hash of the position after `m`. MUST mirror
// applyMove() exactly — see the cross-check in the engine tests.
function hashAfter(h, state, m) {
  const board = state.board;
  const piece = board[m.from];
  const color = piece.color;

  h ^= pieceKey(piece.role, color, m.from);
  if (m.capture) {
    const cap = board[m.to];
    if (cap) h ^= pieceKey(cap.role, cap.color, m.to);
  }
  h ^= pieceKey(m.promotion || piece.role, color, m.to);

  if (m.castle) {
    const home = color === 'white' ? 0 : 56;
    if (m.castle === 'K') { h ^= pieceKey('r', color, home + 7); h ^= pieceKey('r', color, home + 5); }
    else { h ^= pieceKey('r', color, home + 0); h ^= pieceKey('r', color, home + 3); }
  }

  const c = state.castling;
  let K = c.K, Q = c.Q, k = c.k, q = c.q;
  if (piece.role === 'k') { if (color === 'white') { K = Q = false; } else { k = q = false; } }
  for (const idx of [m.from, m.to]) {
    if (idx === 0) Q = false;
    else if (idx === 7) K = false;
    else if (idx === 56) q = false;
    else if (idx === 63) k = false;
  }
  if (K !== c.K) h ^= CASTLE_KEYS.K;
  if (Q !== c.Q) h ^= CASTLE_KEYS.Q;
  if (k !== c.k) h ^= CASTLE_KEYS.k;
  if (q !== c.q) h ^= CASTLE_KEYS.q;

  return h ^ SIDE_KEY;
}

// --- Transposition table -----------------------------------------------------
const EXACT = 0, LOWER = 1, UPPER = 2;
const TT_CAP = 1 << 20; // entry ceiling, to bound memory
let tt = new Map();
let ttEnabled = true;

// Mate scores are stored relative to the node (distance-to-mate from here), so an
// entry reused at a different ply still reports the correct mate distance.
const toTT = (s, ply) => (s >= MATE_THRESH ? s + ply : s <= -MATE_THRESH ? s - ply : s);
const fromTT = (s, ply) => (s >= MATE_THRESH ? s - ply : s <= -MATE_THRESH ? s + ply : s);

// --- evaluation & ordering ---------------------------------------------------
function evalStm(board, turn) {
  let s = 0;
  for (const p of board) if (p) s += p.color === 'white' ? VALUE[p.role] : -VALUE[p.role];
  return turn === 'white' ? s : -s;
}

function hasNonPawn(board, color) {
  for (const p of board) if (p && p.color === color && p.role !== 'p' && p.role !== 'k') return true;
  return false;
}

function scoreMove(m, board, ply, pvKey) {
  const key = keyOf(m);
  if (key === pvKey) return 2e6;
  if (m.capture) {
    const victim = board[m.to], attacker = board[m.from];
    return 1e6 + (victim ? VALUE[victim.role] : 0) * 16 - (attacker ? VALUE[attacker.role] : 0);
  }
  if (m.promotion) return 9e5 + VALUE[m.promotion];
  if (m.jump) return 8e5; // non-capturing jump: tactical, try it early
  const k = killers[ply];
  if (k && (k[0] === key || k[1] === key)) return 7e5;
  return Math.min(history[key], 6e5); // capped so quiet history never outranks the above
}

function orderMoves(moves, board, ply, pvKey) {
  for (const m of moves) m._o = scoreMove(m, board, ply, pvKey);
  moves.sort((a, b) => b._o - a._o);
}

// Resolve captures/jumps/promotions to a quiet position before evaluating.
function qsearch(state, alpha, beta, qdepth) {
  const inCheck = kingAttacked(state.board, state.turn);
  let best;
  if (inCheck) {
    best = -MATE;
  } else {
    best = evalStm(state.board, state.turn); // stand pat
    if (best >= beta) return best;
    if (best > alpha) alpha = best;
  }
  if (qdepth <= 0) return best;

  let moves = legalMoves(state);
  if (moves.length === 0) return inCheck ? -MATE : 0;
  if (!inCheck) moves = moves.filter((m) => m.capture || m.promotion || m.jump);
  orderMoves(moves, state.board, 0, 0);

  for (const m of moves) {
    const score = -qsearch(applyMove(state, m), -beta, -alpha, qdepth - 1);
    if (score > best) best = score;
    if (best > alpha) alpha = best;
    if (alpha >= beta) break;
  }
  return best;
}

function search(state, depth, alpha, beta, ply, canNull, hash, deadline) {
  if (now() > deadline) return 0; // aborted; the root discards this iteration
  if (ply >= MAX_PLY) return evalStm(state.board, state.turn);

  const inCheck = kingAttacked(state.board, state.turn);
  if (inCheck) depth++; // check extension
  if (depth <= 0) return qsearch(state, alpha, beta, QDEPTH);

  const alphaOrig = alpha;
  let ttMoveKey = 0;
  if (ttEnabled) {
    const e = tt.get(hash);
    if (e) {
      ttMoveKey = e.move;
      if (e.depth >= depth) {
        const s = fromTT(e.score, ply);
        if (e.flag === EXACT) return s;
        if (e.flag === LOWER && s >= beta) return s;
        if (e.flag === UPPER && s <= alpha) return s;
      }
    }
  }

  // Null-move pruning: pass the move; if we're still ≥ beta, this node fails high.
  if (canNull && !inCheck && depth >= 3 && beta < MATE_THRESH && hasNonPawn(state.board, state.turn)) {
    const nm = {
      board: state.board, turn: opponent(state.turn),
      castling: state.castling, halfmove: state.halfmove, fullmove: state.fullmove,
    };
    const nh = ttEnabled ? hash ^ SIDE_KEY : 0n;
    const score = -search(nm, depth - 3, -beta, -beta + 1, ply + 1, false, nh, deadline);
    if (score >= beta) return beta;
  }

  const legal = legalMoves(state);
  if (legal.length === 0) return inCheck ? -MATE - depth : 0;
  orderMoves(legal, state.board, ply, ttMoveKey);

  let best = -Infinity, bestKey = 0, moveCount = 0;
  for (const m of legal) {
    moveCount++;
    const child = applyMove(state, m);
    const childHash = ttEnabled ? hashAfter(hash, state, m) : 0n;
    const quiet = !m.capture && !m.promotion && !m.jump;
    let score;
    if (moveCount === 1) {
      score = -search(child, depth - 1, -beta, -alpha, ply + 1, true, childHash, deadline);
    } else {
      // Late move reduction for quiet, late moves (never jumps/captures/promotions).
      const r = (quiet && depth >= 3 && moveCount > 3 && !inCheck) ? 1 : 0;
      score = -search(child, depth - 1 - r, -alpha - 1, -alpha, ply + 1, true, childHash, deadline);
      if (score > alpha && r > 0) score = -search(child, depth - 1, -alpha - 1, -alpha, ply + 1, true, childHash, deadline);
      if (score > alpha && score < beta) score = -search(child, depth - 1, -beta, -alpha, ply + 1, true, childHash, deadline);
    }
    if (score > best) { best = score; bestKey = keyOf(m); }
    if (best > alpha) alpha = best;
    if (alpha >= beta) {
      if (quiet) {
        const key = keyOf(m);
        const k = killers[ply] || (killers[ply] = [0, 0]);
        if (k[0] !== key) { k[1] = k[0]; k[0] = key; }
        history[key] += depth * depth;
      }
      break;
    }
    if (now() > deadline) break;
  }

  if (ttEnabled && tt.size < TT_CAP) {
    const flag = best <= alphaOrig ? UPPER : best >= beta ? LOWER : EXACT;
    tt.set(hash, { depth, score: toTT(best, ply), flag, move: bestKey });
  }
  return best;
}

// Choose a move for the side to move, searching up to `maxDepth` plies but never
// past `maxMs` of wall-clock. `rand` shuffles equal choices so games vary.
// `useTT` exists for benchmarking the transposition table on/off.
export function chooseMove(state, maxDepth = 2, rand = Math.random, maxMs = Infinity, useTT = true) {
  const root = legalMoves(state);
  if (root.length === 0) return null;

  for (let i = root.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [root[i], root[j]] = [root[j], root[i]];
  }

  killers = [];
  history = new Int32Array(64 * 64);
  ttEnabled = useTT;
  tt.clear();
  const rootHash = useTT ? hashOf(state) : 0n;
  const deadline = now() + maxMs;
  let bestMove = root[0];

  // Backstop so an unbounded (maxDepth = Infinity) search still terminates even
  // if the deadline were also infinite; real searches abort on time long before.
  const depthCap = Math.min(maxDepth, 99);
  for (let depth = 1; depth <= depthCap; depth++) {
    orderMoves(root, state.board, 0, keyOf(bestMove));
    let alpha = -Infinity, bestScore = -Infinity, localBest = root[0], aborted = false, moveCount = 0;
    for (const m of root) {
      moveCount++;
      const child = applyMove(state, m);
      const childHash = useTT ? hashAfter(rootHash, state, m) : 0n;
      let score;
      if (moveCount === 1) {
        score = -search(child, depth - 1, -Infinity, -alpha, 1, true, childHash, deadline);
      } else {
        score = -search(child, depth - 1, -alpha - 1, -alpha, 1, true, childHash, deadline);
        if (score > alpha) score = -search(child, depth - 1, -Infinity, -alpha, 1, true, childHash, deadline);
      }
      if (now() > deadline) { aborted = true; break; }
      if (score > bestScore) { bestScore = score; localBest = m; }
      if (score > alpha) alpha = score;
    }
    if (!aborted) bestMove = localBest;
    if (aborted || bestScore >= MATE_THRESH) break;
  }
  return bestMove;
}

// Exposed for tests only (Zobrist hash equivalence check).
export const _internal = { hashOf, hashAfter };
