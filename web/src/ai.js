// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2019-2026 Jean-David Moisan
//
// Iterative-deepening alpha-beta search with several refinements that let it
// look deeper without examining every position:
//   - Quiescence search   — at a leaf, keep resolving captures/jumps/promotions
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
const MAX_PLY = 64;
const QDEPTH = 6; // quiescence depth cap
const now = () => Date.now();

let killers; // killers[ply] = [moveKey, moveKey]
let history; // Int32Array[from*64+to] of cutoff counts

const keyOf = (m) => m.from * 64 + m.to;

// Static evaluation from the side-to-move's perspective.
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

function search(state, depth, alpha, beta, ply, canNull, deadline) {
  if (now() > deadline) return 0; // aborted; the root discards this iteration
  if (ply >= MAX_PLY) return evalStm(state.board, state.turn);

  const inCheck = kingAttacked(state.board, state.turn);
  if (inCheck) depth++; // check extension
  if (depth <= 0) return qsearch(state, alpha, beta, QDEPTH);

  // Null-move pruning: pass the move; if we're still ≥ beta, this node fails high.
  if (canNull && !inCheck && depth >= 3 && beta < MATE - 1000 && hasNonPawn(state.board, state.turn)) {
    const nm = {
      board: state.board, turn: opponent(state.turn),
      castling: state.castling, halfmove: state.halfmove, fullmove: state.fullmove,
    };
    const score = -search(nm, depth - 3, -beta, -beta + 1, ply + 1, false, deadline);
    if (score >= beta) return beta;
  }

  const legal = legalMoves(state);
  if (legal.length === 0) return inCheck ? -MATE - depth : 0;
  orderMoves(legal, state.board, ply, 0);

  let best = -Infinity, moveCount = 0;
  for (const m of legal) {
    moveCount++;
    const child = applyMove(state, m);
    const quiet = !m.capture && !m.promotion && !m.jump;
    let score;
    if (moveCount === 1) {
      score = -search(child, depth - 1, -beta, -alpha, ply + 1, true, deadline);
    } else {
      // Late move reduction for quiet, late moves (never jumps/captures/promotions).
      const r = (quiet && depth >= 3 && moveCount > 3 && !inCheck) ? 1 : 0;
      score = -search(child, depth - 1 - r, -alpha - 1, -alpha, ply + 1, true, deadline);
      if (score > alpha && r > 0) score = -search(child, depth - 1, -alpha - 1, -alpha, ply + 1, true, deadline);
      if (score > alpha && score < beta) score = -search(child, depth - 1, -beta, -alpha, ply + 1, true, deadline);
    }
    if (score > best) best = score;
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
  return best;
}

// Choose a move for the side to move, searching up to `maxDepth` plies but never
// past `maxMs` of wall-clock. `rand` shuffles equal choices so games vary.
export function chooseMove(state, maxDepth = 2, rand = Math.random, maxMs = Infinity) {
  const root = legalMoves(state);
  if (root.length === 0) return null;

  for (let i = root.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [root[i], root[j]] = [root[j], root[i]];
  }

  killers = [];
  history = new Int32Array(64 * 64);
  const deadline = now() + maxMs;
  let bestMove = root[0];

  for (let depth = 1; depth <= maxDepth; depth++) {
    orderMoves(root, state.board, 0, keyOf(bestMove));
    let alpha = -Infinity, bestScore = -Infinity, localBest = root[0], aborted = false, moveCount = 0;
    for (const m of root) {
      moveCount++;
      const child = applyMove(state, m);
      let score;
      if (moveCount === 1) {
        score = -search(child, depth - 1, -Infinity, -alpha, 1, true, deadline);
      } else {
        score = -search(child, depth - 1, -alpha - 1, -alpha, 1, true, deadline);
        if (score > alpha) score = -search(child, depth - 1, -Infinity, -alpha, 1, true, deadline);
      }
      if (now() > deadline) { aborted = true; break; }
      if (score > bestScore) { bestScore = score; localBest = m; }
      if (score > alpha) alpha = score;
    }
    if (!aborted) bestMove = localBest;
    if (aborted || bestScore >= MATE - 1000) break;
  }
  return bestMove;
}
