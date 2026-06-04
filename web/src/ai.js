// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2019-2026 Jean-David Moisan
//
// A small alpha-beta search over the AposChess move generator. Strength is
// modest by design — enough for a casual opponent, cheap enough to run on the
// UI thread between animated moves.

import { legalMoves, applyMove, findKing, isAttacked } from './engine.js';
import { opponent } from './board.js';

const VALUE = { p: 100, n: 300, b: 330, r: 500, q: 900, k: 0 };
const MATE = 1_000_000;

// Static evaluation from White's perspective (positive favours White).
function evaluate(board) {
  let score = 0;
  for (const p of board) {
    if (!p) continue;
    const v = VALUE[p.role];
    score += p.color === 'white' ? v : -v;
  }
  return score;
}

// Negamax with alpha-beta, returning the score for the side to move.
function search(state, depth, alpha, beta) {
  const legal = legalMoves(state);
  if (legal.length === 0) {
    const ksq = findKing(state.board, state.turn);
    const inCheck = ksq >= 0 && isAttacked(state.board, ksq, opponent(state.turn));
    return inCheck ? -MATE - depth : 0; // checkmate (prefer faster mates) or stalemate
  }
  if (depth === 0) {
    const e = evaluate(state.board);
    return state.turn === 'white' ? e : -e;
  }
  // Search captures first to improve pruning.
  legal.sort((a, b) => (b.capture ? 1 : 0) - (a.capture ? 1 : 0));
  let best = -Infinity;
  for (const m of legal) {
    const score = -search(applyMove(state, m), depth - 1, -beta, -alpha);
    if (score > best) best = score;
    if (best > alpha) alpha = best;
    if (alpha >= beta) break;
  }
  return best;
}

// Choose a move for the side to move, picking randomly among (near-)equal best
// moves so games are not identical every time. `rand` defaults to Math.random.
export function chooseMove(state, depth = 2, rand = Math.random) {
  const legal = legalMoves(state);
  if (legal.length === 0) return null;
  legal.sort((a, b) => (b.capture ? 1 : 0) - (a.capture ? 1 : 0));

  let bestScore = -Infinity;
  const scored = [];
  for (const m of legal) {
    // Full window at the root so every score is directly comparable for the tiebreak.
    const score = -search(applyMove(state, m), depth - 1, -Infinity, Infinity);
    scored.push({ m, score });
    if (score > bestScore) bestScore = score;
  }
  const top = scored.filter((s) => s.score >= bestScore - 1).map((s) => s.m);
  return top[Math.floor(rand() * top.length)];
}
