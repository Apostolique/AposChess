// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2019-2026 Jean-David Moisan
//
// AposChess move generation and game rules.
//
// The variant differs from standard chess in these ways (see README.md):
//   - Pawns move one square diagonally (non-capturing), capture one square
//     straight forward, and may advance two straight forward on the first move.
//   - Bishops and Rooks slide normally and may ALSO jump the first piece in a
//     line, landing on the very next square (capturing there or landing empty).
//   - Knights no longer jump: they travel like a rook (clear path, any distance)
//     then step one square to the side.
//   - Queens and Kings each project a 3x3 "safety zone"; a jumping piece may not
//     land on any square in a zone. (A king's own zone means it can never be
//     captured or checked by a jump.)
//
// Standard rules are kept for normal sliding, castling, and check/checkmate.

import {
  sq, fileOf, rankOf, onBoard, squareName, opponent, ORTHO, DIAG, ALL8,
} from './board.js';

const SLIDE = { b: DIAG, r: ORTHO, q: ALL8 };

function mv(from, to, extra) { return Object.assign({ from, to }, extra); }

export function findKing(board, color) {
  for (let i = 0; i < 64; i++) {
    const p = board[i];
    if (p && p.role === 'k' && p.color === color) return i;
  }
  return -1;
}

// Squares that no jumping piece may land on: every square within king-distance 1
// of any king or queen (either colour), including the piece's own square.
export function safetyZones(board) {
  const set = new Set();
  for (let i = 0; i < 64; i++) {
    const p = board[i];
    if (!p || (p.role !== 'q' && p.role !== 'k')) continue;
    const f = fileOf(i), r = rankOf(i);
    for (let df = -1; df <= 1; df++) {
      for (let dr = -1; dr <= 1; dr++) {
        if (onBoard(f + df, r + dr)) set.add(sq(f + df, r + dr));
      }
    }
  }
  return set;
}

// --- per-piece pseudo-move generators (do not consider leaving own king in check) ---

function addPawn(moves, from, to, color, extra) {
  const promoRank = color === 'white' ? 7 : 0;
  if (rankOf(to) === promoRank) {
    for (const role of ['q', 'r', 'b', 'n']) moves.push(mv(from, to, { ...extra, promotion: role }));
  } else {
    moves.push(mv(from, to, extra));
  }
}

function pawnMoves(board, i, color, moves) {
  const fwd = color === 'white' ? 1 : -1;
  const startRank = color === 'white' ? 1 : 6;
  const f = fileOf(i), r = rankOf(i);

  // Diagonal forward: move to an empty square only (cannot capture diagonally).
  for (const df of [-1, 1]) {
    const nf = f + df, nr = r + fwd;
    if (onBoard(nf, nr) && !board[sq(nf, nr)]) addPawn(moves, i, sq(nf, nr), color, {});
  }
  // Straight forward: capture only (must be an enemy piece).
  if (onBoard(f, r + fwd)) {
    const t = board[sq(f, r + fwd)];
    if (t && t.color !== color) addPawn(moves, i, sq(f, r + fwd), color, { capture: true });
  }
  // First move: advance two straight forward over empty squares (non-capturing).
  if (r === startRank && !board[sq(f, r + fwd)] && !board[sq(f, r + 2 * fwd)]) {
    moves.push(mv(i, sq(f, r + 2 * fwd), { double: true }));
  }
}

function knightMoves(board, i, color, moves) {
  const f = fileOf(i), r = rankOf(i);
  const seen = new Set();
  for (const [df, dr] of ORTHO) {
    // Travel like a rook: each square along the way must be empty (no jumping).
    for (let k = 1; ; k++) {
      const cf = f + df * k, cr = r + dr * k;
      if (!onBoard(cf, cr) || board[sq(cf, cr)]) break; // blocked: can't turn here or continue
      // Then step one square to either side, perpendicular to the travel.
      const perps = df === 0 ? [[1, 0], [-1, 0]] : [[0, 1], [0, -1]];
      for (const [pf, pr] of perps) {
        const tf = cf + pf, tr = cr + pr;
        if (!onBoard(tf, tr)) continue;
        const ti = sq(tf, tr);
        if (seen.has(ti)) continue;
        const t = board[ti];
        if (t && t.color === color) continue;
        seen.add(ti);
        moves.push(mv(i, ti, t ? { capture: true } : {}));
      }
    }
  }
}

function sliderMoves(board, i, color, dirs, moves) {
  const f = fileOf(i), r = rankOf(i);
  for (const [df, dr] of dirs) {
    let nf = f + df, nr = r + dr;
    while (onBoard(nf, nr)) {
      const ti = sq(nf, nr), t = board[ti];
      if (!t) { moves.push(mv(i, ti, {})); }
      else { if (t.color !== color) moves.push(mv(i, ti, { capture: true })); break; }
      nf += df; nr += dr;
    }
  }
}

// Jump over the first piece in each direction and land on the next square,
// unless that square is friendly-occupied or inside a safety zone.
function jumpMoves(board, i, color, dirs, zones, moves) {
  const f = fileOf(i), r = rankOf(i);
  for (const [df, dr] of dirs) {
    let nf = f + df, nr = r + dr;
    while (onBoard(nf, nr) && !board[sq(nf, nr)]) { nf += df; nr += dr; }
    if (!onBoard(nf, nr)) continue; // no piece to jump in this direction
    const lf = nf + df, lr = nr + dr;
    if (!onBoard(lf, lr)) continue; // jumped piece sits on the edge
    const li = sq(lf, lr);
    if (zones.has(li)) continue;
    const t = board[li];
    if (t && t.color === color) continue;
    moves.push(mv(i, li, t ? { capture: true, jump: true } : { jump: true }));
  }
}

function kingMoves(board, i, color, moves) {
  const f = fileOf(i), r = rankOf(i);
  for (const [df, dr] of ALL8) {
    const nf = f + df, nr = r + dr;
    if (!onBoard(nf, nr)) continue;
    const t = board[sq(nf, nr)];
    if (t && t.color === color) continue;
    moves.push(mv(i, sq(nf, nr), t ? { capture: true } : {}));
  }
}

// All pseudo-legal moves for `color` (excluding castling, which needs check info).
export function generatePseudoMoves(board, color) {
  const moves = [];
  const zones = safetyZones(board);
  for (let i = 0; i < 64; i++) {
    const p = board[i];
    if (!p || p.color !== color) continue;
    switch (p.role) {
      case 'p': pawnMoves(board, i, color, moves); break;
      case 'n': knightMoves(board, i, color, moves); break;
      case 'b': case 'r':
        sliderMoves(board, i, color, SLIDE[p.role], moves);
        jumpMoves(board, i, color, SLIDE[p.role], zones, moves);
        break;
      case 'q': sliderMoves(board, i, color, SLIDE.q, moves); break;
      case 'k': kingMoves(board, i, color, moves); break;
    }
  }
  return moves;
}

// Is `target` reachable by any pseudo-legal move of `byColor`? Since the target
// in practice is the king square (which always lies in its own safety zone),
// jumps can never reach it — so this correctly captures "is the king in check".
export function isAttacked(board, target, byColor) {
  const moves = generatePseudoMoves(board, byColor);
  for (const m of moves) if (m.to === target) return true;
  return false;
}

export function applyMove(state, m) {
  const board = state.board.slice();
  const piece = board[m.from];
  const color = piece.color;

  board[m.to] = m.promotion ? { role: m.promotion, color } : piece;
  board[m.from] = null;

  if (m.castle) {
    const home = color === 'white' ? 0 : 56;
    if (m.castle === 'K') { board[home + 5] = board[home + 7]; board[home + 7] = null; }
    else { board[home + 3] = board[home + 0]; board[home + 0] = null; }
  }

  const castling = { ...state.castling };
  if (piece.role === 'k') {
    if (color === 'white') { castling.K = false; castling.Q = false; }
    else { castling.k = false; castling.q = false; }
  }
  // A rook leaving or being captured on its home square removes that right.
  for (const idx of [m.from, m.to]) {
    if (idx === 0) castling.Q = false;
    else if (idx === 7) castling.K = false;
    else if (idx === 56) castling.q = false;
    else if (idx === 63) castling.k = false;
  }

  const reset = piece.role === 'p' || m.capture;
  return {
    board,
    turn: opponent(color),
    castling,
    halfmove: reset ? 0 : state.halfmove + 1,
    fullmove: color === 'black' ? state.fullmove + 1 : state.fullmove,
  };
}

function addCastling(state, color, legal) {
  const board = state.board;
  const c = state.castling;
  const enemy = opponent(color);
  const home = color === 'white' ? 0 : 56;
  const kingIdx = home + 4;
  const king = board[kingIdx];
  if (!king || king.role !== 'k' || king.color !== color) return;
  if (isAttacked(board, kingIdx, enemy)) return; // cannot castle out of check

  const canK = color === 'white' ? c.K : c.k;
  const canQ = color === 'white' ? c.Q : c.q;
  const rookOk = (idx) => board[idx] && board[idx].role === 'r' && board[idx].color === color;

  if (canK && !board[home + 5] && !board[home + 6] && rookOk(home + 7)
      && !isAttacked(board, home + 5, enemy) && !isAttacked(board, home + 6, enemy)) {
    legal.push(mv(kingIdx, home + 6, { castle: 'K' }));
  }
  if (canQ && !board[home + 3] && !board[home + 2] && !board[home + 1] && rookOk(home + 0)
      && !isAttacked(board, home + 3, enemy) && !isAttacked(board, home + 2, enemy)) {
    legal.push(mv(kingIdx, home + 2, { castle: 'Q' }));
  }
}

export function legalMoves(state) {
  const color = state.turn;
  const legal = [];
  for (const m of generatePseudoMoves(state.board, color)) {
    const next = applyMove(state, m);
    const ksq = findKing(next.board, color);
    if (ksq >= 0 && !isAttacked(next.board, ksq, opponent(color))) legal.push(m);
  }
  addCastling(state, color, legal);
  return legal;
}

export function gameStatus(state) {
  const color = state.turn;
  const legal = legalMoves(state);
  const ksq = findKing(state.board, color);
  const inCheck = ksq >= 0 && isAttacked(state.board, ksq, opponent(color));
  if (legal.length === 0) {
    return { over: true, check: inCheck, legal, result: inCheck ? 'checkmate' : 'stalemate', winner: inCheck ? opponent(color) : null };
  }
  if (state.halfmove >= 100) {
    return { over: true, check: inCheck, legal, result: 'fifty-move', winner: null };
  }
  return { over: false, check: inCheck, legal, result: null, winner: null };
}

// Map of from-square name -> array of to-square names, for chessground's `dests`.
export function destsMap(legal) {
  const map = new Map();
  for (const m of legal) {
    const from = squareName(m.from), to = squareName(m.to);
    if (!map.has(from)) map.set(from, []);
    const arr = map.get(from);
    if (!arr.includes(to)) arr.push(to);
  }
  return map;
}
