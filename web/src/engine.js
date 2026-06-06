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
//   - Queens and Kings each project a 3x3 "safety zone"; an *enemy* jumping piece
//     may not land on any square in that zone (your own jumps pass through freely).
//     (A king's zone repels enemy jumps, so a king can never be captured or
//     checked by a jump.)
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

// Squares an enemy jumping piece may not land on: every square within
// king-distance 1 of one of `color`'s kings or queens (including the piece's own
// square). A king/queen's zone only repels the *opponent's* jumps — your own
// pieces may freely jump into the zones your kings and queens project.
export function safetyZones(board, color) {
  const zones = new Uint8Array(64); // flag per square; far cheaper than a Set in the hot path
  for (let i = 0; i < 64; i++) {
    const p = board[i];
    if (!p || p.color !== color || (p.role !== 'q' && p.role !== 'k')) continue;
    const f = fileOf(i), r = rankOf(i);
    for (let df = -1; df <= 1; df++) {
      for (let dr = -1; dr <= 1; dr++) {
        if (onBoard(f + df, r + dr)) zones[sq(f + df, r + dr)] = 1;
      }
    }
  }
  return zones;
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
// unless that square is friendly-occupied or inside an enemy safety zone.
function jumpMoves(board, i, color, dirs, zones, moves) {
  const f = fileOf(i), r = rankOf(i);
  for (const [df, dr] of dirs) {
    let nf = f + df, nr = r + dr;
    while (onBoard(nf, nr) && !board[sq(nf, nr)]) { nf += df; nr += dr; }
    if (!onBoard(nf, nr)) continue; // no piece to jump in this direction
    const lf = nf + df, lr = nr + dr;
    if (!onBoard(lf, lr)) continue; // jumped piece sits on the edge
    const li = sq(lf, lr);
    if (zones[li]) continue;
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
  // A jumping piece is repelled only by the enemy's king/queen zones.
  const zones = safetyZones(board, opponent(color));
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
// in practice is the king square (which always lies in its own safety zone, and
// that zone repels the enemy's jumps), `byColor`'s jumps can never reach it — so
// this correctly captures "is the king in check".
export function isAttacked(board, target, byColor) {
  const moves = generatePseudoMoves(board, byColor);
  for (const m of moves) if (m.to === target) return true;
  return false;
}

// Fast check detection: is `color`'s king attacked? A king always sits in its
// own safety zone, so no jump can ever land on it — only normal moves can give
// check. We therefore scan outward from the king for each attack pattern instead
// of generating every enemy move (the hot path during search). This must stay
// equivalent to isAttacked(board, kingSquare, opponent); see the cross-check test.
export function kingAttacked(board, color) {
  return kingAttackedAt(board, color, findKing(board, color));
}

// Same test as kingAttacked, but with the king's square supplied so the legal-move
// filter can locate the king once instead of rescanning the board for every move.
function kingAttackedAt(board, color, k) {
  if (k < 0) return false;
  const enemy = color === 'white' ? 'black' : 'white';
  const kf = fileOf(k), kr = rankOf(k);
  const isEnemy = (f, r, role) => {
    if (!onBoard(f, r)) return false;
    const p = board[sq(f, r)];
    return p && p.color === enemy && p.role === role;
  };

  // Pawn: captures straight forward, so an enemy pawn one square "behind" the
  // king (from the enemy's point of view) attacks it.
  const ef = enemy === 'white' ? 1 : -1;
  if (isEnemy(kf, kr - ef, 'p')) return true;

  // Enemy king on an adjacent square.
  for (const [df, dr] of ALL8) if (isEnemy(kf + df, kr + dr, 'k')) return true;

  // Normal slides: the first piece along each ray, if it is the matching slider.
  for (const [df, dr] of ORTHO) {
    let f = kf + df, r = kr + dr;
    while (onBoard(f, r)) {
      const p = board[sq(f, r)];
      if (p) { if (p.color === enemy && (p.role === 'r' || p.role === 'q')) return true; break; }
      f += df; r += dr;
    }
  }
  for (const [df, dr] of DIAG) {
    let f = kf + df, r = kr + dr;
    while (onBoard(f, r)) {
      const p = board[sq(f, r)];
      if (p) { if (p.color === enemy && (p.role === 'b' || p.role === 'q')) return true; break; }
      f += df; r += dr;
    }
  }

  // Knight (rook path then one step to the side). Reverse the move: step from the
  // king opposite the side-step to the "corner" (must be empty, as in the forward
  // move), then scan straight back — an enemy knight is the first piece if the
  // travel path is clear.
  for (const [sf, sr] of ORTHO) {
    const cf = kf - sf, cr = kr - sr;
    if (!onBoard(cf, cr) || board[sq(cf, cr)]) continue;
    const perps = sf === 0 ? [[1, 0], [-1, 0]] : [[0, 1], [0, -1]];
    for (const [df, dr] of perps) {
      let f = cf - df, r = cr - dr;
      while (onBoard(f, r)) {
        const p = board[sq(f, r)];
        if (p) { if (p.color === enemy && p.role === 'n') return true; break; }
        f -= df; r -= dr;
      }
    }
  }
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
  const board = state.board;
  const kingSq = findKing(board, color);
  const legal = [];
  for (const m of generatePseudoMoves(board, color)) {
    // Make/unmake on the live board to test king safety, instead of cloning the
    // whole board (via applyMove) for every pseudo-move. The promoted role is
    // irrelevant here — only the moved piece's presence at `to` and absence at
    // `from` affect whether OUR king is left in check — so we slide the original
    // piece across and restore it. Pseudo-moves never castle, so no rook to move.
    const moved = board[m.from];
    const captured = board[m.to];
    board[m.to] = moved;
    board[m.from] = null;
    // The king is at kingSq unless the king itself just moved (then at m.to).
    const ok = !kingAttackedAt(board, color, moved.role === 'k' ? m.to : kingSq);
    board[m.from] = moved;
    board[m.to] = captured;
    if (ok) legal.push(m);
  }
  addCastling(state, color, legal);
  return legal;
}

// Draw by insufficient material. Only the unambiguous case is claimed: bare
// king vs bare king (no other pieces of any kind). The variant's pieces move
// differently enough that minor-piece mating potential isn't obvious, so we
// don't extend this to K+minor vs K.
export function insufficientMaterial(board) {
  for (const p of board) {
    if (p && p.role !== 'k') return false;
  }
  return true;
}

export function gameStatus(state) {
  const color = state.turn;
  const legal = legalMoves(state);
  const inCheck = kingAttacked(state.board, color);
  if (legal.length === 0) {
    return { over: true, check: inCheck, legal, result: inCheck ? 'checkmate' : 'stalemate', winner: inCheck ? opponent(color) : null };
  }
  if (insufficientMaterial(state.board)) {
    return { over: true, check: inCheck, legal, result: 'insufficient-material', winner: null };
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
