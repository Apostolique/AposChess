// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2019-2026 Jean-David Moisan
//
// Board representation and FEN handling for AposChess.
//
// Squares are indexed 0..63 with index = rank * 8 + file, where rank 0 is
// White's back rank ("1") and file 0 is the a-file. This matches chessground's
// square names (a1..h8) via squareName().

export const FILES = 'abcdefgh';

// Direction vectors as [dFile, dRank].
export const ORTHO = [[0, 1], [0, -1], [1, 0], [-1, 0]];
export const DIAG = [[1, 1], [1, -1], [-1, 1], [-1, -1]];
export const ALL8 = [...ORTHO, ...DIAG];

export const START_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';

export function sq(file, rank) { return rank * 8 + file; }
export function fileOf(i) { return i % 8; }
export function rankOf(i) { return (i - (i % 8)) / 8; }
export function onBoard(file, rank) { return file >= 0 && file < 8 && rank >= 0 && rank < 8; }
export function squareName(i) { return FILES[fileOf(i)] + (rankOf(i) + 1); }
export function parseSquare(name) { return sq(FILES.indexOf(name[0]), parseInt(name[1], 10) - 1); }
export function opponent(color) { return color === 'white' ? 'black' : 'white'; }

// A piece is { role: 'p'|'n'|'b'|'r'|'q'|'k', color: 'white'|'black' }.
export function pieceFromChar(c) {
  const role = c.toLowerCase();
  return { role, color: c === role ? 'black' : 'white' };
}
export function charFromPiece(p) {
  return p.color === 'white' ? p.role.toUpperCase() : p.role;
}

export function parseFen(fen) {
  const [boardPart, turnPart, castlePart, , halfPart, fullPart] = fen.split(' ');
  const board = new Array(64).fill(null);
  const rows = boardPart.split('/'); // rows[0] is rank 8, rows[7] is rank 1
  for (let r = 0; r < 8; r++) {
    const row = rows[7 - r];
    let f = 0;
    for (const ch of row) {
      if (/\d/.test(ch)) f += parseInt(ch, 10);
      else { board[sq(f, r)] = pieceFromChar(ch); f++; }
    }
  }
  return {
    board,
    turn: turnPart === 'w' ? 'white' : 'black',
    castling: {
      K: castlePart.includes('K'), Q: castlePart.includes('Q'),
      k: castlePart.includes('k'), q: castlePart.includes('q'),
    },
    halfmove: parseInt(halfPart || '0', 10),
    fullmove: parseInt(fullPart || '1', 10),
  };
}

export function toFen(state) {
  const rows = [];
  for (let r = 7; r >= 0; r--) {
    let row = '', empty = 0;
    for (let f = 0; f < 8; f++) {
      const p = state.board[sq(f, r)];
      if (!p) { empty++; continue; }
      if (empty) { row += empty; empty = 0; }
      row += charFromPiece(p);
    }
    if (empty) row += empty;
    rows.push(row);
  }
  const c = state.castling;
  const castle = (c.K ? 'K' : '') + (c.Q ? 'Q' : '') + (c.k ? 'k' : '') + (c.q ? 'q' : '') || '-';
  return `${rows.join('/')} ${state.turn === 'white' ? 'w' : 'b'} ${castle} - ${state.halfmove} ${state.fullmove}`;
}

export function newGameState() { return parseFen(START_FEN); }
