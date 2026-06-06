// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2019-2026 Jean-David Moisan
//
// PGN-flavoured import/export. AposChess is a *variant* (bishop/rook jumps, knight
// slides, safety zones), so standard chess SAN can't represent its moves and stock
// PGN tools would reject both the movetext and a variant FEN. We therefore keep the
// PGN container — tag pairs, numbered movetext, a result token — but write each move
// in the app's long-algebraic form (the from-square is always present). That makes
// the round-trip exact: on import every token is matched back to a real engine move
// via legalMoves(), the same reconstruction online play uses so two games can't
// diverge. The from/to separator is ASCII ('-' quiet, 'x' capture) and promotion is
// '=Q', so the text parses cleanly without the en-dash/× glyphs the on-screen move
// list uses for display.

import { newGameState, parseFen, toFen, squareName } from './board.js';
import { legalMoves, applyMove } from './engine.js';

const START_FEN = toFen(newGameState());
const sqIndex = (name) => (name.charCodeAt(1) - 49) * 8 + (name.charCodeAt(0) - 97);

// --- export ------------------------------------------------------------------

function moveToken(pre, move) {
  if (move.castle === 'K') return 'O-O';
  if (move.castle === 'Q') return 'O-O-O';
  const piece = pre.board[move.from];
  const letter = piece.role === 'p' ? '' : piece.role.toUpperCase();
  const sep = move.capture ? 'x' : '-';
  const promo = move.promotion ? '=' + move.promotion.toUpperCase() : '';
  return letter + squareName(move.from) + sep + squareName(move.to) + promo;
}

function resultToken(status) {
  if (!status || !status.over) return '*';
  if (status.result === 'checkmate') return status.winner === 'white' ? '1-0' : '0-1';
  return '1/2-1/2'; // stalemate / fifty-move / repetition
}

// Wrap movetext at ~80 columns on token boundaries (PGN convention).
function wrap(text) {
  let line = '', out = '';
  for (const w of text.split(' ')) {
    if (line && line.length + 1 + w.length > 80) { out += line + '\n'; line = w; }
    else line = line ? line + ' ' + w : w;
  }
  return out + line;
}

// Build a .pgn string from main.js's `history` (per-ply snapshots, each with
// `.state` and `.lastMove`) and the final game `status`.
export function exportPgn(history, status) {
  const start = history[0].state;
  const result = resultToken(status);

  const tags = [
    ['Event', 'AposChess'],
    ['Site', 'AposChess'],
    ['Date', new Date().toISOString().slice(0, 10).replace(/-/g, '.')],
    ['Variant', 'AposChess'],
    ['Result', result],
  ];
  if (toFen(start) !== START_FEN) { tags.push(['SetUp', '1'], ['FEN', toFen(start)]); }

  let movetext = '';
  for (let k = 1; k < history.length; k++) {
    if (k % 2 === 1) movetext += `${(k + 1) / 2}. `;
    movetext += moveToken(history[k - 1].state, history[k].lastMove) + ' ';
  }
  movetext += result;

  return tags.map(([k, v]) => `[${k} "${v}"]`).join('\n') + '\n\n' + wrap(movetext) + '\n';
}

// --- import ------------------------------------------------------------------

const RESULTS = new Set(['1-0', '0-1', '1/2-1/2', '*']);

// Resolve one movetext token against the legal moves of `state`. Castling matches
// by the move's castle flag; everything else by from/to squares (+ promotion).
function matchToken(state, raw) {
  const token = raw.replace(/[+#!?]+$/g, ''); // tolerate check/mate/annotation suffixes
  const legal = legalMoves(state);
  if (token === 'O-O' || token === '0-0') return legal.find((m) => m.castle === 'K') || null;
  if (token === 'O-O-O' || token === '0-0-0') return legal.find((m) => m.castle === 'Q') || null;

  const m = token.match(/^[KQRBN]?([a-h][1-8])[x-]([a-h][1-8])(?:=([QRBNqrbn]))?$/);
  if (!m) return null;
  const from = sqIndex(m[1]), to = sqIndex(m[2]);
  const promo = m[3] ? m[3].toLowerCase() : null;
  return legal.find((mv) => mv.from === from && mv.to === to && (mv.promotion || null) === promo) || null;
}

// Parse a PGN string into { start, moves } with `moves` as reconstructed engine
// move objects (ready to replay with applyMove). Throws on an unparseable/illegal
// move so the caller can surface a clear error.
export function importPgn(text) {
  const tags = {};
  const tagRe = /\[(\w+)\s+"([^"]*)"\]/g;
  let t;
  while ((t = tagRe.exec(text)) !== null) tags[t[1]] = t[2];

  const start = tags.FEN ? parseFen(tags.FEN) : newGameState();

  // Strip tags, comments {…}, variations (…), NAGs ($n), and move numbers (1. / 1...).
  const body = text
    .replace(tagRe, ' ')
    .replace(/\{[^}]*\}/g, ' ')
    .replace(/\([^)]*\)/g, ' ')
    .replace(/\$\d+/g, ' ')
    .replace(/\d+\.(\.\.)?/g, ' ');

  const tokens = body.split(/\s+/).filter((tok) => tok && !RESULTS.has(tok));

  let state = start;
  const moves = [];
  for (const tok of tokens) {
    const mv = matchToken(state, tok);
    if (!mv) throw new Error(`Unparseable or illegal move: "${tok}"`);
    moves.push(mv);
    state = applyMove(state, mv);
  }
  return { start, moves };
}
