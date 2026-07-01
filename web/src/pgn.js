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

// Moves from the root to a node (root = 0; a node's own move is its depth).
const depthOf = (node) => { let d = 0; for (let n = node; n.parent; n = n.parent) d++; return d; };

// One movetext token with its (optional) move number. White always prints its number
// (`12.`); Black prints `12...` only when it starts a line/variation or follows one.
function numberedToken(node, needNum) {
  const ply = depthOf(node);
  const white = ply % 2 === 1;
  const num = Math.ceil(ply / 2);
  const prefix = white ? `${num}. ` : (needNum ? `${num}... ` : '');
  return prefix + moveToken(node.parent.state, node.lastMove);
}

// Render the line beginning at move node `node`, following main children; at each branch
// its sibling variations are emitted (recursively) as parenthesised `(…)` blocks right
// after the main move — the same traversal the in-app move list uses.
function renderSeq(node, needNum) {
  let out = '';
  while (node) {
    out += (out ? ' ' : '') + numberedToken(node, needNum);
    needNum = false;
    const sibs = node.parent.children;
    if (sibs[0] === node && sibs.length > 1) {
      for (const sib of sibs.slice(1)) out += ' (' + renderSeq(sib, true) + ')';
      needNum = true; // the main line resumes after a variation block
    }
    node = node.children[0];
  }
  return out;
}

// Build a .pgn string from main.js's move-tree `root` (a node with `.state`/`.children`,
// each child a ply with `.lastMove`) and the final game `status`. Variations are written
// as nested `(…)` blocks. `players` carries the White/Black names (e.g. "Human" or
// "AI (depth 7, 6000ms)"); unknown sides fall back to PGN's "?".
export function exportPgn(root, status, players = {}) {
  const start = root.state;
  const result = resultToken(status);

  // The Seven Tag Roster (STR), in PGN's required order, comes first; supplemental
  // tags (Variant, SetUp/FEN) follow. Round is "-" — these are single casual games,
  // not tournament rounds.
  const tags = [
    ['Event', 'AposChess'],
    ['Site', 'AposChess'],
    ['Date', new Date().toISOString().slice(0, 10).replace(/-/g, '.')],
    ['Round', '-'],
    ['White', players.white || '?'],
    ['Black', players.black || '?'],
    ['Result', result],
    ['Variant', 'AposChess'],
  ];
  if (toFen(start) !== START_FEN) { tags.push(['SetUp', '1'], ['FEN', toFen(start)]); }

  const moves = root.children.length ? renderSeq(root.children[0], true) : '';
  const movetext = (moves ? moves + ' ' : '') + result;

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

// Decode one compact self-play token ("e2e4", "e7e8q") against `state`'s legal
// moves — the dataset's move codec (see scripts/gameRecord.mjs encodeMove). Same
// from/to(+promo) matching as matchToken, minus the PGN piece-letter/separator.
function matchCompact(state, token) {
  const m = token.match(/^([a-h][1-8])([a-h][1-8])([qrbn])?$/);
  if (!m) return null;
  const from = sqIndex(m[1]), to = sqIndex(m[2]);
  const promo = m[3] || null;
  return legalMoves(state).find((mv) => mv.from === from && mv.to === to && (mv.promotion || null) === promo) || null;
}

// A bare move-tree node. main.js fills in id/san/check/evals when it installs the tree;
// here we only need the position, the move that produced it, and the parent/child links.
const treeNode = (parent, state, lastMove) => ({ state, lastMove, parent, children: [] });

// Parse one self-play JSONL record (one GAME per line; see scripts/gameRecord.mjs) into
// the same { root, white, black } shape as importPgn — so pasting a single dataset line
// replays/visualizes that recorded game (a linear tree). Participant labels come from
// `players` (the engine@depth vtag per colour) or fall back to the game id `g`.
function importGameLine(line) {
  let rec;
  try { rec = JSON.parse(line); }
  catch { throw new Error('Not a valid PGN or self-play game line'); }
  if (!Array.isArray(rec.moves)) throw new Error('Not a self-play game record (no "moves" array)');

  const root = treeNode(null, rec.start ? parseFen(rec.start) : newGameState(), null);
  let cur = root;
  for (const tok of rec.moves) {
    const mv = matchCompact(cur.state, tok);
    if (!mv) throw new Error(`Unparseable or illegal move: "${tok}"`);
    const child = treeNode(cur, applyMove(cur.state, mv), mv);
    cur.children.push(child);
    cur = child;
  }
  const players = rec.players || {};
  const fallback = rec.g ? `Game ${rec.g}` : null;
  return { root, white: players.w || fallback, black: players.b || fallback };
}

// Split movetext into a token stream, dropping tags, `{…}` comments, `$n` NAGs, move
// numbers, and result tokens, but keeping `(` and `)` as their own tokens so variations
// can be parsed structurally.
function tokenizeMovetext(body) {
  const tokens = [];
  for (let i = 0; i < body.length;) {
    const c = body[i];
    if (c === '{') { const e = body.indexOf('}', i); i = e < 0 ? body.length : e + 1; continue; }
    if (c === '(' || c === ')') { tokens.push(c); i++; continue; }
    if (/\s/.test(c)) { i++; continue; }
    let j = i;
    while (j < body.length && !/[\s(){}]/.test(body[j])) j++;
    let word = body.slice(i, j).replace(/^\d+\.(\.\.)?/, ''); // strip a leading "12."/"12..." glued to the move
    i = j;
    if (!word || /^\$\d+$/.test(word) || /^\d+\.*$/.test(word) || RESULTS.has(word)) continue;
    tokens.push(word);
  }
  return tokens;
}

// Recursive-descent parse of the token stream into the move tree. `cur` is the node whose
// state is the current position; the next move appends a child to it. A `(` opens a
// variation that is an alternative to the LAST move played, so it branches from that
// move's parent; `)` closes the current variation and returns to the caller's line.
function parseMovetext(cur, tokens, pos) {
  while (pos.i < tokens.length) {
    const tok = tokens[pos.i++];
    if (tok === ')') return;
    if (tok === '(') { parseMovetext(cur.parent || cur, tokens, pos); continue; }
    const mv = matchToken(cur.state, tok);
    if (!mv) throw new Error(`Unparseable or illegal move: "${tok}"`);
    const child = treeNode(cur, applyMove(cur.state, mv), mv);
    cur.children.push(child);
    cur = child;
  }
}

// Parse a PGN string (or a single self-play JSONL line) into { root, white, black }: a
// move tree (variations preserved as sibling branches) and participant labels (or null).
// A line that begins with '{' is treated as a self-play record; otherwise it's parsed as
// PGN. Throws on an unparseable/illegal move so the caller can surface a clear error.
export function importPgn(text) {
  const trimmed = text.trim();
  if (trimmed.startsWith('{')) {
    // Tolerate a multi-line paste (a chunk of the dataset) by replaying the first record.
    const first = trimmed.split(/\r?\n/).find((l) => l.trim().startsWith('{'));
    return importGameLine(first.trim());
  }

  const tags = {};
  const tagRe = /\[(\w+)\s+"([^"]*)"\]/g;
  let t;
  while ((t = tagRe.exec(text)) !== null) tags[t[1]] = t[2];

  const root = treeNode(null, tags.FEN ? parseFen(tags.FEN) : newGameState(), null);
  const tokens = tokenizeMovetext(text.replace(tagRe, ' '));
  parseMovetext(root, tokens, { i: 0 });
  return { root, white: tags.White || null, black: tags.Black || null };
}
