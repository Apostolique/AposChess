// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2019-2026 Jean-David Moisan
//
// Game-primary self-play record: the shared boundary every offline tool reads/writes.
//
// One JSONL line per GAME (not per position) — far smaller than the old position-primary
// {fen,r,g,v,vs} (a ~60 B FEN repeated every ply) and it records WHO PLAYED, not only who
// labeled each position:
//
//   { "g":"m5l-27",
//     "players":{"w":"nn8@a14d52","b":"nn8@a14d52"},  // engine×depth vtag per colour
//     "r":1,                                          // WHITE-view result 1/0/-1
//     "moves":["e2e4","e7e5","g1f3",...],             // compact from+to[promo]
//     "v":[-28,42,-11,...],                           // per-position cp, side-to-move view
//     "vs":"nn6@a14d52" }                             // scalar if uniform, else array || v
//
// Invariants:
//   - len(v) === len(moves) + 1. Position 0 is the start; moves connect consecutive
//     RECORDED positions; the final move-to-terminal isn't stored (the terminal had no
//     search). vs is parallel to v.
//   - `start` (a FEN) is omitted for the standard start position (the only current case);
//     readers default to newGameState().
//   - A move is "<from><to>[promo]" (e.g. "e2e4", "e7e8q"); castling is the KING's
//     from→to. from+to+promo uniquely identifies a legal move (asserted by the parity
//     harness), so replay matches it against legalMoves — no extra flags stored.

import { newGameState, parseFen, toFen, squareName, parseSquare } from '../src/board.js';
import { legalMoves, applyMove } from '../src/engine.js';

// --- compact move codec ----------------------------------------------------------
export function encodeMove(m) {
  return squareName(m.from) + squareName(m.to) + (m.promotion || '');
}

// Decode a token back to the real engine Move (with its castle/capture/jump/promotion
// flags) by matching it against the position's legal moves — applyMove needs those flags.
// Throws if no legal move matches (a corrupt/illegal token), so a bad record fails loudly
// instead of silently desyncing the replay.
export function decodeMove(state, token) {
  const from = parseSquare(token.slice(0, 2));
  const to = parseSquare(token.slice(2, 4));
  const promo = token.length > 4 ? token[4] : null;
  for (const m of legalMoves(state)) {
    if (m.from === from && m.to === to && (m.promotion || null) === promo) return m;
  }
  throw new Error(`no legal move matches "${token}" in ${toFen(state)}`);
}

// --- record helpers --------------------------------------------------------------
export const isGameRecord = (rec) => rec != null && Array.isArray(rec.moves);

// The provenance tag for position i (scalar vs applies to the whole game).
export const vsAt = (rec, i) => (Array.isArray(rec.vs) ? rec.vs[i] : rec.vs);

// Set the provenance for position i, promoting a scalar vs to a per-position array the
// moment two positions disagree (a partial refresh). Keeps vs scalar while it stays
// uniform — the common, compact case.
export function setVsAt(rec, i, tag) {
  if (Array.isArray(rec.vs)) { rec.vs[i] = tag; return; }
  if (rec.vs === tag) return;                       // still uniform
  const n = rec.v ? rec.v.length : rec.moves.length + 1;
  rec.vs = new Array(n).fill(rec.vs);
  rec.vs[i] = tag;
}

// Collapse a per-position vs array back to a scalar when every entry is equal — so a
// fully-refreshed game re-shrinks instead of carrying a redundant array forever.
export function normalizeVs(rec) {
  if (!Array.isArray(rec.vs)) return rec;
  const first = rec.vs[0];
  if (rec.vs.every((t) => t === first)) rec.vs = first;
  return rec;
}

// The start position of a game (defaults to the standard start).
export const startState = (rec) => (rec.start ? parseFen(rec.start) : newGameState());

// Replay a game record into its recorded positions — the exact analogue of the old
// position-primary records. Yields { state, r, v, vs, g, ply } for each of the
// len(moves)+1 positions: `state` is the full engine state (board/turn/castling/…), so
// callers featurize state.board/state.turn or toFen(state) for a search; `r` is derived
// side-to-move-relative from the game's White-view `r`. Each state is a fresh object
// (applyMove clones the board), so a caller may hold references across iterations.
export function* expandPositions(rec) {
  let state = startState(rec);
  const n = rec.moves.length;
  for (let i = 0; i <= n; i++) {
    yield {
      g: rec.g,
      ply: i,
      state,
      r: state.turn === 'white' ? rec.r : -rec.r,
      v: rec.v ? rec.v[i] : undefined,
      vs: vsAt(rec, i),
    };
    if (i < n) state = applyMove(state, decodeMove(state, rec.moves[i]));
  }
}

// Serialize a record to one JSONL line with a stable key order (start omitted when
// standard). vs is written scalar when uniform.
export function serializeGameRecord(rec) {
  const out = { g: rec.g };
  if (rec.start) out.start = rec.start;
  if (rec.players) out.players = rec.players;
  out.r = rec.r;
  out.moves = rec.moves;
  if (rec.v) out.v = rec.v;
  if (rec.vs !== undefined) out.vs = rec.vs;
  return JSON.stringify(out);
}

export const parseGameRecord = (line) => JSON.parse(line);

// Tally a game record's per-position `vs` tags into `tagCounts` (tag -> count), for the
// dataset cross-reference scans in rank-engines.mjs / depth-ladder.mjs. Returns the number
// of positions and how many lack a value. A scalar vs counts once per position.
export function tallyVs(rec, tagCounts) {
  const n = rec.v ? rec.v.length : rec.moves.length + 1;
  let missing = 0;
  for (let i = 0; i < n; i++) {
    const v = rec.v ? rec.v[i] : undefined;
    if (v == null) { missing++; continue; }
    const t = vsAt(rec, i);
    if (t) tagCounts.set(t, (tagCounts.get(t) || 0) + 1); else missing++;
  }
  return { positions: n, missing };
}
