// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2019-2026 Jean-David Moisan
//
// Worker for scripts/mine-puzzles.mjs. Takes candidate positions (FENs flagged by
// the main thread's blunder scan) and turns the ones that qualify into puzzles:
//
//   1. Verify   — search at full depth for the best move and its score, then
//                 re-search with that move excluded for the runner-up. A puzzle
//                 needs a UNIQUE winning move: best decisively winning (or mate),
//                 runner-up clearly not. Positions where several moves win, or
//                 where nothing does, are rejected.
//   2. Solution — walk the line forward: the engine's best reply for the
//                 opponent, then the solver's next move for as long as it stays
//                 unique (each solver ply re-passes the uniqueness test, so the
//                 player is never asked to guess between two equally good moves).
//                 Mate lines run to checkmate; winning lines end when the win is
//                 converted (several good moves exist).
//   3. Themes   — tag what makes the line interesting, with the variant's
//                 surprises first-class: jump / jump-capture (the move carries
//                 the generator's jump flag), jump-block (a king/queen move whose
//                 safety zone newly repels an enemy jump), knight (the no-jump
//                 rook-path knight), knight-block (a move that cuts an enemy
//                 knight's travel path — the variant's weird "body-block"
//                 defense), promotion, sacrifice.
//   4. Difficulty — the shallowest search depth whose best move already is (and
//                 stays) the solution move, from the iterative ladder run in (1).
//                 Depth 1 = "take the hanging piece"; the deeper the engine has
//                 to look, the harder the tactic.
//
// The TT is reset per candidate so verification is independent of whatever was
// searched before; within a candidate the ladder/uniqueness/line searches share
// the warm table (correct — entries are position-keyed).

import { parentPort, workerData } from 'node:worker_threads';
import { readFileSync } from 'node:fs';

import { parseFen, toFen, squareName, opponent } from '../src/board.js';
import { legalMoves, applyMove, gameStatus, generatePseudoMoves, safetyZones } from '../src/engine.js';
import { chooseMoveDetailed, _internal } from '../src/ai.js';
import { loadWeights, hasWeights } from '../src/nn.js';

const { weights, depth, win, second, maxSolverMoves } = workerData;
const { resetTT, MATE_THRESH } = _internal;

// Same engine-selection policy as the rest of the tools: the champion net when it
// loads, otherwise the handcrafted eval (NOT nn's material fallback — weak evals
// make wrong puzzles, where the "solution" doesn't actually win).
let engine = 'handcrafted';
if (workerData.eval === 'nn') {
  try {
    loadWeights(JSON.parse(readFileSync(weights, 'utf8')));
    if (hasWeights()) engine = 'nn';
  } catch { /* fall through to handcrafted */ }
}

function mulberry32(a) {
  a >>>= 0;
  return () => { a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

const keyOf = (m) => m.from * 64 + m.to;
const VALUE = { p: 100, n: 300, b: 330, r: 500, q: 900, k: 0 };
const uci = (m) => squareName(m.from) + squareName(m.to) + (m.promotion || '');

// Search wrappers: full strength, deterministic per candidate via the seeded rand.
const best = (state, rand, d = depth) => chooseMoveDetailed(state, d, rand, Infinity, true, [], engine);
const secondBest = (state, rand, excludeMove) =>
  chooseMoveDetailed(state, depth, rand, Infinity, true, [], engine, new Set([keyOf(excludeMove)]));

// Is `m` the only move good enough here? `score` is m's already-computed score.
// Unique = m wins (or mates) while the runner-up clearly doesn't. A position with
// a single legal move is trivially unique (and free to test).
function uniqueness(state, legal, m, score, rand) {
  if (legal.length === 1) return { unique: true, second: -Infinity };
  const alt = secondBest(state, rand, m);
  const s2 = alt.score;
  if (score >= MATE_THRESH) {
    // Mate: reject only if an equally-fast mate exists (s2 >= score means the
    // alternative mates at least as quickly — the solver shouldn't have to guess
    // between them). A slower mate or a mere material win still counts as unique
    // *for line continuation*, but for the FIRST move we also want the win itself
    // to be non-obvious — the caller applies the stricter `second` cap there.
    return { unique: s2 < score, second: s2 };
  }
  return { unique: score >= win && s2 <= second, second: s2 };
}

// Variant-aware theme tags for a solver move `m` played from `state`.
function themesOf(state, m) {
  const t = [];
  const piece = state.board[m.from];
  if (m.jump) t.push(m.capture ? 'jump-capture' : 'jump');
  if (piece.role === 'n') t.push('knight');
  if (m.promotion) t.push('promotion');

  // Block detection: the board changes only at m.from (vacated — only ever ADDS
  // enemy moves) and m.to (occupied — removes enemy moves through/onto it), so an
  // enemy pseudo-move that disappears, while its piece survives, was blocked by
  // this move. (A move whose target became m.to turns into a capture with the
  // same from-to key, so it doesn't read as removed.)
  const enemy = opponent(state.turn);
  const before = generatePseudoMoves(state.board, enemy);
  const after = applyMove(state, m);
  const afterKeys = new Set(generatePseudoMoves(after.board, enemy).map(keyOf));
  const removed = before.filter((em) => !afterKeys.has(keyOf(em))
    && after.board[em.from] && after.board[em.from].color === enemy);

  // jump-block: a king/queen move whose safety zone now covers a landing square an
  // enemy jump could previously reach — the variant's signature defensive resource.
  // Only a blocked jump-CAPTURE of real value counts: every queen move reshapes the
  // zones and incidentally kills some empty-landing jump, which isn't the point.
  if (piece.role === 'q' || piece.role === 'k') {
    const zones = safetyZones(after.board, state.turn);
    if (removed.some((em) => em.jump && em.capture && zones[em.to]
        && after.board[em.to] && VALUE[after.board[em.to].role] >= 300)) t.push('jump-block');
  }
  // knight-block: an enemy knight lost moves because our piece now sits on its
  // travel path — blocking a non-jumping knight needs a body in the way, which
  // looks bizarre coming from regular chess.
  if (removed.some((em) => state.board[em.from].role === 'n' && !em.jump)) t.push('knight-block');

  // sacrifice: the move leaves the moved piece where the opponent can LEGALLY take
  // it for a clear material gain — the classic "surprising move" marker. (Legal,
  // not pseudo: a defended piece "capturable" only by the enemy king isn't loose.)
  if (piece.role !== 'p' && !m.promotion) {
    const net = (m.capture && state.board[m.to] ? VALUE[state.board[m.to].role] : 0) - VALUE[piece.role];
    if (net <= -200 && legalMoves(after).some((em) => em.to === m.to && em.capture)) t.push('sacrifice');
  }
  return t;
}

// Which legal move turns the pre-blunder position into the puzzle position? The
// dataset stores positions, not moves, but consecutive positions of a game pin the
// move down uniquely (two distinct from-squares always leave distinct boards), so
// matching on board+turn+castling (move counters excluded) recovers it. null when
// it can't be derived (e.g. legacy castling-rights drift) — the lead-in is optional.
const posKey = (fen) => fen.split(' ').slice(0, 3).join(' ');
function deriveOpening(prevFen, fen) {
  if (!prevFen) return null;
  let pre;
  try { pre = parseFen(prevFen); } catch { return null; }
  const target = posKey(fen);
  for (const m of legalMoves(pre)) {
    if (posKey(toFen(applyMove(pre, m))) === target) return uci(m);
  }
  return null;
}

function minePuzzle(item) {
  const rand = mulberry32(item.seed);
  const state = parseFen(item.fen);
  const status = gameStatus(state);
  if (status.over) return { reject: 'over' };
  if (status.legal.length < 2) return { reject: 'forced' }; // nothing to find

  resetTT();

  // Difficulty ladder: best move at each depth 1..D (the TT is shared, so this
  // costs about one full iterative-deepening search). The final rung is the
  // verification search itself.
  const ladder = [];
  for (let d = 1; d <= depth; d++) ladder.push(best(state, rand, d));
  const top = ladder[depth - 1];
  const s1 = top.score;
  const isMate = s1 >= MATE_THRESH;

  if (!isMate && s1 < win) return { reject: 'no-win' };
  const u = uniqueness(state, status.legal, top.move, s1, rand);
  // First move: strict — even a mate is only a puzzle if the alternatives don't
  // ALSO leave the solver clearly better (otherwise "any move wins" boredom).
  if (!u.unique || u.second > second) return { reject: 'not-unique' };

  // Shallowest depth from which the engine's choice is (and stays) the solution.
  let solveDepth = depth;
  const bk = keyOf(top.move);
  for (let d = depth - 1; d >= 1 && keyOf(ladder[d - 1].move) === bk; d--) solveDepth = d;

  // --- walk the solution line -------------------------------------------------
  const line = [top.move];
  const themes = new Set(themesOf(state, top.move));
  let cur = applyMove(state, top.move);
  let kind = 'win';
  // Accepted-sacrifice detection: run the line's material ledger (captures only)
  // and remember how deep the solver goes into the red along the way. A combination
  // that gives up a piece (or the queen) before the tactic pays off is exactly the
  // "huge sac into a tactic" puzzle — the per-move en-prise check in themesOf only
  // sees sacs the opponent could DECLINE; this catches the ones the line accepts.
  const capValue = (s, m) => (m.capture && s.board[m.to] ? VALUE[s.board[m.to].role] : 0);
  let matDelta = capValue(state, top.move);
  let minDelta = 0;

  for (;;) {
    let st = gameStatus(cur);
    if (st.over) {
      if (st.result !== 'checkmate') return { reject: 'bogus-end' }; // "winning" into a draw
      kind = 'mate';
      break;
    }
    if (!isMate && (line.length + 1) / 2 > maxSolverMoves) break;

    const reply = best(cur, rand);
    const afterReply = applyMove(cur, reply.move);
    st = gameStatus(afterReply);
    if (st.over) break; // opponent's best self-mates/stalemates — end before it

    const next = best(afterReply, rand);
    if (isMate && next.score < MATE_THRESH) return { reject: 'unsound' }; // search disagrees with itself
    const nu = uniqueness(afterReply, st.legal, next.move, next.score, rand);
    if (!nu.unique) break; // several good moves: the player has converted — stop here

    // The pair is now part of the line — ledger the reply's capture (a break above
    // means the reply never happens, so its capture must not count).
    matDelta -= capValue(cur, reply.move);
    minDelta = Math.min(minDelta, matDelta);
    line.push(reply.move, next.move);
    for (const th of themesOf(afterReply, next.move)) themes.add(th);
    matDelta += capValue(afterReply, next.move);
    cur = applyMove(afterReply, next.move);
  }
  if (minDelta <= -250) themes.add('sacrifice'); // down a minor piece or more mid-line

  const solverMoves = Math.ceil(line.length / 2);
  const puzzle = {
    id: item.id,
    fen: toFen(state),
    moves: line.map(uci),
    kind,
    themes: [...themes],
    difficulty: solveDepth,
    score: Math.round(s1),
  };
  if (kind === 'mate') puzzle.mateIn = solverMoves;
  const opening = deriveOpening(item.prevFen, item.fen);
  if (opening) { puzzle.prefen = item.prevFen; puzzle.opening = opening; }
  return { puzzle };
}

parentPort.on('message', (msg) => {
  if (msg.type !== 'batch') return;
  const results = msg.items.map((item) => {
    try {
      return { idx: item.idx, ...minePuzzle(item) };
    } catch (e) {
      return { idx: item.idx, reject: 'error', error: String(e && e.message || e) };
    }
  });
  parentPort.postMessage({ type: 'done', results });
});
parentPort.postMessage({ type: 'ready', engine });
