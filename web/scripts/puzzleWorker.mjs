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
//                 Mate lines run to checkmate — including lines that only BECOME
//                 a forced mate mid-walk, which are exempt from the solver-move
//                 cap so the puzzle ends with the mate on the board. Winning
//                 lines end when the win is converted (several good moves exist).
//                 The walk searches deeper than the root verification did, so it
//                 doubles as a soundness check: a line that runs into the solver
//                 being mated/stalemated, or a defense save the walk shows to be
//                 lost anyway, rejects the puzzle instead of trimming the line.
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
// Each candidate is mined from BOTH sides of the blunder:
//   - a WIN puzzle from the post-blunder position (punish it) — tagged
//     'razor-edge' when the runner-up move doesn't merely fail to win but LOSES
//     outright (play it right you win, play it wrong you lose);
//   - a DEFENSE puzzle from the PRE-blunder position (the game's own engine
//     fell off this tightrope, which is what nominated it): kind 'defense' when
//     exactly one move holds (best ≥ saveFloor) while every alternative loses
//     (runner-up ≤ -win) — tagged 'mate-threat' when the alternatives lose to
//     forced mate. If the pre-blunder best move turns out to WIN, it's mined as
//     a missed-win puzzle through the normal win path instead. Because the app's
//     play-it-out engine searches deeper than the mining depth, the end-of-line
//     position is re-searched two plies deeper before accepting — a "save" that
//     a deeper look shows lost anyway is rejected, not shipped.
//
// The TT is reset per mined position so verification is independent of whatever
// was searched before; within one position the ladder/uniqueness/line searches
// share the warm table (correct — entries are position-keyed).

import { parentPort, workerData } from 'node:worker_threads';
import { readFileSync } from 'node:fs';

import { parseFen, toFen, squareName, opponent } from '../src/board.js';
import { legalMoves, applyMove, gameStatus, generatePseudoMoves, safetyZones } from '../src/engine.js';
import { _internal } from '../src/ai.js'; // MATE_THRESH constant only (search runs in wasm)
import { makeEngine } from './wasmEngine.mjs';

const { weights, depth, win, second, lineGap, saveFloor, maxSolverMoves } = workerData;
const { MATE_THRESH } = _internal;

// Same engine-selection policy as the rest of the tools: the champion net when it's a
// real net, otherwise the handcrafted eval (NOT nn's material fallback — weak evals make
// wrong puzzles, where the "solution" doesn't actually win). The search runs in the native
// Zig engine (wasm); `engine` is kept for the labels/logging elsewhere.
let engine = 'handcrafted';
if (workerData.eval === 'nn') {
  try { if (JSON.parse(readFileSync(weights, 'utf8')).arch) engine = 'nn'; } catch { /* handcrafted */ }
}
const eng = makeEngine(engine, engine === 'nn' ? weights : null);

function mulberry32(a) {
  a >>>= 0;
  return () => { a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

const keyOf = (m) => m.from * 64 + m.to;
const VALUE = { p: 100, n: 300, b: 330, r: 500, q: 900, k: 0 };
const uci = (m) => squareName(m.from) + squareName(m.to) + (m.promotion || '');

// Search wrappers: full strength, deterministic per candidate via the seeded rand.
// `rand` is unused now (the wasm searcher has its own deterministic root variety); kept in
// the signatures so the call sites are unchanged.
const best = (state, rand, d = depth) => eng.searchMove(state, d, null);
const secondBest = (state, rand, excludeMove) => eng.searchMove(state, depth, [keyOf(excludeMove)]);

// Is `m` the only move good enough to BE the puzzle? (The first move's test — line
// continuations use the looser clearly-best rule in the walk below.) `score` is m's
// already-computed score. Unique = m wins (or mates) while the runner-up clearly
// doesn't. A position with a single legal move is trivially unique (and free to test).
function uniqueness(state, legal, m, score, rand) {
  if (legal.length === 1) return { unique: true, second: -Infinity };
  const alt = secondBest(state, rand, m);
  const s2 = alt.score;
  if (score >= MATE_THRESH) {
    // Mate: an equally-fast second mate (s2 >= score) means the solver would have
    // to guess between them — not unique. A slower mate or a mere material win is
    // fine here; the caller additionally applies the `second` cap so the win as a
    // whole is non-obvious.
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

function mineWin(fen, leadFen, id, seed) {
  const rand = mulberry32(seed);
  const state = parseFen(fen);
  const status = gameStatus(state);
  if (status.over) return { reject: 'over' };
  if (status.legal.length < 2) return { reject: 'forced' }; // nothing to find

  eng.resetTT();

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
  // Razor edge: not only is the key move the single way to win — every other move
  // outright LOSES. Maximum-tension puzzles; u.second is the best of the rest.
  if (u.second <= -win) themes.add('razor-edge');
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

    const reply = best(cur, rand);
    const afterReply = applyMove(cur, reply.move);
    st = gameStatus(afterReply);
    // After the reply the SOLVER is the side to move, so st.over means the walk
    // found the "win" running into a swindle — the solver mated or stalemated. The
    // claim the first move was verified on is false; drop the puzzle.
    if (st.over) return { reject: st.result === 'checkmate' ? 'unsound' : 'bogus-end' };

    const next = best(afterReply, rand);
    if (isMate && next.score < MATE_THRESH) return { reject: 'unsound' }; // search disagrees with itself
    // The solver-move cap only applies while the continuation is NOT a forced mate:
    // a line that walks into a mate runs to checkmate regardless, so "win the queen,
    // then mate in two" doesn't stop one move short of the finish.
    if (next.score < MATE_THRESH && (line.length + 1) / 2 > maxSolverMoves) break;
    // Continuation rule: unlike the FIRST move (which must be the only good one —
    // that's the puzzle's premise), a follow-up only has to be CLEARLY BEST, beating
    // the runner-up by lineGap. Otherwise the line stops the moment a second move
    // also wins, which cuts combinations off right before the payoff ("...and now
    // take the cornered queen" never appears because two captures both won). Two
    // genuinely equal continuations still stop the line — either works, and the app
    // can only script one. Mate lines instead stop on an equally-fast second mate.
    if (st.legal.length > 1) {
      const alt = secondBest(afterReply, rand, next.move);
      // A mate continuation normally ends the line when a twin equally-fast mate
      // exists (we can't say which the solver "should" pick). But if THIS move
      // checkmates on the spot, play it anyway so the puzzle finishes ON the mate
      // instead of one move short — the app accepts any mating move at the final
      // step, so a twin mate-in-1 still solves.
      const ok = next.score >= MATE_THRESH
        ? alt.score < next.score || gameStatus(applyMove(afterReply, next.move)).result === 'checkmate'
        : next.score >= win && next.score - alt.score >= lineGap;
      if (!ok) break; // converted: several moves are fine from here — stop
    }

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

  // A forced-mate position whose line didn't actually reach checkmate bailed on an
  // INTERMEDIATE twin mate (two equally-fast continuations that aren't mate-in-1, so
  // neither can be scripted — mate-in-1 twins were already played through above).
  // Don't ship a mate that stops a move short: a puzzle should have one solution that
  // runs to its end, so drop it rather than truncate.
  if (isMate && kind !== 'mate') return { reject: 'mate-twin' };

  const solverMoves = Math.ceil(line.length / 2);
  const puzzle = {
    id,
    fen: toFen(state),
    moves: line.map(uci),
    kind,
    themes: [...themes],
    difficulty: solveDepth,
    score: Math.round(s1),
  };
  if (kind === 'mate') puzzle.mateIn = solverMoves;
  const opening = deriveOpening(leadFen, fen);
  if (opening) { puzzle.prefen = leadFen; puzzle.opening = opening; }
  return { puzzle };
}

// The pre-blunder position as an only-move puzzle: the game's engine had ONE move
// that held and didn't find it. kind 'defense' = the best move keeps the solver at
// least at saveFloor while every alternative loses decisively; if the best move
// actually WINS, the position is a missed win and is mined through mineWin instead
// (where the razor-edge tag will usually apply).
function mineDefense(item) {
  if (!item.prevFen) return { reject: 'no-prev' };
  const seed = (item.seed ^ 0x55555555) >>> 0; // independent stream from the win mine
  const rand = mulberry32(seed);
  const state = parseFen(item.prevFen);
  const status = gameStatus(state);
  if (status.over) return { reject: 'over' };
  if (status.legal.length < 2) return { reject: 'forced' }; // "only move" with no choice isn't a puzzle

  eng.resetTT();
  const ladder = [];
  for (let d = 1; d <= depth; d++) ladder.push(best(state, rand, d));
  const top = ladder[depth - 1];
  const s1 = top.score;
  if (s1 >= win) return mineWin(item.prevFen, item.prevPrevFen, item.id + 'w', seed); // a missed win, not a save
  if (s1 < saveFloor) return { reject: 'lost-anyway' }; // nothing holds — no save to find
  const alt = secondBest(state, rand, top.move);
  if (alt.score > -win) return { reject: 'not-tight' }; // a second move also survives

  let solveDepth = depth;
  const bk = keyOf(top.move);
  for (let d = depth - 1; d >= 1 && keyOf(ladder[d - 1].move) === bk; d--) solveDepth = d;

  // Walk the tightrope: scripted best replies, and the solver's follow-up stays in
  // the line only while it is STILL the only move that holds. The line ends when
  // the danger has passed (several moves hold), the position resolves (a draw end
  // is a successful save), or the cap is hit.
  const line = [top.move];
  const themes = new Set(themesOf(state, top.move));
  if (alt.score <= -MATE_THRESH) themes.add('mate-threat'); // the mistakes lose to forced mate
  const capValue = (s, m) => (m.capture && s.board[m.to] ? VALUE[s.board[m.to].role] : 0);
  let matDelta = capValue(state, top.move);
  let minDelta = 0;
  let cur = applyMove(state, top.move);

  for (;;) {
    let st = gameStatus(cur);
    if (st.over) break; // an over here is the save succeeding: draw trick, or the opponent mated

    const reply = best(cur, rand);
    const afterReply = applyMove(cur, reply.move);
    st = gameStatus(afterReply);
    if (st.over) {
      if (st.result === 'checkmate') return { reject: 'unsound' }; // the "save" got mated
      break;
    }

    const next = best(afterReply, rand);
    // The walk effectively searches deeper than the root verification did. If it
    // now sees the position is lost ANYWAY after the save, the puzzle's promise
    // ("this move holds") was a horizon mirage — reject, don't ship a doomed save.
    if (next.score <= -win) return { reject: 'unsound' };
    if (next.score >= MATE_THRESH) {
      // The save flipped all the way into a forced mate FOR the solver: run it to
      // checkmate like a mate line (cap-exempt, unique-mate rule), so the puzzle
      // ends with the mate on the board instead of stopping just before it.
      if (st.legal.length > 1) {
        const alt2 = secondBest(afterReply, rand, next.move);
        // Twin equally-fast mate ends the line, unless this move mates on the spot —
        // then play it so the puzzle finishes ON checkmate (the app accepts any final
        // mating move) instead of stopping one move short.
        if (alt2.score >= next.score
            && gameStatus(applyMove(afterReply, next.move)).result !== 'checkmate') break;
      }
    } else {
      if (next.score < saveFloor || next.score >= win) break; // line over: collapsed (noise) or flipped to winning
      if ((line.length + 1) / 2 > maxSolverMoves) break;
      if (st.legal.length > 1) {
        const alt2 = secondBest(afterReply, rand, next.move);
        if (alt2.score > -win) break; // danger passed: more than one move holds now
      }
    }

    matDelta -= capValue(cur, reply.move);
    minDelta = Math.min(minDelta, matDelta);
    line.push(reply.move, next.move);
    for (const th of themesOf(afterReply, next.move)) themes.add(th);
    matDelta += capValue(afterReply, next.move);
    cur = applyMove(afterReply, next.move);
  }
  if (minDelta <= -250) themes.add('sacrifice');

  // Final soundness check: the root claim ("the save holds") was made at `depth`,
  // but the app's play-it-out engine searches deeper than that, so an unsound save
  // gets refuted on the board a few moves later. Re-search the end-of-line position
  // two plies deeper (opponent to move): if they're decisively winning after all,
  // the save was a mirage.
  {
    const st = gameStatus(cur);
    if (!st.over && best(cur, rand, depth + 2).score >= win) return { reject: 'unsound' };
  }

  const puzzle = {
    id: item.id + 'd',
    fen: toFen(state),
    moves: line.map(uci),
    kind: 'defense',
    themes: [...themes],
    difficulty: solveDepth,
    score: Math.round(s1),
  };
  const opening = deriveOpening(item.prevPrevFen, item.prevFen);
  if (opening) { puzzle.prefen = item.prevPrevFen; puzzle.opening = opening; }
  return { puzzle };
}

parentPort.on('message', (msg) => {
  if (msg.type !== 'batch') return;
  const results = msg.items.map((item) => {
    const out = { idx: item.idx };
    try { out.win = mineWin(item.fen, item.prevFen, item.id, item.seed); }
    catch (e) { out.win = { reject: 'error', error: String(e && e.message || e) }; }
    try { out.defense = mineDefense(item); }
    catch (e) { out.defense = { reject: 'error', error: String(e && e.message || e) }; }
    return out;
  });
  parentPort.postMessage({ type: 'done', results });
});
parentPort.postMessage({ type: 'ready', engine });
