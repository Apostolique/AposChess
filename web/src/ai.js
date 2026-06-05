// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2019-2026 Jean-David Moisan
//
// Iterative-deepening alpha-beta search with several refinements that let it
// look deeper without examining every position:
//   - Transposition table  — Zobrist-hash each position; reuse a prior result
//                            (cutoff) when it was searched at least as deep, and
//                            seed move ordering with its best move.
//   - Quiescence search    — at a leaf, keep resolving captures/jumps/promotions
//                            so the evaluation is never taken mid-trade.
//   - Delta pruning        — in quiescence, skip a plain capture that can't get
//                            within a margin of alpha even if it wins the victim.
//                            Jumps/promotions are never pruned (variant tactics).
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
const MATE_THRESH = MATE - 1000; // scores beyond this magnitude encode a forced mate
const MAX_PLY = 64;
const QDEPTH = 6; // quiescence depth cap
const DELTA_MARGIN = 200; // qsearch: skip a capture if even winning it stays this far below alpha
const now = () => Date.now();

let killers; // killers[ply] = [moveKey, moveKey]
let history; // Int32Array[from*64+to] of cutoff counts

const keyOf = (m) => m.from * 64 + m.to;

// --- Zobrist hashing ---------------------------------------------------------
// A deterministic PRNG seeds fixed 64-bit (BigInt) keys, so the same position
// always hashes the same way within and across searches.
function mulberry32(a) {
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return (t ^ (t >>> 14)) >>> 0;
  };
}
const _rnd = mulberry32(0x1a2b3c4d);
const rand64 = () => (BigInt(_rnd()) << 32n) | BigInt(_rnd());

const ROLE_IDX = { p: 0, n: 1, b: 2, r: 3, q: 4, k: 5 };
const PIECE_KEYS = Array.from({ length: 12 * 64 }, rand64);
const SIDE_KEY = rand64(); // XORed in when Black is to move
const CASTLE_KEYS = { K: rand64(), Q: rand64(), k: rand64(), q: rand64() };
const pieceKey = (role, color, sq) =>
  PIECE_KEYS[(ROLE_IDX[role] * 2 + (color === 'white' ? 0 : 1)) * 64 + sq];

function hashOf(state) {
  let h = 0n;
  for (let i = 0; i < 64; i++) {
    const p = state.board[i];
    if (p) h ^= pieceKey(p.role, p.color, i);
  }
  if (state.turn === 'black') h ^= SIDE_KEY;
  const c = state.castling;
  if (c.K) h ^= CASTLE_KEYS.K;
  if (c.Q) h ^= CASTLE_KEYS.Q;
  if (c.k) h ^= CASTLE_KEYS.k;
  if (c.q) h ^= CASTLE_KEYS.q;
  return h;
}

// Incrementally derive the hash of the position after `m`. MUST mirror
// applyMove() exactly — see the cross-check in the engine tests.
function hashAfter(h, state, m) {
  const board = state.board;
  const piece = board[m.from];
  const color = piece.color;

  h ^= pieceKey(piece.role, color, m.from);
  if (m.capture) {
    const cap = board[m.to];
    if (cap) h ^= pieceKey(cap.role, cap.color, m.to);
  }
  h ^= pieceKey(m.promotion || piece.role, color, m.to);

  if (m.castle) {
    const home = color === 'white' ? 0 : 56;
    if (m.castle === 'K') { h ^= pieceKey('r', color, home + 7); h ^= pieceKey('r', color, home + 5); }
    else { h ^= pieceKey('r', color, home + 0); h ^= pieceKey('r', color, home + 3); }
  }

  const c = state.castling;
  let K = c.K, Q = c.Q, k = c.k, q = c.q;
  if (piece.role === 'k') { if (color === 'white') { K = Q = false; } else { k = q = false; } }
  for (const idx of [m.from, m.to]) {
    if (idx === 0) Q = false;
    else if (idx === 7) K = false;
    else if (idx === 56) q = false;
    else if (idx === 63) k = false;
  }
  if (K !== c.K) h ^= CASTLE_KEYS.K;
  if (Q !== c.Q) h ^= CASTLE_KEYS.Q;
  if (k !== c.k) h ^= CASTLE_KEYS.k;
  if (q !== c.q) h ^= CASTLE_KEYS.q;

  return h ^ SIDE_KEY;
}

// --- Transposition table -----------------------------------------------------
// Fixed-size bucket table (one slot per index, addressed by the low hash bits)
// held in typed arrays. Unlike a growing Map it has a hard memory bound, so it
// can *persist across moves* instead of being cleared each search — a later
// search starts "warm", reusing the cutoffs and best moves the previous one (or
// a ponder search on the opponent's turn) already found.
//
// Entries never go stale: each is keyed by the full Zobrist hash, so a value
// computed any number of moves ago is still correct for the same position. The
// `gen` field drives *replacement only*: an entry from an earlier search is
// always overwritable; within the same search we keep the deeper result.
const EXACT = 0, LOWER = 1, UPPER = 2;
const TT_BITS = 20;
const TT_SIZE = 1 << TT_BITS; // ~1M slots
const TT_MASK = BigInt(TT_SIZE - 1);

const ttKey = new BigInt64Array(TT_SIZE);   // full hash (signed reinterpret)
const ttScore = new Int32Array(TT_SIZE);
const ttMove = new Int32Array(TT_SIZE);     // moveKey = from*64+to
const ttDepth = new Int16Array(TT_SIZE);
const ttFlag = new Uint8Array(TT_SIZE);
const ttGen = new Uint16Array(TT_SIZE);     // 0 = empty slot; else search generation
let ttCurGen = 0;
let ttEnabled = true;

const ttReset = () => { ttGen.fill(0); ttCurGen = 0; };
// New generation per search; stays in 1..65535 (0 is reserved for empty slots).
const ttBumpGen = () => { ttCurGen = (ttCurGen % 65535) + 1; };

function ttProbe(hash) {
  const idx = Number(hash & TT_MASK);
  return ttGen[idx] !== 0 && ttKey[idx] === BigInt.asIntN(64, hash) ? idx : -1;
}

function ttStore(hash, depth, score, flag, move) {
  const idx = Number(hash & TT_MASK);
  const k = BigInt.asIntN(64, hash);
  // Replace if the slot is empty, holds this same position, is left over from an
  // earlier search, or holds a shallower result from the current one.
  if (ttGen[idx] === 0 || ttKey[idx] === k || ttGen[idx] !== ttCurGen || depth >= ttDepth[idx]) {
    ttKey[idx] = k;
    ttDepth[idx] = depth;
    ttScore[idx] = score;
    ttFlag[idx] = flag;
    ttMove[idx] = move;
    ttGen[idx] = ttCurGen;
  }
}

// Mate scores are stored relative to the node (distance-to-mate from here), so an
// entry reused at a different ply still reports the correct mate distance.
const toTT = (s, ply) => (s >= MATE_THRESH ? s + ply : s <= -MATE_THRESH ? s - ply : s);
const fromTT = (s, ply) => (s >= MATE_THRESH ? s - ply : s <= -MATE_THRESH ? s + ply : s);

// --- evaluation & ordering ---------------------------------------------------
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
  let best, standPat;
  if (inCheck) {
    best = -MATE;
  } else {
    standPat = best = evalStm(state.board, state.turn); // stand pat
    if (best >= beta) return best;
    if (best > alpha) alpha = best;
  }
  if (qdepth <= 0) return best;

  let moves = legalMoves(state);
  if (moves.length === 0) return inCheck ? -MATE : 0;
  if (!inCheck) moves = moves.filter((m) => m.capture || m.promotion || m.jump);
  orderMoves(moves, state.board, 0, 0);

  for (const m of moves) {
    // Delta pruning: when not in check, a plain capture whose best case (winning
    // the victim outright) still can't climb within DELTA_MARGIN of alpha is
    // hopeless — skip it. Jumps and promotions are never pruned: the variant's
    // tactics live there, and a non-capturing jump has no victim to bound.
    if (!inCheck && m.capture && !m.promotion && !m.jump) {
      const victim = state.board[m.to];
      if (victim && standPat + VALUE[victim.role] + DELTA_MARGIN <= alpha) continue;
    }
    const score = -qsearch(applyMove(state, m), -beta, -alpha, qdepth - 1);
    if (score > best) best = score;
    if (best > alpha) alpha = best;
    if (alpha >= beta) break;
  }
  return best;
}

function search(state, depth, alpha, beta, ply, canNull, hash, deadline) {
  if (now() > deadline) return 0; // aborted; the root discards this iteration
  if (ply >= MAX_PLY) return evalStm(state.board, state.turn);

  const inCheck = kingAttacked(state.board, state.turn);
  if (inCheck) depth++; // check extension
  if (depth <= 0) return qsearch(state, alpha, beta, QDEPTH);

  const alphaOrig = alpha;
  let ttMoveKey = 0;
  if (ttEnabled) {
    const i = ttProbe(hash);
    if (i >= 0) {
      ttMoveKey = ttMove[i];
      if (ttDepth[i] >= depth) {
        const s = fromTT(ttScore[i], ply);
        const flag = ttFlag[i];
        if (flag === EXACT) return s;
        if (flag === LOWER && s >= beta) return s;
        if (flag === UPPER && s <= alpha) return s;
      }
    }
  }

  // Null-move pruning: pass the move; if we're still ≥ beta, this node fails high.
  if (canNull && !inCheck && depth >= 3 && beta < MATE_THRESH && hasNonPawn(state.board, state.turn)) {
    const nm = {
      board: state.board, turn: opponent(state.turn),
      castling: state.castling, halfmove: state.halfmove, fullmove: state.fullmove,
    };
    const nh = ttEnabled ? hash ^ SIDE_KEY : 0n;
    const score = -search(nm, depth - 3, -beta, -beta + 1, ply + 1, false, nh, deadline);
    if (score >= beta) return beta;
  }

  const legal = legalMoves(state);
  if (legal.length === 0) return inCheck ? -MATE - depth : 0;
  orderMoves(legal, state.board, ply, ttMoveKey);

  let best = -Infinity, bestKey = 0, moveCount = 0;
  for (const m of legal) {
    moveCount++;
    const child = applyMove(state, m);
    const childHash = ttEnabled ? hashAfter(hash, state, m) : 0n;
    const quiet = !m.capture && !m.promotion && !m.jump;
    let score;
    if (moveCount === 1) {
      score = -search(child, depth - 1, -beta, -alpha, ply + 1, true, childHash, deadline);
    } else {
      // Late move reduction for quiet, late moves (never jumps/captures/promotions).
      const r = (quiet && depth >= 3 && moveCount > 3 && !inCheck) ? 1 : 0;
      score = -search(child, depth - 1 - r, -alpha - 1, -alpha, ply + 1, true, childHash, deadline);
      if (score > alpha && r > 0) score = -search(child, depth - 1, -alpha - 1, -alpha, ply + 1, true, childHash, deadline);
      if (score > alpha && score < beta) score = -search(child, depth - 1, -beta, -alpha, ply + 1, true, childHash, deadline);
    }
    if (score > best) { best = score; bestKey = keyOf(m); }
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

  // Skip storing once the deadline has passed: a node that broke out of its move
  // loop on time has an incomplete `best`, and with a persistent table a bogus
  // entry would survive into later searches. Time is monotonic, so once we're
  // past the deadline every ancestor's store is skipped too.
  if (ttEnabled && now() <= deadline) {
    const flag = best <= alphaOrig ? UPPER : best >= beta ? LOWER : EXACT;
    ttStore(hash, depth, toTT(best, ply), flag, bestKey);
  }
  return best;
}

// Choose a move for the side to move, searching up to `maxDepth` plies but never
// past `maxMs` of wall-clock. `rand` shuffles equal choices so games vary.
// `useTT` exists for benchmarking the transposition table on/off.
//
// Returns { move, ponder, depth }: `move` is the chosen move, `ponder` is the
// predicted opponent reply (its { from, to } — what to think about during their
// turn) read from the table after the search, and `depth` is the deepest
// iteration completed (used to stop pondering once the line is fully resolved).
// The table is NOT cleared here — it persists across calls (see ttReset).
export function chooseMoveDetailed(state, maxDepth = 2, rand = Math.random, maxMs = Infinity, useTT = true) {
  const root = legalMoves(state);
  if (root.length === 0) return { move: null, ponder: null, depth: 0 };

  for (let i = root.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [root[i], root[j]] = [root[j], root[i]];
  }

  killers = [];
  history = new Int32Array(64 * 64);
  ttEnabled = useTT;
  if (useTT) ttBumpGen();
  const rootHash = useTT ? hashOf(state) : 0n;
  const deadline = now() + maxMs;
  let bestMove = root[0];
  let completed = 0;

  // Backstop so an unbounded (maxDepth = Infinity) search still terminates even
  // if the deadline were also infinite; real searches abort on time long before.
  const depthCap = Math.min(maxDepth, 99);
  for (let depth = 1; depth <= depthCap; depth++) {
    orderMoves(root, state.board, 0, keyOf(bestMove));
    let alpha = -Infinity, bestScore = -Infinity, localBest = root[0], aborted = false, moveCount = 0;
    for (const m of root) {
      moveCount++;
      const child = applyMove(state, m);
      const childHash = useTT ? hashAfter(rootHash, state, m) : 0n;
      let score;
      if (moveCount === 1) {
        score = -search(child, depth - 1, -Infinity, -alpha, 1, true, childHash, deadline);
      } else {
        score = -search(child, depth - 1, -alpha - 1, -alpha, 1, true, childHash, deadline);
        if (score > alpha) score = -search(child, depth - 1, -Infinity, -alpha, 1, true, childHash, deadline);
      }
      if (now() > deadline) { aborted = true; break; }
      if (score > bestScore) { bestScore = score; localBest = m; }
      if (score > alpha) alpha = score;
    }
    if (!aborted) { bestMove = localBest; completed = depth; }
    if (aborted || bestScore >= MATE_THRESH) break;
  }

  // The predicted reply is the best move stored for the position *after* ours.
  let ponder = null;
  if (useTT && bestMove) {
    const i = ttProbe(hashAfter(rootHash, state, bestMove));
    if (i >= 0 && ttMove[i]) ponder = { from: (ttMove[i] / 64) | 0, to: ttMove[i] % 64 };
  }
  return { move: bestMove, ponder, depth: completed };
}

export function chooseMove(state, maxDepth, rand, maxMs, useTT) {
  return chooseMoveDetailed(state, maxDepth, rand, maxMs, useTT).move;
}

// Exposed for tests only: Zobrist hash equivalence check + table reset so a
// benchmark/test can start each game from a cold table despite persistence.
export const _internal = { hashOf, hashAfter, resetTT: ttReset };
