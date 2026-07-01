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

import { legalMoves, applyMove, kingAttacked, generatePseudoMoves, hasLegalMove } from './engine.js';
import { opponent } from './board.js';
import { evaluate as nnEvaluate } from './nn.js';

// Knight = 500 (≈ rook), not the standard-chess 300: this variant's knight is much
// stronger than chess implies — in self-play, up a knight wins ~80%, statistically
// identical to up a rook (~81%) and well above up a bishop (~70%), and an outcome
// least-squares fit puts it at ~520cp. Adopted after a head-to-head SPRT vs the old
// 300 table (+61 Elo ± 37). Every other piece already matched outcomes, so unchanged.
const VALUE = { p: 100, n: 500, b: 330, r: 500, q: 900, k: 0 };
// Handcrafted-eval version, stamped into training-data `v` provenance (scripts/vtag.mjs)
// so a v computed by an older eval is distinguishable later. BUMP whenever the
// handcrafted eval changes (VALUE, PST, MOB, evalStm). v1 = the pre-2026-06-13 table
// (knight=300, untagged in old data); v2 = knight=500.
export const HC_VERSION = 2;
const MATE = 1_000_000;
const MATE_THRESH = MATE - 1000; // scores beyond this magnitude encode a forced mate
const MAX_PLY = 64;
const QDEPTH = 6; // quiescence depth cap
const DELTA_MARGIN = 200; // qsearch: skip a capture if even winning it stays this far below alpha
const now = () => Date.now();

let killers; // killers[ply] = [moveKey, moveKey]
let history; // Int32Array[from*64+to] of cutoff counts
// Repetition detection: repPath[ply] is the Zobrist hash at each ply of the current
// search line, with index 0 seeded to the game's current position. A node whose hash
// matches a same-side-to-move ancestor (or the current position) is scored a draw,
// so the engine stops treating a shuffle as progress — it avoids repeating when
// ahead and seeks it when worse. (Hashes require the TT, so this is gated on it.)
let repPath;
// Positions that already occurred in the *actual* game (Zobrist hashes), supplied
// by the caller. A search node whose hash is in here is a genuine repetition the
// engine should avoid when ahead and seek when behind, even though it never
// appeared in the current search line — without this the engine has no game
// history and will happily shuffle a won position into a threefold draw.
let repSeen;
// True when the value the most recent search() call returned came through a
// repetition draw (its own, or its best/cutoff child's). Such a value is
// path-dependent — valid only for the move order that reached it — so it must NOT
// be written to the persistent transposition table. A path-agnostic TT would
// otherwise reuse that draw score down an unrelated path and hide a real win
// (graph-history interaction). The flag bubbles up via this module-level var,
// read by each caller immediately after its recursive search() returns.
let tainted;

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
  hash ^= evalKey; // namespace the table by active eval (see EVAL_KEYS)
  const idx = Number(hash & TT_MASK);
  return ttGen[idx] !== 0 && ttKey[idx] === BigInt.asIntN(64, hash) ? idx : -1;
}

function ttStore(hash, depth, score, flag, move) {
  hash ^= evalKey; // namespace the table by active eval (see EVAL_KEYS)
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
// Piece-square tables: small positional nudges on top of material. Written from
// White's view with index 0 = a1 (rank 1, White's back rank) and index 63 = h8,
// matching the board layout. Black reads the same table vertically mirrored
// (index ^ 56 flips the rank, keeping the file), so the eval stays symmetric.
//
// Variant motivation: knights begin fully boxed in, so the knight table strongly
// rewards getting them off the back rank and central. Pawns are pushed toward
// promotion. Other pieces get mild centralization. The king is left at zero for
// now — a static king table is only right with a tapered midgame/endgame eval,
// which is a later step.
const PST_N = [
  -50,-40,-30,-30,-30,-30,-40,-50,
  -40,-20,  0,  5,  5,  0,-20,-40,
  -30,  5, 10, 15, 15, 10,  5,-30,
  -30,  0, 15, 20, 20, 15,  0,-30,
  -30,  5, 15, 20, 20, 15,  5,-30,
  -30,  0, 10, 15, 15, 10,  0,-30,
  -40,-20,  0,  0,  0,  0,-20,-40,
  -50,-40,-30,-30,-30,-30,-40,-50,
];
const PST_P = [
    0,  0,  0,  0,  0,  0,  0,  0,
    5,  5,  5,  5,  5,  5,  5,  5,
   10, 10, 10, 12, 12, 10, 10, 10,
   20, 20, 25, 30, 30, 25, 20, 20,
   35, 35, 40, 45, 45, 40, 35, 35,
   55, 55, 60, 65, 65, 60, 55, 55,
   80, 80, 85, 90, 90, 85, 80, 80,
    0,  0,  0,  0,  0,  0,  0,  0,
];
const PST_B = [
  -10, -5, -5, -5, -5, -5, -5,-10,
   -5,  5,  0,  0,  0,  0,  5, -5,
   -5,  5,  5,  5,  5,  5,  5, -5,
   -5,  0,  5,  8,  8,  5,  0, -5,
   -5,  0,  5,  8,  8,  5,  0, -5,
   -5,  5,  5,  5,  5,  5,  5, -5,
   -5,  5,  0,  0,  0,  0,  5, -5,
  -10, -5, -5, -5, -5, -5, -5,-10,
];
const PST_R = [
    0,  0,  0,  5,  5,  0,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  0,
    5, 10, 10, 10, 10, 10, 10,  5,
    0,  0,  0,  5,  5,  0,  0,  0,
];
const PST_Q = [
  -10, -5, -5, -2, -2, -5, -5,-10,
   -5,  0,  0,  0,  0,  0,  0, -5,
   -5,  0,  3,  3,  3,  3,  0, -5,
   -2,  0,  3,  5,  5,  3,  0, -2,
   -2,  0,  3,  5,  5,  3,  0, -2,
   -5,  0,  3,  3,  3,  3,  0, -5,
   -5,  0,  0,  0,  0,  0,  0, -5,
  -10, -5, -5, -2, -2, -5, -5,-10,
];
const PST_K = new Array(64).fill(0);
const PST = { p: PST_P, n: PST_N, b: PST_B, r: PST_R, q: PST_Q, k: PST_K };

const MOB = 3; // centipawns per extra pseudo-legal move (mobility differential)

function evalStm(board, turn) {
  let s = 0;
  for (let i = 0; i < 64; i++) {
    const p = board[i];
    if (!p) continue;
    const v = VALUE[p.role] + PST[p.role][p.color === 'white' ? i : i ^ 56];
    s += p.color === 'white' ? v : -v;
  }
  // Mobility: reward having more moves than the opponent. This is the variant's
  // lifeblood (boxed knights, jump availability), but it costs a move generation
  // per side at every leaf. Pseudo-moves (no check filtering) are enough here.
  s += MOB * (generatePseudoMoves(board, 'white').length - generatePseudoMoves(board, 'black').length);
  return turn === 'white' ? s : -s;
}

// --- handcrafted v3 ----------------------------------------------------------
// An alternative handcrafted eval whose material values and piece-square tables are
// DISTILLED FROM THE CHAMPION NEURAL NET: a per-(role,square) ridge regression of the
// net's white-relative eval over ~200k self-play positions recovered the net's implied
// value of each piece on each square; that map was split into an occupancy-weighted
// base value (VALUE3) + positional residual (PST), rescaled so pawn=100 (keeping the
// search's centipawn scale), and left-right symmetrized. Versus v2 this learns a much
// steeper pawn-advancement gradient, knight≈rook, and a development-hungry queen.
// Worth ~+12 Elo over v2 in self-play (4000 games, depth 4).
// The king PST is left at 0: the net's active-king table cost ~18 Elo without a game-phase
// taper (see PST_K3). Selected via the engine string 'handcrafted3'; same eval contract
// (side-to-move centipawns) as evalStm.
const VALUE3 = { p: 100, n: 477, b: 316, r: 478, q: 816, k: 0 };
const PST_P3 = [
     0,   0,   0,   0,   0,   0,   0,   0,
   -29, -17,   3, -15, -15,   3, -17, -29,
   -43, -13,   3,  -4,  -4,   3, -13, -43,
   -37,  -2,  23,  35,  35,  23,  -2, -37,
   -43,  21,  37,  59,  59,  37,  21, -43,
    57,  89, 141, 117, 117, 141,  89,  57,
   157, 353, 310, 313, 313, 310, 353, 157,
     0,   0,   0,   0,   0,   0,   0,   0,
];
const PST_N3 = [
   -39, -28, -37,  -9,  -9, -37, -28, -39,
   -23,  16,  43,  41,  41,  43,  16, -23,
     2,  28,  41,  57,  57,  41,  28,   2,
    -6,  26,  40,  56,  56,  40,  26,  -6,
   -15,  28,  31,  59,  59,  31,  28, -15,
   -66,  52,  32,  58,  58,  32,  52, -66,
   -12,  34,  58,  70,  70,  58,  34, -12,
    -4,   1,  55,  79,  79,  55,   1,  -4,
];
const PST_B3 = [
   -45, -93,  -9, -75, -75,  -9, -93, -45,
   -64,  -2, -49,   4,   4, -49,  -2, -64,
   -13, -35,   8,  28,  28,   8, -35, -13,
   -49,  -8,  36,  24,  24,  36,  -8, -49,
     6,  -7,  19,  51,  51,  19,  -7,   6,
   -86,  17,  26,  28,  28,  26,  17, -86,
   -27,  10, -25, -39, -39, -25,  10, -27,
   -33, -98, -77, -42, -42, -77, -98, -33,
];
const PST_R3 = [
   -21,  13,   4,  20,  20,   4,  13, -21,
    -7, -21, -18, -36, -36, -18, -21,  -7,
     7,  11,  -1,  21,  21,  -1,  11,   7,
   -48,  22,  17,  56,  56,  17,  22, -48,
    13,  22,  47,  50,  50,  47,  22,  13,
   -94,  66,  18,  44,  44,  18,  66, -94,
    26,  80, -20,   0,   0, -20,  80,  26,
   125,  -2,  33,  27,  27,  33,  -2, 125,
];
const PST_Q3 = [
   -93, -96, -86, -65, -65, -86, -96, -93,
  -111, -55, -34, -28, -28, -34, -55,-111,
   -48, -12,  11,  22,  22,  11, -12, -48,
   -30,   5,  29,  34,  34,  29,   5, -30,
     6,  33,  64,  73,  73,  64,  33,   6,
     6,  81,  84, 107, 107,  84,  81,   6,
    14,  78,  90,  73,  73,  90,  78,  14,
    40,  43,  86,  39,  39,  86,  43,  40,
];
// King PST left at 0 (like v2). The net implies an active/central king, but importing
// that table cost ~18 Elo in self-play (-6 with it vs +12 without, 4000 games each at
// depth 4 vs v2): a static king bonus has no game-phase taper and walks the king out in
// the middlegame. Revisit only with a tapered midgame/endgame eval.
const PST_K3 = new Array(64).fill(0);
const PST3 = { p: PST_P3, n: PST_N3, b: PST_B3, r: PST_R3, q: PST_Q3, k: PST_K3 };

function evalStmV3(board, turn) {
  let s = 0;
  for (let i = 0; i < 64; i++) {
    const p = board[i];
    if (!p) continue;
    const v = VALUE3[p.role] + PST3[p.role][p.color === 'white' ? i : i ^ 56];
    s += p.color === 'white' ? v : -v;
  }
  s += MOB * (generatePseudoMoves(board, 'white').length - generatePseudoMoves(board, 'black').length);
  return turn === 'white' ? s : -s;
}

// --- pluggable evaluation ----------------------------------------------------
// The search funnels every leaf and stand-pat score through `activeEval`, which
// chooseMoveDetailed selects per search from its `engine` argument. This lets the
// menu offer the handcrafted engine and a neural-network engine side by side
// without the search itself changing — the only thing that varies is this one
// function. Both evals share the same contract as evalStm: a centipawn score from
// the side-to-move's perspective.
//
// The neural-net evaluation lives in nn.js (feature extraction + forward pass).
// Until weights are trained it falls back to a material-only score, so the engine
// still plays. Loading weights is the caller's job: the worker fetches them, the
// self-play tools read them from disk — see nn.js. For true NNUE speed an
// accumulator would later be threaded through applyMove (it is currently
// pure-functional); recomputing from scratch is fine to start.
// The nn eval reads weights from a named slot (see nn.js). The slot is chosen per
// search by chooseMoveDetailed (the engine string may be 'nn:<slot>'); 'default'
// matches the single-net behaviour. Slots let the match runner pit two nets at once
// and the app offer a choice of nets.
let nnSlot = 'default';
const evalNN = (board, turn) => nnEvaluate(board, turn, nnSlot);

// Material-only eval: the bare piece count (same VALUE table the nn eval falls back to
// before its weights load), side-to-move relative. Exposed as its own 'material' engine
// so a deliberately weak, positionally-blind opponent can be picked in the UI — distinct
// from the nn fallback (which is an accident waiting for weights), this is a real choice.
function evalMaterial(board, turn) {
  let s = 0;
  for (let i = 0; i < 64; i++) {
    const p = board[i];
    if (p) s += p.color === 'white' ? VALUE[p.role] : -VALUE[p.role];
  }
  return turn === 'white' ? s : -s;
}

const EVALS = { handcrafted: evalStm, handcrafted3: evalStmV3, material: evalMaterial, nn: evalNN };
let activeEval = evalStm;

// The transposition table persists across searches and — in AI-vs-AI on a single
// worker — is shared by both colours, which may use *different* evals. Entries are
// keyed by position hash alone, which says nothing about which eval produced the
// stored score, so without namespacing an nn search and a handcrafted search would
// read each other's scores through the table and corrupt both (cutoffs, bounds,
// move ordering). XOR a per-eval constant into the TT key (only there — see
// ttProbe/ttStore) so each eval occupies a disjoint slice of the table. Handcrafted
// uses 0n, so single-eval behaviour — and the match runner's separate-instance
// engines — stay byte-for-byte unchanged; only the nn keys move out of the way.
const EVAL_KEYS = { handcrafted: 0n, handcrafted3: 0x2545f4914f6cdd1dn, material: 0x6a09e667f3bcc908n, nn: 0x9e3779b97f4a7c15n };
let evalKey = 0n;

// Distinct TT namespace per nn slot, so a single instance that switches nets mid-run
// can't read one net's cached scores under another's (same hazard as the per-eval
// keys above). 'default' stays 0n, so single-net behaviour — and the match runner's
// separate-instance engines — keep byte-for-byte identical keys.
const slotKeys = new Map([['default', 0n]]);
function slotKey(slot) {
  let k = slotKeys.get(slot);
  if (k === undefined) {
    k = 0xcbf29ce484222325n;
    for (let i = 0; i < slot.length; i++) k = ((k ^ BigInt(slot.charCodeAt(i))) * 0x100000001b3n) & 0xffffffffffffffffn;
    k |= 1n; // nonzero, so it never collides with 'default'
    slotKeys.set(slot, k);
  }
  return k;
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
    standPat = best = activeEval(state.board, state.turn); // stand pat
    if (best >= beta) return best;
    if (best > alpha) alpha = best;
  }
  if (qdepth <= 0) return best;

  // In check, every evasion must be searched, so the full legal generator is the
  // right tool (and detects mate).
  if (inCheck) {
    const moves = legalMoves(state);
    if (moves.length === 0) return -MATE;
    orderMoves(moves, state.board, 0, 0);
    for (const m of moves) {
      const score = -qsearch(applyMove(state, m), -beta, -alpha, qdepth - 1);
      if (score > best) best = score;
      if (best > alpha) alpha = best;
      if (alpha >= beta) break;
    }
    return best;
  }

  // Not in check: only tactical moves (captures/jumps/promotions) get searched,
  // so generate pseudo-moves and legality-check just those, lazily, after delta
  // pruning. The previous legalMoves() call here paid a make/unmake king-safety
  // test for every quiet move only to filter them all out — and quiet qsearch
  // nodes are the most visited nodes in the whole search. Castling is never
  // tactical, so pseudo-moves cover everything this loop can search.
  const pseudo = generatePseudoMoves(state.board, state.turn);
  const moves = [];
  for (const m of pseudo) if (m.capture || m.promotion || m.jump) moves.push(m);
  orderMoves(moves, state.board, 0, 0);

  let sawLegal = false;
  for (const m of moves) {
    // Delta pruning: a plain capture whose best case (winning the victim
    // outright) still can't climb within DELTA_MARGIN of alpha is hopeless —
    // skip it. Jumps and promotions are never pruned: the variant's tactics
    // live there, and a non-capturing jump has no victim to bound.
    if (m.capture && !m.promotion && !m.jump) {
      const victim = state.board[m.to];
      if (victim && standPat + VALUE[victim.role] + DELTA_MARGIN <= alpha) continue;
    }
    const child = applyMove(state, m);
    if (kingAttacked(child.board, state.turn)) continue; // illegal: leaves own king in check
    sawLegal = true;
    const score = -qsearch(child, -beta, -alpha, qdepth - 1);
    if (score > best) best = score;
    if (best > alpha) alpha = best;
    if (alpha >= beta) break;
  }

  // Stalemate must still score 0 exactly as legalMoves().length === 0 used to:
  // if nothing legal was searched (no tactical moves, all illegal, or all
  // delta-pruned), ask the early-exit existence test — usually one make/unmake —
  // before standing pat.
  if (!sawLegal && !hasLegalMove(state, pseudo)) return 0;
  return best;
}

function search(state, depth, alpha, beta, ply, canNull, hash, deadline) {
  if (now() > deadline) { tainted = false; return 0; } // aborted; the root discards this iteration
  if (ttEnabled) {
    // Draw by repetition. Two sources, both scored as a draw on the first repeat:
    //   - repSeen: the position already occurred in the real game (so reaching it
    //     again is a draw the engine must weigh — avoid when ahead, seek when behind).
    //   - repPath: a same-position ancestor in this search line (every 2 plies back,
    //     since the side to move must match) or the current root.
    // Either way mark the value tainted so it can't poison the persistent table.
    if (repSeen.size !== 0 && repSeen.has(hash)) { tainted = true; return 0; }
    for (let i = ply - 2; i >= 0; i -= 2) if (repPath[i] === hash) { tainted = true; return 0; }
    repPath[ply] = hash;
  }
  if (ply >= MAX_PLY) { tainted = false; return activeEval(state.board, state.turn); }

  const inCheck = kingAttacked(state.board, state.turn);
  if (inCheck) depth++; // check extension
  if (depth <= 0) { tainted = false; return qsearch(state, alpha, beta, QDEPTH); }

  const alphaOrig = alpha;
  let ttMoveKey = 0;
  if (ttEnabled) {
    const i = ttProbe(hash);
    if (i >= 0) {
      ttMoveKey = ttMove[i];
      if (ttDepth[i] >= depth) {
        const s = fromTT(ttScore[i], ply);
        const flag = ttFlag[i];
        if (flag === EXACT) { tainted = false; return s; }
        if (flag === LOWER && s >= beta) { tainted = false; return s; }
        if (flag === UPPER && s <= alpha) { tainted = false; return s; }
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
    // A fail-high resting on a repetition draw is itself path-dependent; leave the
    // child's `tainted` in place (we don't store on this path) and bail out.
    if (score >= beta) return beta;
  }

  const legal = legalMoves(state);
  if (legal.length === 0) { tainted = false; return inCheck ? -MATE - depth : 0; }
  orderMoves(legal, state.board, ply, ttMoveKey);

  let best = -Infinity, bestKey = 0, moveCount = 0, bestTainted = false;
  for (const m of legal) {
    moveCount++;
    const child = applyMove(state, m);
    const childHash = ttEnabled ? hashAfter(hash, state, m) : 0n;
    const quiet = !m.capture && !m.promotion && !m.jump;
    let score, sTainted;
    if (moveCount === 1) {
      score = -search(child, depth - 1, -beta, -alpha, ply + 1, true, childHash, deadline);
      sTainted = tainted;
    } else {
      // Late move reduction for quiet, late moves (never jumps/captures/promotions).
      const r = (quiet && depth >= 3 && moveCount > 3 && !inCheck) ? 1 : 0;
      score = -search(child, depth - 1 - r, -alpha - 1, -alpha, ply + 1, true, childHash, deadline);
      sTainted = tainted;
      if (score > alpha && r > 0) { score = -search(child, depth - 1, -alpha - 1, -alpha, ply + 1, true, childHash, deadline); sTainted = tainted; }
      if (score > alpha && score < beta) { score = -search(child, depth - 1, -beta, -alpha, ply + 1, true, childHash, deadline); sTainted = tainted; }
    }
    if (score > best) { best = score; bestKey = keyOf(m); bestTainted = sTainted; }
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

  // The node's value is tainted if the move that fixed it (the best move, or the one
  // that caused the beta cutoff — both tracked by bestTainted) came back tainted.
  // Skip the store in that case so a path-dependent draw never lands in the
  // persistent table. Also skip past the deadline: a node that broke out of its
  // move loop on time has an incomplete `best`, and with a persistent table a bogus
  // entry would survive into later searches. Time is monotonic, so once we're past
  // the deadline every ancestor's store is skipped too.
  tainted = bestTainted;
  if (ttEnabled && !bestTainted && now() <= deadline) {
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
//
// `prevHashes` is the Zobrist hashes of positions that already occurred in the real
// game (so the search can recognise — and a winning side avoid — a genuine
// threefold draw it would otherwise be blind to). Pass [] when there's no history.
// Only positions since the last irreversible move (capture/pawn move — i.e. the last
// `halfmove` plies) can ever recur, so the caller need only pass that window; doing
// so keeps the per-node repetition lookup set tiny (usually empty).
export function chooseMoveDetailed(state, maxDepth = 2, rand = Math.random, maxMs = Infinity, useTT = true, prevHashes = [], engine = 'handcrafted', excludeKeys = null, onProgress = null) {
  // engine is 'handcrafted', 'nn', or 'nn:<slot>' (a specific net). Split off the slot.
  const colon = engine.indexOf(':');
  const evalName = colon < 0 ? engine : engine.slice(0, colon);
  // 'loser' (the Lemming): the nn champion eval, but the root keeps the WORST-scoring move
  // instead of the best — it tries to lose as fast as possible (see the root loop below).
  // It is a move-selection mode, not a distinct eval, so it borrows the nn eval + TT slice.
  const minimize = evalName === 'loser';
  const realEval = minimize ? 'nn' : evalName;
  nnSlot = colon < 0 ? 'default' : engine.slice(colon + 1);
  activeEval = EVALS[realEval] || evalStm;
  evalKey = (EVAL_KEYS[realEval] || 0n) ^ slotKey(nnSlot);
  let root = legalMoves(state);
  // No legal moves = terminal. Report the true side-to-move score so a caller that
  // searches/ponders into this position (e.g. the eval bar) pins to the result
  // instead of reading a bare 0 as an even position: -MATE when checkmated, 0 for
  // stalemate. Mirrors the in-tree terminal handling in search().
  if (root.length === 0) {
    return { move: null, ponder: null, depth: 0, score: kingAttacked(state.board, state.turn) ? -MATE : 0 };
  }

  // Optional opening-variety filter: drop root moves whose key (from*64+to) is in
  // excludeKeys, so the caller can forbid a few recently-played openings. Only the
  // root is touched — the search below is unchanged — and if the filter would leave
  // no move (every legal move excluded) the full list is kept, so a move is always
  // returned.
  if (excludeKeys && excludeKeys.size) {
    const kept = root.filter((m) => !excludeKeys.has(keyOf(m)));
    if (kept.length) root = kept;
  }

  for (let i = root.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [root[i], root[j]] = [root[j], root[i]];
  }

  killers = [];
  history = new Int32Array(64 * 64);
  ttEnabled = useTT;
  if (useTT) ttBumpGen();
  const rootHash = useTT ? hashOf(state) : 0n;
  repPath = useTT ? [rootHash] : []; // index 0 = the current (root) position
  repSeen = useTT ? new Set(prevHashes) : new Set(); // positions already seen in the real game
  const deadline = now() + maxMs;
  let bestMove = root[0];
  let completed = 0;
  let rootScore = 0; // side-to-move-relative value (cp) of the last completed depth

  // Backstop so an unbounded (maxDepth = Infinity) search still terminates even
  // if the deadline were also infinite; real searches abort on time long before.
  const depthCap = Math.min(maxDepth, 99);
  for (let depth = 1; depth <= depthCap; depth++) {
    orderMoves(root, state.board, 0, keyOf(bestMove));
    let alpha = -Infinity, bestScore = minimize ? Infinity : -Infinity, localBest = root[0], aborted = false, moveCount = 0;
    for (const m of root) {
      moveCount++;
      const child = applyMove(state, m);
      const childHash = useTT ? hashAfter(rootHash, state, m) : 0n;
      let score;
      if (minimize) {
        // Loser mode: every root move needs its TRUE score (so the worst is exact), so
        // search each with a full window — no alpha tightening, no PVS — and keep the min.
        score = -search(child, depth - 1, -Infinity, Infinity, 1, true, childHash, deadline);
      } else if (moveCount === 1) {
        score = -search(child, depth - 1, -Infinity, -alpha, 1, true, childHash, deadline);
      } else {
        score = -search(child, depth - 1, -alpha - 1, -alpha, 1, true, childHash, deadline);
        if (score > alpha) score = -search(child, depth - 1, -Infinity, -alpha, 1, true, childHash, deadline);
      }
      if (now() > deadline) { aborted = true; break; }
      if (minimize ? score < bestScore : score > bestScore) { bestScore = score; localBest = m; }
      if (!minimize && score > alpha) alpha = score;
    }
    if (!aborted) {
      bestMove = localBest; completed = depth; rootScore = bestScore;
      // Report the best score at each finished iteration so a caller (the worker)
      // can stream a live eval while the deeper iterations are still running.
      if (onProgress) onProgress(rootScore, completed);
    }
    // Stop once the outcome is forced: a found win (normal) or a found loss (loser mode).
    if (aborted || (minimize ? bestScore <= -MATE_THRESH : bestScore >= MATE_THRESH)) break;
  }

  // The predicted reply is the best move stored for the position *after* ours.
  let ponder = null;
  if (useTT && bestMove) {
    const i = ttProbe(hashAfter(rootHash, state, bestMove));
    if (i >= 0 && ttMove[i]) ponder = { from: (ttMove[i] / 64) | 0, to: ttMove[i] % 64 };
  }
  return { move: bestMove, ponder, depth: completed, score: rootScore };
}

export function chooseMove(state, maxDepth, rand, maxMs, useTT, prevHashes, engine, excludeKeys) {
  return chooseMoveDetailed(state, maxDepth, rand, maxMs, useTT, prevHashes, engine, excludeKeys).move;
}

// Exposed for tests only: Zobrist hash equivalence check + table reset so a
// benchmark/test can start each game from a cold table despite persistence.
export const _internal = { hashOf, hashAfter, resetTT: ttReset, evalStm, evalStmV3, evalMaterial, MATE, MATE_THRESH };
