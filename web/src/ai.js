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
//                            effective. History is bounded and carries a malus,
//                            and a countermove table remembers what refuted the
//                            opponent's last move.
//   - Reverse futility     — a node whose static eval is far enough above beta
//                            fails high without a search (the null-move idea,
//                            without the search).
//   - Late move pruning    — at shallow depth, quiet moves past a move-count
//                            threshold are not searched at all.
//   - Futility pruning     — near the frontier, a quiet move that can't lift a
//                            far-below-alpha static eval into the window is skipped.
//   - Aspiration windows   — each root iteration searches a narrow window around
//                            the previous score, re-searching wider when it misses.
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
// Which search played a game, stamped into every harvested record as `se` (scripts/gameRecord.mjs)
// so the Bradley-Terry pool never averages two different engines under one (engine, depth) label.
// The node id carries no search, but the search decides what a depth is worth: era 2 loses a
// fixed-depth-6 match to era 1 by ~380 Elo while reaching depth 8 on a quarter of the nodes. BUMP
// on any change to what the shipped search does — the `searchOpts` defaults below, the margins, or
// the code they gate. Era 1 = the search before 2026-08-08 (unstamped in old data); era 2 =
// rfp+lmr+nullr+asp. Mirrored by SEARCH_ERA in engine/src/ai.zig.
export const SEARCH_ERA = 2;
const MATE = 1_000_000;
const MATE_THRESH = MATE - 1000; // scores beyond this magnitude encode a forced mate
const MAX_PLY = 64;
const QDEPTH = 6; // quiescence depth cap
const DELTA_MARGIN = 200; // qsearch: skip a capture if even winning it stays this far below alpha
const now = () => Date.now();

// --- search-refinement tuning ------------------------------------------------
// Every margin below is in centipawns and is sized against the NN EVAL'S RANGE,
// not against standard-chess practice: the nn eval is tanh-squashed × `scale`, so
// it is hard capped at ±600 cp for every champion so far. A textbook "150 cp per
// ply" margin is a quarter of the whole scale here, which would make these fire
// either always or never. Keep in sync with ai.zig (same names, same values).
const RFP_MAX_DEPTH = 5;  // reverse futility applies at depth ≤ this
const RFP_MARGIN = 60;    // ...with this much slack per remaining ply
const FP_MAX_DEPTH = 2;   // frontier futility applies at depth ≤ this
const FP_MARGIN = 80;     // ...per remaining ply,
const FP_BASE = 60;       // ...plus a fixed floor
const LMP_MAX_DEPTH = 4;  // late-move pruning applies at depth ≤ this
const LMR_DIV = 2.25;     // late-move reduction: r = 0.5 + ln(d)·ln(mc)/LMR_DIV
const HIST_MAX = 16384;   // history stays in [-HIST_MAX, HIST_MAX] via the gravity term
const HIST_GOOD = HIST_MAX / 4; // above this a quiet move is reduced one ply less
const ASP_DELTA = 30;     // first aspiration half-window around the previous score
const MAX_QUIETS = 48;    // quiet moves per node remembered for the history malus

// Late-move reduction table, r = 0.5 + ln(depth)·ln(moveCount)/LMR_DIV, saturating
// at 63. Built once so the search reads an entry instead of two logarithms.
const LMR = Array.from({ length: 64 }, (_, d) =>
  Array.from({ length: 64 }, (_, m) =>
    d === 0 || m === 0 ? 0 : Math.max(0, Math.floor(0.5 + (Math.log(d) * Math.log(m)) / LMR_DIV))));

// Move-count threshold for late-move pruning: 3 + d² (4, 7, 12, 19).
const lmpCount = (depth) => 3 + depth * depth;

// Which refinements are active. Mirrors ai.zig's SearchOpts — the Zig match runner
// exposes it as --search-a/--search-b so a search change can be played against its
// own predecessor in one binary; here it exists so the reference can be run in the
// same configurations. `true` = shipped; `false` = implemented but lost its gate.
//
// Measured 2026-08-08 at --nodes=50000 over 600 games against all-off (the search
// before any of these existed): rfp+nullr+asp+lmr scored **+74 ± 28 Elo** while
// cutting 75% of the nodes to depth 8 at an unchanged 371 nodes/ms; adding lmp, fp
// and hist dropped that to +17 ± 24 at 284 nodes/ms. `lmp` is the whole difference —
// biggest tree cut of the seven and the only one that costs nodes/*second* (pruning
// quiet moves shifts the surviving mix toward eval-heavy quiescence), and the least
// accurate (it agreed with the unpruned search on 9 of 12 midgame positions where the
// others managed 11–12). Blind move-count pruning is a bad trade in a variant whose
// quiet moves carry the jumps. See ai.zig's SearchOpts for the full table.
const searchOpts = { rfp: true, fp: false, lmp: false, lmr: true, hist: false, nullr: true, asp: true };
export function setSearchOpts(o) { Object.assign(searchOpts, o); }

let killers; // killers[ply] = [moveKey, moveKey]
let history; // Int32Array[from*64+to], bounded cutoff score (see histBump)
let counter; // Int32Array[previous move's key] = the quiet reply that refuted it
// Nodes visited by the current search, and the budget that stops it. A NODE IS COUNTED ON
// ENTRY to search() and to qsearch() — the two disjoint kinds of node this engine visits, and
// the only point every visited node passes through exactly once. ai.zig counts at exactly the
// same two entries, so "20000 nodes" names the same tree in both engines; counting cutoffs,
// applyMoves or leaf evals instead would make a JS/Zig comparison at a fixed budget diverge
// for no interesting reason.
//
// WHY the budget exists: at a fixed DEPTH a pruning or move-ordering gain returns the same move
// for fewer nodes, so its whole benefit is invisible and a fixed-depth match would reject every
// correct pruning change; wall-clock movetime prices it but reads machine load into the result.
// A node budget prices speed and accuracy on one scale and is deterministic. nodeCap = Infinity
// (the default, mirroring maxMs) means unbounded, so every existing caller is unaffected.
let nodes = 0;
let nodeCap = Infinity;
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

// The search's single stop condition: out of TIME or out of NODES. Both are monotonic (the
// clock never runs backwards, `nodes` only grows), so once it reads true every ancestor sees it
// too — which is what makes the unwind safe: in-tree nodes return 0, move loops break, the TT
// store is skipped so an incomplete score can't poison the persistent table, and the root marks
// the iteration aborted and keeps the move from the last COMPLETED iteration. Every former
// `now() > deadline` site checks this instead, so a node budget behaves exactly like a time
// limit rather than through a parallel path (ai.zig's outOfBudget is the same predicate).
const outOfBudget = (deadline) => nodes >= nodeCap || now() > deadline;

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

function scoreMove(m, board, ply, pvKey, counterKey) {
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
  if (key === counterKey) return 6.5e5; // countermove: below the killers, above history
  return Math.min(history[key], 6e5); // capped so quiet history never outranks the above
}

// Bounded history update ("gravity"): the correction term pulls the entry toward 0
// in proportion to how far it already is, so the table stays inside
// [-HIST_MAX, HIST_MAX] and a move that stopped working decays instead of coasting
// on an old score. The unbounded `+= depth*depth` it replaces let early cutoffs
// dominate move ordering forever.
function histBump(key, bonus) {
  const b = Math.max(-HIST_MAX, Math.min(HIST_MAX, bonus));
  history[key] += b - Math.trunc((history[key] * Math.abs(b)) / HIST_MAX);
}

function orderMoves(moves, board, ply, pvKey, counterKey) {
  for (const m of moves) m._o = scoreMove(m, board, ply, pvKey, counterKey);
  moves.sort((a, b) => b._o - a._o);
}

// Resolve captures/jumps/promotions to a quiet position before evaluating.
function qsearch(state, alpha, beta, qdepth) {
  // Counted on entry, before anything can return early (see `nodes`). qsearch itself has no
  // abort check — QDEPTH bounds it — so a node-capped search overshoots its cap by at most the
  // quiescence subtree in flight, deterministically (that subtree is a pure function of the
  // position). Same in ai.zig.
  nodes++;
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
    orderMoves(moves, state.board, 0, 0, 0);
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
  orderMoves(moves, state.board, 0, 0, 0);

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

function search(state, depth, alpha, beta, ply, canNull, hash, deadline, prevKey = 0) {
  if (outOfBudget(deadline)) { tainted = false; return 0; } // aborted; the root discards this iteration
  nodes++; // counted on entry, after the abort check — an aborted node is not a visited one
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

  // Static evaluation of THIS node, computed once and shared by the three refinements
  // that need it (reverse futility, the null-move reduction, frontier futility). It's
  // only taken where one of them can fire: in check none apply, and above
  // RFP_MAX_DEPTH only the null move reads it.
  const wantsNull = canNull && depth >= 3 && beta < MATE_THRESH;
  const wantsStatic = !inCheck &&
    ((searchOpts.rfp && depth <= RFP_MAX_DEPTH) ||
     (searchOpts.fp && depth <= FP_MAX_DEPTH) ||
     (searchOpts.nullr && wantsNull));
  const staticEval = wantsStatic ? activeEval(state.board, state.turn) : 0;

  // Reverse futility pruning ("static null move"): if the side to move is so far ahead
  // that giving up RFP_MARGIN per remaining ply still fails high, the node isn't worth
  // a search. It's the null-move idea without the search — and unlike the null move it
  // needs no zugzwang guard, because it never claims a line, only that the margin is
  // out of reach. Skipped near mate scores, where a centipawn margin means nothing.
  if (searchOpts.rfp && !inCheck && depth <= RFP_MAX_DEPTH &&
      beta > -MATE_THRESH && beta < MATE_THRESH &&
      staticEval - RFP_MARGIN * depth >= beta) {
    tainted = false;
    return staticEval;
  }

  // Null-move pruning: pass the move; if we're still ≥ beta, this node fails high.
  // The reduction is flat 3 in the base search; with `nullr` it grows with depth (a
  // deeper node can afford to give up more) and with how far the static eval already
  // is above beta — and the node must be at or above beta to try it at all, which is
  // what gives that eval term its meaning.
  if (wantsNull && !inCheck && hasNonPawn(state.board, state.turn) &&
      (!searchOpts.nullr || staticEval >= beta)) {
    const rNull = searchOpts.nullr
      ? Math.max(3, Math.min(depth - 1, 3 + Math.trunc(depth / 5) + Math.min(Math.trunc((staticEval - beta) / 160), 2)))
      : 3;
    const nm = {
      board: state.board, turn: opponent(state.turn),
      castling: state.castling, halfmove: state.halfmove, fullmove: state.fullmove,
    };
    const nh = ttEnabled ? hash ^ SIDE_KEY : 0n;
    const score = -search(nm, depth - rNull, -beta, -beta + 1, ply + 1, false, nh, deadline, 0);
    // A fail-high resting on a repetition draw is itself path-dependent; leave the
    // child's `tainted` in place (we don't store on this path) and bail out.
    if (score >= beta) return beta;
  }

  const legal = legalMoves(state);
  if (legal.length === 0) { tainted = false; return inCheck ? -MATE - depth : 0; }
  const counterKey = searchOpts.hist && prevKey ? counter[prevKey] : 0;
  orderMoves(legal, state.board, ply, ttMoveKey, counterKey);

  let best = -Infinity, bestKey = 0, moveCount = 0, bestTainted = false;
  const quiets = []; // quiet moves already tried here, for the history malus on a cutoff
  for (const m of legal) {
    moveCount++;
    const quiet = !m.capture && !m.promotion && !m.jump;
    // Shallow-depth pruning of QUIET moves only — the variant's tactics live in
    // captures, promotions and jumps, and none of those is ever skipped here. Both
    // rules require a real score in hand (`best > -MATE_THRESH`), so the first move is
    // always searched and a node can never come back empty.
    if (quiet && !inCheck && best > -MATE_THRESH) {
      // Late move pruning: past lmpCount(depth) quiet moves at a shallow depth, the
      // rest almost never beat what move ordering already put first.
      if (searchOpts.lmp && depth <= LMP_MAX_DEPTH && moveCount > lmpCount(depth)) continue;
      // Frontier futility: a quiet move can't lift a static eval this far below alpha
      // into the window in the plies that are left.
      if (searchOpts.fp && depth <= FP_MAX_DEPTH &&
          staticEval + FP_MARGIN * depth + FP_BASE <= alpha) continue;
    }
    const key = keyOf(m);
    if (quiet && quiets.length < MAX_QUIETS) quiets.push(key);
    const child = applyMove(state, m);
    const childHash = ttEnabled ? hashAfter(hash, state, m) : 0n;
    let score, sTainted;
    if (moveCount === 1) {
      score = -search(child, depth - 1, -beta, -alpha, ply + 1, true, childHash, deadline, key);
      sTainted = tainted;
    } else {
      // Late move reduction for quiet, late moves (never jumps/captures/promotions).
      let r = 0;
      if (quiet && depth >= 3 && moveCount > 3 && !inCheck) {
        if (searchOpts.lmr) {
          r = LMR[Math.min(depth, 63)][Math.min(moveCount, 63)];
          // A quiet move with a strong history is reduced one ply less: the table
          // already says it works, so the late-move assumption is weaker.
          if (searchOpts.hist && history[key] > HIST_GOOD) r--;
          r = Math.max(0, Math.min(r, depth - 2)); // never reduce into qsearch
        } else r = 1;
      }
      score = -search(child, depth - 1 - r, -alpha - 1, -alpha, ply + 1, true, childHash, deadline, key);
      sTainted = tainted;
      if (score > alpha && r > 0) { score = -search(child, depth - 1, -alpha - 1, -alpha, ply + 1, true, childHash, deadline, key); sTainted = tainted; }
      if (score > alpha && score < beta) { score = -search(child, depth - 1, -beta, -alpha, ply + 1, true, childHash, deadline, key); sTainted = tainted; }
    }
    if (score > best) { best = score; bestKey = key; bestTainted = sTainted; }
    if (best > alpha) alpha = best;
    if (alpha >= beta) {
      if (quiet) {
        const k = killers[ply] || (killers[ply] = [0, 0]);
        if (k[0] !== key) { k[1] = k[0]; k[0] = key; }
        if (searchOpts.hist) {
          // Bonus to the move that cut, MALUS to every quiet move tried before it.
          // Rewarding only the winner teaches the table which moves are good but never
          // which are bad, so a quiet move that keeps getting searched early and keeps
          // failing holds its slot forever.
          const bonus = Math.min(depth * depth * 16 + 32 * depth, HIST_MAX / 4);
          histBump(key, bonus);
          for (const qk of quiets) if (qk !== key) histBump(qk, -bonus);
          if (prevKey) counter[prevKey] = key;
        } else {
          history[key] += depth * depth;
        }
      }
      break;
    }
    if (outOfBudget(deadline)) break;
  }

  // The node's value is tainted if the move that fixed it (the best move, or the one
  // that caused the beta cutoff — both tracked by bestTainted) came back tainted.
  // Skip the store in that case so a path-dependent draw never lands in the
  // persistent table. Also skip past the deadline: a node that broke out of its
  // move loop on time (or on the node cap) has an incomplete `best`, and with a persistent
  // table a bogus entry would survive into later searches. Both budgets are monotonic, so once
  // we're out of budget every ancestor's store is skipped too.
  tainted = bestTainted;
  if (ttEnabled && !bestTainted && !outOfBudget(deadline)) {
    const flag = best <= alphaOrig ? UPPER : best >= beta ? LOWER : EXACT;
    ttStore(hash, depth, toTT(best, ply), flag, bestKey);
  }
  return best;
}

// Choose a move for the side to move, searching up to `maxDepth` plies but never
// past `maxMs` of wall-clock and never past `maxNodes` nodes. `rand` shuffles equal
// choices so games vary. `useTT` exists for benchmarking the transposition table on/off.
//
// `maxNodes` is the deterministic budget the offline match runner gates search changes on
// (apos-match --nodes; see `nodes` above for why depth and wall-clock both fail at that). It
// defaults to Infinity — unbounded, exactly like `maxMs` — and 0 is also read as unbounded, so
// it accepts ai.zig's `0 = no limit` convention too. Both budgets abort the same way: the
// partial iteration is discarded and the move from the last completed one is returned.
//
// Returns { move, ponder, depth, score, nodes }: `move` is the chosen move, `ponder` is the
// predicted opponent reply (its { from, to } — what to think about during their
// turn) read from the table after the search, `depth` is the deepest
// iteration completed (used to stop pondering once the line is fully resolved),
// and `nodes` is how many nodes it took (the unit `maxNodes` bounds, and what a
// JS-vs-Zig node comparison reads).
// The table is NOT cleared here — it persists across calls (see ttReset).
//
// `prevHashes` is the Zobrist hashes of positions that already occurred in the real
// game (so the search can recognise — and a winning side avoid — a genuine
// threefold draw it would otherwise be blind to). Pass [] when there's no history.
// Only positions since the last irreversible move (capture/pawn move — i.e. the last
// `halfmove` plies) can ever recur, so the caller need only pass that window; doing
// so keeps the per-node repetition lookup set tiny (usually empty).
export function chooseMoveDetailed(state, maxDepth = 2, rand = Math.random, maxMs = Infinity, useTT = true, prevHashes = [], engine = 'handcrafted', excludeKeys = null, onProgress = null, maxNodes = Infinity) {
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
    return { move: null, ponder: null, depth: 0, score: kingAttacked(state.board, state.turn) ? -MATE : 0, nodes: 0 };
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
  counter = new Int32Array(64 * 64);
  ttEnabled = useTT;
  if (useTT) ttBumpGen();
  const rootHash = useTT ? hashOf(state) : 0n;
  repPath = useTT ? [rootHash] : []; // index 0 = the current (root) position
  repSeen = useTT ? new Set(prevHashes) : new Set(); // positions already seen in the real game
  const deadline = now() + maxMs;
  // Node budget for this search only, set right beside the deadline and reset with the counter,
  // so the two budgets stay one mechanism and the cap is per-search (per move) the way movetime
  // is. `> 0` so 0 reads as unbounded, matching ai.zig's convention.
  nodes = 0;
  nodeCap = maxNodes > 0 ? maxNodes : Infinity;
  let bestMove = root[0];
  let completed = 0;
  let rootScore = 0; // side-to-move-relative value (cp) of the last completed depth

  // Backstop so an unbounded (maxDepth = Infinity) search still terminates even
  // if the deadline were also infinite; real searches abort on time long before.
  // One root pass over the (already ordered) root list inside the window [lo, hi], so the
  // aspiration loop below can call it again with a wider window when its guess was wrong.
  // `hi` is a real beta: a move that beats it ends the pass, since there's no point proving
  // how much better it is under a window that's about to be reopened. With lo = -Infinity
  // and hi = Infinity this is exactly the full-width root PVS pass search has always done.
  const rootWindow = (lo, hi) => {
    let alpha = lo, bestScore = -Infinity, localBest = root[0], moveCount = 0;
    for (const m of root) {
      moveCount++;
      const child = applyMove(state, m);
      const childHash = useTT ? hashAfter(rootHash, state, m) : 0n;
      const key = keyOf(m);
      let score;
      if (moveCount === 1) {
        score = -search(child, depth - 1, -hi, -alpha, 1, true, childHash, deadline, key);
      } else {
        score = -search(child, depth - 1, -alpha - 1, -alpha, 1, true, childHash, deadline, key);
        if (score > alpha && score < hi) score = -search(child, depth - 1, -hi, -alpha, 1, true, childHash, deadline, key);
      }
      if (outOfBudget(deadline)) return { aborted: true, score: 0, move: localBest };
      if (score > bestScore) { bestScore = score; localBest = m; }
      if (score > alpha) alpha = score;
      if (alpha >= hi) break; // fail high — the caller reopens the window
    }
    return { aborted: false, score: bestScore, move: localBest };
  };

  const depthCap = Math.min(maxDepth, 99);
  let depth = 1;
  for (; depth <= depthCap; depth++) {
    orderMoves(root, state.board, 0, keyOf(bestMove), 0);
    let bestScore = minimize ? Infinity : -Infinity, localBest = root[0], aborted = false;
    if (minimize) {
      // Loser mode: every root move needs its TRUE score (so the worst is exact), so
      // search each with a full window — no alpha tightening, no PVS, and no aspiration
      // window (which is an assumption about the BEST move's score).
      for (const m of root) {
        const child = applyMove(state, m);
        const childHash = useTT ? hashAfter(rootHash, state, m) : 0n;
        const score = -search(child, depth - 1, -Infinity, Infinity, 1, true, childHash, deadline, keyOf(m));
        if (outOfBudget(deadline)) { aborted = true; break; }
        if (score < bestScore) { bestScore = score; localBest = m; }
      }
    } else {
      // Aspiration window: the score at depth d is usually close to the score at d-1, so
      // searching [prev-delta, prev+delta] instead of the full window makes every node's
      // window narrower and cuts the tree. The cost is a re-search when the guess is
      // wrong, which is why it only opens once there's a previous score to guess from and
      // the position isn't already a forced mate.
      let lo = -Infinity, hi = Infinity, delta = ASP_DELTA;
      if (searchOpts.asp && depth >= 4 && completed > 0 && Math.abs(rootScore) < MATE_THRESH) {
        lo = rootScore - delta; hi = rootScore + delta;
      }
      for (;;) {
        const rr = rootWindow(lo, hi);
        if (rr.aborted) { aborted = true; break; }
        // Fail low: the true score is below the window, so the move that came back isn't
        // trustworthy — widen downward and keep the previous best.
        if (rr.score <= lo && lo !== -Infinity) {
          hi = Math.trunc((lo + hi) / 2);
          lo = rr.score - delta;
          delta += Math.trunc(delta / 2) + 5;
          continue;
        }
        // Fail high: a move beat the window, so it IS the new best; widen upward.
        if (rr.score >= hi && hi !== Infinity) {
          localBest = rr.move;
          hi = rr.score + delta;
          delta += Math.trunc(delta / 2) + 5;
          continue;
        }
        bestScore = rr.score; localBest = rr.move;
        break;
      }
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
  return { move: bestMove, ponder, depth: completed, score: rootScore, nodes };
}

export function chooseMove(state, maxDepth, rand, maxMs, useTT, prevHashes, engine, excludeKeys, maxNodes) {
  return chooseMoveDetailed(state, maxDepth, rand, maxMs, useTT, prevHashes, engine, excludeKeys, null, maxNodes).move;
}

// Exposed for tests only: Zobrist hash equivalence check + table reset so a
// benchmark/test can start each game from a cold table despite persistence.
export const _internal = { hashOf, hashAfter, resetTT: ttReset, evalStm, evalStmV3, evalMaterial, MATE, MATE_THRESH };
