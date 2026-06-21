// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2019-2026 Jean-David Moisan
//
// Engine equivalence harness — the safety net for porting the engine to another
// language (Zig → native binary for the offline tools + wasm32 for the browser).
//
// It runs the CURRENT JS engine over a deterministic battery of positions and
// writes a reference file (engine-parity.json) the port's test runner reloads and
// must reproduce exactly. Three layers, in increasing specificity:
//
//   1. perft   — node counts per position per depth: catches move-generation bugs
//                (a wrong count means a missing/extra/illegal move somewhere). The
//                shallow `divide` (per-root-move leaf counts) localizes WHICH move.
//   2. moves   — the exact sorted legal-move list per position: a coincidental
//                perft match can still hide a wrong move set; this pins the set.
//   3. zobrist — hashOf(position) AND the invariant that the incremental hashAfter
//                equals the from-scratch hashOf(applyMove) for every legal move
//                (the same equivalence the codebase verifies by hand). The port's
//                incremental hash must match these.
//   4. eval    — handcrafted (evalStm) and nn (evaluate) values per position, so a
//                ported eval is bit-for-bit checkable against the JS one.
//
// Positions are reproducible: startpos plus boards reached by playing a fixed
// number of random *legal* moves from a seeded RNG, so the battery exercises real
// variant features (jumps, safety zones, castling, promotions) and is identical on
// every run and on every machine. Self-checks run first: if the JS engine's own
// Zobrist invariant fails, the reference would be wrong, so we abort rather than
// emit a bad baseline.
//
// Usage (run from web/):
//   node scripts/engine-parity.mjs [--seed=S] [--out=FILE] [--start-depth=D] [--probe-depth=D]
//   npm run parity

import { writeFileSync, readFileSync, existsSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

import { newGameState, toFen, parseFen, squareName } from '../src/board.js';
import { legalMoves, applyMove } from '../src/engine.js';
import { _internal } from '../src/ai.js';
import { loadWeights, hasWeights, evaluate as nnEvaluate } from '../src/nn.js';

const { hashOf, hashAfter, evalStm } = _internal;
const here = dirname(fileURLToPath(import.meta.url));

const args = Object.fromEntries(process.argv.slice(2).map((a) => {
  const m = a.replace(/^--/, '').split('='); return [m[0], m.length > 1 ? m[1] : true];
}));
const seed = args.seed !== undefined ? Number(args.seed) : 1;
const startDepth = args['start-depth'] !== undefined ? Number(args['start-depth']) : 4;
const probeDepth = args['probe-depth'] !== undefined ? Number(args['probe-depth']) : 3;
const outFile = typeof args.out === 'string'
  ? resolve(process.cwd(), args.out)
  : resolve(here, '..', 'engine-parity.json');

// Same generator the other tools use, so "seeded and reproducible" means the same
// thing everywhere in the repo.
function mulberry32(a) {
  a >>>= 0;
  return () => { a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

const MASK64 = (1n << 64n) - 1n;
const hexHash = (state) => (hashOf(state) & MASK64).toString(16).padStart(16, '0');

// A move's identity: from–to (+ promotion). Castling is implied by the king's
// from/to, so this is unique. Sorted lexically for a stable, diffable list.
function moveKey(m) {
  let k = squareName(m.from) + squareName(m.to);
  if (m.promotion) k += `=${m.promotion}`;
  return k;
}
const sortedMoveKeys = (state) => legalMoves(state).map(moveKey).sort();

function perft(state, depth) {
  if (depth <= 0) return 1;
  const moves = legalMoves(state);
  if (depth === 1) return moves.length;
  let n = 0;
  for (const m of moves) n += perft(applyMove(state, m), depth - 1);
  return n;
}

// Per-root-move leaf counts at (depth-1): the standard way to bisect a move-gen
// discrepancy down to the single root move whose subtree diverges.
function perftDivide(state, depth) {
  const out = {};
  for (const m of legalMoves(state)) out[moveKey(m)] = perft(applyMove(state, m), Math.max(0, depth - 1));
  return out;
}

// Play `plies` random legal moves from the start; stops early at a terminal node.
// Each probe re-seeds from (seed + index) so it's independent and reproducible.
function randomPosition(plies, s) {
  const rng = mulberry32(s);
  let state = newGameState();
  for (let i = 0; i < plies; i++) {
    const moves = legalMoves(state);
    if (!moves.length) break;
    state = applyMove(state, moves[Math.floor(rng() * moves.length)]);
  }
  return state;
}

// The Zobrist invariant the codebase relies on: the incrementally-updated hash
// (hashAfter) must equal the hash recomputed from scratch after the move
// (hashOf(applyMove)). Verified here for EVERY legal move of EVERY probe — if it
// fails in JS, the reference we'd emit is already wrong.
function checkZobristInvariant(state) {
  const h = hashOf(state);
  let checked = 0;
  for (const m of legalMoves(state)) {
    const incremental = (hashAfter(h, state, m) & MASK64);
    const recomputed = (hashOf(applyMove(state, m)) & MASK64);
    if (incremental !== recomputed) {
      return { ok: false, move: moveKey(m), incremental: incremental.toString(16), recomputed: recomputed.toString(16) };
    }
    checked++;
  }
  return { ok: true, checked };
}

// --- load the nn champion so its eval is part of the baseline ---------------------
const weightsPath = resolve(here, '..', 'src', 'nn-weights.json');
let weightsLoaded = false;
if (existsSync(weightsPath)) {
  try { loadWeights(JSON.parse(readFileSync(weightsPath, 'utf8'))); weightsLoaded = hasWeights(); }
  catch (e) { console.warn(`  nn weights at ${weightsPath} did not load (${e.message}); nn eval falls back to material.`); }
}

// --- build the position battery ---------------------------------------------------
// startpos, then a spread of midgame depths exercising the variant's mechanics.
const probePlies = [2, 4, 6, 8, 10, 13, 16, 20, 26, 34];
const positions = [{ id: 'start', plies: 0, state: newGameState() }];
probePlies.forEach((plies, i) => {
  positions.push({ id: `p${plies}`, plies, state: randomPosition(plies, seed + 1 + i) });
});

console.log(`engine-parity: seed ${seed} | ${positions.length} positions | `
  + `start depth ${startDepth}, probe depth ${probeDepth} | nn weights ${weightsLoaded ? 'loaded' : 'NOT loaded (material fallback)'}`);

// --- self-check the Zobrist invariant before trusting any output ------------------
let invMoves = 0;
for (const p of positions) {
  const r = checkZobristInvariant(p.state);
  if (!r.ok) {
    console.error(`\nFAIL: Zobrist incremental != recompute at ${p.id} (${toFen(p.state)})`);
    console.error(`  move ${r.move}: hashAfter=${r.incremental} vs hashOf(applyMove)=${r.recomputed}`);
    console.error('  The JS engine itself is inconsistent — refusing to write a bad reference.');
    process.exit(1);
  }
  invMoves += r.checked;
}
console.log(`  Zobrist invariant OK: incremental == recompute across ${invMoves.toLocaleString()} moves.`);

// --- emit the reference -----------------------------------------------------------
const out = { meta: {}, positions: [] };
let totalNodes = 0;
const t0 = Date.now();
positions.forEach((p, i) => {
  const depth = p.id === 'start' ? startDepth : probeDepth;
  const fen = toFen(p.state);
  // Re-parse from the FEN so the recorded hash/eval are exactly what a port gets
  // from the same FEN string (not from our in-memory state), the way the port
  // will rebuild positions.
  const state = parseFen(fen);
  const perftByDepth = {};
  for (let d = 1; d <= depth; d++) { perftByDepth[d] = perft(state, d); totalNodes += perftByDepth[d]; }
  const rec = {
    id: p.id,
    plies: p.plies,
    fen,
    hash: hexHash(state),
    evalHc: Math.round(evalStm(state.board, state.turn)),
    evalNn: Math.round(nnEvaluate(state.board, state.turn)),
    moves: sortedMoveKeys(state),
    perft: perftByDepth,
  };
  // A shallow divide on the first few positions: enough to localize a move-gen
  // bug without bloating the file for every position.
  if (i < 4 && depth >= 2) rec.divide = perftDivide(state, Math.min(depth, 3));
  out.positions.push(rec);
});

out.meta = {
  generatedAt: new Date().toISOString(),
  seed,
  startDepth,
  probeDepth,
  weights: weightsLoaded ? 'src/nn-weights.json' : null,
  positions: out.positions.length,
  zobristMovesChecked: invMoves,
};

writeFileSync(outFile, JSON.stringify(out, null, 2) + '\n');
console.log(`  perft: ${totalNodes.toLocaleString()} nodes in ${((Date.now() - t0) / 1000).toFixed(1)}s.`);
console.log(`Wrote ${outFile} — ${out.positions.length} positions. The port must reproduce hash, moves, perft, and eval for each.`);
