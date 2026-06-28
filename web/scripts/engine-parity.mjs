// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2019-2026 Jean-David Moisan
//
// Engine equivalence harness — the safety net for porting the engine to another
// language (Zig → native binary for the offline tools + wasm32 for the browser).
//
// It runs the CURRENT JS engine over a deterministic battery of positions and
// writes reference files the port's test runner reloads and must reproduce exactly.
// The battery is reproducible: startpos plus boards reached by playing a fixed
// number of random *legal* moves from a seeded RNG, so it exercises real variant
// features (jumps, safety zones, castling, promotions) identically on every run and
// machine.
//
// TWO outputs, because the AposChess RULES ARE FROZEN while the eval keeps evolving:
//
//   engine-parity.json       — the STRUCTURAL contract: per position the legal-move
//                              list, perft node counts (+ a shallow per-root-move
//                              `divide` to localize a move-gen bug), and the Zobrist
//                              hash. Rules never change, so this is a PERMANENT oracle:
//                              commit it once and the port is forever checked against
//                              it; it only changes if the engine has a bug. The meta
//                              carries the exact Zobrist key-generation spec, since the
//                              hashes are only reproducible with identical key gen.
//
//   engine-parity.eval.json  — the EVAL contract: per position the handcrafted
//                              (evalStm) and nn (evaluate) centipawn values. The nn
//                              value tracks the current champion, so this file is
//                              REGENERATED whenever the champion changes; it's tagged
//                              with HC_VERSION and the champion's weights hash so a
//                              stale baseline is obvious. --no-eval skips it.
//
// Self-checks run first: the JS engine's own Zobrist invariant (incremental hashAfter
// == from-scratch hashOf(applyMove)) is verified for every legal move of every
// position; if it fails the reference would be wrong, so we abort instead of writing
// a bad baseline.
//
// Usage (run from web/):
//   node scripts/engine-parity.mjs [--seed=S] [--out=FILE] [--start-depth=D]
//                                  [--probe-depth=D] [--no-eval]
//   npm run parity

import { writeFileSync, readFileSync, existsSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

import { newGameState, toFen, parseFen, squareName } from '../src/board.js';
import { legalMoves, applyMove, gameStatus } from '../src/engine.js';
import { _internal, HC_VERSION } from '../src/ai.js';
import { loadWeights, hasWeights, evaluate as nnEvaluate } from '../src/nn.js';
import { weightsHash } from './vtag.mjs';

const { hashOf, hashAfter, evalStm, evalStmV3, evalMaterial } = _internal;
const here = dirname(fileURLToPath(import.meta.url));

const args = Object.fromEntries(process.argv.slice(2).map((a) => {
  const m = a.replace(/^--/, '').split('='); return [m[0], m.length > 1 ? m[1] : true];
}));
const seed = args.seed !== undefined ? Number(args.seed) : 1;
const startDepth = args['start-depth'] !== undefined ? Number(args['start-depth']) : 4;
const probeDepth = args['probe-depth'] !== undefined ? Number(args['probe-depth']) : 3;
const doEval = !args['no-eval'];
const outFile = typeof args.out === 'string'
  ? resolve(process.cwd(), args.out)
  : resolve(here, '..', 'engine-parity.json');
const evalOutFile = outFile.replace(/\.json$/, '.eval.json');

// The exact recipe a port must replicate to reproduce the hashes (mirrors ai.js).
// Embedded in the structural oracle so the contract is self-describing.
const ZOBRIST_SPEC = {
  rng: 'mulberry32 seeded 0x1a2b3c4d (uint32 stream)',
  rand64: '(BigInt(rnd()) << 32) | BigInt(rnd()) — first draw is the HIGH 32 bits',
  drawOrder: '768 piece keys, then SIDE_KEY, then CASTLE_KEYS in order K, Q, k, q',
  pieceKeyIndex: '(ROLE_IDX[role]*2 + (white?0:1))*64 + sq, ROLE_IDX={p:0,n:1,b:2,r:3,q:4,k:5}, sq=rank*8+file',
  hash: 'XOR piece keys of occupied squares; XOR SIDE_KEY iff black to move; XOR each available castle key',
  width: '64-bit, reported as 16-hex-digit lowercase',
};

// Same generator the other tools (and ai.js's key gen) use.
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

// --- load the nn champion so its eval is part of the eval baseline ----------------
const weightsPath = resolve(here, '..', 'src', 'nn-weights.json');
let weightsLoaded = false;
if (doEval && existsSync(weightsPath)) {
  try { loadWeights(JSON.parse(readFileSync(weightsPath, 'utf8'))); weightsLoaded = hasWeights(); }
  catch (e) { console.warn(`  nn weights at ${weightsPath} did not load (${e.message}); nn eval falls back to material.`); }
}

// Play full random games to termination and collect their final positions, so the
// status classifier (checkmate / stalemate / insufficient-material / fifty-move) is
// tested on REAL reachable terminals — including draws that still HAVE legal moves
// (insufficient/fifty), which perft can never distinguish from an ongoing position.
// Keeps a spread of distinct results plus a couple of extra checkmates.
function collectTerminals(s, maxGames = 200, maxPlies = 400) {
  const found = [];
  const seenResult = new Map();
  for (let g = 0; g < maxGames && found.length < 8; g++) {
    const rng = mulberry32(s + 1000 + g);
    let state = newGameState();
    for (let i = 0; i < maxPlies; i++) {
      const st = gameStatus(state);
      if (st.over) {
        const cap = seenResult.get(st.result) || 0;
        if (cap < (st.result === 'checkmate' ? 3 : 1)) { // variety, but a few mates
          seenResult.set(st.result, cap + 1);
          found.push({ id: `t${found.length + 1}`, plies: i, state, terminal: true });
        }
        break;
      }
      const moves = st.legal;
      state = applyMove(state, moves[Math.floor(rng() * moves.length)]);
    }
  }
  return found;
}

// --- build the position battery ---------------------------------------------------
// startpos, a spread of midgame depths exercising the variant's mechanics, then real
// terminal positions for the status classifier.
const probePlies = [2, 4, 6, 8, 10, 13, 16, 20, 26, 34];
const positions = [{ id: 'start', plies: 0, state: newGameState() }];
probePlies.forEach((plies, i) => {
  positions.push({ id: `p${plies}`, plies, state: randomPosition(plies, seed + 1 + i) });
});
positions.push(...collectTerminals(seed));

console.log(`engine-parity: seed ${seed} | ${positions.length} positions | `
  + `start depth ${startDepth}, probe depth ${probeDepth} | eval ${doEval ? `on (nn weights ${weightsLoaded ? 'loaded' : 'NOT loaded — material fallback'})` : 'off'}`);

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
  // moveKey (from–to + promotion) must INJECTIVELY identify a legal move: the game
  // dataset stores moves in exactly this compact form (gameRecord.mjs) and replays them
  // by matching legalMoves, so a collision would make a stored game un-replayable. This
  // holds by construction (slides stop before a piece, jumps land beyond it; knights
  // dedup; promotions differ by piece; castling lands two squares off) — assert it so a
  // future rule edit that broke it would fail here, not silently corrupt the data.
  const keys = legalMoves(p.state).map(moveKey);
  if (new Set(keys).size !== keys.length) {
    console.error(`\nFAIL: from–to(+promo) move key is not unique at ${p.id} (${toFen(p.state)}).`);
    console.error('  The compact move encoding the dataset relies on would be lossy — aborting.');
    process.exit(1);
  }
}
console.log(`  Zobrist invariant OK: incremental == recompute across ${invMoves.toLocaleString()} moves.`);
console.log('  Move-key uniqueness OK: from–to(+promo) identifies every legal move.');

// --- emit the structural (frozen) reference + the eval (champion-tagged) reference -
const structural = { meta: {}, positions: [] };
const evalRef = { meta: {}, positions: [] };
let totalNodes = 0;
const t0 = Date.now();
positions.forEach((p, i) => {
  const depth = p.id === 'start' ? startDepth : probeDepth;
  const fen = toFen(p.state);
  // Re-parse from the FEN so the recorded hash/eval are exactly what a port gets
  // from the same FEN string, the way the port rebuilds positions.
  const state = parseFen(fen);
  const st = gameStatus(state);
  // Terminal positions have no subtree worth walking — record perft(1) (= 0 for
  // mate/stalemate, the legal-move count for a draw that still has moves) and stop.
  const maxD = p.terminal ? 1 : depth;
  const perftByDepth = {};
  for (let d = 1; d <= maxD; d++) { perftByDepth[d] = perft(state, d); totalNodes += perftByDepth[d]; }
  const rec = {
    id: p.id, plies: p.plies, fen, hash: hexHash(state),
    status: st.result || 'ongoing', check: st.check,
    moves: sortedMoveKeys(state), perft: perftByDepth,
  };
  // A shallow divide on the first few non-terminal positions: enough to localize a
  // move-gen bug without bloating the file for every position.
  if (i < 4 && !p.terminal && depth >= 2) rec.divide = perftDivide(state, Math.min(depth, 3));
  structural.positions.push(rec);
  if (doEval) {
    evalRef.positions.push({
      id: p.id, fen,
      evalHc: Math.round(evalStm(state.board, state.turn)),
      evalHc3: Math.round(evalStmV3(state.board, state.turn)),
      evalMat: Math.round(evalMaterial(state.board, state.turn)),
      evalNn: Math.round(nnEvaluate(state.board, state.turn)),
    });
  }
});

const common = { generatedAt: new Date().toISOString(), seed, positions: structural.positions.length };
structural.meta = { ...common, startDepth, probeDepth, zobristMovesChecked: invMoves, zobrist: ZOBRIST_SPEC };
writeFileSync(outFile, JSON.stringify(structural, null, 2) + '\n');
console.log(`  perft: ${totalNodes.toLocaleString()} nodes in ${((Date.now() - t0) / 1000).toFixed(1)}s.`);
console.log(`Wrote ${outFile} (frozen rules oracle — perft, moves, hash). Commit once; only changes if the engine has a bug.`);

if (doEval) {
  evalRef.meta = {
    ...common,
    hcVersion: HC_VERSION,
    weights: weightsLoaded ? 'src/nn-weights.json' : null,
    weightsHash: weightsLoaded ? weightsHash(weightsPath) : null,
  };
  writeFileSync(evalOutFile, JSON.stringify(evalRef, null, 2) + '\n');
  console.log(`Wrote ${evalOutFile} (eval oracle, hc v${HC_VERSION}`
    + `${weightsLoaded ? ` / champion ${evalRef.meta.weightsHash}` : ''}). Regenerate when the champion changes.`);
}
