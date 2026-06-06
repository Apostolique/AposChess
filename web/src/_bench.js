// Throwaway: perft (correctness invariant) + speed benchmark for engine work.
// Behaviour-preserving optimizations MUST reproduce the perft counts exactly.
// Timings are best-of-N (min = least interference) and the workload is kept light
// so the CPU doesn't thermally throttle mid-measurement and skew comparisons.
import { newGameState } from './board.js';
import { legalMoves, applyMove } from './engine.js';
import { chooseMove, _internal } from './ai.js';

function perft(state, depth) {
  if (depth === 0) return 1;
  const moves = legalMoves(state);
  if (depth === 1) return moves.length;
  let nodes = 0;
  for (const m of moves) nodes += perft(applyMove(state, m), depth - 1);
  return nodes;
}

function rng(seed) { return () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; }; }
function best(runs, fn) {
  let min = Infinity;
  for (let i = 0; i < runs; i++) {
    const t0 = process.hrtime.bigint();
    fn();
    const ms = Number(process.hrtime.bigint() - t0) / 1e6;
    if (ms < min) min = ms;
  }
  return min;
}

const ps = [];
{
  const r = rng(2024);
  let st = newGameState();
  while (ps.length < 6) {
    const ms = legalMoves(st);
    if (!ms.length) { st = newGameState(); continue; }
    st = applyMove(st, ms[Math.floor(r() * ms.length)]);
    if (legalMoves(st).length) ps.push(st);
    if (ps.length % 3 === 0) st = newGameState();
  }
}

const start = newGameState();
// Correctness invariant — these counts must never change.
const counts = [1, 2, 3, 4].map((d) => perft(start, d));
console.log('perft counts d1..d4 (invariant):', counts.join(', '));

// Move-gen speed: perft d4 from start (deterministic, ~sub-second).
const genMs = best(5, () => perft(start, 4));
console.log(`perft d4: ${genMs.toFixed(0)}ms  (${(736594 / genMs / 1000).toFixed(2)} Mnps)`);

// End-to-end search speed (depth 5 over a few positions), the felt latency.
const searchMs = best(3, () => {
  for (let i = 0; i < ps.length; i++) { _internal.resetTT(); chooseMove(ps[i], 5, rng(7 + i), Infinity, true); }
});
console.log(`search d5 x${ps.length}: ${searchMs.toFixed(0)}ms`);
