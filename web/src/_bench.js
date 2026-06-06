// Throwaway: perft (correctness invariant) + speed benchmark for engine work.
// A behavior-preserving optimization MUST reproduce the perft counts exactly;
// the timings show whether it actually got faster.
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

// A spread of reachable positions for representative branching/timing.
function positions(n) {
  const out = [], r = rng(2024);
  let st = newGameState();
  while (out.length < n) {
    const ms = legalMoves(st);
    if (!ms.length) { st = newGameState(); continue; }
    st = applyMove(st, ms[Math.floor(r() * ms.length)]);
    if (legalMoves(st).length) out.push(st);
    if (out.length % 5 === 0) st = newGameState();
  }
  return out;
}

const start = newGameState();
console.log('perft from start (counts are the invariant — must not change):');
for (let d = 1; d <= 5; d++) {
  const t0 = Date.now();
  const n = perft(start, d);
  const ms = Date.now() - t0;
  console.log(`  depth ${d}: ${String(n).padStart(9)} nodes  ${String(ms).padStart(6)}ms` + (ms > 5 ? `  ${(n / ms / 1000).toFixed(2)} Mnps` : ''));
}

// Aggregate perft over varied positions: pure move-gen + make + legality speed.
const ps = positions(12);
{
  const t0 = Date.now();
  let total = 0;
  for (const st of ps) total += perft(st, 3);
  const ms = Date.now() - t0;
  console.log(`perft d3 over ${ps.length} positions: ${total} nodes, ${ms}ms (${(total / ms / 1000).toFixed(2)} Mnps)`);
}

// End-to-end search (includes eval + TT): the latency the user actually feels.
{
  const t0 = Date.now();
  for (let i = 0; i < ps.length; i++) { _internal.resetTT(); chooseMove(ps[i], 6, rng(7 + i), Infinity, true); }
  console.log(`search depth 6 over ${ps.length} positions: ${Date.now() - t0}ms`);
}
