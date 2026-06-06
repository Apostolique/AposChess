// Throwaway: short search workload for CPU profiling (run with: node --prof).
import { newGameState } from './board.js';
import { legalMoves, applyMove } from './engine.js';
import { chooseMove, _internal } from './ai.js';

function rng(seed) { return () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; }; }

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
for (let i = 0; i < ps.length; i++) { _internal.resetTT(); chooseMove(ps[i], 6, rng(7 + i), Infinity, true); }
console.log('done');
