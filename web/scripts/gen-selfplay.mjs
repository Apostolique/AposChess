// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2019-2026 Jean-David Moisan
//
// Self-play data generator for the neural-net evaluation. Plays games with the
// existing engine (the "teacher") from randomized openings, and writes one JSONL
// line per position: the active input feature indices (from nn.js, so the feature
// definition is single-sourced), the game's final result from the SIDE-TO-MOVE's
// view (matching the canonical, side-to-move feature orientation; the White-view
// outcome is sign-flipped for Black-to-move positions), and a
// game id ("g") so the trainer can split train/val by GAME — every position in a
// game shares one label and is highly correlated, so a position-level split would
// leak a game across both sides and make the val loss (and early stopping)
// optimistic. The id is "<seed>-<index>" so it's unique per run; merging runs (or
// appending) never collides as long as seeds differ (a collision would only group
// two games together, which is harmless — it never straddles the split).
//
// The trainer (training/train.py) reads these vectors directly — it needs no chess
// logic. Targets are pure game outcomes from the mover's view (+1 the side to move
// went on to win / 0 draw / -1 it lost) so the net learns from who actually won
// rather than mimicking (and inheriting the blind spots of) the handcrafted
// evaluation. To iterate, regenerate with
// --eval=nn once you have weights, so fresh data comes from the improving net.
//
// Usage (run from web/):
//   npm run train:gen -- [options]
// Options:
//   --games=N       games to play (default 200)
//   --depth=D       fixed-depth search per move (default 4)
//   --movetime=MS   use a time budget per move instead of fixed depth
//   --openings=K    random plies to start each game, for variety (default 8)
//   --eval=NAME     engine evaluation to play with: 'handcrafted' (default) | 'nn'
//   --maxmoves=N    adjudicate a draw after N plies (default 200)
//   --out=FILE      output path (default ../training/data/selfplay.jsonl); appends
//   --seed=S        RNG seed (default Date.now())

import { mkdirSync, appendFileSync, existsSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { readFileSync } from 'node:fs';

import { newGameState } from '../src/board.js';
import { legalMoves, applyMove, gameStatus } from '../src/engine.js';
import { chooseMove, _internal } from '../src/ai.js';
import { featureIndices, loadWeights } from '../src/nn.js';

const here = dirname(fileURLToPath(import.meta.url));
const args = Object.fromEntries(
  process.argv.slice(2).map((a) => {
    const m = a.replace(/^--/, '').split('=');
    return [m[0], m.length > 1 ? m[1] : true];
  }),
);
const num = (v, d) => (v === undefined ? d : Number(v));

const cfg = {
  games: num(args.games, 200),
  depth: args.depth !== undefined ? Number(args.depth) : (args.movetime !== undefined ? null : 4),
  movetime: num(args.movetime, 50),
  openings: num(args.openings, 8),
  evalName: typeof args.eval === 'string' ? args.eval : 'handcrafted',
  maxmoves: num(args.maxmoves, 200),
  // Default lands in the repo-root training/data/ (where train.py reads); a custom
  // --out is resolved against the current directory for intuitive relative paths.
  out: typeof args.out === 'string'
    ? resolve(process.cwd(), args.out)
    : resolve(here, '../../training/data/selfplay.jsonl'),
  seed: num(args.seed, Date.now()),
};

// If playing with the net, load its weights so the teacher is the improving net.
if (cfg.evalName === 'nn') {
  try {
    const w = JSON.parse(readFileSync(resolve(here, '../src/nn-weights.json'), 'utf8'));
    loadWeights(w);
    if (!w.arch) console.warn('nn-weights.json is a placeholder; the net will use its material fallback.');
  } catch { console.warn('Could not read nn-weights.json; the net will use its material fallback.'); }
}

function makeRng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const rng = makeRng(cfg.seed);
const pick = (arr) => arr[Math.floor(rng() * arr.length)];

// Play one game; return { positions: [{board,turn}], result } with result from
// White's perspective (+1 / 0 / -1). The opening plies are random for variety.
function playGame() {
  let state = newGameState();
  const positions = [];
  const seen = [];
  let result = 0;

  for (let ply = 0; ply < cfg.maxmoves; ply++) {
    const status = gameStatus(state);
    if (status.over) {
      result = status.result === 'checkmate' ? (status.winner === 'white' ? 1 : -1) : 0;
      break;
    }
    // Record real (post-opening) positions only; random opening moves aren't the
    // engine's choices, but the positions themselves are still fine training data.
    positions.push({ board: state.board, turn: state.turn });

    let move;
    if (ply < cfg.openings) {
      move = pick(status.legal);
    } else {
      const prev = seen.map((s) => _internal.hashOf(s));
      move = chooseMove(state, cfg.depth ?? 99, rng,
        cfg.depth != null ? Infinity : cfg.movetime, true, prev, cfg.evalName);
    }
    seen.push(state);
    state = applyMove(state, move);
  }
  return { positions, result };
}

mkdirSync(dirname(cfg.out), { recursive: true });
const fresh = !existsSync(cfg.out);
console.log(`Generating ${cfg.games} games -> ${cfg.out}${fresh ? '' : ' (appending)'}`);
console.log(`  ${cfg.depth != null ? `depth ${cfg.depth}` : `${cfg.movetime}ms/move`} | eval ${cfg.evalName} | openings ${cfg.openings} | seed ${cfg.seed}`);

// Format a duration in seconds as "Mm SSs" (or "SSs" under a minute).
function fmt(secs) {
  secs = Math.round(secs);
  return secs < 60 ? `${secs}s` : `${Math.floor(secs / 60)}m ${String(secs % 60).padStart(2, '0')}s`;
}

const t0 = Date.now();
let totalPositions = 0;
for (let g = 0; g < cfg.games; g++) {
  // Live status before each game: which game is running, how many are left, and —
  // once a game or two has finished — elapsed time and an ETA from the running pace.
  const elapsed = (Date.now() - t0) / 1000;
  const eta = g > 0 ? ` | ETA ${fmt((elapsed / g) * (cfg.games - g))}` : '';
  process.stdout.write(
    `\r  game ${g + 1}/${cfg.games} | ${cfg.games - g - 1} left | `
    + `${totalPositions} positions | ${fmt(elapsed)} elapsed${eta}      `,
  );

  const { positions, result } = playGame();
  const gid = `${cfg.seed.toString(36)}-${g}`; // unique per run; groups one game
  let buf = '';
  for (const { board, turn } of positions) {
    // Canonical features are side-to-move-relative, so the label must be too:
    // flip the White-view game result for Black-to-move positions.
    const r = turn === 'white' ? result : -result;
    buf += JSON.stringify({ f: featureIndices(board, turn), r, g: gid }) + '\n';
  }
  appendFileSync(cfg.out, buf);
  totalPositions += positions.length;
}
console.log(`\nDone: ${totalPositions} positions written in ${fmt((Date.now() - t0) / 1000)}.`);
