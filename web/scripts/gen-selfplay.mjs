// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2019-2026 Jean-David Moisan
//
// Self-play data generator for the neural-net evaluation. Plays games with the
// existing engine (the "teacher") from randomized openings, and writes one JSONL
// line per position — the RAW position, net-agnostic: the board "fen" (with correct
// castling rights, since the generator keeps the full game state), the game's final
// result from the SIDE-TO-MOVE's view ("r"; the White-view outcome is sign-flipped
// for Black-to-move positions, matching the canonical features), and a
// game id ("g") so the trainer can split train/val by GAME — every position in a
// game shares one label and is highly correlated, so a position-level split would
// leak a game across both sides and make the val loss (and early stopping)
// optimistic. The id is "<seed>-<index>" so it's unique per run; merging runs (or
// appending) never collides as long as seeds differ (a collision would only group
// two games together, which is harmless — it never straddles the split).
//
// This raw dataset is turned into per-net training inputs by scripts/featurize.mjs
// (`npm run train:featurize`), which applies the current nn.js featureIndices and
// writes selfplay.features.jsonl — so the trainer still needs no chess logic, and a
// feature change is a quick featurize pass instead of regenerating self-play.
// Targets are pure game outcomes from the mover's view (+1 the side to move went on
// to win / 0 draw / -1 it lost) so the net learns from who actually won rather than
// mimicking (and inheriting the blind spots of) the handcrafted evaluation. To
// iterate, regenerate with --eval=nn once you have weights, so fresh data comes from
// the improving net.
//
// Games run in parallel across worker threads (scripts/genWorker.mjs); this main
// thread is the single writer to the output file, so parallel games never interleave
// mid-line. Each game is seeded from --seed and its index, so the dataset is
// reproducible from --seed regardless of --jobs (arrival order in the file may vary,
// but order doesn't matter to the trainer — the "g" id groups each game).
//
// Usage (run from web/):
//   npm run train:gen -- [options]
// Options:
//   --games=N       games to play (default 200); 'inf'/'forever' runs until Ctrl-C
//   --forever       alias for --games=inf: generate indefinitely, flushing each
//                   game as it finishes; press Ctrl-C to stop (in-flight games drain)
//   --jobs=N        parallel worker threads (default: CPU core count)
//   --depth=D       fixed-depth search per move (default 4)
//   --movetime=MS   use a time budget per move instead of fixed depth
//   --openings=K    random plies to start each game, for variety (default 8)
//   --eval=NAME     engine evaluation to play with: 'handcrafted' (default) | 'nn'
//   --maxmoves=N    adjudicate a draw after N plies (default 200)
//   --out=FILE      output path (default ../training/data/selfplay.jsonl); appends
//   --seed=S        RNG seed (default Date.now())

import { mkdirSync, appendFileSync, existsSync, readFileSync, statSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { Worker } from 'node:worker_threads';
import { cpus } from 'node:os';

import { fmtDur, fmtNum, fmtMB, liveStatus } from './fmt.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const args = Object.fromEntries(
  process.argv.slice(2).map((a) => {
    const m = a.replace(/^--/, '').split('=');
    return [m[0], m.length > 1 ? m[1] : true];
  }),
);
const num = (v, d) => (v === undefined ? d : Number(v));
// 'inf'/'infinity'/'forever' (or the --forever flag) means run until interrupted.
const parseGames = (v, d) => {
  if (args.forever) return Infinity;
  if (v === undefined) return d;
  if (typeof v === 'string' && /^(inf|infinity|forever)$/i.test(v)) return Infinity;
  return Number(v);
};

const cfg = {
  games: parseGames(args.games, 200),
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
const jobs = Math.max(1, Math.min(num(args.jobs, cpus().length), cfg.games));

// Warn once up front (the workers load weights silently) if the net was requested
// but its weights are missing/placeholder, so the teacher quality is no surprise.
if (cfg.evalName === 'nn') {
  try {
    const w = JSON.parse(readFileSync(resolve(here, '../src/nn-weights.json'), 'utf8'));
    if (!w.arch) console.warn('nn-weights.json is a placeholder; the net will use its material fallback.');
  } catch { console.warn('Could not read nn-weights.json; the net will use its material fallback.'); }
}

mkdirSync(dirname(cfg.out), { recursive: true });
const fresh = !existsSync(cfg.out);
const forever = !Number.isFinite(cfg.games);
console.log(`Generating ${forever ? 'games forever (Ctrl-C to stop)' : `${cfg.games} games`} -> ${cfg.out}${fresh ? '' : ' (appending)'}`);
console.log(`  ${cfg.depth != null ? `depth ${cfg.depth}` : `${cfg.movetime}ms/move`} | eval ${cfg.evalName} | openings ${cfg.openings} | jobs ${jobs} | seed ${cfg.seed}`);

const status = liveStatus();
const t0 = Date.now();
let totalPositions = 0;
let doneGames = 0;
let nextGame = 0;
let stopping = false; // set on Ctrl-C in forever mode: stop dispatching, let in-flight games drain

await new Promise((resolve_) => {
  const pool = [];
  let live = jobs;
  const retire = () => { if (--live === 0) resolve_(); };

  // In forever mode, a SIGINT stops new dispatches; workers finish their current
  // game (already flushed on completion) and then retire, so the run ends cleanly.
  if (forever) {
    process.on('SIGINT', () => {
      if (stopping) return;
      stopping = true;
      status.clear();
      console.log('  Stopping: draining in-flight games...');
    });
  }

  const dispatch = (w) => {
    if (stopping || nextGame >= cfg.games) { w.terminate(); return; } // exit -> retire
    w.postMessage({ type: 'play', g: nextGame++ });
  };

  for (let i = 0; i < jobs; i++) {
    const w = new Worker(new URL('./genWorker.mjs', import.meta.url), { workerData: { cfg } });
    pool.push(w);
    w.on('message', (msg) => {
      if (msg.type === 'ready') { dispatch(w); return; }
      if (msg.type === 'result') {
        if (msg.lines) appendFileSync(cfg.out, msg.lines); // single writer: no interleave
        totalPositions += msg.nPositions;
        doneGames++;
        const elapsed = (Date.now() - t0) / 1000;
        const head = forever ? `${doneGames} games` : `${doneGames}/${cfg.games} games`;
        const eta = !forever && doneGames < cfg.games
          ? ` | ETA ${fmtDur((elapsed / doneGames) * (cfg.games - doneGames))}` : '';
        status.update(`  ${head} | ${fmtNum(totalPositions)} positions | `
          + `${(doneGames / (elapsed / 60)).toFixed(1)} games/min | ${fmtDur(elapsed)} elapsed${eta}`);
        dispatch(w);
      }
    });
    w.on('error', (err) => { console.error('\ngenerator worker error:', err); w.terminate(); });
    w.on('exit', retire);
  }
});
status.clear();
{
  const el = (Date.now() - t0) / 1000;
  const size = existsSync(cfg.out) ? ` Dataset now ${fmtMB(statSync(cfg.out).size)}.` : '';
  console.log(`Done: ${doneGames} games, ${fmtNum(totalPositions)} positions in ${fmtDur(el)} `
    + `(${(doneGames / (el / 60)).toFixed(1)} games/min).${size}`);
}
