// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2019-2026 Jean-David Moisan
//
// Read-only winrate report for the self-play dataset: what share of games White
// wins, Black wins, and draws. The dataset is GAME-PRIMARY (one record per game;
// scripts/gameRecord.mjs), and each game record's `r` is already the WHITE-view result
// (1 White won, -1 Black won, 0 draw) — so we just tally `r` per game. (Legacy
// position-primary lines, whose `r` is side-to-move-relative, are still handled by
// collapsing per game id and folding to White's view via the FEN's side to move.)
//
//   npm run winrate                 # report on ../training/data/selfplay.jsonl
//   npm run winrate -- --data=FILE   # point at a different *.jsonl dataset
//
// Nothing here writes or spawns — it streams the file line by line (the dataset is
// hundreds of MB), so it's safe to run against the live selfplay.jsonl.

import { createReadStream, existsSync } from 'node:fs';
import { createInterface } from 'node:readline';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const webDir = resolve(here, '..');
const repoDir = resolve(webDir, '..');

const args = Object.fromEntries(
  process.argv.slice(2).map((a) => {
    const m = a.replace(/^--/, '').split('=');
    return [m[0], m.length > 1 ? m[1] : true];
  }),
);
const dataFile =
  typeof args.data === 'string'
    ? resolve(args.data)
    : resolve(repoDir, 'training', 'data', 'selfplay.jsonl');

if (!existsSync(dataFile)) {
  console.error(`No dataset at ${dataFile}. Generate self-play first (or pass --data=FILE).`);
  process.exit(1);
}

// game id -> White-POV result (1 White won, -1 Black won, 0 draw). First record per
// game wins; the value is constant within a game, so any record gives the same answer.
const games = new Map();
let positions = 0;
let bad = 0;

const rl = createInterface({ input: createReadStream(dataFile), crlfDelay: Infinity });
for await (const line of rl) {
  if (!line.trim()) continue;
  positions += 1;
  let rec;
  try {
    rec = JSON.parse(line);
  } catch {
    bad += 1;
    continue;
  }
  const { g, fen, r } = rec;
  if (typeof g !== 'string' || typeof r !== 'number') { bad += 1; continue; }
  if (Array.isArray(rec.moves)) {
    // Game record: r is already White-view; one record per game, count its positions.
    positions += (rec.v ? rec.v.length : rec.moves.length + 1) - 1; // -1: this line counted as 1 above
    games.set(g, Math.sign(r));
    continue;
  }
  // Legacy position-primary line: r is side-to-move-relative; fold to White's view.
  if (typeof fen !== 'string') { bad += 1; continue; }
  if (games.has(g)) continue;
  const stm = fen.split(' ')[1]; // 'w' or 'b'
  const whiteResult = stm === 'b' ? -r : r;
  games.set(g, Math.sign(whiteResult));
}

let white = 0;
let black = 0;
let draw = 0;
for (const result of games.values()) {
  if (result > 0) white += 1;
  else if (result < 0) black += 1;
  else draw += 1;
}

const total = games.size;
const pct = (n) => (total ? ((100 * n) / total).toFixed(1) : '0.0');
// Score from White's perspective (win=1, draw=0.5), the usual chess yardstick.
const whiteScore = total ? (white + draw / 2) / total : 0;

console.log(`Dataset: ${dataFile}`);
console.log(`Positions: ${positions.toLocaleString()}  |  Games: ${total.toLocaleString()}`);
if (bad) console.log(`Skipped (unparseable / missing fields): ${bad.toLocaleString()}`);
console.log('');
console.log(`  White wins : ${String(white).padStart(7)}  (${pct(white)}%)`);
console.log(`  Black wins : ${String(black).padStart(7)}  (${pct(black)}%)`);
console.log(`  Draws      : ${String(draw).padStart(7)}  (${pct(draw)}%)`);
console.log('');
console.log(`  White score: ${(100 * whiteScore).toFixed(1)}%  (win=1, draw=½)`);
