// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2019-2026 Jean-David Moisan
//
// ONE-OFF migration: convert the legacy POSITION-PRIMARY dataset ({fen,r,g,v,vs} per line)
// to the GAME-PRIMARY format ({g,start?,players?,r,moves,v,vs} per game; scripts/gameRecord.mjs).
//
// For each game (records grouped by `g`, in file order) it recovers the move list by finding
// the unique legal move connecting each position to the next, and folds the per-position
// v/vs into arrays. Two realities make this not a clean 1:1:
//   - HARVEST games ('m…' ids) start from a random opening, so the first position isn't the
//     standard start — we store it as `start`.
//   - dedup-cap / old concatenating merges can leave GAPS (a position whose successor in the
//     file isn't one legal move away). A continuous move list can't span a gap, so the game
//     is SPLIT into maximal connected SEGMENTS; segment k>0 gets a "<g>~k" id (still unique,
//     so merge/rank treat them as distinct). Every position is emitted exactly once, so the
//     featurized output (which depends only on board+turn+r+v, not move counters) is identical
//     — verify with --verify-featurize after, or the scratch check in this PR.
//
// `players` is left UNSET for legacy games (the position format never recorded who played),
// so they don't feed corpus ranking; new gen/match data carries it.
//
// Usage (run from web/):
//   node scripts/migrate-to-games.mjs [--in=FILE] [--out=FILE] [--limit=N]
// Defaults: in = ../training/data/selfplay.jsonl, out = <in>.games.jsonl (NOT in place — keep
//           the original as a backup until you've verified, then swap it in). --limit=N stops
//           after N input lines (for a quick sample).

import { createReadStream, createWriteStream, existsSync, statSync } from 'node:fs';
import { createInterface } from 'node:readline';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

import { parseFen, toFen, START_FEN } from '../src/board.js';
import { legalMoves, applyMove } from '../src/engine.js';
import { encodeMove, serializeGameRecord } from './gameRecord.mjs';
import { fmtDur, fmtNum, fmtMB, liveStatus, everyMs } from './fmt.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const args = Object.fromEntries(process.argv.slice(2).map((a) => {
  const m = a.replace(/^--/, '').split('='); return [m[0], m.length > 1 ? m[1] : true];
}));
const inFile = typeof args.in === 'string'
  ? resolve(process.cwd(), args.in) : resolve(here, '../../training/data/selfplay.jsonl');
const outFile = typeof args.out === 'string'
  ? resolve(process.cwd(), args.out) : inFile.replace(/\.jsonl$/, '') + '.games.jsonl';
const limit = args.limit !== undefined ? Number(args.limit) : Infinity;

if (!existsSync(inFile)) { console.error(`No dataset at ${inFile}`); process.exit(1); }

const posKey = (fen) => fen.split(' ').slice(0, 3).join(' ');
// The unique legal move from state A that reaches position B (matched by board+turn+castling,
// so it's robust to move-counter differences in the legacy fens). Null if none connects.
function connectingMove(stateA, fenB) {
  const bk = posKey(fenB);
  for (const m of legalMoves(stateA)) if (posKey(toFen(applyMove(stateA, m))) === bk) return m;
  return null;
}

const out = createWriteStream(outFile);
const write = async (s) => { if (!out.write(s)) await new Promise((r) => out.once('drain', r)); };

// Emit one game's records, split into maximal connected segments. `recs` are the legacy
// position records of one game, in file order.
async function emitGame(g, recs, stats) {
  // Split into segments where each consecutive pair connects by a single legal move.
  let segIdx = 0;
  let i = 0;
  while (i < recs.length) {
    // Grow a segment from i.
    const seg = [recs[i]];
    let state = parseFen(recs[i].fen);
    let j = i + 1;
    const moves = [];
    while (j < recs.length) {
      const m = connectingMove(state, recs[j].fen);
      if (!m) break;             // gap: end this segment
      moves.push(encodeMove(m));
      seg.push(recs[j]);
      state = parseFen(recs[j].fen);
      j++;
    }
    // Build the record for this segment.
    const first = seg[0];
    const firstState = parseFen(first.fen);
    const r = firstState.turn === 'white' ? first.r : -first.r; // White-view
    const v = seg.map((s) => (s.v == null ? null : s.v));
    const tags = seg.map((s) => (s.vs == null ? null : s.vs));
    const uniform = tags.every((t) => t === tags[0]);
    const rec = { g: segIdx === 0 ? g : `${g}~${segIdx}`, r, moves, v, vs: uniform ? tags[0] : tags };
    if (first.fen !== START_FEN) rec.start = first.fen;
    await write(serializeGameRecord(rec) + '\n');
    stats.segments++;
    if (seg.length === 1) stats.singletons++;
    if (j === i + 1 && j < recs.length) stats.gaps++; // a forced break (couldn't extend)
    segIdx++;
    i = j;
  }
  stats.games++;
}

const status = liveStatus();
const tick = everyMs(500);
const t0 = Date.now();
const stats = { lines: 0, games: 0, segments: 0, singletons: 0, gaps: 0, skipped: 0 };
console.log(`migrate-to-games: ${inFile} (${fmtMB(statSync(inFile).size)}) -> ${outFile}`);

let curG = null, buf = [];
const rl = createInterface({ input: createReadStream(inFile), crlfDelay: Infinity });
for await (const line of rl) {
  if (!line) continue;
  if (stats.lines >= limit) break;
  stats.lines++;
  let rec;
  try { rec = JSON.parse(line); } catch { stats.skipped++; continue; }
  if (Array.isArray(rec.moves)) { await write(line + '\n'); continue; } // already a game record
  if (typeof rec.fen !== 'string' || rec.g == null) { stats.skipped++; continue; }
  if (rec.g !== curG) {
    if (buf.length) await emitGame(curG, buf, stats);
    curG = rec.g; buf = [];
  }
  buf.push(rec);
  if (tick()) status.update(`  ${fmtNum(stats.lines)} lines | ${fmtNum(stats.games)} games | ${fmtNum(stats.segments)} segments`);
}
if (buf.length) await emitGame(curG, buf, stats);
await new Promise((r) => out.end(r));
status.clear();

console.log(`Done in ${fmtDur((Date.now() - t0) / 1000)}: ${fmtNum(stats.lines)} positions -> ${fmtNum(stats.games)} games `
  + `(${fmtNum(stats.segments)} segments, ${fmtNum(stats.singletons)} singletons) -> ${outFile} (${fmtMB(statSync(outFile).size)}).`);
if (stats.gaps) console.log(`  ${fmtNum(stats.gaps)} gap-splits (dedup-capped / non-contiguous positions).`);
if (stats.skipped) console.log(`  ${fmtNum(stats.skipped)} record(s) skipped (no fen / unparseable).`);
console.log('Verify featurize is loss-free, then replace the original with this file and re-featurize.');
