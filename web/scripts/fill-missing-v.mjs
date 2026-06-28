// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2019-2026 Jean-David Moisan
//
// Fill a search value `v` on any position that lacks one in the GAME-PRIMARY dataset
// (scripts/gameRecord.mjs) — e.g. the random opening plies the generator doesn't search.
// `v` is a property of the POSITION (the net's fixed-depth assessment), independent of how
// the position was reached, so evaluating these positions is valid and makes the whole
// dataset's TD/bootstrap target uniform (train.py --lambda).
//
// This is the single-threaded convenience subset of `refresh-v.mjs` (whose default mode
// also fills missing v, multithreaded) — handy for a quick top-up. Games whose every v is
// present pass through as raw strings (no parse/replay); only games with a gap are replayed.
//
// Usage (run from web/):  node scripts/fill-missing-v.mjs [--depth=D] [--in=FILE] [--weights=FILE]

import { createReadStream, createWriteStream, existsSync, rmSync, renameSync } from 'node:fs';
import { createInterface } from 'node:readline';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

import { toFen } from '../src/board.js';
import { makeEngine } from './wasmEngine.mjs';
import { isGameRecord, vsAt, setVsAt, normalizeVs, serializeGameRecord, expandPositions } from './gameRecord.mjs';
import { vtag as computeVtag } from './vtag.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const args = Object.fromEntries(process.argv.slice(2).map((a) => {
  const m = a.replace(/^--/, '').split('='); return [m[0], m.length > 1 ? m[1] : true];
}));
const dataFile = typeof args.in === 'string'
  ? resolve(process.cwd(), args.in) : resolve(here, '../../training/data/selfplay.jsonl');
const depth = args.depth !== undefined ? Number(args.depth) : 6;
const weights = typeof args.weights === 'string'
  ? resolve(process.cwd(), args.weights) : resolve(here, '../src/nn-weights.json');

if (!existsSync(dataFile)) { console.error(`No dataset at ${dataFile}`); process.exit(1); }
// Score via the native Zig engine (wasm) — consistent with the generator/gate.
const eng = makeEngine('nn', weights);
const vtag = computeVtag('nn', depth, weights);

console.log(`fill-missing-v: ${dataFile} | depth ${depth} | weights ${weights.replace(/^.*[\\/]/, '')}`);

const tmp = dataFile + '.tmp';
const out = createWriteStream(tmp);
const write = async (s) => { if (!out.write(s)) await new Promise((r) => out.once('drain', r)); };

const rl = createInterface({ input: createReadStream(dataFile), crlfDelay: Infinity });
let games = 0, filled = 0;
const t0 = Date.now();

for await (const line of rl) {
  if (!line) continue;
  // Cheap gate: a game with no missing v has no `null` in its v array, so skip the parse.
  if (!line.includes('null')) { await write(line + '\n'); continue; }
  let rec;
  try { rec = JSON.parse(line); } catch { await write(line + '\n'); continue; }
  if (!isGameRecord(rec) || !Array.isArray(rec.v)) { await write(line + '\n'); continue; }
  games++;
  // Replay; search any position whose v is missing, write it back (vs promotes to an array).
  let changed = false;
  for (const p of expandPositions(rec)) {
    if (rec.v[p.ply] == null) {
      rec.v[p.ply] = eng.score(toFen(p.state), depth);
      setVsAt(rec, p.ply, vtag);
      filled++; changed = true;
      if (filled % 200 === 0) process.stdout.write(`\r  filled ${filled} | ${games} games`);
    }
  }
  if (changed) normalizeVs(rec);
  await write(serializeGameRecord(rec) + '\n');
}
await new Promise((r) => out.end(r));
if (existsSync(dataFile)) rmSync(dataFile);
renameSync(tmp, dataFile);
process.stdout.write('\n');
console.log(`Done: filled v on ${filled} positions across ${games} game(s) in ${((Date.now() - t0) / 1000).toFixed(0)}s. `
  + `Re-featurize next:  npm run train:featurize`);
