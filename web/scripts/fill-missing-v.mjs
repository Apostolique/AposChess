// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2019-2026 Jean-David Moisan
//
// Fill a search value `v` on any position-primary record ({fen,r,g}) that lacks one
// — i.e. the random opening plies the generator doesn't search. `v` is a property of
// the POSITION (the net's fixed-depth assessment), independent of how the position
// was reached or what move was played next, so evaluating these positions is valid
// and makes the whole dataset's TD/bootstrap target uniform (train.py --lambda).
//
// Lines that already carry `"v":` pass through as raw strings (no parse), so only the
// handful missing it are parsed + searched — fast over a multi-million-line file.
//
// Usage (run from web/):  node scripts/fill-missing-v.mjs [--depth=D] [--in=FILE] [--weights=FILE]

import { createReadStream, createWriteStream, existsSync, rmSync, renameSync, readFileSync, statSync } from 'node:fs';
import { createInterface } from 'node:readline';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

import { parseFen } from '../src/board.js';
import { chooseMoveDetailed } from '../src/ai.js';
import { loadWeights } from '../src/nn.js';

const here = dirname(fileURLToPath(import.meta.url));
const args = Object.fromEntries(process.argv.slice(2).map((a) => {
  const m = a.replace(/^--/, '').split('='); return [m[0], m.length > 1 ? m[1] : true];
}));
const dataFile = typeof args.in === 'string'
  ? resolve(process.cwd(), args.in) : resolve(here, '../../training/data/selfplay.jsonl');
const depth = args.depth !== undefined ? Number(args.depth) : 3;
const weights = typeof args.weights === 'string'
  ? resolve(process.cwd(), args.weights) : resolve(here, '../src/nn-weights.json');

if (!existsSync(dataFile)) { console.error(`No dataset at ${dataFile}`); process.exit(1); }
try { loadWeights(JSON.parse(readFileSync(weights, 'utf8'))); } catch { /* material fallback */ }

console.log(`fill-missing-v: ${dataFile} | depth ${depth} | weights ${weights.replace(/^.*[\\/]/, '')}`);

const tmp = dataFile + '.tmp';
const out = createWriteStream(tmp);
const write = async (s) => { if (!out.write(s)) await new Promise((r) => out.once('drain', r)); };

const rl = createInterface({ input: createReadStream(dataFile), crlfDelay: Infinity });
let total = 0, filled = 0;
const t0 = Date.now();

for await (const line of rl) {
  if (!line) continue;
  total++;
  if (line.includes('"v":')) { await write(line + '\n'); continue; } // already has v
  const rec = JSON.parse(line);
  if (typeof rec.fen === 'string') {
    const v = Math.round(chooseMoveDetailed(parseFen(rec.fen), depth, Math.random, Infinity, true, [], 'nn').score);
    rec.v = v; filled++;
    await write(JSON.stringify(rec) + '\n');
    if (filled % 200 === 0) process.stdout.write(`\r  filled ${filled} | ${total} scanned`);
  } else {
    await write(line + '\n'); // no fen and no v (shouldn't occur post-migration) -> leave as-is
  }
}
await new Promise((r) => out.end(r));
if (existsSync(dataFile)) rmSync(dataFile);
renameSync(tmp, dataFile);
process.stdout.write('\n');
console.log(`Done: filled v on ${filled} positions (${total} total) in ${((Date.now() - t0) / 1000).toFixed(0)}s. `
  + `Re-featurize next:  npm run train:featurize`);
