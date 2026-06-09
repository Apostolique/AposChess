// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2019-2026 Jean-David Moisan
//
// Recompute the input features of an existing dataset in place, using the CURRENT
// nn.js feature definition. When you change featureIndices (add/remove/reorder
// features), the cached `f` arrays in the JSONL go stale — but the positions
// themselves don't. This rewrites `f` and leaves the label (`r`) and game id (`g`)
// untouched, so a feature-set change becomes a quick pass over the file instead of
// regenerating hours of self-play. The trainer still needs no chess logic — it
// keeps reading `f` directly.
//
// Two sources for the position, in order of fidelity:
//   1. `fen` (written by genWorker.mjs) — the full position, so even a feature that
//      depends on castling rights / move counters refeaturizes correctly.
//   2. the existing `f` itself — because features are stored in CANONICAL
//      side-to-move orientation, each index decodes to (role, us/them, square) and
//      rebuilds a board that is feature-equivalent to the original (verified to
//      round-trip exactly). This covers any feature that is a canonical function of
//      board + turn (king-relative, mobility, …) and lets pre-FEN datasets be
//      refeaturized without regeneration. It can NOT recover state outside the
//      canonical board (castling rights, etc.); use a FEN-bearing dataset for that.
//
// The file is streamed line by line (never loaded whole) and written to a temp
// file that replaces the original only after a complete, successful pass, so an
// interrupted run never corrupts the dataset.
//
// Usage (run from web/):
//   npm run train:refeaturize
// Options:
//   --in=FILE    dataset to read  (default ../training/data/selfplay.jsonl)
//   --out=FILE   dataset to write (default: same as --in, i.e. in place)

import {
  createReadStream, createWriteStream, existsSync, rmSync, renameSync, statSync,
} from 'node:fs';
import { createInterface } from 'node:readline';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { parseFen } from '../src/board.js';
import { featureIndices, PIECE_SQUARE_FEATURES } from '../src/nn.js';

// Rebuild a canonical board from stored feature indices (the fallback when a record
// has no `fen`). The plain piece-square block (indices < PIECE_SQUARE_FEATURES)
// fully determines the board; any later blocks (e.g. king-relative) just duplicate
// those pieces, so we decode the plain block alone — which also makes this robust
// whether the input `f` is in the old plain-only layout or a newer multi-block one.
// idx = (role*2 + side)*64 + sq, side 0 = us, 1 = them; we relabel us=white and treat
// the position as white-to-move, which is feature-equivalent for any canonical
// feature function (see header).
const ROLES = ['p', 'n', 'b', 'r', 'q', 'k'];
function boardFromFeatures(f) {
  const board = new Array(64).fill(null);
  for (const idx of f) {
    if (idx >= PIECE_SQUARE_FEATURES) continue; // skip non-plain blocks (redundant)
    const sq = idx % 64;
    const s = (idx - sq) / 64;
    const side = s % 2;
    board[sq] = { role: ROLES[(s - side) / 2], color: side === 0 ? 'white' : 'black' };
  }
  return { board, turn: 'white' };
}

const here = dirname(fileURLToPath(import.meta.url));
const args = Object.fromEntries(
  process.argv.slice(2).map((a) => {
    const m = a.replace(/^--/, '').split('=');
    return [m[0], m.length > 1 ? m[1] : true];
  }),
);

const inFile = typeof args.in === 'string'
  ? resolve(process.cwd(), args.in)
  : resolve(here, '../../training/data/selfplay.jsonl');
const outFile = typeof args.out === 'string' ? resolve(process.cwd(), args.out) : inFile;

if (!existsSync(inFile)) {
  console.error(`No dataset at ${inFile}. Generate it first:  npm run train:gen`);
  process.exit(1);
}

const mb = (bytes) => (bytes / 1e6).toFixed(1) + ' MB';
const tmp = outFile + '.tmp';

console.log(`Refeaturizing ${inFile} (${mb(statSync(inFile).size)})`);
console.log(`  -> ${outFile}${outFile === inFile ? ' (in place)' : ''}`);

const out = createWriteStream(tmp);
const rl = createInterface({ input: createReadStream(inFile), crlfDelay: Infinity });

let fromFen = 0, fromFeatures = 0;
for await (const line of rl) {
  if (!line) continue;
  const rec = JSON.parse(line);
  const { board, turn } = typeof rec.fen === 'string'
    ? (fromFen++, parseFen(rec.fen))
    : (fromFeatures++, boardFromFeatures(rec.f));
  rec.f = featureIndices(board, turn);
  // Keep field order stable with the generator (fen omitted when absent).
  const o = rec.fen !== undefined
    ? { f: rec.f, r: rec.r, g: rec.g, fen: rec.fen }
    : { f: rec.f, r: rec.r, g: rec.g };
  if (!out.write(JSON.stringify(o) + '\n')) {
    await new Promise((res) => out.once('drain', res)); // respect backpressure on a big file
  }
}
await new Promise((res) => out.end(res));

// Replace the target only now that the full pass succeeded (Windows rename can't
// overwrite, so remove first).
if (existsSync(outFile)) rmSync(outFile);
renameSync(tmp, outFile);

console.log(`\nDone: ${fromFen + fromFeatures} records rewritten (${mb(statSync(outFile).size)})`
  + ` — ${fromFen} from fen, ${fromFeatures} reconstructed from features.`);
