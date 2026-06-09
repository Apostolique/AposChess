// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2019-2026 Jean-David Moisan
//
// Turn the raw self-play dataset (positions + outcomes) into per-net training inputs.
// The generator (gen-selfplay.mjs) stores only the raw position — net-agnostic:
// { fen, r, g }. This reads each position and applies the CURRENT nn.js
// featureIndices, writing the training-ready { f, r, g } that train.py consumes. So
// the position is the source of truth and features are a derived, regenerable
// artifact: changing featureIndices is just a re-run of this script, never a
// regeneration of self-play, and the trainer still needs no chess logic.
//
// The position comes from `fen` (preferred — carries castling etc.), or, for legacy
// pre-FEN records, is reconstructed from the stored `f`: features are CANONICAL
// side-to-move, so each index decodes to (role, us/them, square) and rebuilds a board
// that is feature-equivalent (verified to round-trip). That fallback can't recover
// state outside the canonical board (castling/move counters), so any feature needing
// those requires FEN-bearing (freshly generated) data.
//
// Streamed line by line (never loaded whole), written to a temp file that replaces
// the target only after a complete pass, so an interrupted run can't corrupt it.
//
// Usage (run from web/):
//   npm run train:featurize
// Options:
//   --in=FILE    raw dataset to read   (default ../training/data/selfplay.jsonl)
//   --out=FILE   features to write      (default ../training/data/selfplay.features.jsonl)

import {
  createReadStream, createWriteStream, existsSync, rmSync, renameSync, statSync, writeFileSync,
} from 'node:fs';
import { createInterface } from 'node:readline';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { parseFen } from '../src/board.js';
import { featureIndices, PIECE_SQUARE_FEATURES, NUM_FEATURES } from '../src/nn.js';

// Rebuild a canonical board from stored feature indices (the legacy fallback when a
// record has no `fen`). The plain piece-square block (indices < PIECE_SQUARE_FEATURES)
// fully determines the board; any later blocks (e.g. king-relative) just duplicate
// those pieces, so we decode the plain block alone — robust to either layout.
// idx = (role*2 + side)*64 + sq, side 0 = us, 1 = them; we relabel us=white and treat
// the position as white-to-move, which is feature-equivalent for canonical features.
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
const outFile = typeof args.out === 'string'
  ? resolve(process.cwd(), args.out)
  : resolve(here, '../../training/data/selfplay.features.jsonl');

if (!existsSync(inFile)) {
  console.error(`No dataset at ${inFile}. Generate it first:  npm run train:gen`);
  process.exit(1);
}

const mb = (bytes) => (bytes / 1e6).toFixed(1) + ' MB';
const tmp = outFile + '.tmp';

console.log(`Featurizing ${inFile} (${mb(statSync(inFile).size)})`);
console.log(`  -> ${outFile}`);

const out = createWriteStream(tmp);
const rl = createInterface({ input: createReadStream(inFile), crlfDelay: Infinity });

let fromFen = 0, fromFeatures = 0;
for await (const line of rl) {
  if (!line) continue;
  const rec = JSON.parse(line);
  const { board, turn } = typeof rec.fen === 'string'
    ? (fromFen++, parseFen(rec.fen))
    : (fromFeatures++, boardFromFeatures(rec.f));
  const f = featureIndices(board, turn);
  if (!out.write(JSON.stringify({ f, r: rec.r, g: rec.g }) + '\n')) {
    await new Promise((res) => out.once('drain', res)); // respect backpressure on a big file
  }
}
await new Promise((res) => out.end(res));

// Replace the target only now that the full pass succeeded (Windows rename can't
// overwrite, so remove first).
if (existsSync(outFile)) rmSync(outFile);
renameSync(tmp, outFile);

// Stamp the input size (nn.js NUM_FEATURES) into a sidecar so train.py uses the
// SAME value — no hand-syncing a constant in two languages. Keep it next to the
// data with a matching name: <data>.meta.json (strip .jsonl, add .meta.json).
const metaFile = outFile.replace(/\.jsonl$/, '') + '.meta.json';
writeFileSync(metaFile, JSON.stringify({ num_features: NUM_FEATURES }, null, 2) + '\n');

const src = fromFen && fromFeatures ? `${fromFen} from fen, ${fromFeatures} from legacy features`
  : fromFeatures ? 'from legacy features (no fen)' : 'from fen';
console.log(`\nDone: ${fromFen + fromFeatures} positions featurized (${mb(statSync(outFile).size)}) — ${src}.`);
console.log(`Wrote ${metaFile} (num_features=${NUM_FEATURES}).`);
