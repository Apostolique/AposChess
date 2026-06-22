// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2019-2026 Jean-David Moisan
//
// Turn the raw self-play dataset (positions + outcomes) into per-net training inputs.
// The generator (apos-gen) stores only the raw position — net-agnostic:
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
// INCREMENTAL: the raw dataset is append-only in the common case (the generator
// appends whole games), so the meta sidecar records how many raw bytes were
// processed, a hash of the tail of that prefix, and the output size. When the
// current raw file still starts with that exact prefix (size grew, tail hash
// matches) and the output is exactly as we left it, only the appended tail is
// featurized and appended — turning the per-cycle cost from "whole dataset" into
// "new games only". Anything else (refresh-v / dedup-cap rewrites, a feature-set
// change via num_features, an interrupted append) falls back to the full pass.
//
// Usage (run from web/):
//   npm run train:featurize
// Options:
//   --in=FILE    raw dataset to read   (default ../training/data/selfplay.jsonl)
//   --out=FILE   features to write      (default ../training/data/selfplay.features.jsonl)
//   --full       force a full rebuild (skip the incremental fast path)

import {
  createReadStream, createWriteStream, existsSync, rmSync, renameSync, statSync, writeFileSync,
  readFileSync, openSync, readSync, closeSync,
} from 'node:fs';
import { createHash } from 'node:crypto';
import { createInterface } from 'node:readline';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { parseFen } from '../src/board.js';
import { featureIndices, PIECE_SQUARE_FEATURES, NUM_FEATURES } from '../src/nn.js';
import { fmtDur, fmtNum, fmtMB, liveStatus, everyMs } from './fmt.mjs';

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

const tmp = outFile + '.tmp';
// The sidecar carries num_features for train.py plus the incremental state. Keep
// it next to the data with a matching name: <data>.meta.json (strip .jsonl, add
// .meta.json).
const metaFile = outFile.replace(/\.jsonl$/, '') + '.meta.json';

// Hash the tail (up to 64 KB) of the first `prefixBytes` of `file` — enough to
// detect an in-place rewrite (refresh-v / dedup-cap touch lines everywhere, and
// the file's own growth moves the boundary) without reading the whole prefix.
const TAIL = 64 * 1024;
function tailHash(file, prefixBytes) {
  const len = Math.min(TAIL, prefixBytes);
  const buf = Buffer.alloc(len);
  const fd = openSync(file, 'r');
  try {
    readSync(fd, buf, 0, len, prefixBytes - len);
  } finally {
    closeSync(fd);
  }
  return createHash('sha1').update(buf).digest('hex');
}

// Snapshot the raw size up front and process exactly that prefix, so the recorded
// state stays correct even if a generator appends while we run.
const rawSize = statSync(inFile).size;

// Try the incremental fast path: same feature layout, the raw file still begins
// with the prefix we already processed, and the output is exactly as we left it.
let meta = null;
try { meta = JSON.parse(readFileSync(metaFile, 'utf8')); } catch { /* no sidecar yet */ }
const inc = !args.full && meta && meta.num_features === NUM_FEATURES && meta.incremental
  && meta.incremental.rawBytes <= rawSize
  && existsSync(outFile) && statSync(outFile).size === meta.incremental.outBytes
  && tailHash(inFile, meta.incremental.rawBytes) === meta.incremental.rawTailHash;
const startAt = inc ? meta.incremental.rawBytes : 0;

if (inc && startAt === rawSize) {
  console.log(`Featurized data is up to date (${fmtMB(rawSize)} raw, ${fmtMB(statSync(outFile).size)} out).`);
  process.exit(0);
}
console.log(inc
  ? `Featurizing ${inFile} incrementally (${fmtMB(rawSize - startAt)} new of ${fmtMB(rawSize)})`
  : `Featurizing ${inFile} (${fmtMB(rawSize)}, full pass)`);
console.log(`  -> ${outFile}`);

// Incremental appends straight to the output (an interrupted append leaves a size
// mismatch, which forces a clean full rebuild next run); a full pass goes through
// a temp file that replaces the target only after completing.
const out = inc
  ? createWriteStream(outFile, { flags: 'a' })
  : createWriteStream(tmp);
const rl = createInterface({
  input: createReadStream(inFile, { start: startAt, end: Math.max(startAt, rawSize - 1) }),
  crlfDelay: Infinity,
});

// Live progress: lines are ASCII (FEN + JSON), so line lengths track the byte
// offset closely enough for a percentage and ETA over the prefix being processed.
const status = liveStatus();
const tick = everyMs(500);
const t0 = Date.now();
const span = rawSize - startAt;
let bytesDone = 0;

let fromFen = 0, fromFeatures = 0;
for await (const line of rl) {
  if (!line) continue;
  bytesDone += line.length + 1;
  const n = fromFen + fromFeatures;
  if (tick() && n) {
    const el = (Date.now() - t0) / 1000;
    status.update(`  ${fmtNum(n)} positions | ${(100 * bytesDone / span).toFixed(1)}% | `
      + `${fmtNum(n / el)}/s | ETA ${fmtDur((span - bytesDone) / (bytesDone / el))}`);
  }
  const rec = JSON.parse(line);
  const { board, turn } = typeof rec.fen === 'string'
    ? (fromFen++, parseFen(rec.fen))
    : (fromFeatures++, boardFromFeatures(rec.f));
  const f = featureIndices(board, turn);
  const o = { f, r: rec.r, g: rec.g };
  if (rec.v != null) o.v = rec.v; // search value, for TD/bootstrap targets (train.py --lambda)
  if (!out.write(JSON.stringify(o) + '\n')) {
    await new Promise((res) => out.once('drain', res)); // respect backpressure on a big file
  }
}
await new Promise((res) => out.end(res));

if (!inc) {
  // Replace the target only now that the full pass succeeded (Windows rename can't
  // overwrite, so remove first).
  if (existsSync(outFile)) rmSync(outFile);
  renameSync(tmp, outFile);
}

// Stamp the input size (nn.js NUM_FEATURES) into the sidecar so train.py uses the
// SAME value — no hand-syncing a constant in two languages — plus the incremental
// state for the next run.
writeFileSync(metaFile, JSON.stringify({
  num_features: NUM_FEATURES,
  incremental: {
    rawBytes: rawSize,
    rawTailHash: tailHash(inFile, rawSize),
    outBytes: statSync(outFile).size,
  },
}, null, 2) + '\n');

status.clear();
console.log(`Done: ${fmtNum(fromFen + fromFeatures)} positions featurized${inc ? ' (incremental)' : ''} `
  + `in ${fmtDur((Date.now() - t0) / 1000)} (${fmtMB(statSync(outFile).size)} total, num_features=${NUM_FEATURES}).`);
if (fromFeatures) {
  console.log(`  ${fmtNum(fromFeatures)} legacy records had no fen — rebuilt from stored features.`);
}
