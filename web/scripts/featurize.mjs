// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2019-2026 Jean-David Moisan
//
// Turn the raw self-play dataset into per-net training inputs.
//
// The dataset is GAME-PRIMARY (scripts/gameRecord.mjs): one JSONL line per game, holding
// the move list + per-position v/vs. This script REPLAYS each game into its positions and
// applies the CURRENT nn.js featureIndices, writing the training-ready { f, r, g } (plus v)
// that train.py consumes. So the position is the source of truth and features are a derived,
// regenerable artifact: changing featureIndices is just a re-run of this script, never a
// regeneration of self-play, and the trainer still needs no chess logic.
//
// It also reads LEGACY position-primary records ({fen,r,g,v} or pre-fen {f,...}) so the
// pre-migration dataset and the migration's loss-free check still work — a game record is
// recognized by its `moves` array; anything else is treated as a single position.
//
// Streamed line by line (never loaded whole), written to a temp file that replaces the
// target only after a complete pass, so an interrupted run can't corrupt it.
//
// INCREMENTAL: the raw dataset is append-only in the common case (the generator appends
// whole games), so the meta sidecar records how many raw bytes were processed, a hash of
// the tail of that prefix, and the output size. When the current raw file still starts with
// that exact prefix (size grew, tail hash matches) and the output is exactly as we left it,
// only the appended tail is featurized and appended. Anything else (refresh-v rewrites, a
// feature-set/quiet/cap change, an interrupted append) falls back to the full pass.
//
// --cap=N folds in the old dedup-cap: count-cap how many times any single training input
// (canonical feature set) appears, to undo the heavy duplication of common positions (the
// start position occurs once per game). It needs a global count, so it forces a full
// TWO-pass run (count, then Bernoulli-thin to ~cap copies preserving the win/draw/loss
// ratio) and disables the incremental fast path; it's recorded in the meta sidecar, so
// changing it forces a full re-featurize. Omit it (or --cap=0) for the uncapped default.
//
// Usage (run from web/):
//   npm run train:featurize
// Options:
//   --in=FILE    raw dataset to read   (default ../training/data/selfplay.jsonl)
//   --out=FILE   features to write      (default ../training/data/selfplay.features.jsonl)
//   --full       force a full rebuild (skip the incremental fast path)
//   --cap=N      count-cap each canonical input to ~N copies (full two-pass; default off)
//   --seed=S     RNG seed for the cap thinning (default 1)
//   --quiet-only drop tactically loud positions (side to move in check, or a winning
//                capture available) — NNUE is queried only on quiet positions at qsearch
//                leaves, so training on loud ones mismatches that distribution + adds noise.
//                Recorded in the meta sidecar; toggling it forces a full re-featurize.

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
import { generatePseudoMoves, kingAttacked } from '../src/engine.js';
import { expandPositions, isGameRecord } from './gameRecord.mjs';
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

// --- Quiet-position filter (--quiet-only) ----------------------------------
// NNUE is a STATIC eval, called only at the leaves of quiescence search — which
// resolves captures/checks before evaluating — so at runtime the net is only ever
// queried on relatively quiet positions. Training on "loud" positions both mismatches
// that distribution and injects label noise. With no stored best move, we approximate
// "best move is a capture" statically: a position is LOUD if the side to move is in
// check, or it has a capture of an enemy piece worth strictly more than the capturer.
// Variant-calibrated values (knight ≈ rook), kept in lockstep with ai.js / nn.js VALUE.
const VALUE = { p: 100, n: 500, b: 330, r: 500, q: 900, k: 0 };
function isQuiet(board, turn) {
  if (kingAttacked(board, turn)) return false; // side to move in check
  for (const m of generatePseudoMoves(board, turn)) {
    const victim = board[m.to];
    if (victim && victim.color !== turn && VALUE[victim.role] > VALUE[board[m.from].role]) {
      return false; // a winning capture is available — position is tactically loud
    }
  }
  return true;
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
const quietOnly = !!args['quiet-only'];
const cap = args.cap !== undefined ? Number(args.cap) : 0; // 0 = uncapped
const seed = args.seed !== undefined ? Number(args.seed) : 1;

if (!existsSync(inFile)) {
  console.error(`No dataset at ${inFile}. Generate it first:  npm run train:gen`);
  process.exit(1);
}

const tmp = outFile + '.tmp';
const metaFile = outFile.replace(/\.jsonl$/, '') + '.meta.json';

// Expand one raw line into its training positions: a game record replays into many
// positions; a legacy line is a single position (fen-bearing or pre-fen features). Yields
// { board, turn, r, v, g } — exactly what featureIndices / the quiet filter / train.py need.
function* positionsOf(rec) {
  if (isGameRecord(rec)) {
    for (const p of expandPositions(rec)) {
      yield { board: p.state.board, turn: p.state.turn, r: p.r, v: p.v, g: rec.g };
    }
  } else {
    const { board, turn } = typeof rec.fen === 'string' ? parseFen(rec.fen) : boardFromFeatures(rec.f);
    yield { board, turn, r: rec.r, v: rec.v, g: rec.g };
  }
}

// 64-bit key over the sorted feature indices (for --cap counting), so the count map holds
// compact keys instead of full feature lists. Collisions across a few million keys are
// negligible. (Same hashing the old dedup-cap used.)
function keyOfF(f) {
  const s = f.slice().sort((a, b) => a - b);
  let h = 0x811c9dc5 >>> 0, g = 0x9e3779b1 >>> 0;
  for (const x of s) {
    h = Math.imul(h ^ x, 0x01000193) >>> 0;
    g = Math.imul(g ^ (x + 0x7f4a7c15), 0x85ebca6b) >>> 0;
  }
  return h.toString(36) + ':' + g.toString(36);
}
function mulberry32(a) {
  a >>>= 0;
  return () => { a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

// Build the output record for a featurized position.
function featurize(p) {
  const f = featureIndices(p.board, p.turn);
  const o = { f, r: p.r, g: p.g };
  if (p.v != null) o.v = p.v; // search value, for TD/bootstrap targets (train.py --lambda)
  return o;
}

const writeMeta = (outBytes, rawBytes, rawTailHash) => writeFileSync(metaFile, JSON.stringify({
  num_features: NUM_FEATURES,
  quiet_only: quietOnly,
  cap,
  incremental: { rawBytes, rawTailHash, outBytes },
}, null, 2) + '\n');

// Hash the tail (up to 64 KB) of the first `prefixBytes` of `file` — enough to detect an
// in-place rewrite without reading the whole prefix.
const TAIL = 64 * 1024;
function tailHash(file, prefixBytes) {
  const len = Math.min(TAIL, prefixBytes);
  const buf = Buffer.alloc(len);
  const fd = openSync(file, 'r');
  try { readSync(fd, buf, 0, len, prefixBytes - len); } finally { closeSync(fd); }
  return createHash('sha1').update(buf).digest('hex');
}

const rawSize = statSync(inFile).size;
const t0 = Date.now();
const status = liveStatus();

// =========================================================================
// --cap: full two-pass (count canonical inputs, then Bernoulli-thin). Disables the
// incremental fast path — a global count can't be maintained incrementally.
// =========================================================================
if (cap > 0) {
  console.log(`Featurizing ${inFile} (${fmtMB(rawSize)}, full pass, cap ${cap})`);
  console.log(`  -> ${outFile}`);
  const counts = new Map();
  let total = 0, games = 0;
  // pass 1: count occurrences of each canonical input (after the quiet filter, so counts
  // match what pass 2 writes).
  {
    const tick = everyMs(500);
    const rl = createInterface({ input: createReadStream(inFile), crlfDelay: Infinity });
    for await (const line of rl) {
      if (!line) continue;
      const rec = JSON.parse(line); games++;
      for (const p of positionsOf(rec)) {
        if (quietOnly && !isQuiet(p.board, p.turn)) continue;
        const k = keyOfF(featureIndices(p.board, p.turn));
        counts.set(k, (counts.get(k) || 0) + 1);
        total++;
      }
      if (tick()) status.update(`  pass 1/2: counting — ${fmtNum(total)} positions in ${fmtNum(games)} games`);
    }
  }
  status.clear();
  // pass 2: write each kept position with probability min(1, cap/count).
  const out = createWriteStream(tmp);
  const rng = mulberry32(seed);
  let written = 0, scanned = 0;
  {
    const tick = everyMs(500);
    const rl = createInterface({ input: createReadStream(inFile), crlfDelay: Infinity });
    for await (const line of rl) {
      if (!line) continue;
      const rec = JSON.parse(line);
      for (const p of positionsOf(rec)) {
        if (quietOnly && !isQuiet(p.board, p.turn)) continue;
        const o = featurize(p);
        scanned++;
        const c = counts.get(keyOfF(o.f));
        if (c <= cap || rng() < cap / c) {
          if (!out.write(JSON.stringify(o) + '\n')) await new Promise((res) => out.once('drain', res));
          written++;
        }
      }
      if (tick()) status.update(`  pass 2/2: thinning — ${fmtNum(written)} kept of ${fmtNum(scanned)}`);
    }
  }
  await new Promise((res) => out.end(res));
  if (existsSync(outFile)) rmSync(outFile);
  renameSync(tmp, outFile);
  writeMeta(statSync(outFile).size, rawSize, tailHash(inFile, rawSize));
  status.clear();
  console.log(`Done: ${fmtNum(written)} positions kept of ${fmtNum(scanned)} (${(100 * (1 - written / Math.max(1, scanned))).toFixed(1)}% capped) `
    + `in ${fmtDur((Date.now() - t0) / 1000)} -> ${outFile} (${fmtMB(statSync(outFile).size)}, num_features=${NUM_FEATURES}).`);
  process.exit(0);
}

// =========================================================================
// Default path: incremental when possible, else a full single pass.
// =========================================================================
let meta = null;
try { meta = JSON.parse(readFileSync(metaFile, 'utf8')); } catch { /* no sidecar yet */ }
const inc = !args.full && meta && meta.num_features === NUM_FEATURES
  && !!meta.quiet_only === quietOnly       // a filter toggle changes output content -> full pass
  && (meta.cap || 0) === 0                  // last run was capped -> can't append; full pass
  && meta.incremental
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

const out = inc ? createWriteStream(outFile, { flags: 'a' }) : createWriteStream(tmp);
const rl = createInterface({
  input: createReadStream(inFile, { start: startAt, end: Math.max(startAt, rawSize - 1) }),
  crlfDelay: Infinity,
});

const tick = everyMs(500);
const span = rawSize - startAt;
let bytesDone = 0;
let games = 0, positions = 0, skippedLoud = 0, fromFeatures = 0;
for await (const line of rl) {
  if (!line) continue;
  bytesDone += line.length + 1;
  if (tick() && positions) {
    const el = (Date.now() - t0) / 1000;
    status.update(`  ${fmtNum(positions)} positions | ${(100 * bytesDone / span).toFixed(1)}% | `
      + `${fmtNum(positions / el)}/s | ETA ${fmtDur((span - bytesDone) / (bytesDone / el))}`);
  }
  const rec = JSON.parse(line);
  if (!isGameRecord(rec) && typeof rec.fen !== 'string') fromFeatures++;
  games++;
  for (const p of positionsOf(rec)) {
    if (quietOnly && !isQuiet(p.board, p.turn)) { skippedLoud++; continue; }
    if (!out.write(JSON.stringify(featurize(p)) + '\n')) {
      await new Promise((res) => out.once('drain', res)); // respect backpressure on a big file
    }
    positions++;
  }
}
await new Promise((res) => out.end(res));

if (!inc) {
  if (existsSync(outFile)) rmSync(outFile);
  renameSync(tmp, outFile);
}

writeMeta(statSync(outFile).size, rawSize, tailHash(inFile, rawSize));

status.clear();
console.log(`Done: ${fmtNum(positions)} positions featurized${inc ? ' (incremental)' : ''} from ${fmtNum(games)} record(s) `
  + `in ${fmtDur((Date.now() - t0) / 1000)} (${fmtMB(statSync(outFile).size)} total, num_features=${NUM_FEATURES}).`);
if (quietOnly) {
  const scanned = positions + skippedLoud;
  console.log(`  --quiet-only: dropped ${fmtNum(skippedLoud)} loud positions `
    + `(${scanned ? (100 * skippedLoud / scanned).toFixed(1) : '0.0'}% of ${fmtNum(scanned)} scanned).`);
}
if (fromFeatures) {
  console.log(`  ${fmtNum(fromFeatures)} legacy records had no fen — rebuilt from stored features.`);
}
