// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2019-2026 Jean-David Moisan
//
// Dataset maintenance: cap how many times any single TRAINING INPUT appears, to undo
// the heavy duplication of common positions (the start position alone occurs once per
// game — tens of thousands of times — so the net sees the opening orders of magnitude
// more than any midgame position). This de-biases the training distribution and
// shrinks the file (faster cycles) without losing the outcome signal.
//
// Two records are "the same input" iff they have the same canonical feature set `f`
// (what the net actually sees — independent of castling / move counters / which colour
// is to move, since features are side-to-move canonical). The SAME position carries
// DIFFERENT results `r` across games, and averaging those is how the net learns its
// expected value — so we don't dedup to one copy. Instead we keep each copy with
// probability cap/count (Bernoulli thinning): an over-represented input is thinned to
// ~cap copies that are a uniform random sample of all its occurrences, preserving the
// win/draw/loss ratio (and `v` distribution); inputs at or under the cap are untouched.
//
// Two streaming passes: pass 1 counts occurrences per input (a hashed key, so the map
// stays small); pass 2 emits each record with probability min(1, cap/count). Seeded
// for reproducibility. --dry-run reports the reduction without writing.
//
// Usage (run from web/):
//   node scripts/dedup-cap.mjs [--cap=N] [--in=FILE] [--out=FILE] [--seed=S] [--dry-run]
// Defaults: cap 64, in/out = ../training/data/selfplay.jsonl (atomic replace), seed 1.

import { createReadStream, createWriteStream, existsSync, rmSync, renameSync, statSync } from 'node:fs';
import { createInterface } from 'node:readline';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

import { parseFen } from '../src/board.js';
import { featureIndices } from '../src/nn.js';
import { fmtDur, fmtNum, fmtMB, liveStatus, everyMs } from './fmt.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const args = Object.fromEntries(process.argv.slice(2).map((a) => {
  const m = a.replace(/^--/, '').split('='); return [m[0], m.length > 1 ? m[1] : true];
}));
const cap = args.cap !== undefined ? Number(args.cap) : 64;
const seed = args.seed !== undefined ? Number(args.seed) : 1;
const dryRun = !!args['dry-run'];
const inFile = typeof args.in === 'string'
  ? resolve(process.cwd(), args.in) : resolve(here, '../../training/data/selfplay.jsonl');
const outFile = typeof args.out === 'string'
  ? resolve(process.cwd(), args.out) : resolve(here, '../../training/data/selfplay.jsonl');

if (!existsSync(inFile)) { console.error(`No dataset at ${inFile}`); process.exit(1); }

// 64-bit key (two 32-bit lanes -> short string) over the sorted feature indices, so the
// count map holds compact keys instead of full feature lists. Collisions across a few
// million keys in 2^64 are negligible.
function keyOf(fen) {
  const { board, turn } = parseFen(fen);
  const f = featureIndices(board, turn).sort((a, b) => a - b);
  let h = 0x811c9dc5 >>> 0, g = 0x9e3779b1 >>> 0;
  for (const x of f) {
    h = Math.imul(h ^ x, 0x01000193) >>> 0;
    g = Math.imul(g ^ (x + 0x7f4a7c15), 0x85ebca6b) >>> 0;
  }
  return h.toString(36) + ':' + g.toString(36);
}

function mulberry32(a) {
  a >>>= 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

console.log(`dedup-cap: ${inFile} (${fmtMB(statSync(inFile).size)}) | cap ${cap} | seed ${seed}${dryRun ? ' | DRY RUN' : ''}`);

const t0 = Date.now();
const status = liveStatus();
let tick = everyMs(500);

// --- pass 1: count occurrences per input -----------------------------------------
const counts = new Map();
let total = 0;
{
  const rl = createInterface({ input: createReadStream(inFile), crlfDelay: Infinity });
  for await (const line of rl) {
    if (!line) continue;
    total++;
    if (tick()) status.update(`  pass 1/2: counting — ${fmtNum(total)} positions read`);
    const rec = JSON.parse(line);
    if (typeof rec.fen !== 'string') continue; // shouldn't happen post-migration
    const k = keyOf(rec.fen);
    counts.set(k, (counts.get(k) || 0) + 1);
  }
}
status.clear();
let unique = counts.size, over = 0, maxCount = 0, kept = 0;
for (const c of counts.values()) {
  if (c > maxCount) maxCount = c;
  if (c > cap) { over++; kept += cap; } else kept += c;
}
console.log(`  ${fmtNum(total)} positions | ${fmtNum(unique)} unique inputs | `
  + `${fmtNum(over)} over cap (most-duplicated: ${fmtNum(maxCount)}x)`);
console.log(`  expected after cap: ~${fmtNum(kept)} positions (${(100 * (1 - kept / total)).toFixed(1)}% removed)`);

if (dryRun) { console.log('  (dry run — nothing written)'); process.exit(0); }

// --- pass 2: Bernoulli-thin over-cap inputs --------------------------------------
const tmp = outFile + '.tmp';
const out = createWriteStream(tmp);
const write = async (s) => { if (!out.write(s)) await new Promise((r) => out.once('drain', r)); };
const rng = mulberry32(seed);
let written = 0;
tick = everyMs(500);
{
  const rl = createInterface({ input: createReadStream(inFile), crlfDelay: Infinity });
  let scanned = 0;
  for await (const line of rl) {
    if (!line) continue;
    scanned++;
    if (tick()) status.update(`  pass 2/2: thinning — ${fmtNum(written)} kept of ${fmtNum(scanned)} read`);
    const rec = JSON.parse(line);
    const c = typeof rec.fen === 'string' ? counts.get(keyOf(rec.fen)) : 1;
    if (c <= cap || rng() < cap / c) { await write(line + '\n'); written++; }
  }
}
await new Promise((r) => out.end(r));
status.clear();
if (existsSync(outFile)) rmSync(outFile);
renameSync(tmp, outFile);
console.log(`Done: ${fmtNum(written)} positions kept (${(100 * (1 - written / total)).toFixed(1)}% removed) `
  + `in ${fmtDur((Date.now() - t0) / 1000)} -> ${outFile} (${fmtMB(statSync(outFile).size)}).`);
console.log('Re-featurize next:  npm run train:featurize');
