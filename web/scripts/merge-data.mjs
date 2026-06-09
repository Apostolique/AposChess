// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2019-2026 Jean-David Moisan
//
// Combine every *.jsonl file in the training data folder into a single dataset and
// delete the leftovers. Handy when pooling self-play data generated on several
// machines (or in several runs): drop all the files into training/data/ and run
// this to fold them into one selfplay.jsonl.
//
// JSONL records are independent and order doesn't matter (the trainer shuffles),
// so merging is just concatenation. Files are *streamed* (never loaded whole) so
// it scales to large datasets. The originals are removed only after the merged
// file is fully written, so an interrupted run never loses data.
//
// Usage (run from web/):
//   npm run train:merge
// Options:
//   --dir=PATH   data folder (default ../training/data, relative to web/)
//   --out=NAME   merged filename within that folder (default selfplay.jsonl)

import {
  createReadStream, createWriteStream, readdirSync, rmSync, renameSync, statSync,
} from 'node:fs';
import { dirname, resolve, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const args = Object.fromEntries(
  process.argv.slice(2).map((a) => {
    const m = a.replace(/^--/, '').split('=');
    return [m[0], m.length > 1 ? m[1] : true];
  }),
);

const dataDir = typeof args.dir === 'string'
  ? resolve(process.cwd(), args.dir)
  : resolve(here, '../../training/data');
const outName = typeof args.out === 'string' ? args.out : 'selfplay.jsonl';
const target = join(dataDir, outName);
const tmp = join(dataDir, '_merged.jsonl.tmp'); // .tmp, so it isn't itself a *.jsonl input

const mb = (bytes) => (bytes / 1e6).toFixed(1) + ' MB';

let entries;
try {
  entries = readdirSync(dataDir);
} catch {
  console.log(`No data directory at ${dataDir}. Nothing to merge.`);
  process.exit(0);
}

// Merge only the RAW position files. `*.features.jsonl` are derived (per-net inputs
// from featurize.mjs) and regenerable, so folding them into the raw dataset would
// duplicate/poison it — skip them.
const files = entries.filter((f) => f.endsWith('.jsonl') && !f.endsWith('.features.jsonl'))
  .map((f) => join(dataDir, f)).sort();
if (files.length === 0) {
  console.log(`No .jsonl files in ${dataDir}. Nothing to merge.`);
  process.exit(0);
}
if (files.length === 1 && files[0] === target) {
  console.log(`Only ${outName} present — nothing to merge.`);
  process.exit(0);
}

console.log(`Merging ${files.length} file(s) into ${outName}:`);
for (const f of files) console.log(`  - ${f.slice(dataDir.length + 1)} (${mb(statSync(f).size)})`);

const out = createWriteStream(tmp);
let lines = 0;

function appendFile(path) {
  return new Promise((res, rej) => {
    let lastByte = 0x0a; // treat an empty file as already newline-terminated
    const rs = createReadStream(path);
    rs.on('data', (chunk) => {
      for (let i = 0; i < chunk.length; i++) if (chunk[i] === 0x0a) lines++;
      if (chunk.length) lastByte = chunk[chunk.length - 1];
    });
    rs.on('error', rej);
    rs.on('end', () => {
      // Guard the join: if a source file didn't end in a newline, add one so its
      // last record doesn't glue onto the next file's first record (and count that
      // final, otherwise-untallied record).
      if (lastByte !== 0x0a) { out.write('\n'); lines++; }
      res();
    });
    rs.pipe(out, { end: false });
  });
}

for (const f of files) await appendFile(f);
await new Promise((res) => out.end(res));

// Replace the sources with the merged file. Delete first (Windows rename can't
// overwrite), then move the temp into place — the merge is already safely on disk.
for (const f of files) rmSync(f);
renameSync(tmp, target);

console.log(`\nDone: ${lines} lines in ${outName} (${mb(statSync(target).size)}). `
  + `Removed ${files.length} source file(s).`);
