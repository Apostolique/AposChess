// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2019-2026 Jean-David Moisan
//
// Combine every *.jsonl file in the training data folder into a single dataset and
// delete the leftovers. Handy when pooling self-play data generated on several
// machines (or in several runs): drop all the files into training/data/ and run
// this to fold them into one selfplay.jsonl.
//
// SMART merge (not a blind concatenation). When you share a base dataset across two
// computers, each may refresh `v` on a different slice and generate fresh games, then
// you want to fold the two copies back together. A naive cat would DUPLICATE every
// shared position and randomly keep one machine's (possibly staler) `v`. Instead we:
//   - identify each record by its game + ply  (g + position-within-the-game), so the
//     same position in two copies of the same game collapses to ONE record;
//   - among the copies, keep the one with the BEST `v` provenance — a real value beats
//     a missing one, a stronger engine's label beats a weaker one (by the `npm run rank`
//     Elo ledger, read from `vs` tags), and at equal strength a deeper search wins;
//   - keep every record whose game/ply isn't already present, so NEW games from either
//     machine are all added.
// The game id `g` is "<seed36>-<index>", deterministic from the generator seed, so the
// same `g` really is the same game (same seed+index => identical play); different
// machines get genuinely-new games by using different --seed values.
//
// Inputs are assumed to be CLEAN per-game datasets (each game listed once, in ply order)
// — which is what gen-selfplay, refresh-v, the gate harvest, and this tool itself all
// emit. (A file that already contains a game twice non-contiguously — e.g. an artifact of
// the old concatenating merge — can't be de-duplicated within itself, only across files.)
//
// Records are keyed in memory, so peak memory is ~the number of UNIQUE positions across
// all inputs (the shared base collapses); the npm script raises the V8 heap ceiling for
// the multi-hundred-MB datasets this is used on. The originals are removed only after the
// merged file is fully written, so an interrupted run never loses data.
//
// `*.features.jsonl` are derived (per-net inputs from featurize.mjs) and regenerable, so
// they're skipped — folding them into the raw dataset would poison it.
//
// Usage (run from web/):
//   npm run train:merge
// Options:
//   --dir=PATH      data folder (default ../training/data, relative to web/)
//   --out=NAME      merged filename within that folder (default selfplay.jsonl)
//   --ledger[=FILE] strength ledger from `npm run rank` used to rank `v` provenance
//                   (default training/data/loop/engine-elo.json; auto-used if present)
//   --no-ledger     ignore the ledger; rank provenance by has-v then search depth only

import {
  createReadStream, createWriteStream, readdirSync, rmSync, renameSync, statSync,
  existsSync, readFileSync,
} from 'node:fs';
import { createInterface } from 'node:readline';
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

// --- provenance ranking (the "best v" rule) --------------------------------------
// A `vs` tag is "<engine><depth>@<version>" (see vtag.mjs): e.g. "nn6@a3f2c1", "hc6@2".
// We score a record's value by (engine Elo, search depth); a record with no `v` scores
// lowest of all. The Elo comes from the strength ledger written by `npm run rank`, which
// holds one entry PER engine×depth — so we look the record's exact tag up first (its true
// strength at that depth) and fall back to the engine's BEST per-version Elo for a depth
// that was never ranked (a deeper-than-ranked search is at least as strong). An untagged /
// unknown label counts as weakest (-Inf), like refresh-v's weakest-first cohort logic.
// Without a ledger every Elo is -Inf, so the rule degrades to "has-v, then deeper wins".
const parseTag = (tag) => {
  const m = /^(nn|hc)(\d+|t)@(.+)$/.exec(tag || '');
  return m ? { eng: m[1], depth: m[2], version: m[3] } : null;
};
// "elo<N>" version = a self-described non-promoted gate candidate; its Elo is in the tag.
const ephemeralElo = (version) => { const m = /^elo(-?\d+)$/.exec(version || ''); return m ? Number(m[1]) : null; };

const ledgerPath = args.ledger === true || args.ledger === undefined
  ? resolve(here, '../../training/data/loop/engine-elo.json')
  : (typeof args.ledger === 'string' ? resolve(process.cwd(), args.ledger) : null);
const ledgerExplicit = args.ledger === true || typeof args.ledger === 'string';
const useLedger = !args['no-ledger'];

let eloByTag = new Map();      // exact "<eng><depth>@<version>" tag -> Elo (the precise strength)
let eloByVersion = new Map();  // version -> best (max) Elo across its ranked depths (fallback)
if (useLedger && ledgerPath) {
  if (existsSync(ledgerPath)) {
    try {
      const ledger = JSON.parse(readFileSync(ledgerPath, 'utf8'));
      // The material fallback ('?') is excluded — its Elo is only an internal ledger stat,
      // so anything it (or an unrecoverable/untagged source) labeled is treated as weakest.
      for (const e of ledger.ranking || []) {
        if (e.elo == null || e.version === '?') continue;
        if (e.tag) eloByTag.set(e.tag, e.elo);
        // Per-version fallback for an unranked depth: the engine's BEST measured Elo, since a
        // deeper-than-ranked search is at least as strong as its deepest ranked one.
        const prev = eloByVersion.has(e.version) ? eloByVersion.get(e.version) : -Infinity;
        eloByVersion.set(e.version, Math.max(prev, e.elo));
      }
      console.log(`Ranking v by ledger ${ledgerPath.slice(dataDir.length + 1) || ledgerPath} `
        + `(${eloByTag.size} engine×depth entr${eloByTag.size === 1 ? 'y' : 'ies'}, ${eloByVersion.size} version(s)).`);
    } catch (e) {
      console.error(`Could not read ledger ${ledgerPath}: ${e.message}`);
      process.exit(1);
    }
  } else if (ledgerExplicit) {
    console.error(`No ledger at ${ledgerPath}. Run 'npm run rank' or pass --no-ledger.`);
    process.exit(1);
  } else {
    console.log('No strength ledger found — ranking v by has-v then search depth only.');
  }
}

// Quality of a record's `v`, as a comparable [elo, depth] pair (higher is better). A
// missing value is the worst possible. Used both to choose between two copies of a
// position and to count how often a refreshed value displaced a staler one.
const NONE = { elo: -Infinity, depth: -Infinity };
function vQuality(rec) {
  if (rec.v === undefined || rec.v === null) return NONE;
  const t = parseTag(rec.vs);
  if (!t) return NONE;
  const depth = t.depth === 't' ? 0 : Number(t.depth);
  // A self-described ephemeral candidate ("nn6@elo37") carries its strength in the tag.
  const eph = ephemeralElo(t.version);
  if (eph !== null) return { elo: eph, depth };
  // Exact engine×depth Elo if that depth was ranked; otherwise the engine's best per-version
  // Elo (unranked depth). Unknown engine -> -Inf (weakest), same as a missing value.
  const elo = eloByTag.has(rec.vs) ? eloByTag.get(rec.vs)
    : (eloByVersion.has(t.version) ? eloByVersion.get(t.version) : -Infinity);
  return { elo, depth };
}
// Strictly better => replace; equal or worse => keep the incumbent (so the first file in
// sorted order wins ties, and identical labels — same engine+version+depth => same v —
// never churn).
const better = (a, b) => a.elo > b.elo || (a.elo === b.elo && a.depth > b.depth);

let entries;
try {
  entries = readdirSync(dataDir);
} catch {
  console.log(`No data directory at ${dataDir}. Nothing to merge.`);
  process.exit(0);
}

// Merge only the RAW position files; *.features.jsonl are derived and regenerable.
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

// --- build the merged set --------------------------------------------------------
// key = "<game>:<ply>"  (ply = the record's 0-based position within its game IN THIS FILE).
// best.get(key) = { line, q }  where line is the winning record (newline-stripped) and q
// is its v-quality. Map iteration order is insertion order, so output = first-seen order
// (the sorted-first file's records first, then any games unique to later files appended) —
// games stay grouped and in ply order, which doesn't matter to the (shuffling) trainer but
// keeps the file readable. Records with no game id can't be de-duplicated, so each is kept
// under a unique synthetic key.
const best = new Map();
let totalLines = 0, kept = 0, collapsed = 0, upgrades = 0, malformed = 0, noGame = 0;

async function ingest(path) {
  const ply = new Map(); // game id -> next ply index, reset per file (each game listed once)
  const rl = createInterface({ input: createReadStream(path), crlfDelay: Infinity });
  for await (const line of rl) {
    if (!line) continue;
    totalLines++;
    let rec;
    try { rec = JSON.parse(line); } catch { best.set(` bad${malformed++}`, { line, q: NONE }); kept++; continue; }
    if (rec.g == null) { best.set(` nog${noGame++}`, { line, q: NONE }); kept++; continue; }
    const p = ply.get(rec.g) || 0;
    ply.set(rec.g, p + 1);
    const key = `${rec.g}:${p}`;
    const q = vQuality(rec);
    const cur = best.get(key);
    if (!cur) { best.set(key, { line, q }); kept++; continue; }
    collapsed++;                       // a duplicate of an already-seen position
    if (better(q, cur.q)) { cur.line = line; cur.q = q; upgrades++; } // adopt the better v
  }
}

for (const f of files) await ingest(f);

// --- write out, then atomically replace the sources ------------------------------
const out = createWriteStream(tmp);
for (const { line } of best.values()) {
  if (!out.write(line + '\n')) await new Promise((res) => out.once('drain', res));
}
await new Promise((res) => out.end(res));

// Delete the sources first (Windows rename can't overwrite), then move the temp into
// place — the merge is already safely on disk.
for (const f of files) rmSync(f);
renameSync(tmp, target);

console.log(`\nDone: ${kept} unique record(s) in ${outName} (${mb(statSync(target).size)}).`);
console.log(`  ${totalLines} input line(s) -> ${collapsed} duplicate position(s) collapsed`
  + `${upgrades ? `, ${upgrades} kept a better v` : ''}.`);
if (noGame || malformed) {
  console.log(`  Passed through unchanged: ${noGame} record(s) without a game id`
    + `${malformed ? `, ${malformed} unparseable line(s)` : ''}.`);
}
console.log(`Removed ${files.length} source file(s). Re-featurize next: npm run train:featurize`);
