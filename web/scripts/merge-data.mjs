// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2019-2026 Jean-David Moisan
//
// Combine every *.jsonl file in the training data folder into a single dataset and
// delete the leftovers. Handy when pooling self-play data generated on several
// machines (or in several runs): drop all the files into training/data/ and run
// this to fold them into one selfplay.jsonl.
//
// SMART merge (not a blind concatenation). The dataset is GAME-PRIMARY (one line per game;
// scripts/gameRecord.mjs). When you share a base dataset across two computers, each may
// refresh `v` on a different slice and generate fresh games, then you want to fold the two
// copies back together. A naive cat would DUPLICATE every shared game. Instead we:
//   - identify each record by its game id `g`, so the two copies of a shared game collapse
//     to ONE record;
//   - reconcile their per-position `v` PLY BY PLY, keeping each position's BEST `v`
//     provenance — a real value beats a missing one, a stronger engine's label beats a
//     weaker one (by the `npm run rank:pool` Elo ledger, read from `vs` tags), and at equal
//     strength a deeper search wins. So if machine A refreshed plies 10-20 and machine B
//     plies 30-40 of the same game, the merge keeps both improvements;
//   - keep every game not already present, so NEW games from either machine are all added.
// The game id `g` is "<seed36>-<index>", deterministic from the generator seed, so the
// same `g` really is the same game (same seed+index => identical play, hence identical
// moves/players/result — only `v`/`vs` can differ); different machines get genuinely-new
// games by using different --seed values.
//
// Inputs are assumed to be CLEAN per-game datasets (each game listed once) — which is what
// gen, refresh-v, the gate/rank harvests, and this tool all emit.
//
// The rank:pool game archive (loop/ladder-games.jsonl) is folded in as a READ-ONLY extra
// input: its games join the dataset (deduped by game id), but the file is kept on disk, not
// deleted — it stays the ladder's own regenerable record (and rank:pool's --corpus subtracts
// its pool store before folding, so those games are never double-counted in the ratings).
//
// Records are keyed by game in memory (far fewer than positions, since each game is one
// entry); the npm script raises the V8 heap ceiling for the large datasets this runs on.
// The originals are removed only after the merged file is fully written, so an interrupted
// run never loses data.
//
// `*.features.jsonl` are derived (per-net inputs from featurize.mjs) and regenerable, so
// they're skipped — folding them into the raw dataset would poison it.
//
// Usage (run from web/):
//   npm run train:merge
// Options:
//   --dir=PATH      data folder (default ../training/data, relative to web/)
//   --out=NAME      merged filename within that folder (default selfplay.jsonl)
//   --ledger[=FILE] strength ledger from `npm run rank:pool` used to rank `v` provenance
//                   (default training/data/loop/engine-elo.ladder.json; auto-used if present)
//   --no-ledger     ignore the ledger; rank provenance by has-v then search depth only
//   --drop-unlabeled  purge games whose engines are unknown (no `players`) — the pre-refactor
//                   games, plus any legacy position-primary / unkeyed / unparseable line. Shrinks
//                   the dataset but leaves every survivor with known provenance. Rewrites even a
//                   lone selfplay.jsonl (so it works as a standalone in-place purge).

import {
  createReadStream, createWriteStream, readdirSync, rmSync, renameSync, statSync,
  existsSync, readFileSync,
} from 'node:fs';
import { createInterface } from 'node:readline';
import { dirname, resolve, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { isGameRecord, vsAt, setVsAt, normalizeVs, serializeGameRecord } from './gameRecord.mjs';

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

// --drop-unlabeled: purge games whose ENGINES are unknown — the pre-refactor games, whose old
// position-primary format recorded who LABELED each position (`vs`) but not who PLAYED
// (`players`). Without players a game can't inform the corpus rating or be quality-controlled by
// engine, so this drops it (and any legacy position-primary / unkeyed / unparseable line). It
// shrinks the dataset now, but every surviving game has known provenance. The material floor
// (players like "nn6@?") IS labeled and is kept; only a missing/literal-'?' player is unlabeled.
const dropUnlabeled = !!args['drop-unlabeled'];
const labeledPlayer = (p) => !!p && p !== '?';
const hasPlayers = (rec) => rec.players != null && labeledPlayer(rec.players.w) && labeledPlayer(rec.players.b);

const mb = (bytes) => (bytes / 1e6).toFixed(1) + ' MB';

// --- provenance ranking (the "best v" rule) --------------------------------------
// A `vs` tag is "<engine><depth>@<version>" (see vtag.mjs): e.g. "nn6@a3f2c1", "hc6@2".
// We score a position's value by (engine Elo, search depth); a position with no `v` scores
// lowest of all. The Elo comes from the strength ledger written by `npm run rank:pool`, which
// holds one entry PER engine×depth — so we look the exact tag up first (its true strength
// at that depth) and fall back to the engine's BEST per-version Elo for a depth that was
// never ranked (a deeper-than-ranked search is at least as strong). An untagged / unknown
// label counts as weakest (-Inf), like refresh-v's weakest-first cohort logic. Without a
// ledger every Elo is -Inf, so the rule degrades to "has-v, then deeper wins".
const parseTag = (tag) => {
  const m = /^(nn|hc)(\d+|t)@(.+)$/.exec(tag || '');
  return m ? { eng: m[1], depth: m[2], version: m[3] } : null;
};
// "elo<N>" version = a self-described non-promoted gate candidate; its Elo is in the tag.
const ephemeralElo = (version) => { const m = /^elo(-?\d+)$/.exec(version || ''); return m ? Number(m[1]) : null; };

const ledgerPath = args.ledger === true || args.ledger === undefined
  ? resolve(here, '../../training/data/loop/engine-elo.ladder.json')
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
    console.error(`No ledger at ${ledgerPath}. Run 'npm run rank:pool' or pass --no-ledger.`);
    process.exit(1);
  } else {
    console.log('No strength ledger found — ranking v by has-v then search depth only.');
  }
}

// Quality of a single position's `v`, as a comparable [elo, depth] pair (higher is better).
// A missing value is the worst possible. Used to choose between two copies of a position.
const NONE = { elo: -Infinity, depth: -Infinity };
function plyQuality(v, vsTag) {
  if (v === undefined || v === null) return NONE;
  const t = parseTag(vsTag);
  if (!t) return NONE;
  const depth = t.depth === 't' ? 0 : Number(t.depth);
  // A self-described ephemeral candidate ("nn6@elo37") carries its strength in the tag.
  const eph = ephemeralElo(t.version);
  if (eph !== null) return { elo: eph, depth };
  // Exact engine×depth Elo if that depth was ranked; otherwise the engine's best per-version
  // Elo (unranked depth). Unknown engine -> -Inf (weakest), same as a missing value.
  const elo = eloByTag.has(vsTag) ? eloByTag.get(vsTag)
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

// Merge only the RAW game files; *.features.jsonl are derived and regenerable.
const files = entries.filter((f) => f.endsWith('.jsonl') && !f.endsWith('.features.jsonl'))
  .map((f) => join(dataDir, f)).sort();

// Read-only extra inputs: the rank:pool harvest (loop/ladder-games.jsonl) is the ladder's
// self-contained game archive (every game the strength pool has played, in lockstep with its
// store). We FOLD it into the dataset — deduped by game id like any source — but never DELETE
// it, so it stays the regenerable ladder archive. Ingested LAST, so existing dataset labels are
// the tie incumbents and a ladder position only upgrades a label when it's strictly stronger by
// the ledger. (The dataset and the pool store can hold the same ladder games, but rank:pool's
// --corpus subtracts the store before folding, so ranking never double-counts them.)
const readonlyInputs = [join(dataDir, 'loop', 'ladder-games.jsonl')].filter(existsSync);

if (files.length === 0 && readonlyInputs.length === 0) {
  console.log(`No .jsonl files in ${dataDir}. Nothing to merge.`);
  process.exit(0);
}
// Only the target and nothing to fold in — a no-op, UNLESS --drop-unlabeled, which rewrites
// even a lone selfplay.jsonl to purge the unlabeled games.
if (!dropUnlabeled && readonlyInputs.length === 0 && files.length === 1 && files[0] === target) {
  console.log(`Only ${outName} present — nothing to merge.`);
  process.exit(0);
}

const allInputs = [...files, ...readonlyInputs];
console.log(`Merging ${allInputs.length} file(s) into ${outName}`
  + `${readonlyInputs.length ? ` (${readonlyInputs.length} read-only, kept on disk)` : ''}:`);
for (const f of files) console.log(`  - ${f.slice(dataDir.length + 1)} (${mb(statSync(f).size)})`);
for (const f of readonlyInputs) console.log(`  - ${f.slice(dataDir.length + 1)} (${mb(statSync(f).size)}, read-only)`);

// --- build the merged set --------------------------------------------------------
// Keyed by game id `g`. best.get(g) = the winning record object; on a repeat we reconcile
// its per-position v/vs ply by ply (same g => identical moves/players/result). Map iteration
// order is insertion order, so output = first-seen order (sorted-first file's games first,
// then games unique to later files) — which doesn't matter to the (shuffling) trainer.
// Records with no game id (or unparseable lines) can't be de-duplicated, so each is kept
// verbatim.
const best = new Map();
let totalLines = 0, games = 0, collapsed = 0, upgrades = 0, malformed = 0, noGame = 0, dropped = 0;
const verbatim = []; // non-game lines kept as-is

// Reconcile `src` into `dst` (same game): adopt each position's v/vs when it's better.
function mergeInto(dst, src) {
  if (!Array.isArray(dst.v) || !Array.isArray(src.v)) return;
  const n = Math.min(dst.v.length, src.v.length);
  for (let i = 0; i < n; i++) {
    const dq = plyQuality(dst.v[i], vsAt(dst, i));
    const sq = plyQuality(src.v[i], vsAt(src, i));
    if (better(sq, dq)) { dst.v[i] = src.v[i]; setVsAt(dst, i, vsAt(src, i)); upgrades++; }
  }
  normalizeVs(dst);
}

async function ingest(path) {
  const rl = createInterface({ input: createReadStream(path), crlfDelay: Infinity });
  for await (const line of rl) {
    if (!line) continue;
    totalLines++;
    let rec;
    try { rec = JSON.parse(line); }
    catch { if (dropUnlabeled) { dropped++; continue; } verbatim.push(line); malformed++; continue; }
    if (!isGameRecord(rec) || rec.g == null) {
      if (dropUnlabeled) { dropped++; continue; } // legacy position-primary / unkeyed line
      verbatim.push(line); noGame++; continue;
    }
    if (dropUnlabeled && !hasPlayers(rec)) { dropped++; continue; } // pre-refactor: engines unknown
    const cur = best.get(rec.g);
    if (!cur) { best.set(rec.g, rec); games++; continue; }
    collapsed++;            // a duplicate of an already-seen game
    mergeInto(cur, rec);    // reconcile per-position v/vs
  }
}

for (const f of allInputs) await ingest(f);

// --- write out, then atomically replace the sources ------------------------------
const out = createWriteStream(tmp);
const write = async (s) => { if (!out.write(s)) await new Promise((res) => out.once('drain', res)); };
for (const rec of best.values()) await write(serializeGameRecord(rec) + '\n');
for (const line of verbatim) await write(line + '\n');
await new Promise((res) => out.end(res));

// Delete the (deletable) sources first (Windows rename can't overwrite), then move the temp
// into place — the merge is already safely on disk. Read-only inputs (the ladder archive) are
// NOT removed: they were folded in but stay on disk as their own regenerable record.
for (const f of files) rmSync(f);
renameSync(tmp, target);

console.log(`\nDone: ${games} unique game(s) in ${outName} (${mb(statSync(target).size)}).`);
console.log(`  ${totalLines} input line(s) -> ${collapsed} duplicate game(s) collapsed`
  + `${upgrades ? `, ${upgrades} position(s) kept a better v` : ''}.`);
if (dropUnlabeled) console.log(`  Dropped ${dropped} unlabeled record(s) (no engines / pre-refactor).`);
if (noGame || malformed) {
  console.log(`  Passed through unchanged: ${noGame} record(s) without a game id`
    + `${malformed ? `, ${malformed} unparseable line(s)` : ''}.`);
}
console.log(`Removed ${files.length} source file(s)`
  + `${readonlyInputs.length ? `; kept ${readonlyInputs.length} read-only archive(s)` : ''}. `
  + `Re-featurize next: npm run train:featurize`);
