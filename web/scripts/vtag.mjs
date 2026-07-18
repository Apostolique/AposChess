// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2019-2026 Jean-David Moisan
//
// Provenance tag for a computed search value `v`, stamped into the raw dataset as each
// record's `vs` field so the dataset stays auditable as `v` is rewritten over time
// (generation, refresh-v, gate harvest — each by a possibly-different, EVOLVING eval).
//
// Format: "<engine><depth>@<version>"   e.g. "nn6@a3f2c1", "hc6@2".
//   engine   'nn' | 'hc'
//   depth    the fixed search depth (or 't' for a time-based search)
//   version  handcrafted -> HC_VERSION (bumped by hand when the eval changes);
//            nn -> a short content hash of the weights file, so every distinct champion
//                  is automatically its own version. '?' = the bare material floor (hc<d>@?,
//                  no net), so v from the fallback eval is never mistaken for a real net.
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { HC_VERSION } from '../src/ai.js';

// Short content hash of a weights file = its nn version. Uncached/pure: the loop's
// champion file keeps ONE path but its content changes across promotions, so the loop
// archives by calling this fresh each time. '?' = missing (material fallback).
export function weightsHash(weightsPath) {
  if (!weightsPath) return '?';
  try { return createHash('sha1').update(readFileSync(weightsPath)).digest('hex').slice(0, 6); }
  catch { return '?'; }
}

// Cached wrapper for the per-position tagging hot path (short-lived gen/refresh
// workers, where a given weights file is constant for the run).
const hashCache = new Map(); // weightsPath -> short hash
function nnVersion(weightsPath) {
  if (!weightsPath) return '?';
  let h = hashCache.get(weightsPath);
  if (h === undefined) { h = weightsHash(weightsPath); hashCache.set(weightsPath, h); }
  return h;
}

export function vtag(evalName, depth, weightsPath) {
  // material = the bare piece-count floor: hc-family, no net, version '?' (mirrors main_match.zig
  // vtagFmt), so it's distinct from real handcrafted hc<d>@<HC_VERSION>.
  if (evalName === 'material') return `hc${depth != null ? depth : 't'}@?`;
  const eng = evalName && evalName.startsWith('nn') ? 'nn' : 'hc';
  const ver = eng === 'nn' ? nnVersion(weightsPath) : HC_VERSION;
  return `${eng}${depth != null ? depth : 't'}@${ver}`;
}

// Parse a "<engine><depth>@<version>" tag into its parts (null if not a valid tag).
export function parseVtag(tag) {
  const m = /^(nn|hc)(\d+|t)@(.+)$/.exec(tag || '');
  return m ? { eng: m[1], depth: m[2], version: m[3] } : null;
}

// PLAYER-strength resolver over a rank:pool ledger (engine-elo.ladder.json): returns a
// function tag -> Elo, or null when the ledger doesn't know the engine. This is for judging
// how strong a game's PLAYERS were (featurize --min-elo), which differs from refresh-v's
// label-side semantics in two ways: an unknown tag is null ("can't judge, keep the game"),
// not -Inf "relabel on sight"; and the material floor ('?' versions) IS resolved by exact
// tag — hc1@? etc. are real rated pool nodes, and games they played are exactly the weak
// games the filter exists to catch. Version-level fallback (max Elo across ranked depths)
// still excludes '?' — it would smear the floor across all hc depths.
export function ledgerEloResolver(ledgerPath) {
  const ledger = JSON.parse(readFileSync(ledgerPath, 'utf8')); // caller handles a missing file
  const byTag = new Map(), byVersion = new Map();
  for (const e of ledger.ranking || []) {
    if (e.elo == null) continue;
    if (e.tag) byTag.set(e.tag, e.elo);
    if (e.version === '?') continue;
    const prev = byVersion.has(e.version) ? byVersion.get(e.version) : -Infinity;
    byVersion.set(e.version, Math.max(prev, e.elo));
  }
  return (tag) => {
    const t = parseVtag(tag);
    if (!t) return null;
    const eph = ephemeralElo(t.version);
    if (eph !== null) return eph;
    if (byTag.has(tag)) return byTag.get(tag);
    return byVersion.has(t.version) ? byVersion.get(t.version) : null;
  };
}

// Ephemeral-engine version marker. A non-promoted gate candidate is never archived and
// can't be re-instantiated, so a content-hash version would be UNRECOVERABLE — refresh-v
// and merge-data would read it as -Inf "weakest, relabel on sight". Instead its `vs`
// version encodes the candidate's measured strength directly: "elo<N>", an integer absolute
// Elo on the SAME hc-anchored scale the rank ledger uses (pin hc6 := 1500). Consumers read the strength
// straight off the tag — no ledger lookup, no archived weights needed. The 'elo' prefix is
// collision-proof against a real nn version (a sha1 hex hash can't contain 'l' or 'o') and
// against the integer hc version.
//   ephemeralVersion(37) -> "elo37"   ephemeralElo("elo37") -> 37   ephemeralElo("a3f2c1") -> null
export function ephemeralVersion(elo) { return `elo${Math.round(elo)}`; }
export function ephemeralElo(version) {
  const m = /^elo(-?\d+)$/.exec(version || '');
  return m ? Number(m[1]) : null;
}
