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
//                  is automatically its own version. '?' = weights missing (material
//                  fallback), so v from a fallback eval is never mistaken for a real net.
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
  const eng = evalName && evalName.startsWith('nn') ? 'nn' : 'hc';
  const ver = eng === 'nn' ? nnVersion(weightsPath) : HC_VERSION;
  return `${eng}${depth != null ? depth : 't'}@${ver}`;
}
