// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2019-2026 Jean-David Moisan
//
// Runs the AI search off the main thread so deeper lookahead never freezes the UI. The
// search itself runs in the Zig engine compiled to wasm (`web/engine`, exported as
// `apos.wasm`). The JS engine (`board.js`/`engine.js`) is still used here for the variant
// move objects and Zobrist hashing; `ai.js` is a fallback if the wasm fails to load.
// Two request kinds, both tagged with a `seq` so a reply for a superseded
// position is discarded by the page:
//   { type:'search', seq, state, depth, maxMs, engine, net, posHistory, exclude, wasmUrl }
//       → { type:'search', seq, move, ponder, score }  — a real move to play; `ponder` is
//       the predicted opponent reply { from, to }; `score` is the root value (cp, stm-rel).
//       While searching it streams { type:'progress', seq, score, depth } after each
//       completed iterative-deepening depth so the eval bar climbs live.
//   { type:'ponder', seq, state, depth, maxMs, engine, net, posHistory, wasmUrl }
//       → { type:'ponder', seq, reached, score }  — thinking on the opponent's turn to warm
//       the transposition table; `reached` is the deepest completed iteration.
// `engine` picks the evaluation ('handcrafted' | 'nn'); for 'nn', `net` is the full URL of
// the selected weights file. `wasmUrl` is the full URL of `apos.wasm` (the page resolves it
// against the document base — a worker's own relative URL would resolve under assets/).
//
// The wasm searcher's transposition table is NOT cleared between messages, so a real search
// reuses what pondering found (and vice versa); switching eval or net rebuilds it (a fresh
// table). Hard resets (new game / stop) recreate the worker for a fresh table.

import { parseFen, toFen } from './board.js';
import { legalMoves } from './engine.js';
import { chooseMoveDetailed, _internal } from './ai.js'; // fallback only
const hashOf = _internal.hashOf; // matches the Zig zobrist (parity-verified)

// Zig board.Role enum order — promo index from the wasm maps back to the JS role letter.
const ROLE_CHARS = ['p', 'n', 'b', 'r', 'q', 'k'];

// --- wasm engine ------------------------------------------------------------------
let wasm = null; // the instance exports, once loaded
let wasmReady = null; // in-flight load promise (single-flight)
let wasmFailed = false;
let curSeq = 0; // tags the progress messages streamed from inside a search
let scratch = null; // reused wasm buffers (fen / hashes / exclude), so a long game can't leak
let curEvalKind = -1; // 0 = handcrafted, 1 = nn (what the wasm searcher is currently set to)
let curNetUrl = null;
const netCache = new Map(); // url -> Promise<ArrayBuffer>

function loadWasm(url) {
  if (wasm || wasmFailed) return Promise.resolve(wasm);
  if (wasmReady) return wasmReady;
  const env = {
    aposNowMs: () => performance.now(),
    // Streamed synchronously from inside the wasm search after each completed depth.
    aposProgress: (score, depth) => self.postMessage({ type: 'progress', seq: curSeq, score, depth }),
  };
  wasmReady = (async () => {
    try {
      const resp = await fetch(url);
      let inst;
      try {
        ({ instance: inst } = await WebAssembly.instantiateStreaming(resp.clone(), { env }));
      } catch {
        // Streaming needs an application/wasm MIME; fall back to a plain buffer compile.
        const bytes = await resp.arrayBuffer();
        ({ instance: inst } = await WebAssembly.instantiate(bytes, { env }));
      }
      wasm = inst.exports;
    } catch {
      wasmFailed = true; // the JS fallback takes over
    }
    return wasm;
  })();
  return wasmReady;
}

// Fixed scratch buffers, allocated once and reused (recreating the byte views each call,
// since growing wasm memory detaches old views). Caps are far above any real input.
function ensureScratch() {
  if (!scratch) scratch = { fen: wasm.allocBytes(128), hash: wasm.allocBytes(1024 * 8), excl: wasm.allocBytes(256 * 4) };
}
function writeFen(fen) {
  const bytes = new TextEncoder().encode(fen);
  new Uint8Array(wasm.memory.buffer, scratch.fen, bytes.length).set(bytes);
  return bytes.length;
}
function writeHashes(hashes) {
  const n = Math.min(hashes.length, 1024);
  const dv = new DataView(wasm.memory.buffer, scratch.hash, n * 8);
  for (let i = 0; i < n; i++) dv.setBigUint64(i * 8, BigInt.asUintN(64, hashes[i]), true);
  return n;
}
function writeExcl(keys) {
  const n = Math.min(keys.length, 256);
  const dv = new DataView(wasm.memory.buffer, scratch.excl, n * 4);
  for (let i = 0; i < n; i++) dv.setInt32(i * 4, keys[i], true);
  return n;
}

// Bring the wasm searcher to the requested eval + net. setEval / loadWeights each rebuild
// the searcher (a fresh TT), so they're called only on an actual change — matching the JS
// path's reset-TT-on-net-switch and otherwise persistent table.
async function ensureEvalNet(engine, netUrl) {
  const kind = engine === 'nn' ? 1 : 0;
  if (kind !== curEvalKind) { wasm.setEval(kind); curEvalKind = kind; curNetUrl = null; }
  if (kind === 1 && netUrl && netUrl !== curNetUrl) {
    let p = netCache.get(netUrl);
    if (!p) { p = fetch(netUrl).then((r) => r.arrayBuffer()); netCache.set(netUrl, p); }
    let buf;
    try { buf = new Uint8Array(await p); }
    catch { return; } // keep the material fallback if the fetch fails
    const ptr = wasm.allocBytes(buf.length);
    new Uint8Array(wasm.memory.buffer, ptr, buf.length).set(buf);
    if (wasm.loadWeights(ptr, buf.length)) curNetUrl = netUrl;
  }
}

// A finite ms budget for the wasm clock; 0 means "depth-only" (no deadline).
const msBudget = (maxMs) => (Number.isFinite(maxMs) ? Math.max(0, maxMs | 0) : 0);
// A finite, positive depth for the wasm u32 param. "No depth limit" — the time-only
// "Custom, depth 0" mode, which main.js sends as Infinity (and which 0 also means, per
// the native tools' depth==0 sentinel) — maps to the 99-ply cap, NOT 0. Passing Infinity
// straight to the wasm i32 arg coerces to 0, and a literal 0 is 0 too; either way that
// makes depth_cap 0 and the search returns its first legal move instantly (the timeout
// never gets a chance to run). Anything <= 0 or non-finite therefore means "unlimited".
const depthBudget = (depth) => (Number.isFinite(depth) && depth > 0 ? depth | 0 : 99);

// Reconstruct the full variant move object (castle/jump/promotion flags) from the wasm's
// from/to/promo by matching the JS engine's own legal moves for this position.
function decodeMove(state, packed) {
  if (packed === 0xffff) return null;
  const from = (packed >> 8) & 0xff;
  const to = packed & 0xff;
  const promoIdx = wasm.lastPromo();
  const promo = promoIdx ? ROLE_CHARS[promoIdx - 1] : null;
  const legal = legalMoves(state);
  return (
    legal.find((m) => m.from === from && m.to === to && (promo ? m.promotion === promo : !m.promotion)) ||
    legal.find((m) => m.from === from && m.to === to) ||
    null
  );
}

// --- JS fallback (only if the wasm can't load) ------------------------------------
function fallback(data) {
  const { type, seq, state, depth, maxMs, posHistory, engine, exclude } = data;
  const prevHashes = posHistory ? posHistory.map((f) => hashOf(parseFen(f))) : [];
  if (type === 'ponder') {
    const { depth: reached, score } = chooseMoveDetailed(state, depth, Math.random, maxMs, true, prevHashes, engine);
    self.postMessage({ type: 'ponder', seq, reached, score });
    return;
  }
  const excludeKeys = exclude && exclude.length ? new Set(exclude) : null;
  const onProgress = (score, d) => self.postMessage({ type: 'progress', seq, score, depth: d });
  const { move, ponder, score } = chooseMoveDetailed(state, depth, Math.random, maxMs, true, prevHashes, engine, excludeKeys, onProgress);
  self.postMessage({ type: 'search', seq, move, ponder, score });
}

self.onmessage = async ({ data }) => {
  const { type, seq, state, depth, maxMs, posHistory, engine, net, exclude, wasmUrl } = data;
  curSeq = seq;
  const x = wasmUrl ? await loadWasm(wasmUrl) : null;
  if (!x) { fallback(data); return; }

  await ensureEvalNet(engine, net);
  ensureScratch();
  const fenLen = writeFen(toFen(state));
  // posHistory is the live game's prior positions as FENs; the search wants Zobrist hashes.
  const prevHashes = posHistory ? posHistory.map((f) => hashOf(parseFen(f))) : [];
  const hashCount = writeHashes(prevHashes);
  const ms = msBudget(maxMs);
  const d = depthBudget(depth);

  if (type === 'ponder') {
    const reached = x.ponderSearch(scratch.fen, fenLen, d, ms, scratch.hash, hashCount);
    self.postMessage({ type: 'ponder', seq, reached, score: x.lastScore() });
    return;
  }

  const exCount = writeExcl(exclude && exclude.length ? exclude : []);
  const packed = x.search(scratch.fen, fenLen, d, ms, scratch.hash, hashCount, scratch.excl, exCount);
  const move = decodeMove(state, packed);
  const pp = x.lastPonder();
  const ponder = pp === 0xffff ? null : { from: (pp >> 8) & 0xff, to: pp & 0xff };
  self.postMessage({ type: 'search', seq, move, ponder, score: x.lastScore() });
};
