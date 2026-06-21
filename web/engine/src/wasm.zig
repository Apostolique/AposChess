// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2019-2026 Jean-David Moisan
//
// wasm32 entry for the browser bot: the SAME engine core (board/engine/ai/nn) the native
// tools use, exported over linear memory. The app's Web Worker (src/aiWorker.js) drives
// it — a drop-in for the old ai.js search with full parity:
//   * nn eval — JS writes the weights JSON into wasm memory and calls loadWeights.
//   * movetime — ai.zig reads the clock from the imported env.aposNowMs (no std.Io here).
//   * live eval bar — the search calls env.aposProgress(score, depth) after each depth.
//   * ponder — the predicted opponent reply rides back as a side output.
//   * repetition + opening variety — prev-position hashes and excluded root-move keys are
//     passed in as little-endian byte buffers JS fills.
// The JS side reconstructs the full variant move object (castle/jump/promotion flags) by
// matching from/to against its own legalMoves, so only from/to/promo need to cross over.

const std = @import("std");
const board = @import("board.zig");
const engine = @import("engine.zig");
const ai = @import("ai.zig");
const nn = @import("nn.zig");

const gpa = std.heap.wasm_allocator;

// JS host imports (provided in the Worker's WebAssembly.instantiate `env`). aposNowMs is
// referenced inside ai.zig (the movetime clock); aposProgress is streamed from the search.
extern "env" fn aposProgress(score: i32, depth: u32) void;
fn progressCb(score: i32, depth: u32) callconv(.c) void {
    aposProgress(score, depth);
}

var searcher: ?ai.Searcher = null;
var net: ?nn.Net = null;
var eval_kind: ai.EvalKind = .handcrafted;

// Side outputs of the last search()/ponderSearch(), read via the getters below.
var out_score: i32 = 0;
var out_reached: u32 = 0;
var out_ponder: u32 = 0xFFFF; // packed (from << 8) | to, 0xFFFF = none
var out_promo: u32 = 0; // 0 = none, else @intFromEnum(Role) + 1

// Scratch for the per-search inputs decoded from JS byte buffers (cap is far above any
// real repetition window / candidate-exclude list).
var hbuf: [1024]u64 = undefined;
var ebuf: [256]i32 = undefined;

fn ensureSearcher() void {
    if (searcher != null) return;
    const np: ?*const nn.Net = if (eval_kind == .nn and net != null) &net.? else null;
    searcher = ai.Searcher.init(gpa, null, eval_kind, np, 1) catch null;
    if (searcher) |*s| s.on_progress = &progressCb;
}

fn rebuildSearcher() void {
    if (searcher) |*s| {
        s.deinit();
        searcher = null;
    }
    ensureSearcher();
}

/// Allocate `n` bytes in wasm memory and return the byte offset, so JS can write a FEN,
/// the weights JSON, or the hash/exclude buffers into it before calling a search.
export fn allocBytes(n: usize) usize {
    const buf = gpa.alloc(u8, n) catch return 0;
    return @intFromPtr(buf.ptr);
}

/// Select the evaluation: 0 = handcrafted, 1 = nn. Rebuilds the searcher so the TT is
/// eval-namespaced correctly and the right net is bound.
export fn setEval(kind: u32) void {
    eval_kind = if (kind == 1) .nn else .handcrafted;
    rebuildSearcher();
}

/// Parse the weights JSON in [ptr, ptr+len) and make it the active net. Returns 1 on
/// success, 0 on parse/load failure (the eval then stays on its material fallback).
export fn loadWeights(ptr: [*]const u8, len: usize) u32 {
    const parsed = std.json.parseFromSlice(std.json.Value, gpa, ptr[0..len], .{}) catch return 0;
    defer parsed.deinit();
    const n = nn.load(gpa, parsed.value) catch return 0;
    net = n; // (a previously-loaded net's arrays leak; nets switch rarely in the UI)
    rebuildSearcher();
    return 1;
}

fn decodeHashes(hash_ptr: [*]const u8, hash_count: usize) []const u64 {
    const n = @min(hash_count, hbuf.len);
    var i: usize = 0;
    while (i < n) : (i += 1) hbuf[i] = std.mem.readInt(u64, hash_ptr[i * 8 ..][0..8], .little);
    return hbuf[0..n];
}

fn decodeExcl(ex_ptr: [*]const u8, ex_count: usize) []const i32 {
    const n = @min(ex_count, ebuf.len);
    var i: usize = 0;
    while (i < n) : (i += 1) ebuf[i] = std.mem.readInt(i32, ex_ptr[i * 4 ..][0..4], .little);
    return ebuf[0..n];
}

fn run(fen: []const u8, depth: u32, max_ms: u32, prev: []const u64, excl: []const i32) ai.Result {
    ensureSearcher();
    var s = &searcher.?;
    const st = board.parseFen(fen);
    const res = s.chooseMoveExcl(&st, depth, @intCast(max_ms), prev, excl);
    out_score = res.score;
    out_reached = res.depth;
    out_ponder = if (res.ponder) |pm| (@as(u32, pm.from) << 8) | pm.to else 0xFFFF;
    return res;
}

/// Search the position to `depth` (or until `max_ms` elapses if > 0). `hash_*` is the
/// repetition window (little-endian u64s); `ex_*` the excluded root-move keys (LE i32s).
/// Returns the best move packed as (from << 8) | to, or 0xFFFF if there is none; the
/// score, reached depth, ponder reply, and promotion role are read via the getters.
export fn search(
    fen_ptr: [*]const u8,
    fen_len: usize,
    depth: u32,
    max_ms: u32,
    hash_ptr: [*]const u8,
    hash_count: usize,
    ex_ptr: [*]const u8,
    ex_count: usize,
) u32 {
    const prev = decodeHashes(hash_ptr, hash_count);
    const excl = decodeExcl(ex_ptr, ex_count);
    const res = run(fen_ptr[0..fen_len], depth, max_ms, prev, excl);
    const m = res.move orelse {
        out_promo = 0;
        return 0xFFFF;
    };
    out_promo = if (m.promotion) |r| @as(u32, @intFromEnum(r)) + 1 else 0;
    return (@as(u32, m.from) << 8) | m.to;
}

/// Ponder: think on the position to warm the TT; returns the deepest completed depth.
/// (The move is irrelevant — the value is the warmed table + the refined eval-bar score.)
export fn ponderSearch(
    fen_ptr: [*]const u8,
    fen_len: usize,
    depth: u32,
    max_ms: u32,
    hash_ptr: [*]const u8,
    hash_count: usize,
) u32 {
    const prev = decodeHashes(hash_ptr, hash_count);
    _ = run(fen_ptr[0..fen_len], depth, max_ms, prev, &.{});
    return out_reached;
}

/// Invalidate the searcher's transposition table (the puzzle miner resets it between
/// positions so a prior search can't leak in). No-op before the first search.
export fn resetTT() void {
    if (searcher) |*s| s.clearTT();
}

export fn lastScore() i32 {
    return out_score;
}
export fn lastReached() u32 {
    return out_reached;
}
export fn lastPonder() u32 {
    return out_ponder;
}
export fn lastPromo() u32 {
    return out_promo;
}
