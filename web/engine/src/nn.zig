// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2019-2026 Jean-David Moisan
//
// Neural-net evaluation — a direct port of web/src/nn.js (KING_BUCKETS = 0, the
// shipped plain 768 layout): sparse input layer → dense ReLU hidden layers →
// scalar head, tanh-squashed × scale → centipawns, already side-to-move-relative.
// All arithmetic is f64 in the SAME order as nn.js, so the pre-tanh value is
// bit-identical; only libm tanh may differ by a ULP, so the parity check allows
// ±1 cp. JS rounds with Math.round = floor(x + 0.5); we match that.

const std = @import("std");
const board = @import("board.zig");

const Color = board.Color;
const Role = board.Role;
const Piece = board.Piece;

// PIECE_INDEX in nn.js: p,n,b,r,q,k -> 0..5.
fn pieceIndex(role: Role) usize {
    return switch (role) {
        .p => 0,
        .n => 1,
        .b => 2,
        .r => 3,
        .q => 4,
        .k => 5,
    };
}

const Layer = struct {
    w: []f64, // input-major, flattened: w[i*out + o]
    b: []f64,
    out: usize,
};

// Integer (quantized) layer — see the integer pipeline in nn.js compileInt.
const LayerI = struct {
    w: []i64,
    b: []i64,
    out: usize,
};

pub const Net = struct {
    layers: []Layer = &.{}, // float net (is_int = false)
    ilayers: []LayerI = &.{}, // quantized net (is_int = true)
    scale: f64,
    is_int: bool = false,
    qa: i64 = 0, // activation scale (clipped ReLU upper bound)
    qw: i64 = 0, // dense-weight scale
};

fn toF64(v: std.json.Value) f64 {
    return switch (v) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        .number_string => |s| std.fmt.parseFloat(f64, s) catch 0,
        else => 0,
    };
}

fn toI64(v: std.json.Value) i64 {
    return switch (v) {
        .integer => |i| i,
        .float => |f| @intFromFloat(f),
        .number_string => |s| std.fmt.parseInt(i64, s, 10) catch 0,
        else => 0,
    };
}

/// Build a Net from a parsed weights object ({arch, scale, layers:[{w,b}]}). An
/// "int" net additionally carries {quant:{qa,qw}} and integer weights — see nn.js.
pub fn load(alloc: std.mem.Allocator, v: std.json.Value) !Net {
    const obj = v.object;
    const layers_j = obj.get("layers").?.array;
    const scale = toF64(obj.get("scale").?);

    if (obj.get("int")) |intv| {
        if (intv == .bool and intv.bool) {
            const q = obj.get("quant").?.object;
            const layers = try alloc.alloc(LayerI, layers_j.items.len);
            for (layers_j.items, 0..) |lj, li| {
                const wj = lj.object.get("w").?.array;
                const bj = lj.object.get("b").?.array;
                const w = try alloc.alloc(i64, wj.items.len);
                for (wj.items, 0..) |x, k| w[k] = toI64(x);
                const b = try alloc.alloc(i64, bj.items.len);
                for (bj.items, 0..) |x, k| b[k] = toI64(x);
                layers[li] = .{ .w = w, .b = b, .out = bj.items.len };
            }
            return .{
                .ilayers = layers,
                .scale = scale,
                .is_int = true,
                .qa = toI64(q.get("qa").?),
                .qw = toI64(q.get("qw").?),
            };
        }
    }

    const layers = try alloc.alloc(Layer, layers_j.items.len);
    for (layers_j.items, 0..) |lj, li| {
        const wj = lj.object.get("w").?.array;
        const bj = lj.object.get("b").?.array;
        const w = try alloc.alloc(f64, wj.items.len);
        for (wj.items, 0..) |x, k| w[k] = toF64(x);
        const b = try alloc.alloc(f64, bj.items.len);
        for (bj.items, 0..) |x, k| b[k] = toF64(x);
        layers[li] = .{ .w = w, .b = b, .out = bj.items.len };
    }
    return .{ .layers = layers, .scale = scale };
}

/// Centipawn evaluation from the side-to-move's perspective.
pub fn evaluate(net: *const Net, b: *const [64]?Piece, turn: Color) i32 {
    if (net.is_int) return evaluateInt(net, b, turn);
    const H0 = net.layers[0].out;
    var bufA: [1024]f64 = undefined;
    var bufB: [1024]f64 = undefined;

    // Input (sparse) layer: bias + active feature columns, ReLU'd.
    const L0 = net.layers[0];
    var h: usize = 0;
    while (h < H0) : (h += 1) bufA[h] = L0.b[h];
    const flip = turn == .black;
    var i: usize = 0;
    while (i < 64) : (i += 1) {
        const p = b[i] orelse continue;
        const side: usize = if (p.color == turn) 0 else 1; // 0 = us, 1 = them
        const sqx: usize = if (flip) i ^ 56 else i;
        const kind = (pieceIndex(p.role) * 2 + side) * 64 + sqx;
        const off = kind * H0;
        var hh: usize = 0;
        while (hh < H0) : (hh += 1) bufA[hh] += L0.w[off + hh];
    }
    {
        var hh: usize = 0;
        while (hh < H0) : (hh += 1) {
            if (bufA[hh] < 0) bufA[hh] = 0;
        }
    }

    // Dense ReLU hidden layers (all but input and head).
    var act: []f64 = bufA[0..H0];
    var a_is_src = true;
    var dim = H0;
    var li: usize = 1;
    while (li < net.layers.len - 1) : (li += 1) {
        const L = net.layers[li];
        const out_dim = L.out;
        const next: []f64 = if (a_is_src) bufB[0..out_dim] else bufA[0..out_dim];
        var o: usize = 0;
        while (o < out_dim) : (o += 1) next[o] = L.b[o];
        var ii: usize = 0;
        while (ii < dim) : (ii += 1) {
            const a = act[ii];
            if (a == 0) continue; // post-ReLU inputs are mostly zero
            const off = ii * out_dim;
            o = 0;
            while (o < out_dim) : (o += 1) next[o] += a * L.w[off + o];
        }
        o = 0;
        while (o < out_dim) : (o += 1) {
            if (next[o] < 0) next[o] = 0;
        }
        act = next;
        dim = out_dim;
        a_is_src = !a_is_src;
    }

    // Scalar head.
    const Lf = net.layers[net.layers.len - 1];
    var out: f64 = Lf.b[0];
    var ii: usize = 0;
    while (ii < dim) : (ii += 1) out += act[ii] * Lf.w[ii];

    const scaled = std.math.tanh(out) * net.scale;
    return @intFromFloat(@floor(scaled + 0.5)); // Math.round
}

// --- integer (quantized) NNUE arithmetic -------------------------------------------
// The forward pass is split into the dual perspective ACCUMULATOR (layer 0, the part the
// incremental search maintains via make/unmake) and EVAL-FROM-ACCUMULATOR (ReLU → dense
// → head). The non-incremental `evaluateInt` just composes them. All a direct port of
// nn.js compileInt: every intermediate is an exact integer (i64, the same value as
// nn.js's float64 integers), so the pre-tanh value is bit-identical to JS and only tanh
// can differ by a ULP (±1 cp). Keep the three (nn.js, here, ai.zig) in sync.
const ACC_MAX: usize = 1024; // upper bound on the first-layer width H0

/// First-layer width of a (quantized) net — the accumulator size.
pub fn h0(net: *const Net) usize {
    return net.ilayers[0].out;
}

/// Add (or subtract) one piece's layer-0 columns to BOTH perspective accumulators,
/// in raw (pre-clip) form. `accw` is white-perspective (us = white), `accb` is
/// black-perspective (us = black, squares vertically flipped) — the standard NNUE dual
/// accumulator. The single source for the feature layout, alongside featureIndices.
pub fn accAddPiece(net: *const Net, accw: []i64, accb: []i64, role: Role, color: Color, sq: usize, add: bool) void {
    const L0 = net.ilayers[0];
    const H0 = L0.out;
    // White perspective.
    {
        const side: usize = if (color == .white) 0 else 1;
        const off = ((pieceIndex(role) * 2 + side) * 64 + sq) * H0;
        var h: usize = 0;
        if (add) {
            while (h < H0) : (h += 1) accw[h] += L0.w[off + h];
        } else {
            while (h < H0) : (h += 1) accw[h] -= L0.w[off + h];
        }
    }
    // Black perspective (flip square, swap us/them).
    {
        const side: usize = if (color == .black) 0 else 1;
        const off = ((pieceIndex(role) * 2 + side) * 64 + (sq ^ 56)) * H0;
        var h: usize = 0;
        if (add) {
            while (h < H0) : (h += 1) accb[h] += L0.w[off + h];
        } else {
            while (h < H0) : (h += 1) accb[h] -= L0.w[off + h];
        }
    }
}

/// Recompute both perspective accumulators (raw, pre-clip) from scratch for `b`.
pub fn accRefresh(net: *const Net, accw: []i64, accb: []i64, b: *const [64]?Piece) void {
    const L0 = net.ilayers[0];
    const H0 = L0.out;
    var h: usize = 0;
    while (h < H0) : (h += 1) {
        accw[h] = L0.b[h];
        accb[h] = L0.b[h];
    }
    var i: usize = 0;
    while (i < 64) : (i += 1) {
        const p = b[i] orelse continue;
        accAddPiece(net, accw, accb, p.role, p.color, i, true);
    }
}

/// Centipawns from a raw side-to-move accumulator: ReLU → dense ReLU layers
/// (requantized ÷ QW) → scalar head (dequantized ÷ QW·QA) → tanh·scale. Activations are
/// at fixed-point scale QA (plain ReLU, not clamped — faithful to the float net).
pub fn evalFromAcc(net: *const Net, acc_stm: []const i64) i32 {
    const QA = net.qa;
    const QW = net.qw;
    var bufA: [ACC_MAX]i64 = undefined;
    var bufB: [ACC_MAX]i64 = undefined;

    const H0 = net.ilayers[0].out;
    var h: usize = 0;
    while (h < H0) : (h += 1) {
        const v = acc_stm[h];
        bufA[h] = if (v < 0) 0 else v; // ReLU (fixed-point scale QA)
    }

    var act: []i64 = bufA[0..H0];
    var a_is_src = true;
    var dim = H0;
    var li: usize = 1;
    while (li < net.ilayers.len - 1) : (li += 1) {
        const L = net.ilayers[li];
        const out_dim = L.out;
        const next: []i64 = if (a_is_src) bufB[0..out_dim] else bufA[0..out_dim];
        var o: usize = 0;
        while (o < out_dim) : (o += 1) next[o] = L.b[o];
        var ii: usize = 0;
        while (ii < dim) : (ii += 1) {
            const a = act[ii];
            if (a == 0) continue;
            const off = ii * out_dim;
            o = 0;
            while (o < out_dim) : (o += 1) next[o] += a * L.w[off + o];
        }
        o = 0;
        while (o < out_dim) : (o += 1) {
            const v = next[o];
            next[o] = if (v <= 0) 0 else @divFloor(v, QW); // ReLU + requant (÷QW)
        }
        act = next;
        dim = out_dim;
        a_is_src = !a_is_src;
    }

    const Lf = net.ilayers[net.ilayers.len - 1];
    var pre: i64 = Lf.b[0];
    var ii: usize = 0;
    while (ii < dim) : (ii += 1) pre += act[ii] * Lf.w[ii];
    const real = @as(f64, @floatFromInt(pre)) / @as(f64, @floatFromInt(QW * QA));
    const scaled = std.math.tanh(real) * net.scale;
    return @intFromFloat(@floor(scaled + 0.5));
}

/// Non-incremental integer eval (from-scratch accumulator). Used at the search root,
/// the parity oracle, and any caller outside the make/unmake search loop.
fn evaluateInt(net: *const Net, b: *const [64]?Piece, turn: Color) i32 {
    var accw: [ACC_MAX]i64 = undefined;
    var accb: [ACC_MAX]i64 = undefined;
    const H0 = net.ilayers[0].out;
    accRefresh(net, accw[0..H0], accb[0..H0], b);
    return evalFromAcc(net, if (turn == .white) accw[0..H0] else accb[0..H0]);
}
