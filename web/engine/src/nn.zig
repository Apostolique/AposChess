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

pub const Net = struct {
    layers: []Layer,
    scale: f64,
};

fn toF64(v: std.json.Value) f64 {
    return switch (v) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        .number_string => |s| std.fmt.parseFloat(f64, s) catch 0,
        else => 0,
    };
}

/// Build a Net from a parsed weights object ({arch, scale, layers:[{w,b}]}).
pub fn load(alloc: std.mem.Allocator, v: std.json.Value) !Net {
    const obj = v.object;
    const layers_j = obj.get("layers").?.array;
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
    return .{ .layers = layers, .scale = toF64(obj.get("scale").?) };
}

/// Centipawn evaluation from the side-to-move's perspective.
pub fn evaluate(net: *const Net, b: *const [64]?Piece, turn: Color) i32 {
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
