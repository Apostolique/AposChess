// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2019-2026 Jean-David Moisan
//
// Handcrafted evaluation (evalStm) — a direct port from web/src/ai.js: material
// + piece-square tables + a mobility differential (MOB centipawns per extra
// pseudo-legal move). Centipawns from the side-to-move's perspective.

const std = @import("std");
const board = @import("board.zig");
const engine = @import("engine.zig");

const Color = board.Color;
const Role = board.Role;
const Piece = board.Piece;

const MOB: i32 = 3;

fn roleIndex(role: Role) usize {
    return switch (role) {
        .p => 0,
        .n => 1,
        .b => 2,
        .r => 3,
        .q => 4,
        .k => 5,
    };
}

const VALUE = [_]i32{ 100, 500, 330, 500, 900, 0 }; // p, n, b, r, q, k

/// Material value of a role (used by search move ordering / delta pruning).
pub fn value(role: Role) i32 {
    return VALUE[roleIndex(role)];
}

const PST_P = [64]i32{
    0,  0,  0,  0,  0,  0,  0,  0,
    5,  5,  5,  5,  5,  5,  5,  5,
    10, 10, 10, 12, 12, 10, 10, 10,
    20, 20, 25, 30, 30, 25, 20, 20,
    35, 35, 40, 45, 45, 40, 35, 35,
    55, 55, 60, 65, 65, 60, 55, 55,
    80, 80, 85, 90, 90, 85, 80, 80,
    0,  0,  0,  0,  0,  0,  0,  0,
};
const PST_N = [64]i32{
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20, 0,   5,   5,   0,   -20, -40,
    -30, 5,   10,  15,  15,  10,  5,   -30,
    -30, 0,   15,  20,  20,  15,  0,   -30,
    -30, 5,   15,  20,  20,  15,  5,   -30,
    -30, 0,   10,  15,  15,  10,  0,   -30,
    -40, -20, 0,   0,   0,   0,   -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
};
const PST_B = [64]i32{
    -10, -5, -5, -5, -5, -5, -5, -10,
    -5,  5,  0,  0,  0,  0,  5,  -5,
    -5,  5,  5,  5,  5,  5,  5,  -5,
    -5,  0,  5,  8,  8,  5,  0,  -5,
    -5,  0,  5,  8,  8,  5,  0,  -5,
    -5,  5,  5,  5,  5,  5,  5,  -5,
    -5,  5,  0,  0,  0,  0,  5,  -5,
    -10, -5, -5, -5, -5, -5, -5, -10,
};
const PST_R = [64]i32{
    0, 0,  0,  5,  5,  0,  0,  0,
    0, 0,  0,  0,  0,  0,  0,  0,
    0, 0,  0,  0,  0,  0,  0,  0,
    0, 0,  0,  0,  0,  0,  0,  0,
    0, 0,  0,  0,  0,  0,  0,  0,
    0, 0,  0,  0,  0,  0,  0,  0,
    5, 10, 10, 10, 10, 10, 10, 5,
    0, 0,  0,  5,  5,  0,  0,  0,
};
const PST_Q = [64]i32{
    -10, -5, -5, -2, -2, -5, -5, -10,
    -5,  0,  0,  0,  0,  0,  0,  -5,
    -5,  0,  3,  3,  3,  3,  0,  -5,
    -2,  0,  3,  5,  5,  3,  0,  -2,
    -2,  0,  3,  5,  5,  3,  0,  -2,
    -5,  0,  3,  3,  3,  3,  0,  -5,
    -5,  0,  0,  0,  0,  0,  0,  -5,
    -10, -5, -5, -2, -2, -5, -5, -10,
};
const PST_K = [_]i32{0} ** 64;

fn pstOf(role: Role, sq: usize) i32 {
    return switch (role) {
        .p => PST_P[sq],
        .n => PST_N[sq],
        .b => PST_B[sq],
        .r => PST_R[sq],
        .q => PST_Q[sq],
        .k => PST_K[sq],
    };
}

pub fn evalStm(b: *const [64]?Piece, turn: Color) i32 {
    var s: i32 = 0;
    var i: usize = 0;
    while (i < 64) : (i += 1) {
        const p = b[i] orelse continue;
        const sqv: usize = if (p.color == .white) i else i ^ 56;
        const v = VALUE[roleIndex(p.role)] + pstOf(p.role, sqv);
        s += if (p.color == .white) v else -v;
    }
    // Mobility: pseudo-legal move differential (white minus black).
    var wm: engine.MoveList = .{};
    engine.generatePseudoMoves(b, .white, &wm);
    var bm: engine.MoveList = .{};
    engine.generatePseudoMoves(b, .black, &bm);
    s += MOB * (@as(i32, @intCast(wm.len)) - @as(i32, @intCast(bm.len)));
    return if (turn == .white) s else -s;
}
