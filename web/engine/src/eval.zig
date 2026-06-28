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

// Material-only eval: the bare piece count (VALUE table, the nn eval's pre-weights
// fallback), side-to-move relative. Selected by EvalKind.material. Mirrors evalMaterial
// in web/src/ai.js — keep in lockstep.
pub fn evalMaterial(b: *const [64]?Piece, turn: Color) i32 {
    var s: i32 = 0;
    var i: usize = 0;
    while (i < 64) : (i += 1) {
        const p = b[i] orelse continue;
        const v = VALUE[roleIndex(p.role)];
        s += if (p.color == .white) v else -v;
    }
    return if (turn == .white) s else -s;
}

// --- handcrafted v3 ----------------------------------------------------------
// Material + PSTs distilled from the champion neural net (per-(role,square) ridge
// regression of the net's eval, decomposed into base value + positional residual,
// rescaled to pawn=100, L-R symmetrized). Mirrors evalStmV3 in web/src/ai.js — keep in
// lockstep. Selected by EvalKind.handcrafted3.
const VALUE3 = [_]i32{ 100, 477, 316, 478, 816, 0 }; // p, n, b, r, q, k
const PST_P3 = [64]i32{
    0,   0,   0,   0,   0,   0,   0,   0,
    -29, -17, 3,   -15, -15, 3,   -17, -29,
    -43, -13, 3,   -4,  -4,  3,   -13, -43,
    -37, -2,  23,  35,  35,  23,  -2,  -37,
    -43, 21,  37,  59,  59,  37,  21,  -43,
    57,  89,  141, 117, 117, 141, 89,  57,
    157, 353, 310, 313, 313, 310, 353, 157,
    0,   0,   0,   0,   0,   0,   0,   0,
};
const PST_N3 = [64]i32{
    -39, -28, -37, -9,  -9,  -37, -28, -39,
    -23, 16,  43,  41,  41,  43,  16,  -23,
    2,   28,  41,  57,  57,  41,  28,  2,
    -6,  26,  40,  56,  56,  40,  26,  -6,
    -15, 28,  31,  59,  59,  31,  28,  -15,
    -66, 52,  32,  58,  58,  32,  52,  -66,
    -12, 34,  58,  70,  70,  58,  34,  -12,
    -4,  1,   55,  79,  79,  55,  1,   -4,
};
const PST_B3 = [64]i32{
    -45, -93, -9,  -75, -75, -9,  -93, -45,
    -64, -2,  -49, 4,   4,   -49, -2,  -64,
    -13, -35, 8,   28,  28,  8,   -35, -13,
    -49, -8,  36,  24,  24,  36,  -8,  -49,
    6,   -7,  19,  51,  51,  19,  -7,  6,
    -86, 17,  26,  28,  28,  26,  17,  -86,
    -27, 10,  -25, -39, -39, -25, 10,  -27,
    -33, -98, -77, -42, -42, -77, -98, -33,
};
const PST_R3 = [64]i32{
    -21, 13,  4,   20,  20,  4,   13,  -21,
    -7,  -21, -18, -36, -36, -18, -21, -7,
    7,   11,  -1,  21,  21,  -1,  11,  7,
    -48, 22,  17,  56,  56,  17,  22,  -48,
    13,  22,  47,  50,  50,  47,  22,  13,
    -94, 66,  18,  44,  44,  18,  66,  -94,
    26,  80,  -20, 0,   0,   -20, 80,  26,
    125, -2,  33,  27,  27,  33,  -2,  125,
};
const PST_Q3 = [64]i32{
    -93,  -96, -86, -65, -65, -86, -96, -93,
    -111, -55, -34, -28, -28, -34, -55, -111,
    -48,  -12, 11,  22,  22,  11,  -12, -48,
    -30,  5,   29,  34,  34,  29,  5,   -30,
    6,    33,  64,  73,  73,  64,  33,  6,
    6,    81,  84,  107, 107, 84,  81,  6,
    14,   78,  90,  73,  73,  90,  78,  14,
    40,   43,  86,  39,  39,  86,  43,  40,
};
// King PST left at 0 (like v2). The net DOES imply an active/central king table, but
// importing it cost ~18 Elo in self-play (-6 with it vs +12 without, 4000 games each at
// depth 4 vs v2): a static king bonus has no game-phase taper, so it walks the king out
// in the middlegame. A tapered eval would be needed to use it. Without it, v3's material
// + piece PSTs are ~+12 Elo over v2.
const PST_K3 = [_]i32{0} ** 64;

fn pstOf3(role: Role, sq: usize) i32 {
    return switch (role) {
        .p => PST_P3[sq],
        .n => PST_N3[sq],
        .b => PST_B3[sq],
        .r => PST_R3[sq],
        .q => PST_Q3[sq],
        .k => PST_K3[sq],
    };
}

pub fn evalStmV3(b: *const [64]?Piece, turn: Color) i32 {
    var s: i32 = 0;
    var i: usize = 0;
    while (i < 64) : (i += 1) {
        const p = b[i] orelse continue;
        const sqv: usize = if (p.color == .white) i else i ^ 56;
        const v = VALUE3[roleIndex(p.role)] + pstOf3(p.role, sqv);
        s += if (p.color == .white) v else -v;
    }
    var wm: engine.MoveList = .{};
    engine.generatePseudoMoves(b, .white, &wm);
    var bm: engine.MoveList = .{};
    engine.generatePseudoMoves(b, .black, &bm);
    s += MOB * (@as(i32, @intCast(wm.len)) - @as(i32, @intCast(bm.len)));
    return if (turn == .white) s else -s;
}
