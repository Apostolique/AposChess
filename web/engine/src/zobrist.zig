// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2019-2026 Jean-David Moisan
//
// Zobrist hashing — a direct port of the key generation and hashing in
// web/src/ai.js. The keys MUST match the JS engine bit-for-bit, so this
// replicates its exact recipe (see the spec in engine-parity.json meta.zobrist):
//   - mulberry32 seeded 0x1a2b3c4d, returning a raw uint32 (no /2^32 here).
//   - rand64() = (u64(rnd()) << 32) | u64(rnd()) — first draw is the HIGH word.
//   - draw order: 768 piece keys, then SIDE_KEY, then CASTLE_KEYS K, Q, k, q.
//   - piece key index = (roleIdx*2 + (white?0:1))*64 + sq,
//     roleIdx = {p:0,n:1,b:2,r:3,q:4,k:5}.

const std = @import("std");
const board = @import("board.zig");
const engine = @import("engine.zig");

const Color = board.Color;
const Role = board.Role;
const State = board.State;
const Move = engine.Move;

// mulberry32 (ai.js): operates on a 32-bit word; `>>` on u32 is the logical
// shift that matches JS `>>>`, and `*%` matches Math.imul (mod 2^32).
fn next32(a: *u32) u32 {
    a.* = a.* +% 0x6d2b79f5;
    var t: u32 = (a.* ^ (a.* >> 15)) *% (1 | a.*);
    t = (t +% ((t ^ (t >> 7)) *% (61 | t))) ^ t;
    return t ^ (t >> 14);
}

fn next64(a: *u32) u64 {
    const hi: u64 = next32(a);
    const lo: u64 = next32(a);
    return (hi << 32) | lo;
}

const Keys = struct {
    piece: [12 * 64]u64,
    side: u64,
    castle: [4]u64, // K, Q, k, q
};

const KEYS: Keys = blk: {
    @setEvalBranchQuota(2_000_000);
    var a: u32 = 0x1a2b3c4d;
    var k: Keys = undefined;
    var i: usize = 0;
    while (i < k.piece.len) : (i += 1) k.piece[i] = next64(&a);
    k.side = next64(&a);
    i = 0;
    while (i < 4) : (i += 1) k.castle[i] = next64(&a);
    break :blk k;
};

fn roleIdx(role: Role) usize {
    return switch (role) {
        .p => 0,
        .n => 1,
        .b => 2,
        .r => 3,
        .q => 4,
        .k => 5,
    };
}

fn pieceKey(role: Role, color: Color, sq: usize) u64 {
    const ci: usize = if (color == .white) 0 else 1;
    return KEYS.piece[(roleIdx(role) * 2 + ci) * 64 + sq];
}

// The side-to-move key, for null-move hashing (flip the side bit only).
pub fn sideKey() u64 {
    return KEYS.side;
}

pub fn hashOf(state: *const State) u64 {
    var h: u64 = 0;
    var i: usize = 0;
    while (i < 64) : (i += 1) {
        if (state.board[i]) |p| h ^= pieceKey(p.role, p.color, i);
    }
    if (state.turn == .black) h ^= KEYS.side;
    const c = state.castling;
    if (c.K) h ^= KEYS.castle[0];
    if (c.Q) h ^= KEYS.castle[1];
    if (c.k) h ^= KEYS.castle[2];
    if (c.q) h ^= KEYS.castle[3];
    return h;
}

// Incrementally derive the hash after `m` from the hash before it. MUST mirror
// applyMove() exactly — verified by the parity runner's incremental invariant.
pub fn hashAfter(h0: u64, state: *const State, m: Move) u64 {
    const b = &state.board;
    const piece = b[m.from].?;
    const color = piece.color;
    var h = h0;

    h ^= pieceKey(piece.role, color, m.from);
    if (m.capture) {
        if (b[m.to]) |cap| h ^= pieceKey(cap.role, cap.color, m.to);
    }
    h ^= pieceKey(if (m.promotion) |pr| pr else piece.role, color, m.to);

    if (m.castle != 0) {
        const home: usize = if (color == .white) 0 else 56;
        if (m.castle == 'K') {
            h ^= pieceKey(.r, color, home + 7);
            h ^= pieceKey(.r, color, home + 5);
        } else {
            h ^= pieceKey(.r, color, home + 0);
            h ^= pieceKey(.r, color, home + 3);
        }
    }

    const c = state.castling;
    var nK = c.K;
    var nQ = c.Q;
    var nk = c.k;
    var nq = c.q;
    if (piece.role == .k) {
        if (color == .white) {
            nK = false;
            nQ = false;
        } else {
            nk = false;
            nq = false;
        }
    }
    for ([_]u8{ m.from, m.to }) |id| {
        if (id == 0) {
            nQ = false;
        } else if (id == 7) {
            nK = false;
        } else if (id == 56) {
            nq = false;
        } else if (id == 63) {
            nk = false;
        }
    }
    if (nK != c.K) h ^= KEYS.castle[0];
    if (nQ != c.Q) h ^= KEYS.castle[1];
    if (nk != c.k) h ^= KEYS.castle[2];
    if (nq != c.q) h ^= KEYS.castle[3];

    return h ^ KEYS.side;
}
