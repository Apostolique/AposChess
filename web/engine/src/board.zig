// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2019-2026 Jean-David Moisan
//
// Board representation and FEN handling — a direct port of web/src/board.js.
// Squares are indexed 0..63 with index = rank*8 + file, rank 0 = White's back
// rank ("1"), file 0 = the a-file. The variant has no en passant, so that FEN
// field is always written as '-' and ignored on parse (mirrors board.js).

const std = @import("std");

pub const Color = enum { white, black };
pub const Role = enum { p, n, b, r, q, k };

pub const Piece = struct { role: Role, color: Color };

pub const Castling = struct {
    K: bool = false,
    Q: bool = false,
    k: bool = false,
    q: bool = false,
};

pub const State = struct {
    board: [64]?Piece,
    turn: Color,
    castling: Castling,
    halfmove: u32,
    fullmove: u32,
};

pub inline fn sq(file: usize, rank: usize) usize {
    return rank * 8 + file;
}

pub inline fn fileOf(i: usize) usize {
    return i % 8;
}

pub inline fn rankOf(i: usize) usize {
    return i / 8;
}

pub fn opponent(c: Color) Color {
    return if (c == .white) .black else .white;
}

/// "a1".."h8" as a 2-byte array.
pub fn squareName(i: usize) [2]u8 {
    return .{ 'a' + @as(u8, @intCast(fileOf(i))), '1' + @as(u8, @intCast(rankOf(i))) };
}

fn roleFromChar(lower: u8) Role {
    return switch (lower) {
        'p' => .p,
        'n' => .n,
        'b' => .b,
        'r' => .r,
        'q' => .q,
        'k' => .k,
        else => unreachable,
    };
}

pub fn charFromRole(role: Role) u8 {
    return switch (role) {
        .p => 'p',
        .n => 'n',
        .b => 'b',
        .r => 'r',
        .q => 'q',
        .k => 'k',
    };
}

fn pieceFromChar(c: u8) Piece {
    const lower = std.ascii.toLower(c);
    // Uppercase letter in FEN = White; lowercase = Black.
    return .{ .role = roleFromChar(lower), .color = if (c == lower) .black else .white };
}

fn charFromPiece(p: Piece) u8 {
    const base = charFromRole(p.role);
    return if (p.color == .white) std.ascii.toUpper(base) else base;
}

pub fn parseFen(fen: []const u8) State {
    var it = std.mem.splitScalar(u8, fen, ' ');
    const board_part = it.next().?;
    const turn_part = it.next().?;
    const castle_part = it.next().?;
    _ = it.next(); // en passant — unused in this variant
    const half_part = it.next() orelse "0";
    const full_part = it.next() orelse "1";

    var board: [64]?Piece = @splat(null);
    // rows[0] is rank 8, rows[7] is rank 1.
    var rows = std.mem.splitScalar(u8, board_part, '/');
    var row_slices: [8][]const u8 = undefined;
    var ri: usize = 0;
    while (rows.next()) |row| : (ri += 1) row_slices[ri] = row;

    var r: usize = 0;
    while (r < 8) : (r += 1) {
        const row = row_slices[7 - r];
        var f: usize = 0;
        for (row) |ch| {
            if (ch >= '1' and ch <= '8') {
                f += ch - '0';
            } else {
                board[sq(f, r)] = pieceFromChar(ch);
                f += 1;
            }
        }
    }

    return .{
        .board = board,
        .turn = if (turn_part[0] == 'w') .white else .black,
        .castling = .{
            .K = std.mem.indexOfScalar(u8, castle_part, 'K') != null,
            .Q = std.mem.indexOfScalar(u8, castle_part, 'Q') != null,
            .k = std.mem.indexOfScalar(u8, castle_part, 'k') != null,
            .q = std.mem.indexOfScalar(u8, castle_part, 'q') != null,
        },
        .halfmove = std.fmt.parseInt(u32, half_part, 10) catch 0,
        .fullmove = std.fmt.parseInt(u32, full_part, 10) catch 1,
    };
}

/// Serialize into `buf` (≥ ~96 bytes); returns the written slice.
pub fn toFen(state: *const State, buf: []u8) []u8 {
    var i: usize = 0;
    var r: isize = 7;
    while (r >= 0) : (r -= 1) {
        var empty: u8 = 0;
        var f: usize = 0;
        while (f < 8) : (f += 1) {
            const p = state.board[sq(f, @intCast(r))];
            if (p == null) {
                empty += 1;
                continue;
            }
            if (empty != 0) {
                buf[i] = '0' + empty;
                i += 1;
                empty = 0;
            }
            buf[i] = charFromPiece(p.?);
            i += 1;
        }
        if (empty != 0) {
            buf[i] = '0' + empty;
            i += 1;
        }
        if (r > 0) {
            buf[i] = '/';
            i += 1;
        }
    }
    buf[i] = ' ';
    i += 1;
    buf[i] = if (state.turn == .white) 'w' else 'b';
    i += 1;
    buf[i] = ' ';
    i += 1;

    const c = state.castling;
    const castle_start = i;
    if (c.K) {
        buf[i] = 'K';
        i += 1;
    }
    if (c.Q) {
        buf[i] = 'Q';
        i += 1;
    }
    if (c.k) {
        buf[i] = 'k';
        i += 1;
    }
    if (c.q) {
        buf[i] = 'q';
        i += 1;
    }
    if (i == castle_start) {
        buf[i] = '-';
        i += 1;
    }

    const tail = std.fmt.bufPrint(buf[i..], " - {d} {d}", .{ state.halfmove, state.fullmove }) catch unreachable;
    return buf[0 .. i + tail.len];
}

pub fn newGameState() State {
    return parseFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
}
