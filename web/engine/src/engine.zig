// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2019-2026 Jean-David Moisan
//
// Move generation and game rules — a direct port of web/src/engine.js.
// Variant rules (see README.md / docs/engine.md):
//   - Pawns move one square diagonally (non-capturing), capture one square
//     straight forward, advance two straight on the first move.
//   - Bishops/Rooks slide normally AND may jump the first piece in a line,
//     landing on the very next square (unless friendly-occupied or in an enemy
//     safety zone).
//   - Knights travel like a rook (clear path) then step one square to the side.
//   - Queens/Kings project a 3x3 "safety zone" that repels ENEMY jumps.

const std = @import("std");
const board = @import("board.zig");

const Color = board.Color;
const Role = board.Role;
const Piece = board.Piece;
const State = board.State;
const opponent = board.opponent;

pub const Move = struct {
    from: u8,
    to: u8,
    promotion: ?Role = null,
    capture: bool = false,
    castle: u8 = 0, // 0 = none, else 'K' or 'Q'
    jump: bool = false,
    double: bool = false,
};

// Fixed-capacity move buffer — no per-node allocation. 1024 is far above any
// reachable legal-move count in this variant.
pub const MoveList = struct {
    items: [1024]Move = undefined,
    len: usize = 0,
    pub fn push(self: *MoveList, m: Move) void {
        self.items[self.len] = m;
        self.len += 1;
    }
    pub fn slice(self: *const MoveList) []const Move {
        return self.items[0..self.len];
    }
};

const Dir = struct { df: i32, dr: i32 };
const ORTHO = [_]Dir{ .{ .df = 0, .dr = 1 }, .{ .df = 0, .dr = -1 }, .{ .df = 1, .dr = 0 }, .{ .df = -1, .dr = 0 } };
const DIAG = [_]Dir{ .{ .df = 1, .dr = 1 }, .{ .df = 1, .dr = -1 }, .{ .df = -1, .dr = 1 }, .{ .df = -1, .dr = -1 } };
const ALL8 = ORTHO ++ DIAG;
// A knight's perpendicular side-step relative to its travel direction.
const PERP_V = [_]Dir{ .{ .df = 1, .dr = 0 }, .{ .df = -1, .dr = 0 } }; // travel vertical (df == 0)
const PERP_H = [_]Dir{ .{ .df = 0, .dr = 1 }, .{ .df = 0, .dr = -1 } };

inline fn onBoard(f: i32, r: i32) bool {
    return f >= 0 and f < 8 and r >= 0 and r < 8;
}
inline fn idx(f: i32, r: i32) u8 {
    return @intCast(r * 8 + f);
}
inline fn fileI(i: u8) i32 {
    return @intCast(i % 8);
}
inline fn rankI(i: u8) i32 {
    return @intCast(i / 8);
}

pub fn findKing(b: *const [64]?Piece, color: Color) i32 {
    var i: usize = 0;
    while (i < 64) : (i += 1) {
        if (b[i]) |p| {
            if (p.role == .k and p.color == color) return @intCast(i);
        }
    }
    return -1;
}

// Squares an enemy jump may not land on: within king-distance 1 of one of
// `color`'s kings or queens.
fn safetyZones(b: *const [64]?Piece, color: Color) [64]bool {
    var zones: [64]bool = @splat(false);
    var i: usize = 0;
    while (i < 64) : (i += 1) {
        const p = b[i] orelse continue;
        if (p.color != color or (p.role != .q and p.role != .k)) continue;
        const f = fileI(@intCast(i));
        const r = rankI(@intCast(i));
        var df: i32 = -1;
        while (df <= 1) : (df += 1) {
            var dr: i32 = -1;
            while (dr <= 1) : (dr += 1) {
                if (onBoard(f + df, r + dr)) zones[idx(f + df, r + dr)] = true;
            }
        }
    }
    return zones;
}

fn addPawn(moves: *MoveList, from: u8, to: u8, color: Color, capture: bool) void {
    const promo_rank: usize = if (color == .white) 7 else 0;
    if (board.rankOf(to) == promo_rank) {
        for ([_]Role{ .q, .r, .b, .n }) |role| {
            moves.push(.{ .from = from, .to = to, .promotion = role, .capture = capture });
        }
    } else {
        moves.push(.{ .from = from, .to = to, .capture = capture });
    }
}

fn pawnMoves(b: *const [64]?Piece, i: u8, color: Color, moves: *MoveList) void {
    const fwd: i32 = if (color == .white) 1 else -1;
    const start_rank: i32 = if (color == .white) 1 else 6;
    const f = fileI(i);
    const r = rankI(i);

    // Diagonal forward to an empty square (cannot capture diagonally).
    for ([_]i32{ -1, 1 }) |df| {
        const nf = f + df;
        const nr = r + fwd;
        if (onBoard(nf, nr) and b[idx(nf, nr)] == null) addPawn(moves, i, idx(nf, nr), color, false);
    }
    // Straight forward: capture only.
    if (onBoard(f, r + fwd)) {
        if (b[idx(f, r + fwd)]) |t| {
            if (t.color != color) addPawn(moves, i, idx(f, r + fwd), color, true);
        }
    }
    // First move: two straight forward over empty squares.
    if (r == start_rank and b[idx(f, r + fwd)] == null and b[idx(f, r + 2 * fwd)] == null) {
        moves.push(.{ .from = i, .to = idx(f, r + 2 * fwd), .double = true });
    }
}

fn knightMoves(b: *const [64]?Piece, i: u8, color: Color, moves: *MoveList) void {
    const f = fileI(i);
    const r = rankI(i);
    var seen: [64]bool = @splat(false);
    for (ORTHO) |d| {
        var k: i32 = 1;
        while (true) : (k += 1) {
            const cf = f + d.df * k;
            const cr = r + d.dr * k;
            if (!onBoard(cf, cr) or b[idx(cf, cr)] != null) break; // blocked
            const perps: []const Dir = if (d.df == 0) &PERP_V else &PERP_H;
            for (perps) |p| {
                const tf = cf + p.df;
                const tr = cr + p.dr;
                if (!onBoard(tf, tr)) continue;
                const ti = idx(tf, tr);
                if (seen[ti]) continue;
                if (b[ti]) |t| {
                    if (t.color == color) continue;
                }
                seen[ti] = true;
                moves.push(.{ .from = i, .to = ti, .capture = b[ti] != null });
            }
        }
    }
}

fn sliderMoves(b: *const [64]?Piece, i: u8, color: Color, dirs: []const Dir, moves: *MoveList) void {
    const f0 = fileI(i);
    const r0 = rankI(i);
    for (dirs) |d| {
        var nf = f0 + d.df;
        var nr = r0 + d.dr;
        while (onBoard(nf, nr)) {
            const ti = idx(nf, nr);
            if (b[ti]) |t| {
                if (t.color != color) moves.push(.{ .from = i, .to = ti, .capture = true });
                break;
            }
            moves.push(.{ .from = i, .to = ti });
            nf += d.df;
            nr += d.dr;
        }
    }
}

fn jumpMoves(b: *const [64]?Piece, i: u8, color: Color, dirs: []const Dir, zones: *const [64]bool, moves: *MoveList) void {
    const f0 = fileI(i);
    const r0 = rankI(i);
    for (dirs) |d| {
        var nf = f0 + d.df;
        var nr = r0 + d.dr;
        while (onBoard(nf, nr) and b[idx(nf, nr)] == null) {
            nf += d.df;
            nr += d.dr;
        }
        if (!onBoard(nf, nr)) continue; // no piece to jump
        const lf = nf + d.df;
        const lr = nr + d.dr;
        if (!onBoard(lf, lr)) continue; // jumped piece on the edge
        const li = idx(lf, lr);
        if (zones[li]) continue;
        if (b[li]) |t| {
            if (t.color == color) continue;
            moves.push(.{ .from = i, .to = li, .capture = true, .jump = true });
        } else {
            moves.push(.{ .from = i, .to = li, .jump = true });
        }
    }
}

fn kingMoves(b: *const [64]?Piece, i: u8, color: Color, moves: *MoveList) void {
    const f0 = fileI(i);
    const r0 = rankI(i);
    for (ALL8) |d| {
        const nf = f0 + d.df;
        const nr = r0 + d.dr;
        if (!onBoard(nf, nr)) continue;
        const ti = idx(nf, nr);
        if (b[ti]) |t| {
            if (t.color == color) continue;
            moves.push(.{ .from = i, .to = ti, .capture = true });
        } else {
            moves.push(.{ .from = i, .to = ti });
        }
    }
}

pub fn generatePseudoMoves(b: *const [64]?Piece, color: Color, moves: *MoveList) void {
    const zones = safetyZones(b, opponent(color));
    var i: usize = 0;
    while (i < 64) : (i += 1) {
        const p = b[i] orelse continue;
        if (p.color != color) continue;
        const s: u8 = @intCast(i);
        switch (p.role) {
            .p => pawnMoves(b, s, color, moves),
            .n => knightMoves(b, s, color, moves),
            .b => {
                sliderMoves(b, s, color, &DIAG, moves);
                jumpMoves(b, s, color, &DIAG, &zones, moves);
            },
            .r => {
                sliderMoves(b, s, color, &ORTHO, moves);
                jumpMoves(b, s, color, &ORTHO, &zones, moves);
            },
            .q => sliderMoves(b, s, color, &ALL8, moves),
            .k => kingMoves(b, s, color, moves),
        }
    }
}

pub fn isAttacked(b: *const [64]?Piece, target: u8, by_color: Color) bool {
    var moves: MoveList = .{};
    generatePseudoMoves(b, by_color, &moves);
    for (moves.slice()) |m| {
        if (m.to == target) return true;
    }
    return false;
}

pub fn kingAttacked(b: *const [64]?Piece, color: Color) bool {
    return kingAttackedAt(b, color, findKing(b, color));
}

// Fast check test (king never sits where a jump could reach, so only normal
// moves give check). Must stay equivalent to isAttacked(b, kingSq, enemy).
fn kingAttackedAt(b: *const [64]?Piece, color: Color, k: i32) bool {
    if (k < 0) return false;
    const enemy = opponent(color);
    const ku: u8 = @intCast(k);
    const kf = fileI(ku);
    const kr = rankI(ku);

    // Pawn: captures straight forward, so an enemy pawn one rank "behind" the king.
    const pr = kr - (if (enemy == .white) @as(i32, 1) else -1);
    if (pr >= 0 and pr <= 7) {
        if (b[idx(kf, pr)]) |p| {
            if (p.color == enemy and p.role == .p) return true;
        }
    }

    // Enemy king adjacent.
    for (ALL8) |d| {
        const f = kf + d.df;
        const r = kr + d.dr;
        if (onBoard(f, r)) {
            if (b[idx(f, r)]) |p| {
                if (p.color == enemy and p.role == .k) return true;
            }
        }
    }

    // Normal slides: first piece on each ray if it's the matching slider.
    for (ORTHO) |d| {
        var f = kf + d.df;
        var r = kr + d.dr;
        while (onBoard(f, r)) {
            if (b[idx(f, r)]) |p| {
                if (p.color == enemy and (p.role == .r or p.role == .q)) return true;
                break;
            }
            f += d.df;
            r += d.dr;
        }
    }
    for (DIAG) |d| {
        var f = kf + d.df;
        var r = kr + d.dr;
        while (onBoard(f, r)) {
            if (b[idx(f, r)]) |p| {
                if (p.color == enemy and (p.role == .b or p.role == .q)) return true;
                break;
            }
            f += d.df;
            r += d.dr;
        }
    }

    // Knight: reverse the move — step opposite the side-step to the "corner"
    // (must be empty), then scan straight back for the first piece.
    for (ORTHO) |s| {
        const cf = kf - s.df;
        const cr = kr - s.dr;
        if (!onBoard(cf, cr) or b[idx(cf, cr)] != null) continue;
        const perps: []const Dir = if (s.df == 0) &PERP_V else &PERP_H;
        for (perps) |d| {
            var f = cf - d.df;
            var r = cr - d.dr;
            while (onBoard(f, r)) {
                if (b[idx(f, r)]) |p| {
                    if (p.color == enemy and p.role == .n) return true;
                    break;
                }
                f -= d.df;
                r -= d.dr;
            }
        }
    }
    return false;
}

pub fn applyMove(state: *const State, m: Move) State {
    var b = state.board;
    const piece = b[m.from].?;
    const color = piece.color;

    b[m.to] = if (m.promotion) |role| Piece{ .role = role, .color = color } else piece;
    b[m.from] = null;

    if (m.castle != 0) {
        const home: usize = if (color == .white) 0 else 56;
        if (m.castle == 'K') {
            b[home + 5] = b[home + 7];
            b[home + 7] = null;
        } else {
            b[home + 3] = b[home + 0];
            b[home + 0] = null;
        }
    }

    var castling = state.castling;
    if (piece.role == .k) {
        if (color == .white) {
            castling.K = false;
            castling.Q = false;
        } else {
            castling.k = false;
            castling.q = false;
        }
    }
    for ([_]u8{ m.from, m.to }) |id| {
        if (id == 0) {
            castling.Q = false;
        } else if (id == 7) {
            castling.K = false;
        } else if (id == 56) {
            castling.q = false;
        } else if (id == 63) {
            castling.k = false;
        }
    }

    const reset = piece.role == .p or m.capture;
    return .{
        .board = b,
        .turn = opponent(color),
        .castling = castling,
        .halfmove = if (reset) 0 else state.halfmove + 1,
        .fullmove = if (color == .black) state.fullmove + 1 else state.fullmove,
    };
}

// --- make / unmake -----------------------------------------------------------------
// In-place move application with an undo record, so the search mutates one board
// instead of cloning a fresh State at every node. makeMove returns the Undo that
// unmakeMove needs to restore the exact prior position. Mirrors applyMove's rules.
pub const Undo = struct {
    captured: ?Piece,
    castling: board.Castling,
    halfmove: u32,
    fullmove: u32,
};

pub fn makeMove(state: *State, m: Move) Undo {
    const undo = Undo{
        .captured = state.board[m.to],
        .castling = state.castling,
        .halfmove = state.halfmove,
        .fullmove = state.fullmove,
    };
    const piece = state.board[m.from].?;
    const color = piece.color;

    state.board[m.to] = if (m.promotion) |role| Piece{ .role = role, .color = color } else piece;
    state.board[m.from] = null;

    if (m.castle != 0) {
        const home: usize = if (color == .white) 0 else 56;
        if (m.castle == 'K') {
            state.board[home + 5] = state.board[home + 7];
            state.board[home + 7] = null;
        } else {
            state.board[home + 3] = state.board[home + 0];
            state.board[home + 0] = null;
        }
    }

    if (piece.role == .k) {
        if (color == .white) {
            state.castling.K = false;
            state.castling.Q = false;
        } else {
            state.castling.k = false;
            state.castling.q = false;
        }
    }
    for ([_]u8{ m.from, m.to }) |id| {
        if (id == 0) {
            state.castling.Q = false;
        } else if (id == 7) {
            state.castling.K = false;
        } else if (id == 56) {
            state.castling.q = false;
        } else if (id == 63) {
            state.castling.k = false;
        }
    }

    const reset = piece.role == .p or m.capture;
    state.halfmove = if (reset) 0 else state.halfmove + 1;
    state.fullmove = if (color == .black) state.fullmove + 1 else state.fullmove;
    state.turn = opponent(color);
    return undo;
}

pub fn unmakeMove(state: *State, m: Move, undo: Undo) void {
    state.turn = opponent(state.turn); // back to the side that moved
    const color = state.turn;

    const moved = state.board[m.to].?;
    state.board[m.from] = if (m.promotion != null) Piece{ .role = .p, .color = color } else moved;
    state.board[m.to] = undo.captured;

    if (m.castle != 0) {
        const home: usize = if (color == .white) 0 else 56;
        if (m.castle == 'K') {
            state.board[home + 7] = state.board[home + 5];
            state.board[home + 5] = null;
        } else {
            state.board[home + 0] = state.board[home + 3];
            state.board[home + 3] = null;
        }
    }

    state.castling = undo.castling;
    state.halfmove = undo.halfmove;
    state.fullmove = undo.fullmove;
}

fn rookOk(b: *const [64]?Piece, i: u8, color: Color) bool {
    if (b[i]) |p| return p.role == .r and p.color == color;
    return false;
}

fn addCastling(state: *const State, color: Color, out: *MoveList) void {
    const b = &state.board;
    const c = state.castling;
    const enemy = opponent(color);
    const home: u8 = if (color == .white) 0 else 56;
    const king_idx = home + 4;
    const king = b[king_idx] orelse return;
    if (king.role != .k or king.color != color) return;
    if (isAttacked(b, king_idx, enemy)) return; // cannot castle out of check

    const can_k = if (color == .white) c.K else c.k;
    const can_q = if (color == .white) c.Q else c.q;

    if (can_k and b[home + 5] == null and b[home + 6] == null and rookOk(b, home + 7, color) and
        !isAttacked(b, home + 5, enemy) and !isAttacked(b, home + 6, enemy))
    {
        out.push(.{ .from = king_idx, .to = home + 6, .castle = 'K' });
    }
    if (can_q and b[home + 3] == null and b[home + 2] == null and b[home + 1] == null and rookOk(b, home + 0, color) and
        !isAttacked(b, home + 3, enemy) and !isAttacked(b, home + 2, enemy))
    {
        out.push(.{ .from = king_idx, .to = home + 2, .castle = 'Q' });
    }
}

pub fn legalMoves(state: *const State, out: *MoveList) void {
    const color = state.turn;
    var b = state.board; // mutable working copy for make/unmake
    const king_sq = findKing(&b, color);
    var pseudo: MoveList = .{};
    generatePseudoMoves(&b, color, &pseudo);
    for (pseudo.slice()) |m| {
        const moved = b[m.from].?;
        const captured = b[m.to];
        b[m.to] = moved;
        b[m.from] = null;
        const k: i32 = if (moved.role == .k) @as(i32, m.to) else king_sq;
        const ok = !kingAttackedAt(&b, color, k);
        b[m.from] = moved;
        b[m.to] = captured;
        if (ok) out.push(m);
    }
    addCastling(state, color, out);
}

// Is there ANY legal move? Early-exits on the first one (cheap vs counting all).
// `pseudo` lets a caller reuse an already-generated pseudo-move list; on no legal
// pseudo-move it falls back to the full generator (castling can be the only legal
// move). Mirrors hasLegalMove in engine.js.
pub fn hasLegalMove(state: *const State, pseudo: ?*const MoveList) bool {
    const color = state.turn;
    var b = state.board;
    const king_sq = findKing(&b, color);
    var local: MoveList = .{};
    const moves: *const MoveList = if (pseudo) |p| p else blk: {
        generatePseudoMoves(&b, color, &local);
        break :blk &local;
    };
    for (moves.slice()) |m| {
        const moved = b[m.from].?;
        const captured = b[m.to];
        b[m.to] = moved;
        b[m.from] = null;
        const k: i32 = if (moved.role == .k) @as(i32, m.to) else king_sq;
        const ok = !kingAttackedAt(&b, color, k);
        b[m.from] = moved;
        b[m.to] = captured;
        if (ok) return true;
    }
    var full: MoveList = .{};
    legalMoves(state, &full);
    return full.len > 0;
}

fn insufficientMaterial(b: *const [64]?Piece) bool {
    for (b) |sqp| {
        if (sqp) |p| {
            if (p.role != .k) return false;
        }
    }
    return true;
}

pub const Status = enum { ongoing, checkmate, stalemate, insufficient_material, fifty_move };

pub fn statusName(s: Status) []const u8 {
    return switch (s) {
        .ongoing => "ongoing",
        .checkmate => "checkmate",
        .stalemate => "stalemate",
        .insufficient_material => "insufficient-material",
        .fifty_move => "fifty-move",
    };
}

pub const StatusResult = struct { status: Status, check: bool };

pub fn gameStatus(state: *const State) StatusResult {
    const color = state.turn;
    var legal: MoveList = .{};
    legalMoves(state, &legal);
    const in_check = kingAttacked(&state.board, color);
    if (legal.len == 0) {
        return .{ .status = if (in_check) .checkmate else .stalemate, .check = in_check };
    }
    if (insufficientMaterial(&state.board)) return .{ .status = .insufficient_material, .check = in_check };
    if (state.halfmove >= 100) return .{ .status = .fifty_move, .check = in_check };
    return .{ .status = .ongoing, .check = in_check };
}

// Uses make/unmake (not applyMove), so the perft oracle validates them exactly.
pub fn perft(state: *State, depth: u32) u64 {
    if (depth == 0) return 1;
    var moves: MoveList = .{};
    legalMoves(state, &moves);
    if (depth == 1) return moves.len;
    var n: u64 = 0;
    for (moves.slice()) |m| {
        const u = makeMove(state, m);
        n += perft(state, depth - 1);
        unmakeMove(state, m, u);
    }
    return n;
}
