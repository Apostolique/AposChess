// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2019-2026 Jean-David Moisan
//
// Quick search driver: search one position to a fixed depth and report the move,
// score, node count, and nodes/sec. Confirms the search runs and measures speed.
//   zig build bench -- --depth=8 [--nn] [--fen="..."]

const std = @import("std");
const board = @import("board.zig");
const engine = @import("engine.zig");
const ai = @import("ai.zig");
const nn = @import("nn.zig");

fn moveStr(m: engine.Move, buf: []u8) []const u8 {
    const fr = board.squareName(m.from);
    const to = board.squareName(m.to);
    buf[0] = fr[0];
    buf[1] = fr[1];
    buf[2] = to[0];
    buf[3] = to[1];
    if (m.promotion) |role| {
        buf[4] = '=';
        buf[5] = board.charFromRole(role);
        return buf[0..6];
    }
    return buf[0..4];
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.arena.allocator();
    const io = init.io;

    var depth: u32 = 8;
    var use_nn = false;
    var fen: []const u8 = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

    const argv = try init.minimal.args.toSlice(init.arena.allocator());
    for (argv[1..]) |arg| {
        if (std.mem.startsWith(u8, arg, "--depth=")) {
            depth = std.fmt.parseInt(u32, arg["--depth=".len..], 10) catch depth;
        } else if (std.mem.eql(u8, arg, "--nn")) {
            use_nn = true;
        } else if (std.mem.startsWith(u8, arg, "--fen=")) {
            fen = arg["--fen=".len..];
        }
    }

    var net: nn.Net = undefined;
    var net_ptr: ?*const nn.Net = null;
    if (use_nn) {
        const wdata = try std.Io.Dir.cwd().readFileAlloc(io, "../src/nn-weights.json", gpa, .unlimited);
        const wparsed = try std.json.parseFromSlice(std.json.Value, gpa, wdata, .{});
        net = try nn.load(gpa, wparsed.value);
        net_ptr = &net;
    }

    var s = try ai.Searcher.init(gpa, io, if (use_nn) .nn else .handcrafted, net_ptr, 1);
    defer s.deinit();

    const st = board.parseFen(fen);
    const t0 = std.Io.Clock.now(.awake, io).nanoseconds;
    const res = s.chooseMove(&st, depth, 0, &.{});
    const ms: u64 = @intCast(@max(1, @divTrunc(std.Io.Clock.now(.awake, io).nanoseconds - t0, 1_000_000)));

    var buf: [8]u8 = undefined;
    const mv = if (res.move) |m| moveStr(m, &buf) else "(none)";
    const nps = res.nodes * 1000 / ms;
    std.debug.print("eval={s} depth={d} bestmove={s} score={d}cp nodes={d} time={d}ms nps={d}\n", .{
        if (use_nn) "nn" else "hc", res.depth, mv, res.score, res.nodes, ms, nps,
    });
}
