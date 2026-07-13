// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2019-2026 Jean-David Moisan
//
// Quick search driver: search one position to a fixed depth and report the move,
// score, node count, nodes/sec, AND per-node time (ns/node). Confirms the search runs
// and measures speed. ns/node is the "frame time" of the eval: an ABSOLUTE per-node
// cost that (for a given arch+quant) is independent of the weights, so it says how many
// nodes the net can search in the browser's per-move budget — read it like a frame-time
// meter, not against another net.
//   zig build bench -- --depth=8 [--nn | --weights=PATH] [--fen="..."]
// --weights=PATH benches an arbitrary net (implies --nn); pass an absolute path when
// running from outside the engine dir (the --nn default is relative to cwd).

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
    var weights_path: []const u8 = "../src/nn-weights.json";
    var fen: []const u8 = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

    const argv = try init.minimal.args.toSlice(init.arena.allocator());
    for (argv[1..]) |arg| {
        if (std.mem.startsWith(u8, arg, "--depth=")) {
            depth = std.fmt.parseInt(u32, arg["--depth=".len..], 10) catch depth;
        } else if (std.mem.eql(u8, arg, "--nn")) {
            use_nn = true;
        } else if (std.mem.startsWith(u8, arg, "--weights=")) {
            weights_path = arg["--weights=".len..]; // bench an arbitrary net
            use_nn = true;
        } else if (std.mem.startsWith(u8, arg, "--fen=")) {
            fen = arg["--fen=".len..];
        }
    }

    var net: nn.Net = undefined;
    var net_ptr: ?*const nn.Net = null;
    if (use_nn) {
        const wdata = try std.Io.Dir.cwd().readFileAlloc(io, weights_path, gpa, .unlimited);
        const wparsed = try std.json.parseFromSlice(std.json.Value, gpa, wdata, .{});
        net = try nn.load(gpa, wparsed.value);
        net_ptr = &net;
    }

    var s = try ai.Searcher.init(gpa, io, if (use_nn) .nn else .handcrafted, net_ptr, 1);
    defer s.deinit();

    const st = board.parseFen(fen);
    const t0 = std.Io.Clock.now(.awake, io).nanoseconds;
    const res = s.chooseMove(&st, depth, 0, &.{});
    const elapsed_ns = std.Io.Clock.now(.awake, io).nanoseconds - t0;
    const ms: u64 = @intCast(@max(1, @divTrunc(elapsed_ns, 1_000_000)));

    var buf: [8]u8 = undefined;
    const mv = if (res.move) |m| moveStr(m, &buf) else "(none)";
    const nps = res.nodes * 1000 / ms;
    // Per-node time — the eval's "frame time". Independent of the weights for a given
    // arch+quant, so it reports how deep the browser's per-move budget can reach.
    const ns_per_node: f64 = if (res.nodes > 0)
        @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, @floatFromInt(res.nodes))
    else
        0;
    std.debug.print("eval={s} depth={d} bestmove={s} score={d}cp nodes={d} time={d}ms nps={d} ns/node={d:.1}\n", .{
        if (use_nn) "nn" else "hc", res.depth, mv, res.score, res.nodes, ms, nps, ns_per_node,
    });
}
