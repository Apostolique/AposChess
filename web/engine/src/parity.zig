// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2019-2026 Jean-David Moisan
//
// Parity test runner: loads the JS-generated oracle (../engine-parity.json) and
// checks the Zig engine reproduces it field for field. Grows one layer at a time:
//   [x] FEN round-trip  — parseFen → toFen must equal the oracle FEN.
//   [x] move generation — legal-move set (order-independent) + perft counts.
//   [x] game status     — checkmate / stalemate / insufficient / fifty / ongoing.
//   [ ] zobrist hash, eval (against engine-parity.eval.json).
// Exits non-zero on any mismatch so it can gate a build/CI step.

const std = @import("std");
const board = @import("board.zig");
const engine = @import("engine.zig");
const zobrist = @import("zobrist.zig");
const eval = @import("eval.zig");
const nn = @import("nn.zig");

fn moveKey(m: engine.Move, buf: []u8) []const u8 {
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

// Is `key` produced by one of the generated legal moves?
fn hasMove(legal: *const engine.MoveList, key: []const u8) bool {
    var buf: [6]u8 = undefined;
    for (legal.slice()) |m| {
        if (std.mem.eql(u8, key, moveKey(m, &buf))) return true;
    }
    return false;
}

pub fn main(init: std.process.Init) !void {
    // The whole run is one-shot, so allocate from the process arena (freed at exit)
    // — no per-allocation cleanup, and no Debug-allocator leak noise.
    const gpa = init.arena.allocator();
    const io = init.io;

    const data = std.Io.Dir.cwd().readFileAlloc(io, "../engine-parity.json", gpa, .unlimited) catch |e| {
        std.debug.print("Could not read ../engine-parity.json ({s}). Run `npm run parity` in web/ first.\n", .{@errorName(e)});
        std.process.exit(2);
    };
    defer gpa.free(data);

    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, data, .{});
    defer parsed.deinit();

    const positions = parsed.value.object.get("positions").?.array;
    var fen_fail: usize = 0;
    var move_fail: usize = 0;
    var perft_fail: usize = 0;
    var status_fail: usize = 0;
    var hash_fail: usize = 0;
    var zinv_fail: usize = 0;
    var buf: [128]u8 = undefined;

    for (positions.items) |pos| {
        const obj = pos.object;
        const fen = obj.get("fen").?.string;
        const id = obj.get("id").?.string;
        const st = board.parseFen(fen);

        // 1. FEN round-trip.
        const out = board.toFen(&st, &buf);
        if (!std.mem.eql(u8, out, fen)) {
            fen_fail += 1;
            std.debug.print("FEN MISMATCH [{s}]\n  want: {s}\n  got:  {s}\n", .{ id, fen, out });
        }

        // 2. Legal-move set (order-independent): equal count, and every oracle
        //    move is generated (no dup moves, so count+subset => equal sets).
        var legal: engine.MoveList = .{};
        engine.legalMoves(&st, &legal);
        const omoves = obj.get("moves").?.array;
        if (omoves.items.len != legal.len) {
            move_fail += 1;
            std.debug.print("MOVE COUNT [{s}] want {d} got {d}\n", .{ id, omoves.items.len, legal.len });
        } else {
            for (omoves.items) |okv| {
                if (!hasMove(&legal, okv.string)) {
                    move_fail += 1;
                    std.debug.print("MOVE MISSING [{s}] {s}\n", .{ id, okv.string });
                    break;
                }
            }
        }

        // 3. perft, every recorded depth.
        var pit = obj.get("perft").?.object.iterator();
        while (pit.next()) |entry| {
            const depth = std.fmt.parseInt(u32, entry.key_ptr.*, 10) catch continue;
            const want: u64 = @intCast(entry.value_ptr.*.integer);
            var pst = st; // perft make/unmakes in place, restoring it
            const got = engine.perft(&pst, depth);
            if (got != want) {
                perft_fail += 1;
                std.debug.print("PERFT [{s}] depth {d}: want {d} got {d}\n", .{ id, depth, want, got });
            }
        }

        // 4. Game status + check flag.
        const gs = engine.gameStatus(&st);
        const want_status = obj.get("status").?.string;
        const want_check = obj.get("check").?.bool;
        if (!std.mem.eql(u8, engine.statusName(gs.status), want_status) or gs.check != want_check) {
            status_fail += 1;
            std.debug.print("STATUS [{s}] want {s}/check={} got {s}/check={}\n", .{ id, want_status, want_check, engine.statusName(gs.status), gs.check });
        }

        // 5. Zobrist: hash matches the oracle, and the incremental update equals
        //    a from-scratch recompute for every legal move (the JS invariant).
        const want_hash = std.fmt.parseInt(u64, obj.get("hash").?.string, 16) catch 0;
        const h = zobrist.hashOf(&st);
        if (h != want_hash) {
            hash_fail += 1;
            std.debug.print("HASH [{s}] want {x:0>16} got {x:0>16}\n", .{ id, want_hash, h });
        }
        for (legal.slice()) |m| {
            const inc = zobrist.hashAfter(h, &st, m);
            const next = engine.applyMove(&st, m);
            if (inc != zobrist.hashOf(&next)) {
                zinv_fail += 1;
                std.debug.print("ZOBRIST INVARIANT [{s}] move {s}\n", .{ id, moveKey(m, &buf) });
                break;
            }
        }
    }

    // --- eval parity: handcrafted + nn vs the champion-tagged eval oracle ------
    // Loaded separately because engine-parity.eval.json regenerates per champion.
    var eval_hc_fail: usize = 0;
    var eval_hc3_fail: usize = 0;
    var eval_nn_fail: usize = 0;
    var eval_positions: usize = 0;
    {
        const wdata = std.Io.Dir.cwd().readFileAlloc(io, "../src/nn-weights.json", gpa, .unlimited) catch |e| {
            std.debug.print("Could not read ../src/nn-weights.json ({s}).\n", .{@errorName(e)});
            std.process.exit(2);
        };
        const wparsed = try std.json.parseFromSlice(std.json.Value, gpa, wdata, .{});
        const net = try nn.load(gpa, wparsed.value);

        const edata = std.Io.Dir.cwd().readFileAlloc(io, "../engine-parity.eval.json", gpa, .unlimited) catch |e| {
            std.debug.print("Could not read ../engine-parity.eval.json ({s}). Run `npm run parity` in web/.\n", .{@errorName(e)});
            std.process.exit(2);
        };
        const eparsed = try std.json.parseFromSlice(std.json.Value, gpa, edata, .{});
        const epos = eparsed.value.object.get("positions").?.array;
        eval_positions = epos.items.len;

        for (epos.items) |pos| {
            const obj = pos.object;
            const id = obj.get("id").?.string;
            const st = board.parseFen(obj.get("fen").?.string);
            const want_hc = obj.get("evalHc").?.integer;
            const want_hc3 = obj.get("evalHc3").?.integer;
            const want_nn = obj.get("evalNn").?.integer;
            const got_hc: i64 = eval.evalStm(&st.board, st.turn);
            const got_hc3: i64 = eval.evalStmV3(&st.board, st.turn);
            const got_nn: i64 = nn.evaluate(&net, &st.board, st.turn);
            if (got_hc != want_hc) {
                eval_hc_fail += 1;
                std.debug.print("EVAL hc [{s}] want {d} got {d}\n", .{ id, want_hc, got_hc });
            }
            if (got_hc3 != want_hc3) {
                eval_hc3_fail += 1;
                std.debug.print("EVAL hc3 [{s}] want {d} got {d}\n", .{ id, want_hc3, got_hc3 });
            }
            if (@max(got_nn, want_nn) - @min(got_nn, want_nn) > 1) { // ±1 cp tolerance (libm tanh)
                eval_nn_fail += 1;
                std.debug.print("EVAL nn [{s}] want {d} got {d}\n", .{ id, want_nn, got_nn });
            }
        }
    }

    const n = positions.items.len;
    std.debug.print(
        \\parity vs JS oracle ({d} positions):
        \\  FEN round-trip : {d} fail
        \\  legal moves    : {d} fail
        \\  perft          : {d} fail
        \\  game status    : {d} fail
        \\  zobrist hash   : {d} fail
        \\  zobrist incr.  : {d} fail
        \\  eval handcraft : {d} fail ({d} positions)
        \\  eval hc3       : {d} fail
        \\  eval nn (±1cp) : {d} fail
        \\
    , .{ n, fen_fail, move_fail, perft_fail, status_fail, hash_fail, zinv_fail, eval_hc_fail, eval_positions, eval_hc3_fail, eval_nn_fail });

    if (fen_fail + move_fail + perft_fail + status_fail + hash_fail + zinv_fail + eval_hc_fail + eval_hc3_fail + eval_nn_fail != 0) std.process.exit(1);
    std.debug.print("ALL GREEN\n", .{});
}
