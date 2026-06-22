// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2019-2026 Jean-David Moisan
//
// Native match runner behind `npm run match`: plays engine A vs engine B over
// seeded random openings (color-reversed pairs), in parallel across threads, with
// optional SPRT early-stopping, and writes the result-file
// {games,wins,draws,losses,score,elo,llr,sprt} that train:loop reads.
//
//   apos-match --games=800 --depth=4 --eval-a=nn --eval-b=nn \
//     --weights-a=cand.json --weights-b=src/nn-weights.json --sprt --elo1=20 \
//     --result-file=match.json --save-games=../training/data/selfplay.jsonl --jobs=14
// Paths are relative to the current directory (run from web/).
//
// --save-games harvests the played positions as raw training data ({fen,r,g,v,vs}):
// every position carries the WINNING engine's value
// (its direct depth-d search on the plies it moved, the free depth-(d-1) value from the
// previous ply on the plies the loser moved); the loser's own opinion is dropped.

const std = @import("std");
const builtin = @import("builtin");
const board = @import("board.zig");
const engine = @import("engine.zig");
const zobrist = @import("zobrist.zig");
const ai = @import("ai.zig");
const nn = @import("nn.zig");

const State = board.State;

// Native programs write raw UTF-8 bytes to the console, but Windows interprets them
// through the console's *output* code page (often 437/1252, not UTF-8) — so a `±`
// renders as mojibake. Node never hits this because it uses the wide-char console API.
// Switching the output code page to 65001 (UTF-8) at startup makes our UTF-8 output
// render correctly. No-op (and unreferenced) off Windows.
extern "kernel32" fn SetConsoleOutputCP(wCodePageID: std.os.windows.UINT) callconv(.winapi) std.os.windows.BOOL;
fn enableUtf8Console() void {
    if (builtin.os.tag == .windows) _ = SetConsoleOutputCP(65001);
}

// --- Elo / SPRT --------------------------------------------------------------------
fn scoreFromElo(e: f64) f64 {
    return 1.0 / (1.0 + std.math.pow(f64, 10.0, -e / 400.0));
}
fn eloFromScore(p: f64) f64 {
    if (p <= 0) return -800;
    if (p >= 1) return 800;
    return -400.0 * std.math.log10(1.0 / p - 1.0);
}
fn llr(scores: []const f64, elo0: f64, elo1: f64) f64 {
    const n = scores.len;
    if (n < 2) return 0;
    const fn_: f64 = @floatFromInt(n);
    const mu0 = scoreFromElo(elo0);
    const mu1 = scoreFromElo(elo1);
    var s: f64 = 0;
    for (scores) |x| s += x;
    const mean = s / fn_;
    var var_sum: f64 = 0;
    for (scores) |x| var_sum += (x - mean) * (x - mean);
    const variance = @max(var_sum / fn_, 1e-3);
    return ((mu1 - mu0) / variance) * (s - (fn_ * (mu0 + mu1)) / 2.0);
}

// The reported Elo with its 95% confidence interval:
// the ± error bar comes from the standard error of the mean score, mapped through the
// Elo curve. z = 1.96 is the 95% two-sided multiplier.
const EloCI = struct { elo: f64, margin: f64, lo: f64, hi: f64 };
fn eloWithCI(scores: []const f64) EloCI {
    const n = scores.len;
    if (n == 0) return .{ .elo = 0, .margin = 0, .lo = 0, .hi = 0 };
    const fn_: f64 = @floatFromInt(n);
    var s: f64 = 0;
    for (scores) |x| s += x;
    const p = s / fn_;
    var var_sum: f64 = 0;
    for (scores) |x| var_sum += (x - p) * (x - p);
    const se = @sqrt(var_sum / fn_ / fn_); // standard error of the mean score
    const z = 1.96;
    const lo = eloFromScore(p - z * se);
    const hi = eloFromScore(p + z * se);
    return .{ .elo = eloFromScore(p), .margin = (hi - lo) / 2.0, .lo = lo, .hi = hi };
}

// "45s", "3m 02s", "1h 04m" — same unit picking as scripts/fmt.mjs's fmtDur, for the
// live progress line's elapsed/ETA fields.
fn fmtDur(buf: []u8, secs_f: f64) []const u8 {
    const secs: u64 = @intFromFloat(@max(0, @round(secs_f)));
    if (secs < 60) return std.fmt.bufPrint(buf, "{d}s", .{secs}) catch buf[0..0];
    const m = secs / 60;
    if (m < 60) return std.fmt.bufPrint(buf, "{d}m {d:0>2}s", .{ m, secs % 60 }) catch buf[0..0];
    return std.fmt.bufPrint(buf, "{d}h {d:0>2}m", .{ m / 60, m % 60 }) catch buf[0..0];
}

// --- Game harvesting (--save-games) -------------------------------------------------
// One searched position: the position, the result from its side-to-move view, the
// mover's search value, and which engine ('a'/'b') moved. The first ply of a game also
// carries the OTHER engine's value of the opening (it never searches that ply itself).
const PlyRec = struct {
    fen: [128]u8 = undefined,
    fen_len: u8 = 0,
    r: i32 = 0,
    r_turn_white: bool = false, // side-to-move was White (to set `r` once the game ends)
    v: i32 = 0,
    mover: u8 = 0, // 'a' or 'b'
    has_other: bool = false,
    v_other: i32 = 0,
    mover_other: u8 = 0,

    fn fenSlice(self: *const PlyRec) []const u8 {
        return self.fen[0..self.fen_len];
    }
};
const Game = struct {
    pair: usize,
    color: u8, // 'w' (A is White) or 'b' (A is Black)
    recs: std.ArrayList(PlyRec),
};

// Per-side search budget: a fixed depth (depth > 0) OR a per-move time budget (depth ==
// 0, search to `movetime` ms). Engine A and B each carry their own, so the rank gauntlet
// can pit cheap fixed-depth contenders against a deep stable anchor (--depth-b).
const Budget = struct { depth: u32, movetime: i64 };

fn searchBudget(s: *ai.Searcher, st: *const State, b: Budget, seen: []const u64, no_tt: bool) ai.Result {
    const d: u32 = if (b.depth > 0) b.depth else 99;
    const ms: i64 = if (b.depth > 0) 0 else b.movetime;
    return if (no_tt) s.chooseMoveNoTT(st, d, ms, seen) else s.chooseMove(st, d, ms, seen);
}

fn randomOpening(rng: std.Random, plies: u32) State {
    var st = board.newGameState();
    var i: u32 = 0;
    while (i < plies) : (i += 1) {
        var moves: engine.MoveList = .{};
        engine.legalMoves(&st, &moves);
        if (moves.len == 0) break;
        st = engine.applyMove(&st, moves.items[rng.intRangeLessThan(usize, 0, moves.len)]);
        if (engine.gameStatus(&st).status != .ongoing) return board.newGameState();
    }
    return st;
}

// Play one game. `white_is_a` decides which engine has White. Returns +1 white win /
// 0 draw / -1 black win. When `recs` is non-null (--save-games) it collects one record
// per searched position (the harvest path).
fn playGame(
    white: *ai.Searcher,
    black: *ai.Searcher,
    white_is_a: bool,
    start: State,
    bw: Budget, // White's search budget
    bb: Budget, // Black's search budget
    max_plies: u32,
    alloc: std.mem.Allocator,
    nodes: *u64,
    recs: ?*std.ArrayList(PlyRec),
) !i32 {
    var seen: std.ArrayList(u64) = .empty;
    defer seen.deinit(alloc);
    var st = start;
    var result_white: i32 = 0;
    var plies: u32 = 0;
    while (plies < max_plies) : (plies += 1) {
        const gs = engine.gameStatus(&st);
        if (gs.status == .checkmate) {
            result_white = if (st.turn == .white) -1 else 1;
            break;
        }
        if (gs.status != .ongoing) break;

        const a_to_move = (st.turn == .white) == white_is_a;
        const mover = if (st.turn == .white) white else black;
        const other = if (st.turn == .white) black else white;
        const mover_b = if (st.turn == .white) bw else bb;
        const other_b = if (st.turn == .white) bb else bw;
        const res = searchBudget(mover, &st, mover_b, seen.items, false);
        nodes.* += res.nodes;
        const m = res.move orelse break;

        if (recs) |list| {
            var rec = PlyRec{
                .v = res.score,
                .mover = if (a_to_move) 'a' else 'b',
            };
            var fenbuf: [128]u8 = undefined;
            const fen = board.toFen(&st, &fenbuf);
            @memcpy(rec.fen[0..fen.len], fen);
            rec.fen_len = @intCast(fen.len);
            // r from the side-to-move's view is set once the game's outcome is known.
            rec.r_turn_white = st.turn == .white;
            if (plies == 0) {
                // The side NOT to move never searches this position in the game; value it
                // once now (TT-free so the probe can't perturb that engine's real play) so
                // the harvester has a real depth-d value whichever engine ends up winning.
                rec.v_other = searchBudget(other, &st, other_b, &.{}, true).score;
                rec.mover_other = if (a_to_move) 'b' else 'a';
                rec.has_other = true;
            }
            try list.append(alloc, rec);
        }

        try seen.append(alloc, zobrist.hashOf(&st));
        const next = engine.applyMove(&st, m);
        if (next.halfmove == 0) seen.clearRetainingCapacity();
        st = next;
    }

    if (recs) |list| {
        for (list.items) |*rec| rec.r = if (rec.r_turn_white) result_white else -result_white;
    }
    return result_white;
}

const Cfg = struct {
    budget_a: Budget,
    budget_b: Budget,
    openings: u32,
    max_plies: u32,
    seed: u64,
    kind_a: ai.EvalKind,
    kind_b: ai.EvalKind,
    net_a: ?*const nn.Net,
    net_b: ?*const nn.Net,
    save_games: bool,
    io: std.Io,
};

const Shared = struct {
    mutex: std.Io.Mutex = .init,
    next_pair: usize = 0,
    total_pairs: usize,
    scores: *std.ArrayList(f64),
    games: *std.ArrayList(Game), // harvested games (--save-games)
    alloc: std.mem.Allocator,
    nodes: u64 = 0,
    stop: bool = false,
    decided: ?[]const u8 = null,
    sprt: bool,
    elo0: f64,
    elo1: f64,
    upper: f64,
    lower: f64,
    cfg: Cfg,
    t0_ns: i128 = 0, // match start, for live elapsed/ETA
    live_len: usize = 0, // chars in the current in-place status line (for repaint padding)
};

// In-place status line on stderr (carriage-return repaint), mirroring fmt.mjs's
// liveStatus: write `\r` + text, padding over any longer previous content so stale
// characters never linger. Called under sh.mutex so the worker threads never interleave.
fn printLive(sh: *Shared, text: []const u8) void {
    var buf: [384]u8 = undefined;
    var n: usize = 0;
    buf[n] = '\r';
    n += 1;
    const t = text[0..@min(text.len, 300)];
    @memcpy(buf[n .. n + t.len], t);
    n += t.len;
    if (t.len < sh.live_len) {
        const pad = @min(sh.live_len - t.len, buf.len - n);
        @memset(buf[n .. n + pad], ' ');
        n += pad;
    }
    std.debug.print("{s}", .{buf[0..n]});
    sh.live_len = t.len;
}

// Erase the live line so a permanent milestone line can be printed without leftovers.
fn clearLive(sh: *Shared) void {
    if (sh.live_len == 0) return;
    var buf: [384]u8 = undefined;
    var n: usize = 0;
    buf[n] = '\r';
    n += 1;
    const pad = @min(sh.live_len, buf.len - 2);
    @memset(buf[n .. n + pad], ' ');
    n += pad;
    buf[n] = '\r';
    n += 1;
    std.debug.print("{s}", .{buf[0..n]});
    sh.live_len = 0;
}

fn worker(sh: *Shared) void {
    const pa = std.heap.page_allocator;
    var sa = ai.Searcher.init(pa, sh.cfg.io, sh.cfg.kind_a, sh.cfg.net_a, 1) catch return;
    defer sa.deinit();
    var sb = ai.Searcher.init(pa, sh.cfg.io, sh.cfg.kind_b, sh.cfg.net_b, 1) catch return;
    defer sb.deinit();

    while (true) {
        sh.mutex.lockUncancelable(sh.cfg.io);
        const done = sh.stop or sh.next_pair >= sh.total_pairs;
        const pair = sh.next_pair;
        if (!done) sh.next_pair += 1;
        sh.mutex.unlock(sh.cfg.io);
        if (done) break;

        var op_prng = std.Random.DefaultPrng.init(sh.cfg.seed +% pair);
        const opening = randomOpening(op_prng.random(), sh.cfg.openings);
        var nodes: u64 = 0;

        // Collect per-game records only when harvesting.
        var recs_w: std.ArrayList(PlyRec) = .empty;
        var recs_b: std.ArrayList(PlyRec) = .empty;
        const pw: ?*std.ArrayList(PlyRec) = if (sh.cfg.save_games) &recs_w else null;
        const pb: ?*std.ArrayList(PlyRec) = if (sh.cfg.save_games) &recs_b else null;

        // Reseed per game so openings/variety are a function of the pair index, not
        // thread timing (the persistent TT still makes exact games order-sensitive).
        sa.reseed(sh.cfg.seed +% pair *% 4 +% 0);
        sb.reseed(sh.cfg.seed +% pair *% 4 +% 1);
        // A = White: White's budget is A's, Black's is B's.
        const r1 = playGame(&sa, &sb, true, opening, sh.cfg.budget_a, sh.cfg.budget_b, sh.cfg.max_plies, pa, &nodes, pw) catch 0;
        sb.reseed(sh.cfg.seed +% pair *% 4 +% 2);
        sa.reseed(sh.cfg.seed +% pair *% 4 +% 3);
        // A = Black: White's budget is B's, Black's is A's.
        const r2 = playGame(&sb, &sa, false, opening, sh.cfg.budget_b, sh.cfg.budget_a, sh.cfg.max_plies, pa, &nodes, pb) catch 0;

        const s1: f64 = if (r1 > 0) 1 else if (r1 < 0) 0 else 0.5; // A is white
        const s2: f64 = if (r2 < 0) 1 else if (r2 > 0) 0 else 0.5; // A is black

        sh.mutex.lockUncancelable(sh.cfg.io);
        sh.scores.append(sh.alloc, s1) catch {};
        sh.scores.append(sh.alloc, s2) catch {};
        sh.nodes += nodes;
        if (sh.cfg.save_games) {
            sh.games.append(sh.alloc, .{ .pair = pair, .color = 'w', .recs = recs_w }) catch {};
            sh.games.append(sh.alloc, .{ .pair = pair, .color = 'b', .recs = recs_b }) catch {};
        }
        if (sh.sprt and sh.decided == null and sh.scores.items.len >= 16) {
            const l = llr(sh.scores.items, sh.elo0, sh.elo1);
            if (l >= sh.upper) {
                sh.decided = "H1";
                sh.stop = true;
            } else if (l <= sh.lower) {
                sh.decided = "H0";
                sh.stop = true;
            }
        }
        // Two-tier progress. Printed under the mutex so the
        // worker threads' output never interleaves.
        const ng = sh.scores.items.len;
        var w: usize = 0;
        var dr: usize = 0;
        var ls: usize = 0;
        var ssum: f64 = 0;
        for (sh.scores.items) |sc| {
            ssum += sc;
            if (sc == 1) w += 1 else if (sc == 0.5) dr += 1 else ls += 1;
        }
        const pp = ssum / @as(f64, @floatFromInt(ng));
        const total: usize = sh.total_pairs * 2;
        const now_ns: i128 = @intCast(std.Io.Clock.now(.awake, sh.cfg.io).nanoseconds);
        const elapsed_s: f64 = @as(f64, @floatFromInt(now_ns - sh.t0_ns)) / 1e9;

        // 1) Live, in-place line refreshed after EVERY finished pair, so a long match
        //    never looks frozen between milestones. Lightweight: counts + score + elapsed
        //    (+ ETA, + LLR with --sprt); no Elo/CI — that's the milestone's job.
        var ebuf: [16]u8 = undefined;
        var etaseg: []const u8 = "";
        var etastore: [40]u8 = undefined;
        if (sh.decided == null and ng < total) {
            const eta_s = elapsed_s / @as(f64, @floatFromInt(ng)) * @as(f64, @floatFromInt(total - ng));
            var b2: [16]u8 = undefined;
            etaseg = std.fmt.bufPrint(&etastore, " | ETA {s}", .{fmtDur(&b2, eta_s)}) catch "";
        }
        var llrseg: []const u8 = "";
        var llrstore: [64]u8 = undefined;
        if (sh.sprt) {
            llrseg = std.fmt.bufPrint(&llrstore, " | LLR {d:.2} [{d:.2}, {d:.2}]", .{
                llr(sh.scores.items, sh.elo0, sh.elo1), sh.lower, sh.upper,
            }) catch "";
        }
        var livebuf: [320]u8 = undefined;
        const live = std.fmt.bufPrint(&livebuf, "  game {d}/{d} | A +{d} ={d} -{d} | {d:.1}% | {s} elapsed{s}{s}", .{
            ng, total, w, dr, ls, pp * 100, fmtDur(&ebuf, elapsed_s), etaseg, llrseg,
        }) catch livebuf[0..0];
        printLive(sh, live);

        // 2) Committed milestone every 5 pairs (or on an SPRT decision): clear the live
        //    line and print a permanent snapshot with the full Elo ± 95% CI.
        if (sh.decided != null or ng % 10 == 0) {
            clearLive(sh);
            const ci = eloWithCI(sh.scores.items);
            const sign = if (ci.elo >= 0) "+" else "";
            if (sh.sprt) {
                std.debug.print("  after {d} games  A: +{d} ={d} -{d}  score {d:.1}%  Elo {s}{d:.0} ± {d:.0}  95% CI [{d:.0}, {d:.0}]  LLR {d:.2} [{d:.2}, {d:.2}]\n", .{
                    ng, w, dr, ls, pp * 100, sign, ci.elo, ci.margin, ci.lo, ci.hi, llr(sh.scores.items, sh.elo0, sh.elo1), sh.lower, sh.upper,
                });
            } else {
                std.debug.print("  after {d} games  A: +{d} ={d} -{d}  score {d:.1}%  Elo {s}{d:.0} ± {d:.0}  95% CI [{d:.0}, {d:.0}]\n", .{
                    ng, w, dr, ls, pp * 100, sign, ci.elo, ci.margin, ci.lo, ci.hi,
                });
            }
        }
        sh.mutex.unlock(sh.cfg.io);
    }
}

fn loadNet(io: std.Io, gpa: std.mem.Allocator, path: []const u8) !nn.Net {
    const data = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .unlimited);
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, data, .{});
    return try nn.load(gpa, parsed.value);
}

fn argStr(arg: []const u8, key: []const u8) ?[]const u8 {
    if (std.mem.startsWith(u8, arg, key)) return arg[key.len..];
    return null;
}

// Short sha1 content hash of the weights file = its nn version (mirrors vtag.mjs):
// the first 6 lowercase-hex chars. "??????" if unreadable.
fn weightsHash(io: std.Io, gpa: std.mem.Allocator, path: []const u8) [6]u8 {
    const data = std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .unlimited) catch
        return .{ '?', '?', '?', '?', '?', '?' };
    var digest: [20]u8 = undefined;
    std.crypto.hash.Sha1.hash(data, &digest, .{});
    const hexchars = "0123456789abcdef";
    var out: [6]u8 = undefined;
    inline for (0..3) |i| {
        out[i * 2] = hexchars[digest[i] >> 4];
        out[i * 2 + 1] = hexchars[digest[i] & 0xf];
    }
    return out;
}

// Provenance tag "<engine><depth>@<version>" (vtag.mjs) into `buf`. depth == 0 means a
// time-based search, marked 't' (matching vtag.mjs).
fn vtagFmt(buf: []u8, kind: ai.EvalKind, depth: u32, io: std.Io, gpa: std.mem.Allocator, weights: []const u8) []const u8 {
    if (kind == .nn) {
        const h = weightsHash(io, gpa, weights);
        if (depth == 0) return std.fmt.bufPrint(buf, "nnt@{s}", .{h}) catch unreachable;
        return std.fmt.bufPrint(buf, "nn{d}@{s}", .{ depth, h }) catch unreachable;
    }
    if (depth == 0) return std.fmt.bufPrint(buf, "hct@{d}", .{ai.HC_VERSION}) catch unreachable;
    return std.fmt.bufPrint(buf, "hc{d}@{d}", .{ depth, ai.HC_VERSION }) catch unreachable;
}

fn appendBase36(out: *std.ArrayList(u8), alloc: std.mem.Allocator, value: u64) !void {
    if (value == 0) {
        try out.append(alloc, '0');
        return;
    }
    var tmp: [13]u8 = undefined;
    var v = value;
    var i: usize = tmp.len;
    const digits = "0123456789abcdefghijklmnopqrstuvwxyz";
    while (v > 0) {
        i -= 1;
        tmp[i] = digits[@intCast(v % 36)];
        v /= 36;
    }
    try out.appendSlice(alloc, tmp[i..]);
}

fn appendInt(out: *std.ArrayList(u8), alloc: std.mem.Allocator, v: i64) !void {
    var buf: [24]u8 = undefined;
    try out.appendSlice(alloc, std.fmt.bufPrint(&buf, "{d}", .{v}) catch unreachable);
}

// Append the harvested games as raw training data (the gate harvest train:loop folds in).
fn writeHarvest(sh: *Shared, gpa: std.mem.Allocator, io: std.Io, path: []const u8, p: f64, depth_a: u32, depth_b: u32, eval_a: ai.EvalKind, eval_b: ai.EvalKind, weights_a: []const u8, weights_b: []const u8) !void {
    const winner: u8 = if (p > 0.5) 'a' else 'b';
    const win_kind = if (winner == 'a') eval_a else eval_b;
    const win_weights = if (winner == 'a') weights_a else weights_b;
    // The winner's value is tagged with ITS OWN search depth (contenders and the anchor can
    // differ — e.g. a depth-6 anchor beating depth-4 contenders harvests as nn6@/hc6@).
    const win_depth = if (winner == 'a') depth_a else depth_b;
    var vtag_buf: [40]u8 = undefined;
    const w_vtag = vtagFmt(&vtag_buf, win_kind, win_depth, io, gpa, win_weights);
    var vtag_prev_buf: [40]u8 = undefined;
    const w_vtag_prev: ?[]const u8 = if (win_depth >= 2)
        vtagFmt(&vtag_prev_buf, win_kind, win_depth - 1, io, gpa, win_weights)
    else
        null;

    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);
    var n_pos: usize = 0;
    var n_kept: usize = 0;
    var n_derived: usize = 0;
    for (sh.games.items) |g| {
        const recs = g.recs.items;
        for (recs, 0..) |rec, i| {
            try out.appendSlice(gpa, "{\"fen\":\"");
            try out.appendSlice(gpa, rec.fenSlice());
            try out.appendSlice(gpa, "\",\"r\":");
            try appendInt(&out, gpa, rec.r);
            try out.appendSlice(gpa, ",\"g\":\"m");
            try appendBase36(&out, gpa, sh.cfg.seed);
            try out.append(gpa, '-');
            try appendInt(&out, gpa, @as(i64, @intCast(g.pair)));
            try out.append(gpa, g.color); // 'w' or 'b'
            try out.append(gpa, '"');
            // Value: winner's direct (its plies), its off-turn opening probe (ply 0), or
            // the derived depth-(d-1) value from the previous ply (loser's plies).
            if (rec.mover == winner) {
                try out.appendSlice(gpa, ",\"v\":");
                try appendInt(&out, gpa, rec.v);
                try out.appendSlice(gpa, ",\"vs\":\"");
                try out.appendSlice(gpa, w_vtag);
                try out.append(gpa, '"');
                n_kept += 1;
            } else if (i == 0 and rec.has_other and rec.mover_other == winner) {
                try out.appendSlice(gpa, ",\"v\":");
                try appendInt(&out, gpa, rec.v_other);
                try out.appendSlice(gpa, ",\"vs\":\"");
                try out.appendSlice(gpa, w_vtag);
                try out.append(gpa, '"');
                n_kept += 1;
            } else if (w_vtag_prev != null and i > 0 and recs[i - 1].mover == winner) {
                try out.appendSlice(gpa, ",\"v\":");
                try appendInt(&out, gpa, -recs[i - 1].v);
                try out.appendSlice(gpa, ",\"vs\":\"");
                try out.appendSlice(gpa, w_vtag_prev.?);
                try out.append(gpa, '"');
                n_derived += 1;
            }
            try out.appendSlice(gpa, "}\n");
            n_pos += 1;
        }
    }

    if (std.fs.path.dirname(path)) |dir| std.Io.Dir.cwd().createDirPath(io, dir) catch {};
    var file = try std.Io.Dir.cwd().createFile(io, path, .{ .truncate = false, .read = true });
    defer file.close(io);
    const offset: u64 = (file.stat(io) catch unreachable).size;
    try file.writePositionalAll(io, out.items, offset);
    if (w_vtag_prev) |wp| {
        std.debug.print("Saved {d} positions from {d} games to {s} (winner {c}: {d} direct {s}, {d} derived {s}).\n", .{
            n_pos, sh.games.items.len, path, std.ascii.toUpper(winner), n_kept, w_vtag, n_derived, wp,
        });
    } else {
        std.debug.print("Saved {d} positions from {d} games to {s} (winner {c}: {d} direct {s}).\n", .{
            n_pos, sh.games.items.len, path, std.ascii.toUpper(winner), n_kept, w_vtag,
        });
    }
}

pub fn main(init: std.process.Init) !void {
    enableUtf8Console(); // so `±` and friends render instead of mojibake on Windows
    const gpa = init.arena.allocator();
    const io = init.io;

    var games: u32 = 100;
    // Per-side search budget: --depth sets A (and B unless --depth-b);
    // --movetime sets A's time budget (and B's unless --movetime-b). depth==0 => use movetime.
    var depth_a: ?u32 = null;
    var depth_b_opt: ?u32 = null;
    var movetime_a: i64 = 50; // selfplay's default think time when no depth is given
    var movetime_b_opt: ?i64 = null;
    var seed: u64 = 1;
    var openings: u32 = 6; // matches the JS `npm run match` default
    var maxmoves: u32 = 200;
    var jobs: usize = std.Thread.getCpuCount() catch 1;
    var eval_a: ai.EvalKind = .handcrafted;
    var eval_b: ai.EvalKind = .handcrafted;
    var weights_a: []const u8 = "src/nn-weights.json";
    var weights_b: []const u8 = "src/nn-weights.json";
    var sprt = false;
    var elo0: f64 = 0;
    var elo1: f64 = 15;
    var alpha: f64 = 0.05;
    var beta: f64 = 0.05;
    var result_file: ?[]const u8 = null;
    var save_games: ?[]const u8 = null;

    const argv = try init.minimal.args.toSlice(gpa);
    for (argv[1..]) |arg| {
        if (argStr(arg, "--games=")) |v| games = std.fmt.parseInt(u32, v, 10) catch games;
        if (argStr(arg, "--depth=")) |v| depth_a = std.fmt.parseInt(u32, v, 10) catch depth_a;
        if (argStr(arg, "--depth-b=")) |v| depth_b_opt = std.fmt.parseInt(u32, v, 10) catch depth_b_opt;
        if (argStr(arg, "--movetime=")) |v| movetime_a = std.fmt.parseInt(i64, v, 10) catch movetime_a;
        if (argStr(arg, "--movetime-b=")) |v| movetime_b_opt = std.fmt.parseInt(i64, v, 10) catch movetime_b_opt;
        if (argStr(arg, "--seed=")) |v| seed = std.fmt.parseInt(u64, v, 10) catch seed;
        if (argStr(arg, "--openings=")) |v| openings = std.fmt.parseInt(u32, v, 10) catch openings;
        if (argStr(arg, "--maxmoves=")) |v| maxmoves = std.fmt.parseInt(u32, v, 10) catch maxmoves;
        if (argStr(arg, "--jobs=")) |v| jobs = std.fmt.parseInt(usize, v, 10) catch jobs;
        if (argStr(arg, "--eval-a=")) |v| eval_a = if (std.mem.eql(u8, v, "nn")) .nn else .handcrafted;
        if (argStr(arg, "--eval-b=")) |v| eval_b = if (std.mem.eql(u8, v, "nn")) .nn else .handcrafted;
        if (argStr(arg, "--weights-a=")) |v| weights_a = v;
        if (argStr(arg, "--weights-b=")) |v| weights_b = v;
        if (std.mem.eql(u8, arg, "--sprt")) sprt = true;
        if (argStr(arg, "--elo0=")) |v| elo0 = std.fmt.parseFloat(f64, v) catch elo0;
        if (argStr(arg, "--elo1=")) |v| elo1 = std.fmt.parseFloat(f64, v) catch elo1;
        if (argStr(arg, "--alpha=")) |v| alpha = std.fmt.parseFloat(f64, v) catch alpha;
        if (argStr(arg, "--beta=")) |v| beta = std.fmt.parseFloat(f64, v) catch beta;
        if (argStr(arg, "--result-file=")) |v| result_file = v;
        if (argStr(arg, "--save-games=")) |v| save_games = v;
    }
    if (jobs < 1) jobs = 1;

    // Resolve each side's budget. A fixed depth wins over movetime; B inherits A's depth
    // when only --depth was given, A's movetime when only --movetime was given.
    const budget_a: Budget = if (depth_a) |d| .{ .depth = d, .movetime = 0 } else .{ .depth = 0, .movetime = movetime_a };
    const budget_b: Budget = if (depth_b_opt orelse depth_a) |d|
        .{ .depth = d, .movetime = 0 }
    else
        .{ .depth = 0, .movetime = movetime_b_opt orelse movetime_a };

    const net_a: ?*const nn.Net = if (eval_a == .nn) blk: {
        const n = try gpa.create(nn.Net);
        n.* = try loadNet(io, gpa, weights_a);
        break :blk n;
    } else null;
    const net_b: ?*const nn.Net = if (eval_b == .nn) blk: {
        const n = try gpa.create(nn.Net);
        n.* = try loadNet(io, gpa, weights_b);
        break :blk n;
    } else null;

    var scores: std.ArrayList(f64) = .empty;
    var harvest: std.ArrayList(Game) = .empty;
    var shared = Shared{
        .total_pairs = (games + 1) / 2,
        .scores = &scores,
        .games = &harvest,
        .alloc = gpa,
        .sprt = sprt,
        .elo0 = elo0,
        .elo1 = elo1,
        .upper = @log((1 - beta) / alpha),
        .lower = @log(beta / (1 - alpha)),
        .cfg = .{
            .budget_a = budget_a,
            .budget_b = budget_b,
            .openings = openings,
            .max_plies = maxmoves,
            .seed = seed,
            .kind_a = eval_a,
            .kind_b = eval_b,
            .net_a = net_a,
            .net_b = net_b,
            .save_games = save_games != null,
            .io = io,
        },
    };

    const t0 = std.Io.Clock.now(.awake, io).nanoseconds;
    shared.t0_ns = @intCast(t0); // so the live progress line can show elapsed/ETA
    const threads = try gpa.alloc(std.Thread, jobs);
    for (threads) |*t| t.* = try std.Thread.spawn(.{}, worker, .{&shared});
    for (threads) |t| t.join();
    const ms: u64 = @intCast(@max(1, @divTrunc(std.Io.Clock.now(.awake, io).nanoseconds - t0, 1_000_000)));

    // --- report -------------------------------------------------------------------
    const n = scores.items.len;
    var sum: f64 = 0;
    var wins: usize = 0;
    var draws: usize = 0;
    var losses: usize = 0;
    for (scores.items) |s| {
        sum += s;
        if (s == 1) wins += 1 else if (s == 0.5) draws += 1 else losses += 1;
    }
    const p = if (n > 0) sum / @as(f64, @floatFromInt(n)) else 0;
    const elo = eloFromScore(p);
    const ci = eloWithCI(scores.items);
    const sign = if (ci.elo >= 0) "+" else "";
    const verdict = if (sprt) (shared.decided orelse "inconclusive") else "n/a";
    std.debug.print("A vs B: {d} games | +{d} ={d} -{d} | score {d:.1}% | Elo {s}{d:.0} ± {d:.0} (95% CI [{d:.0}, {d:.0}]) | SPRT {s} | nodes {d} nps {d}\n", .{
        n, wins, draws, losses, p * 100, sign, ci.elo, ci.margin, ci.lo, ci.hi, verdict, shared.nodes, shared.nodes * 1000 / ms,
    });

    if (save_games) |sg| {
        if (shared.games.items.len > 0) try writeHarvest(&shared, gpa, io, sg, p, budget_a.depth, budget_b.depth, eval_a, eval_b, weights_a, weights_b);
    }

    if (result_file) |rf| {
        var buf: [512]u8 = undefined;
        const sprt_field = if (sprt) shared.decided orelse "inconclusive" else null;
        const json = if (sprt_field) |sf|
            try std.fmt.bufPrint(&buf,
                \\{{"games":{d},"wins":{d},"draws":{d},"losses":{d},"score":{d},"elo":{d},"llr":{d},"sprt":"{s}"}}
            , .{ n, wins, draws, losses, p, elo, llr(scores.items, elo0, elo1), sf })
        else
            try std.fmt.bufPrint(&buf,
                \\{{"games":{d},"wins":{d},"draws":{d},"losses":{d},"score":{d},"elo":{d},"llr":null,"sprt":null}}
            , .{ n, wins, draws, losses, p, elo });
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = rf, .data = json });
    }
}
