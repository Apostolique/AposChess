// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2019-2026 Jean-David Moisan
//
// Native self-play data generator behind `npm run train:gen`. Plays whole games from
// seeded random openings in parallel across threads and appends one JSONL line per GAME
// to the dataset (game-primary; see scripts/gameRecord.mjs) — far smaller than the old
// per-position FEN lines, and it records WHO PLAYED, not only who labeled each position.
//
//   apos-gen --games=200 --depth=6 --eval=nn --openings=8 [--opening-topk=N] \
//     [--movetime=MS] [--maxmoves=200] [--out=../training/data/selfplay.jsonl] \
//     [--seed=S] [--jobs=N]
// Paths are relative to the current directory (run from web/). With --eval=nn the
// teacher is the champion at src/nn-weights.json.
//
// Each line is one game record (features come later via scripts/featurize.mjs):
//   g        "<seed-base36>-<index>" game id (groups a game for the train/val split)
//   players  {w,b} engine×depth tag of each side — both the same self-play eval here
//   r        WHITE-view result (+1 white win / 0 / -1); per-position view derived on replay
//   moves    ["e2e4",...] compact from+to[promo], connecting consecutive recorded positions
//   v        per recorded position, the search value (cp, side-to-move-relative); len = moves+1
//   vs       provenance tag "<engine><depth>@<version>" (mirrors scripts/vtag.mjs)

const std = @import("std");
const board = @import("board.zig");
const engine = @import("engine.zig");
const zobrist = @import("zobrist.zig");
const ai = @import("ai.zig");
const nn = @import("nn.zig");

const State = board.State;
const NN_WEIGHTS = "src/nn-weights.json"; // the champion (--eval=nn)

const Cfg = struct {
    games: u64,
    depth: u32, // 0 => use movetime instead
    movetime: i64,
    openings: u32,
    opening_topk: u32,
    maxmoves: u32,
    kind: ai.EvalKind,
    net: ?*const nn.Net,
    seed: u64,
    vtag: []const u8,
    io: std.Io,
};

const Shared = struct {
    mutex: std.Io.Mutex = .init,
    next_game: u64 = 0,
    cfg: Cfg,
    file: std.Io.File,
    offset: u64, // next append position (initial file size); advanced under the mutex
    // progress
    done_games: u64 = 0,
    total_positions: u64 = 0,
    wins: u64 = 0,
    draws: u64 = 0,
    losses: u64 = 0,
    nodes: u64 = 0,
    t0_ns: i128 = 0, // generation start, for live elapsed/ETA
    live_len: usize = 0, // chars in the current in-place status line (for repaint padding)
    heartbeat_stop: bool = false, // workers done -> the heartbeat thread should exit
};

// "45s", "3m 02s", "1h 04m" — same unit picking as scripts/fmt.mjs's fmtDur, for the live
// progress line's elapsed/ETA fields.
fn fmtDur(buf: []u8, secs_f: f64) []const u8 {
    const secs: u64 = @intFromFloat(@max(0, @round(secs_f)));
    if (secs < 60) return std.fmt.bufPrint(buf, "{d}s", .{secs}) catch buf[0..0];
    const m = secs / 60;
    if (m < 60) return std.fmt.bufPrint(buf, "{d}m {d:0>2}s", .{ m, secs % 60 }) catch buf[0..0];
    return std.fmt.bufPrint(buf, "{d}h {d:0>2}m", .{ m / 60, m % 60 }) catch buf[0..0];
}

// In-place status line on stderr (carriage-return repaint), mirroring fmt.mjs's liveStatus:
// write `\r` + text, padding over any longer previous content so stale characters never
// linger. Called under sh.mutex so worker threads + the heartbeat never interleave.
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

// Erase the live line so the final summary line prints without leftovers.
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

// Repaint the in-place live status line from the current shared counters. MUST be called with
// sh.mutex held. Shared by the per-game worker update AND the heartbeat thread, so the
// elapsed/ETA clock and the in-flight count keep advancing even while games are still being
// searched — a depth-8 generation batch otherwise shows nothing between "Generating…" and
// "Done" for many minutes. Returns without touching the line once every game is finished.
fn paintLive(sh: *Shared) void {
    const done = sh.done_games;
    const total = sh.cfg.games;
    if (done >= total) return;
    const now_ns: i128 = @intCast(std.Io.Clock.now(.awake, sh.cfg.io).nanoseconds);
    const elapsed_s: f64 = @as(f64, @floatFromInt(now_ns - sh.t0_ns)) / 1e9;

    // In-flight games = dispatched − finished ≈ busy workers. During the opening stretch
    // (done still 0) this is the key liveness signal: it shows N games are actively searching.
    const running = if (sh.next_game > done) sh.next_game - done else 0;

    var etaseg: []const u8 = "";
    var etastore: [40]u8 = undefined;
    if (done > 0) {
        const eta_s = elapsed_s / @as(f64, @floatFromInt(done)) * @as(f64, @floatFromInt(total - done));
        var b2: [16]u8 = undefined;
        etaseg = std.fmt.bufPrint(&etastore, " | ETA {s}", .{fmtDur(&b2, eta_s)}) catch "";
    }
    var ebuf: [16]u8 = undefined;
    var livebuf: [320]u8 = undefined;
    const live = std.fmt.bufPrint(&livebuf, "  game {d}/{d} | {d} running | W{d} B{d} D{d} | {d} pos | {s} elapsed{s}", .{
        done, total, running, sh.wins, sh.losses, sh.draws, sh.total_positions, fmtDur(&ebuf, elapsed_s), etaseg,
    }) catch livebuf[0..0];
    printLive(sh, live);
}

// Heartbeat thread: repaint the live status line about once a second so a long generation
// batch never looks frozen between game completions. It only holds the mutex briefly to
// repaint; exits once main signals heartbeat_stop after the workers join.
fn heartbeat(sh: *Shared) void {
    while (true) {
        sh.cfg.io.sleep(std.Io.Duration.fromMilliseconds(1000), .awake) catch {};
        sh.mutex.lockUncancelable(sh.cfg.io);
        const stop = sh.heartbeat_stop;
        if (!stop) paintLive(sh);
        sh.mutex.unlock(sh.cfg.io);
        if (stop) break;
    }
}

// Per-game seed: decorrelate the base seed with the game index so each game is
// reproducible regardless of worker / job count.
fn gameSeed(seed: u64, g: u64) u64 {
    return (seed ^ (g +% 1) *% 0x9e3779b1) & 0xffffffff;
}

fn appendBase36(buf: *std.ArrayList(u8), alloc: std.mem.Allocator, value: u64) !void {
    if (value == 0) {
        try buf.append(alloc, '0');
        return;
    }
    var tmp: [13]u8 = undefined; // u64 in base36 fits in 13 digits
    var v = value;
    var i: usize = tmp.len;
    const digits = "0123456789abcdefghijklmnopqrstuvwxyz";
    while (v > 0) {
        i -= 1;
        tmp[i] = digits[@intCast(v % 36)];
        v /= 36;
    }
    try buf.appendSlice(alloc, tmp[i..]);
}

// Append a compact move token "<from><to>[promo]" (e.g. "e2e4", "e7e8q") — the dataset's
// move encoding (mirrors scripts/gameRecord.mjs encodeMove; no '=' before the promo char).
// Castling is the king's from→to, which uniquely identifies it on replay.
fn appendMoveToken(out: *std.ArrayList(u8), alloc: std.mem.Allocator, m: engine.Move) !void {
    const fr = board.squareName(m.from);
    const to = board.squareName(m.to);
    try out.appendSlice(alloc, &fr);
    try out.appendSlice(alloc, &to);
    if (m.promotion) |role| try out.append(alloc, board.charFromRole(role));
}

// Play one game with the given searcher (same eval on both sides — self-play), append
// the game's JSONL record to `out`. Returns the White-view result (+1/0/-1).
fn playGame(s: *ai.Searcher, cfg: *const Cfg, g: u64, alloc: std.mem.Allocator, out: *std.ArrayList(u8), positions: *u64, nodes: *u64) !i32 {
    var prng = std.Random.DefaultPrng.init(gameSeed(cfg.seed, g));
    const rng = prng.random();

    var states: std.ArrayList(State) = .empty;
    defer states.deinit(alloc);
    var scores: std.ArrayList(i32) = .empty; // parallel to states (search value, cp/stm)
    defer scores.deinit(alloc);
    var moves_played: std.ArrayList(engine.Move) = .empty; // the move applied from each state
    defer moves_played.deinit(alloc);
    var seen: std.ArrayList(u64) = .empty; // Zobrist of every earlier position
    defer seen.deinit(alloc);

    var st = board.newGameState();
    var result: i32 = 0;
    const max_ms: i64 = if (cfg.depth != 0) 0 else cfg.movetime;
    const depth: u32 = if (cfg.depth != 0) cfg.depth else 99;

    var ply: u32 = 0;
    while (ply < cfg.maxmoves) : (ply += 1) {
        const gs = engine.gameStatus(&st);
        if (gs.status == .checkmate) {
            result = if (st.turn == .white) -1 else 1;
            break;
        }
        if (gs.status != .ongoing) break; // stalemate / draw -> 0

        try states.append(alloc, st);

        // Repetition window: only positions since the last irreversible move can recur.
        const tail_start = seen.items.len -| (st.halfmove + 1);
        const prev = seen.items[tail_start..];

        const r = s.chooseMove(&st, depth, max_ms, prev);
        nodes.* += r.nodes;
        try scores.append(alloc, r.score);

        // Pick the move: normal play uses the engine's best; opening plies vary for
        // diversity (uniform-random, or uniform over the top-K best when requested).
        var move: ?engine.Move = r.move;
        if (ply < cfg.openings) {
            if (cfg.opening_topk > 0 and r.move != null) {
                var cands: [64]engine.Move = undefined;
                var ncand: usize = 0;
                cands[ncand] = r.move.?;
                ncand += 1;
                var exclude: [64]i32 = undefined;
                var nex: usize = 0;
                exclude[nex] = @as(i32, r.move.?.from) * 64 + @as(i32, r.move.?.to);
                nex += 1;
                while (ncand < cfg.opening_topk and ncand < cands.len) {
                    const nx = s.chooseMoveExcl(&st, depth, max_ms, prev, exclude[0..nex]);
                    const m = nx.move orelse break;
                    cands[ncand] = m;
                    ncand += 1;
                    exclude[nex] = @as(i32, m.from) * 64 + @as(i32, m.to);
                    nex += 1;
                }
                move = cands[rng.intRangeLessThan(usize, 0, ncand)];
            } else {
                // Uniform-random over all legal moves (the default opening behavior).
                var legal: engine.MoveList = .{};
                engine.legalMoves(&st, &legal);
                if (legal.len > 0) move = legal.items[rng.intRangeLessThan(usize, 0, legal.len)];
            }
        }

        const m = move orelse break;
        try seen.append(alloc, zobrist.hashOf(&st));
        try moves_played.append(alloc, m);
        st = engine.applyMove(&st, m);
    }

    // Serialize the whole game as ONE record { g, players, r, moves, v, vs }. `moves`
    // connects consecutive recorded positions, so there is one fewer than positions (the
    // final move leads to the unrecorded terminal); `v` is per recorded position, the
    // search value (side-to-move view). `vs` is a scalar tag — self-play uses one eval.
    const npos = states.items.len;
    if (npos == 0) return result;
    try out.appendSlice(alloc, "{\"g\":\"");
    try appendBase36(out, alloc, cfg.seed);
    try out.append(alloc, '-');
    try fmtInt(out, alloc, @as(i64, @intCast(g)));
    try out.appendSlice(alloc, "\",\"players\":{\"w\":\"");
    try out.appendSlice(alloc, cfg.vtag);
    try out.appendSlice(alloc, "\",\"b\":\"");
    try out.appendSlice(alloc, cfg.vtag);
    try out.appendSlice(alloc, "\"},\"r\":");
    try fmtInt(out, alloc, result);
    try out.appendSlice(alloc, ",\"moves\":[");
    var mi: usize = 0;
    while (mi + 1 < npos) : (mi += 1) {
        if (mi > 0) try out.append(alloc, ',');
        try out.append(alloc, '"');
        try appendMoveToken(out, alloc, moves_played.items[mi]);
        try out.append(alloc, '"');
    }
    try out.appendSlice(alloc, "],\"v\":[");
    for (scores.items, 0..) |sc, i| {
        if (i > 0) try out.append(alloc, ',');
        try fmtInt(out, alloc, sc);
    }
    try out.appendSlice(alloc, "],\"vs\":\"");
    try out.appendSlice(alloc, cfg.vtag);
    try out.appendSlice(alloc, "\"}\n");
    positions.* = npos;
    return result;
}

fn fmtInt(out: *std.ArrayList(u8), alloc: std.mem.Allocator, v: i64) !void {
    var buf: [24]u8 = undefined;
    const s = std.fmt.bufPrint(&buf, "{d}", .{v}) catch unreachable;
    try out.appendSlice(alloc, s);
}

fn worker(sh: *Shared) void {
    const pa = std.heap.page_allocator;
    var s = ai.Searcher.init(pa, sh.cfg.io, sh.cfg.kind, sh.cfg.net, 1) catch return;
    defer s.deinit();

    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(pa);

    while (true) {
        sh.mutex.lockUncancelable(sh.cfg.io);
        const g = sh.next_game;
        const done = g >= sh.cfg.games;
        if (!done) sh.next_game += 1;
        sh.mutex.unlock(sh.cfg.io);
        if (done) break;

        // Reseed the searcher per game so its root variety is a function of the game
        // index, not thread timing (the persistent TT still makes exact games
        // order-sensitive).
        s.reseed(sh.cfg.seed +% g);
        out.clearRetainingCapacity();
        var positions: u64 = 0;
        var nodes: u64 = 0;
        const result = playGame(&s, &sh.cfg, g, pa, &out, &positions, &nodes) catch 0;

        sh.mutex.lockUncancelable(sh.cfg.io);
        if (out.items.len > 0) {
            sh.file.writePositionalAll(sh.cfg.io, out.items, sh.offset) catch {};
            sh.offset += out.items.len;
        }
        sh.done_games += 1;
        sh.total_positions += positions;
        sh.nodes += nodes;
        if (result > 0) sh.wins += 1 else if (result < 0) sh.losses += 1 else sh.draws += 1;
        // Live in-place line after every finished game (and ~1×/s by the heartbeat thread in
        // between, so even a slow depth-8 batch never looks frozen).
        paintLive(sh);
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
// the first 6 lowercase-hex chars (= first 3 digest bytes). "??????" if unreadable.
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

pub fn main(init: std.process.Init) !void {
    const gpa = init.arena.allocator();
    const io = init.io;

    var games: u64 = 200;
    var depth: u32 = 6;
    var movetime: i64 = 50;
    var use_movetime = false;
    var openings: u32 = 8;
    var opening_topk: u32 = 0;
    var maxmoves: u32 = 200;
    var eval_name: ai.EvalKind = .handcrafted;
    var out_path: []const u8 = "../training/data/selfplay.jsonl";
    // Default seed varies per run (clock-derived); --seed overrides for reproducibility.
    var seed: u64 = @intCast(@mod(std.Io.Clock.now(.awake, io).nanoseconds, 1_000_000_000_000));
    var jobs: usize = std.Thread.getCpuCount() catch 1;

    const argv = try init.minimal.args.toSlice(gpa);
    for (argv[1..]) |arg| {
        if (argStr(arg, "--games=")) |v| games = std.fmt.parseInt(u64, v, 10) catch games;
        if (argStr(arg, "--depth=")) |v| depth = std.fmt.parseInt(u32, v, 10) catch depth;
        if (argStr(arg, "--movetime=")) |v| {
            movetime = std.fmt.parseInt(i64, v, 10) catch movetime;
            use_movetime = true;
        }
        if (argStr(arg, "--openings=")) |v| openings = std.fmt.parseInt(u32, v, 10) catch openings;
        if (argStr(arg, "--opening-topk=")) |v| opening_topk = std.fmt.parseInt(u32, v, 10) catch opening_topk;
        if (argStr(arg, "--maxmoves=")) |v| maxmoves = std.fmt.parseInt(u32, v, 10) catch maxmoves;
        if (argStr(arg, "--eval=")) |v| eval_name = if (std.mem.eql(u8, v, "nn")) .nn else .handcrafted;
        if (argStr(arg, "--out=")) |v| out_path = v;
        if (argStr(arg, "--seed=")) |v| seed = std.fmt.parseInt(u64, v, 10) catch seed;
        if (argStr(arg, "--jobs=")) |v| jobs = std.fmt.parseInt(usize, v, 10) catch jobs;
    }
    if (jobs < 1) jobs = 1;
    if (jobs > games) jobs = @intCast(@max(1, games));
    // --movetime overrides depth (depth==0 sentinel means "use movetime").
    const search_depth: u32 = if (use_movetime) 0 else depth;

    const net: ?*const nn.Net = if (eval_name == .nn) blk: {
        const n = try gpa.create(nn.Net);
        n.* = try loadNet(io, gpa, NN_WEIGHTS);
        break :blk n;
    } else null;

    // Provenance tag: "<engine><depth|t>@<version>" (vtag.mjs). hc version is HC_VERSION
    // (2); nn version is the short content hash of the weights file.
    var tagbuf: [32]u8 = undefined;
    const eng = if (eval_name == .nn) "nn" else "hc";
    const vtag = blk: {
        if (eval_name == .nn) {
            const h = weightsHash(io, gpa, NN_WEIGHTS);
            if (use_movetime) {
                break :blk try std.fmt.bufPrint(&tagbuf, "{s}t@{s}", .{ eng, h });
            } else {
                break :blk try std.fmt.bufPrint(&tagbuf, "{s}{d}@{s}", .{ eng, search_depth, h });
            }
        } else {
            if (use_movetime) {
                break :blk try std.fmt.bufPrint(&tagbuf, "{s}t@{d}", .{ eng, ai.HC_VERSION });
            } else {
                break :blk try std.fmt.bufPrint(&tagbuf, "{s}{d}@{d}", .{ eng, search_depth, ai.HC_VERSION });
            }
        }
    };

    // Ensure the output directory exists, then open the dataset for appending. The file
    // keeps its existing content (truncate=false); we write positionally from its current
    // end, so parallel workers append without a shared seek position.
    if (std.fs.path.dirname(out_path)) |dir| {
        std.Io.Dir.cwd().createDirPath(io, dir) catch {};
    }
    // read=true so the handle can be stat'd (Windows denies stat on a write-only handle).
    var file = try std.Io.Dir.cwd().createFile(io, out_path, .{ .truncate = false, .read = true });
    defer file.close(io);
    const start_offset: u64 = (file.stat(io) catch unreachable).size;
    const fresh = start_offset == 0;

    std.debug.print("Generating {d} games -> {s}{s}\n  {s} | eval {s} | openings {d}{s} | jobs {d} | seed {d}\n", .{
        games,                                                   out_path,
        if (fresh) "" else " (appending)",                       if (use_movetime) "movetime" else "depth",
        if (eval_name == .nn) "nn" else "handcrafted",           openings,
        if (opening_topk > 0) " (topk)" else " (random)",        jobs,
        seed,
    });

    var shared = Shared{
        .cfg = .{
            .games = games,
            .depth = search_depth,
            .movetime = movetime,
            .openings = openings,
            .opening_topk = opening_topk,
            .maxmoves = maxmoves,
            .kind = eval_name,
            .net = net,
            .seed = seed,
            .vtag = vtag,
            .io = io,
        },
        .file = file,
        .offset = start_offset,
    };

    const t0 = std.Io.Clock.now(.awake, io).nanoseconds;
    shared.t0_ns = @intCast(t0); // so the live progress line can show elapsed/ETA
    const threads = try gpa.alloc(std.Thread, jobs);
    for (threads) |*t| t.* = try std.Thread.spawn(.{}, worker, .{&shared});
    // Heartbeat repaints the live line ~1×/s so a slow batch never looks frozen between game
    // completions. Signal it to stop once the workers are done, then join it.
    const hb = try std.Thread.spawn(.{}, heartbeat, .{&shared});
    for (threads) |t| t.join();
    shared.mutex.lockUncancelable(io);
    shared.heartbeat_stop = true;
    shared.mutex.unlock(io);
    hb.join();
    clearLive(&shared); // erase the live line before the final summary
    const ms: u64 = @intCast(@max(1, @divTrunc(std.Io.Clock.now(.awake, io).nanoseconds - t0, 1_000_000)));

    const gpm: f64 = @as(f64, @floatFromInt(shared.done_games)) / (@as(f64, @floatFromInt(ms)) / 60000.0);
    std.debug.print("Done: {d} games, {d} positions in {d}ms ({d:.1} games/min). Results: W {d} B {d} D {d}. nps {d}\n", .{
        shared.done_games, shared.total_positions, ms, gpm, shared.wins, shared.losses, shared.draws, shared.nodes * 1000 / ms,
    });
}
