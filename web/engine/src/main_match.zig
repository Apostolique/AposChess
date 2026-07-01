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
// --save-games harvests each game as one game-primary record (scripts/gameRecord.mjs:
// g/players/r/moves/v/vs). Games run from the standard start; the random opening plies are
// played from a scripted line but still SEARCHED, so every position — opening included —
// carries the value from the engine that was to move (the mover), at that engine's own depth,
// tagged with its provenance (`vs`). The dataset machinery then sorts label quality by
// provenance — merge-data prefers the stronger engine on dedup and refresh-v relabels the
// weakest cohort first — so there's no need to second-guess a label by who won the game. The
// game result is kept separately in `r`.

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

// --- interrupt handling (keep completed games on Ctrl-C) ----------------------------
// Set when the user interrupts (Ctrl-C / console close on Windows, SIGINT on POSIX). The
// heartbeat polls it (via stopRequested) and finalizes the match from the games COMPLETED so
// far, then exits — so an interrupt during the loop's GATE, or a standalone `npm run match`,
// keeps every finished game instead of the OS killing the process and losing the whole matchup.
// (The rank pool's orchestrated stop arrives as a stop-file, also handled by stopRequested.)
var g_interrupt = std.atomic.Value(bool).init(false);

const win = std.os.windows;
extern "kernel32" fn SetConsoleCtrlHandler(
    handler: ?*const fn (ctrl_type: win.DWORD) callconv(.winapi) win.BOOL,
    add: win.BOOL,
) callconv(.winapi) win.BOOL;

fn consoleCtrlHandler(ctrl_type: win.DWORD) callconv(.winapi) win.BOOL {
    _ = ctrl_type; // Ctrl-C, Ctrl-Break, close — all mean "stop and save what's done"
    g_interrupt.store(true, .seq_cst);
    return .TRUE; // handled: don't terminate; the heartbeat finalizes + exits within ~1s
}

fn posixSigint(_: std.posix.SIG) callconv(.c) void {
    g_interrupt.store(true, .seq_cst); // async-signal-safe: just set the flag
}

// Catch the OS interrupt so the match can finalize (keep completed games) instead of dying.
fn installInterruptHandler() void {
    if (builtin.os.tag == .windows) {
        _ = SetConsoleCtrlHandler(consoleCtrlHandler, .TRUE);
    } else {
        const act = std.posix.Sigaction{
            .handler = .{ .handler = posixSigint },
            .mask = std.posix.sigemptyset(),
            .flags = 0,
        };
        std.posix.sigaction(std.posix.SIG.INT, &act, null);
    }
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
// mover's search value, and which engine ('a'/'b') moved (its provenance on harvest).
const PlyRec = struct {
    r: i32 = 0,
    r_turn_white: bool = false, // side-to-move was White (to set `r` once the game ends)
    v: i32 = 0,
    mover: u8 = 0, // 'a' or 'b'
    move: engine.Move = undefined, // the move played from this position (for the game record)
};
const Game = struct {
    pair: usize,
    color: u8, // 'w' (A is White) or 'b' (A is Black)
    result_white: i32 = 0, // White-view result of this game (+1/0/-1)
    recs: std.ArrayList(PlyRec),
};

// A scripted random opening line, shared (color-reversed) by both games of a pair so the
// pairing stays balanced. We play these exact moves at the start of each game but still
// SEARCH each position, so the opening plies are recorded with the to-move engine's own
// value (its point of view of the position it's about to move from) — see playGame.
const MAX_OPENING = 64;
const Opening = struct {
    moves: [MAX_OPENING]engine.Move = undefined,
    len: u32 = 0,
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

// Build a random opening as a MOVE SEQUENCE from the standard start. Returned (not applied)
// so both color-reversed games of a pair replay the identical line. If a random ply ends the
// game, the whole opening is abandoned (len 0) and the pair plays from the standard start —
// mirroring the old fast-forward behavior, which discarded such openings.
fn randomOpening(rng: std.Random, plies: u32) Opening {
    var op = Opening{};
    var st = board.newGameState();
    const want = @min(plies, MAX_OPENING);
    var i: u32 = 0;
    while (i < want) : (i += 1) {
        var moves: engine.MoveList = .{};
        engine.legalMoves(&st, &moves);
        if (moves.len == 0) break;
        const m = moves.items[rng.intRangeLessThan(usize, 0, moves.len)];
        st = engine.applyMove(&st, m);
        if (engine.gameStatus(&st).status != .ongoing) return .{}; // opening ended the game
        op.moves[op.len] = m;
        op.len += 1;
    }
    return op;
}

// Play one game. `white_is_a` decides which engine has White. Returns +1 white win /
// 0 draw / -1 black win. When `recs` is non-null (--save-games) it collects one record
// per searched position (the harvest path).
fn playGame(
    white: *ai.Searcher,
    black: *ai.Searcher,
    white_is_a: bool,
    opening: *const Opening, // scripted random opening plies (still searched, see below)
    bw: Budget, // White's search budget
    bb: Budget, // Black's search budget
    max_plies: u32,
    alloc: std.mem.Allocator,
    nodes: *u64,
    recs: ?*std.ArrayList(PlyRec),
) !i32 {
    var seen: std.ArrayList(u64) = .empty;
    defer seen.deinit(alloc);
    var st = board.newGameState();
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
        const mover_b = if (st.turn == .white) bw else bb;
        // Always search — even in the opening — so each position is recorded with the mover's
        // own value. During the scripted opening we discard the engine's choice and play the
        // predetermined random move (so the pair shares one opening line); after it we play best.
        const res = searchBudget(mover, &st, mover_b, seen.items, false);
        nodes.* += res.nodes;
        const m = if (plies < opening.len) opening.moves[plies] else (res.move orelse break);

        if (recs) |list| {
            // One record per searched position: its mover's value + the move played from
            // it (positions and moves are reassembled into a game record on harvest). r
            // from the side-to-move's view is set once the game's outcome is known.
            try list.append(alloc, .{
                .v = res.score,
                .mover = if (a_to_move) 'a' else 'b',
                .move = m,
                .r_turn_white = st.turn == .white,
            });
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
    // When set, the heartbeat thread polls this path; once it exists, the match finalizes
    // IMMEDIATELY — it writes the result-file + harvest from the games already COMPLETED and
    // exits, abandoning the games still in flight. This is how an orchestrator (rank:pool /
    // the loop) requests a stop that takes effect right away while keeping every finished
    // game, instead of either killing the process (losing the whole matchup) or waiting for
    // the in-flight games to drain.
    stop_file: ?[]const u8,
    // Finalize context (so the heartbeat can write the same result/harvest the normal end does).
    weights_a: []const u8,
    weights_b: []const u8,
    result_file: ?[]const u8,
    save_games_path: ?[]const u8,
    io: std.Io,
};

const Shared = struct {
    mutex: std.Io.Mutex = .init,
    // Work is dispatched one GAME at a time (not one color-reversed pair), so the tail of a
    // matchup tapers over single games rather than pairs — finished workers idle for at most one
    // game-time, not one pair-time, keeping cores fuller near the end. Game index gi maps to
    // pair = gi >> 1, color = gi & 1 (0 = A is White, 1 = A is Black); both games of a pair still
    // share one seeded opening, so the color-reversed balance is unchanged.
    next_game: usize = 0,
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
    heartbeat_stop: bool = false, // workers done -> the heartbeat thread should exit
    finalized: bool = false, // report/harvest/result-file written once (normal end OR stop)
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

// Repaint the in-place live status line from the current shared counters. MUST be called
// with sh.mutex held. Shared by the per-pair worker update AND the heartbeat thread, so the
// elapsed/ETA clock and the in-flight count keep advancing even when no pair has finished —
// a slow depth-8-vs-depth-8 matchup otherwise looks frozen for minutes between completions
// (the first batch of games all search at once and none finish for a long while). Returns
// without touching the line once the match is settled, so it never clobbers the final
// milestone (the committed snapshot the worker prints on completion / an SPRT decision).
fn paintLive(sh: *Shared) void {
    const ng = sh.scores.items.len;
    const total: usize = sh.total_pairs * 2;
    if (sh.decided != null or ng >= total) return;
    var w: usize = 0;
    var dr: usize = 0;
    var ls: usize = 0;
    var ssum: f64 = 0;
    for (sh.scores.items) |sc| {
        ssum += sc;
        if (sc == 1) w += 1 else if (sc == 0.5) dr += 1 else ls += 1;
    }
    const pp = if (ng > 0) ssum / @as(f64, @floatFromInt(ng)) else 0;
    const now_ns: i128 = @intCast(std.Io.Clock.now(.awake, sh.cfg.io).nanoseconds);
    const elapsed_s: f64 = @as(f64, @floatFromInt(now_ns - sh.t0_ns)) / 1e9;

    // In-flight games = dispatched − finished = busy workers. This is the key liveness signal
    // during the opening stretch when ng is still 0: it shows N games are actively searching.
    const running = if (sh.next_game > ng) sh.next_game - ng else 0;

    var etaseg: []const u8 = "";
    var etastore: [40]u8 = undefined;
    if (ng > 0) {
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
    var ebuf: [16]u8 = undefined;
    var livebuf: [320]u8 = undefined;
    const live = std.fmt.bufPrint(&livebuf, "  game {d}/{d} | {d} running | A +{d} ={d} -{d} | {d:.1}% | {s} elapsed{s}{s}", .{
        ng, total, running, w, dr, ls, pp * 100, fmtDur(&ebuf, elapsed_s), etaseg, llrseg,
    }) catch livebuf[0..0];
    printLive(sh, live);
}

// Has a stop been requested — an OS interrupt (Ctrl-C / SIGINT) or the orchestrator's
// stop-file appearing? (success on `access` = the file is there.)
fn stopRequested(sh: *Shared) bool {
    if (g_interrupt.load(.seq_cst)) return true;
    const p = sh.cfg.stop_file orelse return false;
    std.Io.Dir.cwd().access(sh.cfg.io, p, .{}) catch return false;
    return true;
}

// Heartbeat thread: repaint the live status line about once a second so a long matchup never
// looks frozen between pair completions, and poll the orchestrator's stop-file. When the
// stop-file appears it finalizes (writes the result-file + harvest from the COMPLETED games)
// and exits the process immediately — abandoning the games still in flight. Holding the mutex
// across the exit means no worker can append a pair between the snapshot and the exit, so what
// lands on disk is exactly the set of fully completed games. It only holds the mutex briefly
// otherwise (workers keep ownership of the milestone lines), and exits once main signals
// heartbeat_stop at the normal end.
fn heartbeat(sh: *Shared) void {
    while (true) {
        sh.cfg.io.sleep(std.Io.Duration.fromMilliseconds(1000), .awake) catch {};
        const stop_now = stopRequested(sh);
        sh.mutex.lockUncancelable(sh.cfg.io);
        if (sh.heartbeat_stop) {
            sh.mutex.unlock(sh.cfg.io);
            break;
        }
        if (stop_now) {
            sh.stop = true; // marks the early stop (finalizeLocked notes it; no SPRT decision)
            if (!sh.finalized) finalizeLocked(sh);
            std.process.exit(0); // hold the lock through exit so no in-flight pair sneaks in
        }
        paintLive(sh);
        sh.mutex.unlock(sh.cfg.io);
    }
}

fn worker(sh: *Shared) void {
    const pa = std.heap.page_allocator;
    var sa = ai.Searcher.init(pa, sh.cfg.io, sh.cfg.kind_a, sh.cfg.net_a, 1) catch return;
    defer sa.deinit();
    var sb = ai.Searcher.init(pa, sh.cfg.io, sh.cfg.kind_b, sh.cfg.net_b, 1) catch return;
    defer sb.deinit();

    while (true) {
        sh.mutex.lockUncancelable(sh.cfg.io);
        const done = sh.stop or sh.next_game >= sh.total_pairs * 2;
        const gi = sh.next_game; // game index: pair = gi >> 1, color = gi & 1
        if (!done) sh.next_game += 1;
        sh.mutex.unlock(sh.cfg.io);
        if (done) break;

        const pair = gi >> 1;
        const a_is_white = (gi & 1) == 0; // even game of a pair = A has White, odd = A has Black

        var op_prng = std.Random.DefaultPrng.init(sh.cfg.seed +% pair);
        const opening = randomOpening(op_prng.random(), sh.cfg.openings);
        var nodes: u64 = 0;

        // Collect per-game records only when harvesting.
        var recs: std.ArrayList(PlyRec) = .empty;
        const pr: ?*std.ArrayList(PlyRec) = if (sh.cfg.save_games) &recs else null;

        // Reseed per game so openings/variety are a function of the (pair, color), not thread
        // timing — the same offsets the pair-dispatched version used, so a game's seeding is
        // identical regardless of which worker plays it (the persistent TT still makes exact
        // games order-sensitive, as before).
        var r: i32 = undefined;
        if (a_is_white) {
            sa.reseed(sh.cfg.seed +% pair *% 4 +% 0);
            sb.reseed(sh.cfg.seed +% pair *% 4 +% 1);
            // A = White: White's budget is A's, Black's is B's.
            r = playGame(&sa, &sb, true, &opening, sh.cfg.budget_a, sh.cfg.budget_b, sh.cfg.max_plies, pa, &nodes, pr) catch 0;
        } else {
            sb.reseed(sh.cfg.seed +% pair *% 4 +% 2);
            sa.reseed(sh.cfg.seed +% pair *% 4 +% 3);
            // A = Black: White's budget is B's, Black's is A's.
            r = playGame(&sb, &sa, false, &opening, sh.cfg.budget_b, sh.cfg.budget_a, sh.cfg.max_plies, pa, &nodes, pr) catch 0;
        }

        // A's score for this game: +1 win / 0.5 draw / 0 loss, from A's color this game.
        const s: f64 = if (a_is_white)
            (if (r > 0) @as(f64, 1) else if (r < 0) @as(f64, 0) else @as(f64, 0.5))
        else
            (if (r < 0) @as(f64, 1) else if (r > 0) @as(f64, 0) else @as(f64, 0.5));

        sh.mutex.lockUncancelable(sh.cfg.io);
        sh.scores.append(sh.alloc, s) catch {};
        sh.nodes += nodes;
        if (sh.cfg.save_games) {
            // The game shares its pair's scripted opening line and begins at the standard
            // start (no `start` field), so the record holds the full game. color 'w' = A had
            // White, 'b' = A had Black (so writeHarvest assigns the player tags correctly).
            const g = Game{ .pair = pair, .color = if (a_is_white) 'w' else 'b', .result_white = r, .recs = recs };
            sh.games.append(sh.alloc, g) catch {};
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

        // 1) Live, in-place line refreshed after every finished game (and ~1×/s by the
        //    heartbeat thread in between, so even a long match never looks frozen).
        paintLive(sh);

        // 2) Committed milestone every 10 games (or on an SPRT decision): clear the live
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
    if (kind == .handcrafted3) {
        if (depth == 0) return std.fmt.bufPrint(buf, "hc3t@{d}", .{ai.HC_VERSION}) catch unreachable;
        return std.fmt.bufPrint(buf, "hc3-{d}@{d}", .{ depth, ai.HC_VERSION }) catch unreachable;
    }
    if (depth == 0) return std.fmt.bufPrint(buf, "hct@{d}", .{ai.HC_VERSION}) catch unreachable;
    return std.fmt.bufPrint(buf, "hc{d}@{d}", .{ depth, ai.HC_VERSION }) catch unreachable;
}

// --eval-a/--eval-b parse: "nn" -> nn, "hc3" -> handcrafted3, "material" -> material,
// anything else -> handcrafted.
fn parseEval(v: []const u8) ai.EvalKind {
    if (std.mem.eql(u8, v, "nn")) return .nn;
    if (std.mem.eql(u8, v, "hc3")) return .handcrafted3;
    if (std.mem.eql(u8, v, "material")) return .material;
    return .handcrafted;
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

// Compact move token "<from><to>[promo]" (mirrors scripts/gameRecord.mjs encodeMove).
fn appendMoveToken(out: *std.ArrayList(u8), alloc: std.mem.Allocator, m: engine.Move) !void {
    const fr = board.squareName(m.from);
    const to = board.squareName(m.to);
    try out.appendSlice(alloc, &fr);
    try out.appendSlice(alloc, &to);
    if (m.promotion) |role| try out.append(alloc, board.charFromRole(role));
}

// Append the harvested games as game-primary training data (the gate harvest train:loop
// folds in; see scripts/gameRecord.mjs). One line per game records WHO PLAYED (players)
// and labels every position with the value from the engine that actually searched it (the
// mover), at that engine's own depth, tagged with its provenance — so per-position `vs` is
// an array (the movers alternate). No winner-based derivation: a label's trustworthiness is
// captured by its `vs` tag (and the engine's Elo), not by who won.
fn writeHarvest(sh: *Shared, gpa: std.mem.Allocator, io: std.Io, path: []const u8, depth_a: u32, depth_b: u32, eval_a: ai.EvalKind, eval_b: ai.EvalKind, weights_a: []const u8, weights_b: []const u8) !void {
    var tag_a_buf: [40]u8 = undefined;
    var tag_b_buf: [40]u8 = undefined;
    const tag_a = vtagFmt(&tag_a_buf, eval_a, depth_a, io, gpa, weights_a);
    const tag_b = vtagFmt(&tag_b_buf, eval_b, depth_b, io, gpa, weights_b);

    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);
    var n_pos: usize = 0;
    var n_a: usize = 0;
    var dropped: usize = 0;
    for (sh.games.items) |g| {
        const npos = g.recs.items.len;
        if (npos == 0) continue;
        // Keep only color-balanced openings: an early stop (SPRT / abort) can finish one game of a
        // pair without its color-reversed partner, since work is dispatched one game at a time. Such
        // a lone game would skew the opening's color balance, so skip any game whose pair partner
        // isn't also present. (A natural full run dispatches every pair complete, so nothing drops.)
        var pair_games: usize = 0;
        for (sh.games.items) |o| {
            if (o.pair == g.pair and o.recs.items.len > 0) pair_games += 1;
        }
        if (pair_games < 2) {
            dropped += 1;
            continue;
        }
        // players: color 'w' means engine A had White, so w=A's tag, b=B's tag (swapped otherwise).
        const pw = if (g.color == 'w') tag_a else tag_b;
        const pb = if (g.color == 'w') tag_b else tag_a;
        try out.appendSlice(gpa, "{\"g\":\"m");
        try appendBase36(&out, gpa, sh.cfg.seed);
        try out.append(gpa, '-');
        try appendInt(&out, gpa, @as(i64, @intCast(g.pair)));
        try out.append(gpa, g.color); // 'w' or 'b'
        // Games begin at the standard start (the opening is searched and recorded inline),
        // so no `start` field — readers default to the standard start position.
        try out.appendSlice(gpa, "\",\"players\":{\"w\":\"");
        try out.appendSlice(gpa, pw);
        try out.appendSlice(gpa, "\",\"b\":\"");
        try out.appendSlice(gpa, pb);
        try out.appendSlice(gpa, "\"},\"r\":");
        try appendInt(&out, gpa, g.result_white);
        // moves connect consecutive recorded positions (one fewer than positions).
        try out.appendSlice(gpa, ",\"moves\":[");
        var mi: usize = 0;
        while (mi + 1 < npos) : (mi += 1) {
            if (mi > 0) try out.append(gpa, ',');
            try out.append(gpa, '"');
            try appendMoveToken(&out, gpa, g.recs.items[mi].move);
            try out.append(gpa, '"');
        }
        // v: the mover's own search value per recorded position.
        try out.appendSlice(gpa, "],\"v\":[");
        for (g.recs.items, 0..) |rec, i| {
            if (i > 0) try out.append(gpa, ',');
            try appendInt(&out, gpa, rec.v);
        }
        // vs: per-position provenance (mover's engine×depth tag), parallel to v.
        try out.appendSlice(gpa, "],\"vs\":[");
        for (g.recs.items, 0..) |rec, i| {
            if (i > 0) try out.append(gpa, ',');
            try out.append(gpa, '"');
            try out.appendSlice(gpa, if (rec.mover == 'a') tag_a else tag_b);
            try out.append(gpa, '"');
            if (rec.mover == 'a') n_a += 1;
        }
        try out.appendSlice(gpa, "]}\n");
        n_pos += npos;
    }

    if (std.fs.path.dirname(path)) |dir| std.Io.Dir.cwd().createDirPath(io, dir) catch {};
    var file = try std.Io.Dir.cwd().createFile(io, path, .{ .truncate = false, .read = true });
    defer file.close(io);
    const offset: u64 = (file.stat(io) catch unreachable).size;
    try file.writePositionalAll(io, out.items, offset);
    var dropbuf: [48]u8 = undefined;
    const dropmsg = if (dropped > 0)
        (std.fmt.bufPrint(&dropbuf, " (dropped {d} unpaired tail game(s))", .{dropped}) catch "")
    else
        "";
    std.debug.print("Saved {d} positions from {d} games to {s} ({d} A {s}, {d} B {s}){s}.\n", .{
        n_pos, sh.games.items.len - dropped, path, n_a, tag_a, n_pos - n_a, tag_b, dropmsg,
    });
}

// Write the report + harvest + result-file ONCE (guarded by sh.finalized). MUST be called with
// sh.mutex held. Called at the normal end (all games played) and on a stop-file stop (from the
// heartbeat, which then exits immediately) — either way it reads the games COMPLETED so far
// from sh.scores/sh.games, so a stop keeps every finished game and just drops the in-flight ones.
fn finalizeLocked(sh: *Shared) void {
    if (sh.finalized) return;
    sh.finalized = true;
    clearLive(sh); // erase any lingering live line before the permanent report
    const io = sh.cfg.io;
    const gpa = sh.alloc;
    const now_ns: i128 = @intCast(std.Io.Clock.now(.awake, io).nanoseconds);
    const ms: u64 = @intCast(@max(1, @divTrunc(now_ns - sh.t0_ns, 1_000_000)));

    const n = sh.scores.items.len;
    var sum: f64 = 0;
    var wins: usize = 0;
    var draws: usize = 0;
    var losses: usize = 0;
    for (sh.scores.items) |s| {
        sum += s;
        if (s == 1) wins += 1 else if (s == 0.5) draws += 1 else losses += 1;
    }
    const p = if (n > 0) sum / @as(f64, @floatFromInt(n)) else 0;
    const elo = eloFromScore(p);
    const ci = eloWithCI(sh.scores.items);
    const sign = if (ci.elo >= 0) "+" else "";
    // A stop with no SPRT decision means we were stopped early: say so, since fewer games than
    // requested were played (the in-flight ones were abandoned, the completed ones kept).
    if (sh.stop and sh.decided == null) {
        std.debug.print("Stopped early: kept {d} completed game(s), abandoned those in flight.\n", .{n});
    }
    const verdict = if (sh.sprt) (sh.decided orelse "inconclusive") else "n/a";
    std.debug.print("A vs B: {d} games | +{d} ={d} -{d} | score {d:.1}% | Elo {s}{d:.0} ± {d:.0} (95% CI [{d:.0}, {d:.0}]) | SPRT {s} | nodes {d} nps {d}\n", .{
        n, wins, draws, losses, p * 100, sign, ci.elo, ci.margin, ci.lo, ci.hi, verdict, sh.nodes, sh.nodes * 1000 / ms,
    });

    if (sh.cfg.save_games_path) |sg| {
        if (sh.games.items.len > 0)
            writeHarvest(sh, gpa, io, sg, sh.cfg.budget_a.depth, sh.cfg.budget_b.depth, sh.cfg.kind_a, sh.cfg.kind_b, sh.cfg.weights_a, sh.cfg.weights_b) catch {};
    }

    if (sh.cfg.result_file) |rf| {
        var buf: [512]u8 = undefined;
        const sprt_field = if (sh.sprt) sh.decided orelse "inconclusive" else null;
        const json = if (sprt_field) |sf|
            std.fmt.bufPrint(&buf,
                \\{{"games":{d},"wins":{d},"draws":{d},"losses":{d},"score":{d},"elo":{d},"llr":{d},"sprt":"{s}"}}
            , .{ n, wins, draws, losses, p, elo, llr(sh.scores.items, sh.elo0, sh.elo1), sf }) catch return
        else
            std.fmt.bufPrint(&buf,
                \\{{"games":{d},"wins":{d},"draws":{d},"losses":{d},"score":{d},"elo":{d},"llr":null,"sprt":null}}
            , .{ n, wins, draws, losses, p, elo }) catch return;
        std.Io.Dir.cwd().writeFile(io, .{ .sub_path = rf, .data = json }) catch {};
    }
}

pub fn main(init: std.process.Init) !void {
    enableUtf8Console(); // so `±` and friends render instead of mojibake on Windows
    installInterruptHandler(); // Ctrl-C finalizes (keeps completed games) instead of killing
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
    // No default weights on purpose: --eval=nn must name its net explicitly. A silent default
    // (formerly "src/nn-weights.json", the champion) turns any "forgot --weights" into the
    // champion playing under the wrong label — exactly what made the pool's material node rank
    // as the champion. Non-nn evals (handcrafted/hc3/material) need no net and leave it null.
    var weights_a: ?[]const u8 = null;
    var weights_b: ?[]const u8 = null;
    var sprt = false;
    var elo0: f64 = 0;
    var elo1: f64 = 15;
    var alpha: f64 = 0.05;
    var beta: f64 = 0.05;
    var result_file: ?[]const u8 = null;
    var save_games: ?[]const u8 = null;
    var stop_file: ?[]const u8 = null;

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
        if (argStr(arg, "--eval-a=")) |v| eval_a = parseEval(v);
        if (argStr(arg, "--eval-b=")) |v| eval_b = parseEval(v);
        if (argStr(arg, "--weights-a=")) |v| weights_a = v;
        if (argStr(arg, "--weights-b=")) |v| weights_b = v;
        if (std.mem.eql(u8, arg, "--sprt")) sprt = true;
        if (argStr(arg, "--elo0=")) |v| elo0 = std.fmt.parseFloat(f64, v) catch elo0;
        if (argStr(arg, "--elo1=")) |v| elo1 = std.fmt.parseFloat(f64, v) catch elo1;
        if (argStr(arg, "--alpha=")) |v| alpha = std.fmt.parseFloat(f64, v) catch alpha;
        if (argStr(arg, "--beta=")) |v| beta = std.fmt.parseFloat(f64, v) catch beta;
        if (argStr(arg, "--result-file=")) |v| result_file = v;
        if (argStr(arg, "--save-games=")) |v| save_games = v;
        if (argStr(arg, "--stop-file=")) |v| stop_file = v;
    }
    if (jobs < 1) jobs = 1;

    // Resolve each side's budget. A fixed depth wins over movetime; B inherits A's depth
    // when only --depth was given, A's movetime when only --movetime was given.
    const budget_a: Budget = if (depth_a) |d| .{ .depth = d, .movetime = 0 } else .{ .depth = 0, .movetime = movetime_a };
    const budget_b: Budget = if (depth_b_opt orelse depth_a) |d|
        .{ .depth = d, .movetime = 0 }
    else
        .{ .depth = 0, .movetime = movetime_b_opt orelse movetime_a };

    if (eval_a == .nn and weights_a == null) {
        std.debug.print("error: --eval-a=nn requires --weights-a=<file> (no silent champion fallback).\n", .{});
        std.process.exit(2);
    }
    if (eval_b == .nn and weights_b == null) {
        std.debug.print("error: --eval-b=nn requires --weights-b=<file> (no silent champion fallback).\n", .{});
        std.process.exit(2);
    }
    const net_a: ?*const nn.Net = if (eval_a == .nn) blk: {
        const n = try gpa.create(nn.Net);
        n.* = try loadNet(io, gpa, weights_a.?);
        break :blk n;
    } else null;
    const net_b: ?*const nn.Net = if (eval_b == .nn) blk: {
        const n = try gpa.create(nn.Net);
        n.* = try loadNet(io, gpa, weights_b.?);
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
            .stop_file = stop_file,
            .weights_a = weights_a orelse "",
            .weights_b = weights_b orelse "",
            .result_file = result_file,
            .save_games_path = save_games,
            .io = io,
        },
    };

    std.debug.print("Playing {d} games | openings {d} | jobs {d} | seed {d}\n", .{ games, openings, jobs, seed });

    const t0 = std.Io.Clock.now(.awake, io).nanoseconds;
    shared.t0_ns = @intCast(t0); // so the live progress line can show elapsed/ETA
    const threads = try gpa.alloc(std.Thread, jobs);
    for (threads) |*t| t.* = try std.Thread.spawn(.{}, worker, .{&shared});
    // Heartbeat repaints the live line ~1×/s so a slow matchup never looks frozen between
    // pair completions. Signal it to stop once the workers are done, then join it.
    const hb = try std.Thread.spawn(.{}, heartbeat, .{&shared});
    for (threads) |t| t.join();
    // Normal end (every game played): stop the heartbeat, then finalize under the mutex. A
    // stop-file stop instead finalizes from inside the heartbeat and exits before reaching here.
    shared.mutex.lockUncancelable(io);
    shared.heartbeat_stop = true;
    shared.mutex.unlock(io);
    hb.join();
    shared.mutex.lockUncancelable(io);
    finalizeLocked(&shared);
    shared.mutex.unlock(io);
}
