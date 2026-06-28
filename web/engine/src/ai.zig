// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2019-2026 Jean-David Moisan
//
// Search — a port of web/src/ai.js: iterative-deepening alpha-beta with a
// pluggable eval (handcrafted PST or nn), persistent Zobrist-hashed transposition
// table, quiescence, PVS, null-move, LMR, check extensions, killer/history move
// ordering, and repetition detection. State lives in a `Searcher` (the JS module
// globals) so two can coexist — the match runner pits two head-to-head, each with
// its own TT, exactly like the JS match runner's separate module instances.
//
// Search is intentionally NOT bit-exact vs JS (ordering/TT/float differences);
// it's validated by strength (a Zig-vs-Zig match scores ~50%) and by the engine
// layers below it being exactly equal to JS (perft/hash/eval all green).

const std = @import("std");
const builtin = @import("builtin");
const board = @import("board.zig");
const engine = @import("engine.zig");
const zobrist = @import("zobrist.zig");
const eval = @import("eval.zig");
const nn = @import("nn.zig");

// On wasm there is no std.Io clock; the JS host supplies the monotonic time source so a
// time-budgeted (movetime) search works in the browser. Declared only on wasm, so native
// builds never reference (or need to link) the import.
const is_wasm = builtin.target.cpu.arch.isWasm();
const host = if (is_wasm) struct {
    extern "env" fn aposNowMs() f64;
} else struct {};

const Color = board.Color;
const Role = board.Role;
const Piece = board.Piece;
const State = board.State;
const Move = engine.Move;
const opponent = board.opponent;

const MATE: i32 = 1_000_000;
const MATE_THRESH: i32 = MATE - 1000;
const INF: i32 = 2_000_000;
const MAX_PLY: usize = 64;
const QDEPTH: i32 = 6;
const DELTA_MARGIN: i32 = 200;
const MOB = 3;

// When true, every incremental-accumulator eval is cross-checked against a from-scratch
// recompute and a mismatch panics — flip on to catch accumulator desync in parity/bench.
const ACC_DEBUG = false;

const EXACT: u8 = 0;
const LOWER: u8 = 1;
const UPPER: u8 = 2;
const TT_BITS: u6 = 20;
const TT_SIZE: usize = @as(usize, 1) << TT_BITS;
const TT_MASK: u64 = TT_SIZE - 1;

pub const EvalKind = enum { handcrafted, nn, handcrafted3, material };

// Handcrafted-eval version (mirrors HC_VERSION in src/ai.js) — bumped by hand when the
// PST eval changes, so a `v` from it is stamped with which version produced it (vtag).
pub const HC_VERSION = 2;

fn keyOf(m: Move) i32 {
    return @as(i32, m.from) * 64 + @as(i32, m.to);
}

// Mate scores stored relative to the node (distance-to-mate from here).
fn toTT(s: i32, ply: usize) i32 {
    const p: i32 = @intCast(ply);
    if (s >= MATE_THRESH) return s + p;
    if (s <= -MATE_THRESH) return s - p;
    return s;
}
fn fromTT(s: i32, ply: usize) i32 {
    const p: i32 = @intCast(ply);
    if (s >= MATE_THRESH) return s - p;
    if (s <= -MATE_THRESH) return s + p;
    return s;
}

fn hasNonPawn(b: *const [64]?Piece, color: Color) bool {
    for (b) |sqp| {
        if (sqp) |p| {
            if (p.color == color and p.role != .p and p.role != .k) return true;
        }
    }
    return false;
}

pub const Result = struct { move: ?Move, score: i32, depth: u32, nodes: u64, ponder: ?Move = null };

pub const Searcher = struct {
    alloc: std.mem.Allocator,
    // Transposition table (typed arrays, persistent across searches).
    tt_key: []u64,
    tt_score: []i32,
    tt_move: []i32,
    tt_depth: []i16,
    tt_flag: []u8,
    tt_gen: []u16,
    cur_gen: u16 = 0,
    tt_enabled: bool = true,
    // Per-search ordering / repetition state.
    killers: [MAX_PLY][2]i32 = std.mem.zeroes([MAX_PLY][2]i32),
    history: [64 * 64]i32 = undefined,
    rep_path: [MAX_PLY]u64 = undefined,
    rep_seen: []const u64 = &.{},
    tainted: bool = false,
    // Eval selection.
    eval_kind: EvalKind,
    eval_key: u64,
    net: ?*const nn.Net,
    // Incremental NNUE accumulators (raw, pre-clip), maintained through make/unmake when
    // the selected net is quantized — one per fixed perspective (us = white / us = black).
    // The leaf eval reads the side-to-move one (see evalNn). Float nets recompute instead.
    acc_white: [1024]i64 = undefined,
    acc_black: [1024]i64 = undefined,
    nn_h0: usize = 0, // first-layer width (accumulator size); 0 when not incremental
    nn_inc: bool = false, // eval_kind == .nn and the net is quantized
    // Optional so the wasm/freestanding build (no Io) can run fixed-depth searches.
    io: ?std.Io,
    // Time + stats + variety. deadline_ns == maxInt means "no time limit" and the
    // clock is never read (fixed-depth search — the gen/gate path — pays nothing).
    deadline_ns: i96 = std.math.maxInt(i96),
    nodes: u64 = 0,
    prng: std.Random.DefaultPrng,
    // Optional progress hook: called with (score, depth) after each completed root depth
    // (the browser worker streams it to the live eval bar). callconv(.c) so a wasm host
    // wrapper can be assigned.
    on_progress: ?*const fn (i32, u32) callconv(.c) void = null,

    pub fn init(alloc: std.mem.Allocator, io: ?std.Io, eval_kind: EvalKind, net: ?*const nn.Net, seed: u64) !Searcher {
        const s = Searcher{
            .alloc = alloc,
            .io = io,
            .tt_key = try alloc.alloc(u64, TT_SIZE),
            .tt_score = try alloc.alloc(i32, TT_SIZE),
            .tt_move = try alloc.alloc(i32, TT_SIZE),
            .tt_depth = try alloc.alloc(i16, TT_SIZE),
            .tt_flag = try alloc.alloc(u8, TT_SIZE),
            .tt_gen = try alloc.alloc(u16, TT_SIZE),
            .eval_kind = eval_kind,
            .eval_key = switch (eval_kind) {
                .handcrafted => 0,
                .nn => 0x9e3779b97f4a7c15,
                .handcrafted3 => 0x2545f4914f6cdd1d,
                .material => 0x6a09e667f3bcc908,
            },
            .net = net,
            .nn_inc = eval_kind == .nn and net != null and net.?.is_int,
            .nn_h0 = if (eval_kind == .nn and net != null and net.?.is_int) nn.h0(net.?) else 0,
            .prng = std.Random.DefaultPrng.init(seed),
        };
        @memset(s.tt_gen, 0);
        return s;
    }

    pub fn reseed(self: *Searcher, seed: u64) void {
        self.prng = std.Random.DefaultPrng.init(seed);
    }

    pub fn deinit(self: *Searcher) void {
        self.alloc.free(self.tt_key);
        self.alloc.free(self.tt_score);
        self.alloc.free(self.tt_move);
        self.alloc.free(self.tt_depth);
        self.alloc.free(self.tt_flag);
        self.alloc.free(self.tt_gen);
    }

    fn evalPos(self: *Searcher, b: *const [64]?Piece, turn: Color) i32 {
        return switch (self.eval_kind) {
            .handcrafted => eval.evalStm(b, turn),
            .handcrafted3 => eval.evalStmV3(b, turn),
            .material => eval.evalMaterial(b, turn),
            .nn => self.evalNn(b, turn),
        };
    }

    // nn eval at a leaf: a quantized net reads the maintained side-to-move accumulator
    // (the incremental fast path); a float net recomputes from scratch as before.
    fn evalNn(self: *Searcher, b: *const [64]?Piece, turn: Color) i32 {
        const net = self.net.?;
        if (!self.nn_inc) return nn.evaluate(net, b, turn);
        const acc = if (turn == .white) self.acc_white[0..self.nn_h0] else self.acc_black[0..self.nn_h0];
        const v = nn.evalFromAcc(net, acc);
        if (ACC_DEBUG) {
            const ref = nn.evaluate(net, b, turn);
            if (ref != v) std.debug.panic("nn accumulator desync: incremental={d} from-scratch={d}", .{ v, ref });
        }
        return v;
    }

    // make/unmake that also keep the NNUE accumulators in sync (quantized net only).
    // The deltas are read from the PRE-move board: on make that's the live board before
    // engine.makeMove; on unmake it's the board engine.unmakeMove just restored.
    fn nnMake(self: *Searcher, state: *State, m: Move) engine.Undo {
        if (self.nn_inc) self.accApplyMove(&state.board, m, true);
        return engine.makeMove(state, m);
    }
    fn nnUnmake(self: *Searcher, state: *State, m: Move, u: engine.Undo) void {
        engine.unmakeMove(state, m, u);
        if (self.nn_inc) self.accApplyMove(&state.board, m, false);
    }

    // Apply (add=true) or reverse (add=false) move m's piece deltas to both accumulators.
    // Mirrors engine.makeMove's board edits exactly: moved piece leaves `from` and (with
    // promotion) arrives at `to`, any captured piece leaves `to`, and a castle hops the rook.
    fn accApplyMove(self: *Searcher, b: *const [64]?Piece, m: Move, add: bool) void {
        const net = self.net.?;
        const accw = self.acc_white[0..self.nn_h0];
        const accb = self.acc_black[0..self.nn_h0];
        const moved = b[m.from].?;
        const color = moved.color;
        nn.accAddPiece(net, accw, accb, moved.role, color, m.from, !add); // leaves from
        const placed_role = m.promotion orelse moved.role;
        nn.accAddPiece(net, accw, accb, placed_role, color, m.to, add); // arrives at to
        if (b[m.to]) |cap| nn.accAddPiece(net, accw, accb, cap.role, cap.color, m.to, !add); // captured leaves to
        if (m.castle != 0) {
            const home: usize = if (color == .white) 0 else 56;
            const rf: usize = if (m.castle == 'K') home + 7 else home + 0;
            const rt: usize = if (m.castle == 'K') home + 5 else home + 3;
            const rook = b[rf].?;
            nn.accAddPiece(net, accw, accb, rook.role, rook.color, rf, !add);
            nn.accAddPiece(net, accw, accb, rook.role, rook.color, rt, add);
        }
    }

    // Invalidate every TT entry cheaply (no realloc) — used by the puzzle miner between
    // positions so a prior search's deep values can't leak into the next.
    pub fn clearTT(self: *Searcher) void {
        @memset(self.tt_gen, 0);
        self.cur_gen = 0;
    }

    // Monotonic nanoseconds: the std.Io clock on native, the JS-host clock on wasm.
    fn monoNs(self: *Searcher) i96 {
        if (self.io) |io| return std.Io.Clock.now(.awake, io).nanoseconds;
        if (is_wasm) return @intFromFloat(host.aposNowMs() * 1_000_000.0);
        return 0;
    }

    fn timeUp(self: *Searcher) bool {
        if (self.deadline_ns == std.math.maxInt(i96)) return false; // no time limit
        return self.monoNs() > self.deadline_ns;
    }

    fn repSeenHas(self: *Searcher, hash: u64) bool {
        for (self.rep_seen) |h| if (h == hash) return true;
        return false;
    }

    fn ttProbe(self: *Searcher, hash: u64) isize {
        if (!self.tt_enabled) return -1; // TT-free search: never read the table
        const h = hash ^ self.eval_key;
        const idx: usize = @intCast(h & TT_MASK);
        if (self.tt_gen[idx] != 0 and self.tt_key[idx] == h) return @intCast(idx);
        return -1;
    }

    fn ttStore(self: *Searcher, hash: u64, depth: i32, score: i32, flag: u8, move: i32) void {
        if (!self.tt_enabled) return; // TT-free search: leave the table untouched
        const h = hash ^ self.eval_key;
        const idx: usize = @intCast(h & TT_MASK);
        if (self.tt_gen[idx] == 0 or self.tt_key[idx] == h or self.tt_gen[idx] != self.cur_gen or depth >= self.tt_depth[idx]) {
            self.tt_key[idx] = h;
            self.tt_depth[idx] = @intCast(depth);
            self.tt_score[idx] = score;
            self.tt_flag[idx] = flag;
            self.tt_move[idx] = move;
            self.tt_gen[idx] = self.cur_gen;
        }
    }

    fn scoreMove(self: *Searcher, m: Move, b: *const [64]?Piece, ply: usize, pv_key: i32) i32 {
        const key = keyOf(m);
        if (key == pv_key) return 2_000_000;
        if (m.capture) {
            const victim: i32 = if (b[m.to]) |v| eval.value(v.role) else 0;
            const attacker: i32 = if (b[m.from]) |a| eval.value(a.role) else 0;
            return 1_000_000 + victim * 16 - attacker;
        }
        if (m.promotion) |pr| return 900_000 + eval.value(pr);
        if (m.jump) return 800_000;
        const k = self.killers[ply];
        if (k[0] == key or k[1] == key) return 700_000;
        return @min(self.history[@intCast(key)], 600_000);
    }

    // Stable descending sort by move score (matches JS Array.sort semantics on _o).
    fn orderMoves(self: *Searcher, moves: []Move, b: *const [64]?Piece, ply: usize, pv_key: i32) void {
        var scores: [1024]i32 = undefined;
        for (moves, 0..) |m, i| scores[i] = self.scoreMove(m, b, ply, pv_key);
        var i: usize = 1;
        while (i < moves.len) : (i += 1) {
            const km = moves[i];
            const ks = scores[i];
            var j = i;
            while (j > 0 and scores[j - 1] < ks) : (j -= 1) {
                moves[j] = moves[j - 1];
                scores[j] = scores[j - 1];
            }
            moves[j] = km;
            scores[j] = ks;
        }
    }

    fn qsearch(self: *Searcher, state: *State, alpha0: i32, beta: i32, qdepth: i32) i32 {
        self.nodes += 1;
        var alpha = alpha0;
        const in_check = engine.kingAttacked(&state.board, state.turn);
        var best: i32 = undefined;
        var stand_pat: i32 = 0;
        if (in_check) {
            best = -MATE;
        } else {
            stand_pat = self.evalPos(&state.board, state.turn);
            best = stand_pat;
            if (best >= beta) return best;
            if (best > alpha) alpha = best;
        }
        if (qdepth <= 0) return best;

        if (in_check) {
            var moves: engine.MoveList = .{};
            engine.legalMoves(state, &moves);
            if (moves.len == 0) return -MATE;
            self.orderMoves(moves.items[0..moves.len], &state.board, 0, 0);
            for (moves.slice()) |m| {
                const u = self.nnMake(state, m);
                const score = -self.qsearch(state, -beta, -alpha, qdepth - 1);
                self.nnUnmake(state, m, u);
                if (score > best) best = score;
                if (best > alpha) alpha = best;
                if (alpha >= beta) break;
            }
            return best;
        }

        var pseudo: engine.MoveList = .{};
        engine.generatePseudoMoves(&state.board, state.turn, &pseudo);
        var tac: engine.MoveList = .{};
        for (pseudo.slice()) |m| {
            if (m.capture or m.promotion != null or m.jump) tac.push(m);
        }
        self.orderMoves(tac.items[0..tac.len], &state.board, 0, 0);

        var saw_legal = false;
        const mover = state.turn;
        for (tac.slice()) |m| {
            if (m.capture and m.promotion == null and !m.jump) {
                if (state.board[m.to]) |victim| {
                    if (stand_pat + eval.value(victim.role) + DELTA_MARGIN <= alpha) continue;
                }
            }
            const u = self.nnMake(state, m);
            if (engine.kingAttacked(&state.board, mover)) { // illegal: mover left in check
                self.nnUnmake(state, m, u);
                continue;
            }
            saw_legal = true;
            const score = -self.qsearch(state, -beta, -alpha, qdepth - 1);
            self.nnUnmake(state, m, u);
            if (score > best) best = score;
            if (best > alpha) alpha = best;
            if (alpha >= beta) break;
        }

        if (!saw_legal and !engine.hasLegalMove(state, &pseudo)) return 0; // stalemate
        return best;
    }

    fn search(self: *Searcher, state: *State, depth0: i32, alpha0: i32, beta: i32, ply: usize, can_null: bool, hash: u64) i32 {
        if (self.timeUp()) {
            self.tainted = false;
            return 0;
        }
        self.nodes += 1;
        var alpha = alpha0;
        var depth = depth0;
        if (self.tt_enabled) {
            if (self.repSeenHas(hash)) {
                self.tainted = true;
                return 0;
            }
            var i: isize = @as(isize, @intCast(ply)) - 2;
            while (i >= 0) : (i -= 2) {
                if (self.rep_path[@intCast(i)] == hash) {
                    self.tainted = true;
                    return 0;
                }
            }
            self.rep_path[ply] = hash;
        }
        if (ply >= MAX_PLY) {
            self.tainted = false;
            return self.evalPos(&state.board, state.turn);
        }

        const in_check = engine.kingAttacked(&state.board, state.turn);
        if (in_check) depth += 1; // check extension
        if (depth <= 0) {
            self.tainted = false;
            return self.qsearch(state, alpha, beta, QDEPTH);
        }

        const alpha_orig = alpha;
        var tt_move_key: i32 = 0;
        if (self.tt_enabled) {
            const idx = self.ttProbe(hash);
            if (idx >= 0) {
                const ix: usize = @intCast(idx);
                tt_move_key = self.tt_move[ix];
                if (self.tt_depth[ix] >= depth) {
                    const s = fromTT(self.tt_score[ix], ply);
                    const flag = self.tt_flag[ix];
                    if (flag == EXACT) {
                        self.tainted = false;
                        return s;
                    }
                    if (flag == LOWER and s >= beta) {
                        self.tainted = false;
                        return s;
                    }
                    if (flag == UPPER and s <= alpha) {
                        self.tainted = false;
                        return s;
                    }
                }
            }
        }

        // Null-move pruning.
        if (can_null and !in_check and depth >= 3 and beta < MATE_THRESH and hasNonPawn(&state.board, state.turn)) {
            const nh = if (self.tt_enabled) hash ^ zobrist.sideKey() else 0;
            const saved_turn = state.turn;
            state.turn = opponent(state.turn); // null move: only the side to move flips
            const score = -self.search(state, depth - 3, -beta, -beta + 1, ply + 1, false, nh);
            state.turn = saved_turn;
            if (score >= beta) return beta;
        }

        var legal: engine.MoveList = .{};
        engine.legalMoves(state, &legal);
        if (legal.len == 0) {
            self.tainted = false;
            return if (in_check) -MATE - depth else 0;
        }
        self.orderMoves(legal.items[0..legal.len], &state.board, ply, tt_move_key);

        var best: i32 = -INF;
        var best_key: i32 = 0;
        var move_count: i32 = 0;
        var best_tainted = false;
        for (legal.slice()) |m| {
            move_count += 1;
            const child_hash = if (self.tt_enabled) zobrist.hashAfter(hash, state, m) else 0;
            const quiet = !m.capture and m.promotion == null and !m.jump;
            const u = self.nnMake(state, m);
            var score: i32 = undefined;
            var s_tainted: bool = undefined;
            if (move_count == 1) {
                score = -self.search(state, depth - 1, -beta, -alpha, ply + 1, true, child_hash);
                s_tainted = self.tainted;
            } else {
                const r: i32 = if (quiet and depth >= 3 and move_count > 3 and !in_check) 1 else 0;
                score = -self.search(state, depth - 1 - r, -alpha - 1, -alpha, ply + 1, true, child_hash);
                s_tainted = self.tainted;
                if (score > alpha and r > 0) {
                    score = -self.search(state, depth - 1, -alpha - 1, -alpha, ply + 1, true, child_hash);
                    s_tainted = self.tainted;
                }
                if (score > alpha and score < beta) {
                    score = -self.search(state, depth - 1, -beta, -alpha, ply + 1, true, child_hash);
                    s_tainted = self.tainted;
                }
            }
            self.nnUnmake(state, m, u);
            if (score > best) {
                best = score;
                best_key = keyOf(m);
                best_tainted = s_tainted;
            }
            if (best > alpha) alpha = best;
            if (alpha >= beta) {
                if (quiet) {
                    const key = keyOf(m);
                    if (self.killers[ply][0] != key) {
                        self.killers[ply][1] = self.killers[ply][0];
                        self.killers[ply][0] = key;
                    }
                    self.history[@intCast(key)] += depth * depth;
                }
                break;
            }
            if (self.timeUp()) break;
        }

        self.tainted = best_tainted;
        if (self.tt_enabled and !best_tainted and !self.timeUp()) {
            const flag: u8 = if (best <= alpha_orig) UPPER else if (best >= beta) LOWER else EXACT;
            self.ttStore(hash, depth, toTT(best, ply), flag, best_key);
        }
        return best;
    }

    // Choose a move: iterative deepening to `max_depth`, never past `max_ms`
    // (<= 0 means no time limit). `prev_hashes` are positions already seen in the
    // real game (repetition awareness). Mirrors chooseMoveDetailed.
    pub fn chooseMove(self: *Searcher, state: *const State, max_depth: u32, max_ms: i64, prev_hashes: []const u64) Result {
        return self.chooseMoveExcl(state, max_depth, max_ms, prev_hashes, &.{});
    }

    // As `chooseMove`, but ignores any root move whose `keyOf` (from*64+to) is in
    // `exclude`. Used by gen's `--opening-topk`: re-searching with the best moves so
    // far excluded yields the Nth-best, the same idiom as the JS puzzle miner.
    pub fn chooseMoveExcl(self: *Searcher, state: *const State, max_depth: u32, max_ms: i64, prev_hashes: []const u64, exclude: []const i32) Result {
        var root: engine.MoveList = .{};
        engine.legalMoves(state, &root);
        if (exclude.len > 0) {
            var w: usize = 0;
            for (root.items[0..root.len]) |m| {
                var skip = false;
                for (exclude) |k| {
                    if (keyOf(m) == k) {
                        skip = true;
                        break;
                    }
                }
                if (!skip) {
                    root.items[w] = m;
                    w += 1;
                }
            }
            root.len = w;
        }
        if (root.len == 0) return .{ .move = null, .score = 0, .depth = 0, .nodes = 0 };

        // Shuffle root for variety among equal choices.
        var rnd = self.prng.random();
        if (root.len > 1) {
            var i: usize = root.len - 1;
            while (i > 0) : (i -= 1) {
                const j = rnd.intRangeAtMost(usize, 0, i);
                const tmp = root.items[i];
                root.items[i] = root.items[j];
                root.items[j] = tmp;
            }
        }

        self.killers = std.mem.zeroes([MAX_PLY][2]i32);
        @memset(self.history[0..], 0);
        self.cur_gen = (self.cur_gen % 65535) + 1;
        const root_hash = zobrist.hashOf(state);
        self.rep_path[0] = root_hash;
        self.rep_seen = prev_hashes;
        self.deadline_ns = if (max_ms <= 0 or (self.io == null and !is_wasm))
            std.math.maxInt(i96)
        else
            self.monoNs() + @as(i96, max_ms) * 1_000_000;
        self.nodes = 0;

        var best_move = root.items[0];
        var completed: u32 = 0;
        var root_score: i32 = 0;
        var work = state.*; // one copy; the search make/unmakes it and restores to root
        // Seed the NNUE accumulators from the root; the search keeps them in sync through
        // make/unmake, so they're valid at every leaf (rebuilt fresh per chooseMove call).
        if (self.nn_inc) nn.accRefresh(self.net.?, self.acc_white[0..self.nn_h0], self.acc_black[0..self.nn_h0], &work.board);

        const depth_cap = @min(max_depth, 99);
        var depth: u32 = 1;
        while (depth <= depth_cap) : (depth += 1) {
            self.orderMoves(root.items[0..root.len], &state.board, 0, keyOf(best_move));
            var alpha: i32 = -INF;
            var best_score: i32 = -INF;
            var local_best = root.items[0];
            var aborted = false;
            var move_count: i32 = 0;
            const d: i32 = @as(i32, @intCast(depth)) - 1;
            for (root.slice()) |m| {
                move_count += 1;
                const child_hash = zobrist.hashAfter(root_hash, state, m);
                const u = self.nnMake(&work, m);
                var score: i32 = undefined;
                if (move_count == 1) {
                    score = -self.search(&work, d, -INF, -alpha, 1, true, child_hash);
                } else {
                    score = -self.search(&work, d, -alpha - 1, -alpha, 1, true, child_hash);
                    if (score > alpha) score = -self.search(&work, d, -INF, -alpha, 1, true, child_hash);
                }
                self.nnUnmake(&work, m, u);
                if (self.timeUp()) {
                    aborted = true;
                    break;
                }
                if (score > best_score) {
                    best_score = score;
                    local_best = m;
                }
                if (score > alpha) alpha = score;
            }
            if (!aborted) {
                best_move = local_best;
                completed = depth;
                root_score = best_score;
                if (self.on_progress) |cb| cb(root_score, depth); // stream the live eval bar
            }
            if (aborted or best_score >= MATE_THRESH) break;
        }

        // Ponder: the opponent's predicted reply = the best move the warmed TT stored for
        // the position after our best move (its PV child). from*64+to; no promotion needed.
        // (TT-free searches probe nothing, so ponder is simply null there.)
        var ponder: ?Move = null;
        {
            const child = engine.applyMove(state, best_move);
            const ix = self.ttProbe(zobrist.hashOf(&child));
            if (ix >= 0) {
                const mk = self.tt_move[@intCast(ix)];
                if (mk > 0) ponder = .{ .from = @intCast(@divTrunc(mk, 64)), .to = @intCast(@mod(mk, 64)) };
            }
        }

        return .{ .move = best_move, .score = root_score, .depth = completed, .nodes = self.nodes, .ponder = ponder };
    }

    // Score a position WITHOUT reading or writing the transposition table, so the probe
    // can't perturb this engine's real games (used once per game by the match harvest to
    // value the one opening ply the winner didn't search).
    pub fn chooseMoveNoTT(self: *Searcher, state: *const State, max_depth: u32, max_ms: i64, prev_hashes: []const u64) Result {
        self.tt_enabled = false;
        defer self.tt_enabled = true;
        return self.chooseMoveExcl(state, max_depth, max_ms, prev_hashes, &.{});
    }
};
