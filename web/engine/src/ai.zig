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
// Refinements above that base are individually switchable (`SearchOpts`), so one
// binary can play its new search against its own predecessor — `apos-match
// --search-b=none` — with the two sides differing in nothing else. Gate a search
// change by --nodes (equal work) or --movetime, NEVER by depth: a selective search
// visits far fewer nodes at the same nominal depth, so a depth-paced match measures
// how much work each side did rather than how well it used it.
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

// --- search-refinement tuning ------------------------------------------------------
// Every margin here is in centipawns and is sized against the NN EVAL'S RANGE, not
// against standard-chess practice. The nn eval is tanh-squashed x `scale`, so it is hard
// capped at +/-600 cp for every champion so far (see docs/nn-training.md) — a textbook
// "150 cp per ply" margin is a quarter of the entire scale here and would make the
// shallow-depth prunings below fire either always or never. These are ~55-90 cp/ply.
const RFP_MAX_DEPTH: i32 = 5; // reverse futility applies at depth <= this
const RFP_MARGIN: i32 = 60; // ...with this much slack per remaining ply
const FP_MAX_DEPTH: i32 = 2; // frontier futility applies at depth <= this
const FP_MARGIN: i32 = 80; // ...per remaining ply,
const FP_BASE: i32 = 60; // ...plus a fixed floor (a quiet move is rarely worth more)
const LMP_MAX_DEPTH: i32 = 4; // late-move pruning applies at depth <= this
const LMR_DIV: f64 = 2.25; // late-move reduction: r = 0.5 + ln(d)*ln(mc)/LMR_DIV
const HIST_MAX: i32 = 16384; // history is kept in [-HIST_MAX, HIST_MAX] by the gravity term
const HIST_GOOD: i32 = HIST_MAX / 4; // above this a quiet move is reduced one ply less
const ASP_DELTA: i32 = 30; // first aspiration half-window around the previous score

// Quiet moves searched at one node, remembered so a beta cutoff can apply the history
// MALUS to the ones that failed. Beyond this many the malus is simply not applied —
// a node with 48 quiet moves before a cutoff has nothing useful to teach the table.
const MAX_QUIETS: usize = 48;

// Late-move reduction table, r = 0.5 + ln(depth)*ln(move_count)/LMR_DIV, saturating at
// depth/move-count 63. Built at comptime so the search reads a byte instead of two logs.
const LMR = blk: {
    @setEvalBranchQuota(200000);
    var t: [64][64]u8 = undefined;
    for (0..64) |d| {
        for (0..64) |m| {
            if (d == 0 or m == 0) {
                t[d][m] = 0;
                continue;
            }
            const r = 0.5 + @log(@as(f64, @floatFromInt(d))) * @log(@as(f64, @floatFromInt(m))) / LMR_DIV;
            t[d][m] = if (r <= 0) 0 else @intFromFloat(r);
        }
    }
    break :blk t;
};

// Move-count threshold for late-move pruning: past this many quiet moves at a shallow
// depth the rest are not searched at all. 3 + d^2 is the usual shape (4, 7, 12, 19).
fn lmpCount(depth: i32) i32 {
    return 3 + depth * depth;
}

// --- search feature switches --------------------------------------------------------
// Each refinement below the base alpha-beta is individually switchable so that ONE binary
// can play a search change against its own predecessor: `apos-match --search-b=none` is
// "the search as it shipped before this file grew these", and `--search-a=-lmp` ablates a
// single feature out of the full set. Without this, gating a search change means keeping
// two builds around and trusting they differ in nothing else.
//
// A default of TRUE means the refinement is part of the shipped search; FALSE means it is
// implemented and switchable but LOST its gate. Measured 2026-08-08, all at --nodes=50000
// over 600 games against `none` (this file's search before any of them), on champion Uma:
//
//   rfp,nullr,asp,lmr    +74 +/- 28 Elo   -75% nodes to depth 8   371 nodes/ms   <- shipped
//   + lmp + fp + hist    +17 +/- 24 Elo   -91% nodes to depth 8   284 nodes/ms
//   lmr,lmp,hist,asp     ~13% score (abandoned at 52 games)
//
// The lesson is that `lmp` is the whole difference, in both directions. It is the biggest
// tree cut of the seven (-63% on its own) and also the only one that costs nodes/SECOND
// (368 -> 310, because pruning quiet moves shifts the surviving node mix toward eval-heavy
// quiescence), and it is the least accurate (it agreed with the unpruned search on 9 of 12
// midgame positions where the others managed 11-12). Blind move-count pruning is simply a
// bad trade in a variant whose quiet moves carry the jumps. `fp` earns its 31%-on-its-own
// only when `lmp` is off, and then costs 12% of nodes/ms for it. `hist` is off on its own
// gate below. Everything shipped is FREE per node — the set measures 371 nodes/ms against
// the baseline's 373 — so its equal-work gate is also its equal-time gate.
// Which search the games were played by, stamped into every harvested game record as `se`
// (scripts/gameRecord.mjs) so the rating pool never averages two different engines under one
// (engine, depth) label. A pool node is (engine, depth) and says nothing about the search, but
// the search decides what a depth is WORTH: era 2 loses a fixed-depth-6 match to era 1 by ~380
// Elo while reaching depth 8 on a quarter of the nodes. Mixing them silently is a measurement
// bug, not drift. BUMP on any change to what the shipped search does — the defaults below, the
// margins, or the code they gate. Era 1 = the search before 2026-08-08 (unstamped in old data);
// era 2 = rfp+lmr+nullr+asp. Mirrored by SEARCH_ERA in src/ai.js.
pub const SEARCH_ERA: u32 = 2;

pub const SearchOpts = struct {
    rfp: bool = true, // reverse futility pruning (static null move)
    fp: bool = false, // frontier futility pruning of quiet moves — see the note above
    lmp: bool = false, // late move pruning (move-count based) — see the note above
    lmr: bool = true, // depth/move-count LMR table (off = the flat 1-ply reduction)
    hist: bool = false, // history gravity + malus + countermove ordering
    nullr: bool = true, // depth- and eval-scaled null-move reduction (off = flat R=3)
    asp: bool = true, // aspiration windows at the root

    pub const NAMES = [_][]const u8{ "rfp", "fp", "lmp", "lmr", "hist", "nullr", "asp" };

    // Every switch off — the search as it was before any of these existed, and what a
    // search change is gated against.
    pub fn none() SearchOpts {
        return .{ .rfp = false, .fp = false, .lmp = false, .lmr = false, .hist = false, .nullr = false, .asp = false };
    }
    // Every switch on, including the ones that lost their gate. Only useful for re-running
    // that measurement — it is deliberately NOT the default.
    pub fn all() SearchOpts {
        return .{ .rfp = true, .fp = true, .lmp = true, .lmr = true, .hist = true, .nullr = true, .asp = true };
    }

    fn setByName(self: *SearchOpts, name: []const u8, on: bool) bool {
        if (std.mem.eql(u8, name, "rfp")) self.rfp = on else if (std.mem.eql(u8, name, "fp")) self.fp = on else if (std.mem.eql(u8, name, "lmp")) self.lmp = on else if (std.mem.eql(u8, name, "lmr")) self.lmr = on else if (std.mem.eql(u8, name, "hist")) self.hist = on else if (std.mem.eql(u8, name, "nullr")) self.nullr = on else if (std.mem.eql(u8, name, "asp")) self.asp = on else return false;
        return true;
    }

    // Parse a comma list. A spec that NAMES features ("rfp,lmr") starts from nothing and
    // enables exactly those; a spec that only SUBTRACTS ("-lmp") starts from the SHIPPED
    // set, so `--search-a=-rfp` reads as "the engine, minus rfp". "all" (every switch) and
    // "none" (no switch) set the base explicitly. Returns null on an unknown token, so a
    // typo is a hard error rather than a silently different engine.
    pub fn parse(spec: []const u8) ?SearchOpts {
        var adds = false;
        var it = std.mem.splitScalar(u8, spec, ',');
        while (it.next()) |raw| {
            const t = std.mem.trim(u8, raw, " ");
            if (t.len == 0 or t[0] == '-') continue;
            if (std.mem.eql(u8, t, "all") or std.mem.eql(u8, t, "none") or
                std.mem.eql(u8, t, "off") or std.mem.eql(u8, t, "default")) continue;
            adds = true;
        }
        var o: SearchOpts = if (adds) SearchOpts.none() else .{};
        it = std.mem.splitScalar(u8, spec, ',');
        while (it.next()) |raw| {
            const t = std.mem.trim(u8, raw, " ");
            if (t.len == 0) continue;
            if (std.mem.eql(u8, t, "all")) {
                o = SearchOpts.all();
                continue;
            }
            if (std.mem.eql(u8, t, "default")) {
                o = .{};
                continue;
            }
            if (std.mem.eql(u8, t, "none") or std.mem.eql(u8, t, "off")) {
                o = SearchOpts.none();
                continue;
            }
            const on = t[0] != '-';
            const name = if (t[0] == '-' or t[0] == '+') t[1..] else t;
            if (!o.setByName(name, on)) return null;
        }
        return o;
    }

    // "all", "none", or the comma list of what is on — for the match runner's header.
    pub fn describe(self: SearchOpts, buf: []u8) []const u8 {
        const on = [_]bool{ self.rfp, self.fp, self.lmp, self.lmr, self.hist, self.nullr, self.asp };
        var n: usize = 0;
        for (on) |b| {
            if (b) n += 1;
        }
        if (n == on.len) return "all";
        if (n == 0) return "none";
        var w: usize = 0;
        for (on, NAMES) |b, name| {
            if (!b) continue;
            if (w > 0 and w < buf.len) {
                buf[w] = ',';
                w += 1;
            }
            for (name) |c| {
                if (w < buf.len) {
                    buf[w] = c;
                    w += 1;
                }
            }
        }
        return buf[0..w];
    }
};

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
    // Countermove table: counter[previous move's key] = the quiet reply that refuted it.
    // Indexed by the PARENT's move, so it is a per-move memory the killers (per-ply) miss.
    counter: [64 * 64]i32 = undefined,
    rep_path: [MAX_PLY]u64 = undefined,
    rep_seen: []const u64 = &.{},
    tainted: bool = false,
    // Which search refinements are active. All on by default; the match runner sets them
    // per side so a search change can be played against its own predecessor (SearchOpts).
    opts: SearchOpts = .{},
    // Eval selection.
    eval_kind: EvalKind,
    eval_key: u64,
    net: ?*const nn.Net,
    // Loser ("Lemming") mode: keep the WORST-scoring root move instead of the best, so the
    // engine tries to lose as fast as possible. A move-selection mode layered on the nn eval
    // (set by the host after init); the search itself is unchanged. See chooseMoveExcl.
    minimize: bool = false,
    // Incremental NNUE accumulators (raw, pre-ReLU), maintained through make/unmake when
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
    // Fixed-NODE budget: the search aborts once `nodes` reaches `node_cap`, exactly the way
    // it aborts on `deadline_ns` (see outOfBudget). maxInt(u64) means "no node limit" — the
    // default, and unreachable by a u64 counter, so every existing fixed-depth / movetime
    // path behaves bit-for-bit as before and pays one u64 compare per abort check.
    //
    // WHY a node budget exists at all: at a fixed DEPTH a pruning or move-ordering gain is
    // invisible (the same tree, the same move — only cheaper), so a fixed-depth gate rejects
    // every correct pruning change; and wall-clock movetime on a machine that is also
    // training is too noisy to gate on. A node budget prices the speed gain and the accuracy
    // loss on one scale AND is deterministic, so a match is reproducible under load.
    node_cap: u64 = std.math.maxInt(u64),
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

    // The search's single stop condition: out of TIME or out of NODES. Both are monotonic
    // (the clock never runs backwards, `nodes` only grows), so once it reads true every
    // ancestor sees it too — which is what makes the unwind safe: in-tree nodes return 0,
    // move loops break, the TT store is skipped so an incomplete score can't poison the
    // persistent table, and the root marks the iteration aborted and keeps the move from the
    // last COMPLETED iteration. Every former timeUp() call site checks this instead, so a
    // node budget behaves exactly like a time limit rather than through a parallel path.
    // The node compare comes first because it is the cheap one: against the maxInt default it
    // is a single always-false u64 comparison, and the no-time-limit test behind it still
    // returns without reading the clock — so an unlimited search pays one extra compare.
    fn outOfBudget(self: *Searcher) bool {
        return self.nodes >= self.node_cap or self.timeUp();
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

    fn scoreMove(self: *Searcher, m: Move, b: *const [64]?Piece, ply: usize, pv_key: i32, counter_key: i32) i32 {
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
        if (key == counter_key) return 650_000; // countermove: below the killers, above history
        return @min(self.history[@intCast(key)], 600_000);
    }

    // Bounded history update ("gravity"): the correction term pulls the entry toward 0 in
    // proportion to how far it already is, so the table stays inside [-HIST_MAX, HIST_MAX]
    // and a move that stopped working decays instead of coasting on an old score. The
    // unbounded `+= depth*depth` it replaces let early cutoffs dominate ordering forever.
    fn histBump(self: *Searcher, key: i32, bonus: i32) void {
        const idx: usize = @intCast(key);
        const b = @max(-HIST_MAX, @min(HIST_MAX, bonus));
        const mag: i32 = if (b < 0) -b else b;
        self.history[idx] += b - @divTrunc(self.history[idx] * mag, HIST_MAX);
    }

    // Stable descending sort by move score (matches JS Array.sort semantics on _o).
    fn orderMoves(self: *Searcher, moves: []Move, b: *const [64]?Piece, ply: usize, pv_key: i32, counter_key: i32) void {
        var scores: [1024]i32 = undefined;
        for (moves, 0..) |m, i| scores[i] = self.scoreMove(m, b, ply, pv_key, counter_key);
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
        // A NODE IS COUNTED ON ENTRY, once, before anything can return early — here for
        // quiescence nodes and at the top of search() for full-width ones, the two disjoint
        // kinds of node this engine visits. Entry is the only point every visited node passes
        // through exactly once, so it is the only definition the JS reference (ai.js, same two
        // places) can mirror without ambiguity; counting cutoffs, make/unmakes or leaf evals
        // instead would make "20000 nodes" mean different trees in the two engines and a
        // JS/Zig comparison at a fixed budget would diverge for no interesting reason.
        // qsearch itself has no abort check (QDEPTH bounds it), so a node-capped search
        // overshoots its cap by at most the quiescence subtree in flight — deterministically,
        // since that subtree is a pure function of the position.
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
            self.orderMoves(moves.items[0..moves.len], &state.board, 0, 0, 0);
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
        self.orderMoves(tac.items[0..tac.len], &state.board, 0, 0, 0);

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

    fn search(self: *Searcher, state: *State, depth0: i32, alpha0: i32, beta: i32, ply: usize, can_null: bool, hash: u64, prev_key: i32) i32 {
        if (self.outOfBudget()) {
            self.tainted = false;
            return 0;
        }
        self.nodes += 1; // counted on entry, after the abort check — an aborted node is not a visited one
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

        // Static evaluation of THIS node, computed once and shared by the three refinements
        // that need it (reverse futility, the null-move reduction, frontier futility). It is
        // only worth computing where one of them can actually fire: in check none of them
        // apply, and above RFP_MAX_DEPTH only the null move reads it. With a quantized net
        // this is an accumulator read; with the handcrafted eval it is two move generations,
        // which is why it stays behind that guard instead of being taken unconditionally.
        const wants_null = can_null and depth >= 3 and beta < MATE_THRESH;
        const wants_static = !in_check and
            ((self.opts.rfp and depth <= RFP_MAX_DEPTH) or
                (self.opts.fp and depth <= FP_MAX_DEPTH) or
                (self.opts.nullr and wants_null));
        const static_eval: i32 = if (wants_static) self.evalPos(&state.board, state.turn) else 0;

        // Reverse futility pruning ("static null move"): if the side to move is so far ahead
        // that giving up RFP_MARGIN per remaining ply still fails high, the node is not worth
        // a search. It is the null-move idea without the search — and unlike the null move it
        // needs no zugzwang guard, because it never claims a line, only that the margin is
        // unreachable. Skipped near mate scores, where a cp margin means nothing.
        if (self.opts.rfp and !in_check and depth <= RFP_MAX_DEPTH and
            beta > -MATE_THRESH and beta < MATE_THRESH and
            static_eval - RFP_MARGIN * depth >= beta)
        {
            self.tainted = false;
            return static_eval;
        }

        // Null-move pruning. The reduction is flat 3 in the base search; with `nullr` it
        // scales with depth (a deeper node can afford to give up more) and with how far the
        // static eval already is above beta (the further ahead, the more certain the pass
        // still fails high) — and the node has to be at or above beta to try it at all,
        // which is where the eval term gets its meaning.
        if (wants_null and !in_check and hasNonPawn(&state.board, state.turn) and
            (!self.opts.nullr or static_eval >= beta))
        {
            const r_null: i32 = if (self.opts.nullr)
                @max(3, @min(depth - 1, 3 + @divTrunc(depth, 5) + @min(@divTrunc(static_eval - beta, 160), 2)))
            else
                3;
            const nh = if (self.tt_enabled) hash ^ zobrist.sideKey() else 0;
            const saved_turn = state.turn;
            state.turn = opponent(state.turn); // null move: only the side to move flips
            const score = -self.search(state, depth - r_null, -beta, -beta + 1, ply + 1, false, nh, 0);
            state.turn = saved_turn;
            if (score >= beta) return beta;
        }

        var legal: engine.MoveList = .{};
        engine.legalMoves(state, &legal);
        if (legal.len == 0) {
            self.tainted = false;
            return if (in_check) -MATE - depth else 0;
        }
        const counter_key: i32 = if (self.opts.hist and prev_key != 0) self.counter[@intCast(prev_key)] else 0;
        self.orderMoves(legal.items[0..legal.len], &state.board, ply, tt_move_key, counter_key);

        var best: i32 = -INF;
        var best_key: i32 = 0;
        var move_count: i32 = 0;
        var best_tainted = false;
        // Quiet moves already tried at this node, for the history malus on a cutoff.
        var quiets: [MAX_QUIETS]i32 = undefined;
        var n_quiets: usize = 0;
        for (legal.slice()) |m| {
            move_count += 1;
            const quiet = !m.capture and m.promotion == null and !m.jump;
            // Shallow-depth pruning of QUIET moves only — the variant's tactics live in
            // captures, promotions and jumps, and none of those are ever skipped here. Both
            // rules require a real score in hand (`best > -MATE_THRESH`), so the first move
            // is always searched and a node can never come back empty.
            if (quiet and !in_check and best > -MATE_THRESH) {
                // Late move pruning: past lmpCount(depth) quiet moves at a shallow depth,
                // the rest almost never beat what ordering already put first.
                if (self.opts.lmp and depth <= LMP_MAX_DEPTH and move_count > lmpCount(depth)) continue;
                // Frontier futility: a quiet move that cannot lift a static eval this far
                // below alpha into the window within the remaining plies.
                if (self.opts.fp and depth <= FP_MAX_DEPTH and
                    static_eval + FP_MARGIN * depth + FP_BASE <= alpha) continue;
            }
            const key = keyOf(m);
            if (quiet and n_quiets < MAX_QUIETS) {
                quiets[n_quiets] = key;
                n_quiets += 1;
            }
            const child_hash = if (self.tt_enabled) zobrist.hashAfter(hash, state, m) else 0;
            const u = self.nnMake(state, m);
            var score: i32 = undefined;
            var s_tainted: bool = undefined;
            if (move_count == 1) {
                score = -self.search(state, depth - 1, -beta, -alpha, ply + 1, true, child_hash, key);
                s_tainted = self.tainted;
            } else {
                var r: i32 = 0;
                if (quiet and depth >= 3 and move_count > 3 and !in_check) {
                    if (self.opts.lmr) {
                        r = LMR[@intCast(@min(depth, 63))][@intCast(@min(move_count, 63))];
                        // A quiet move with a strong history is reduced one ply less: the
                        // table already says it works, so the late-move assumption is weaker.
                        if (self.opts.hist and self.history[@intCast(key)] > HIST_GOOD) r -= 1;
                        r = @max(0, @min(r, depth - 2)); // never reduce into qsearch
                    } else r = 1;
                }
                score = -self.search(state, depth - 1 - r, -alpha - 1, -alpha, ply + 1, true, child_hash, key);
                s_tainted = self.tainted;
                if (score > alpha and r > 0) {
                    score = -self.search(state, depth - 1, -alpha - 1, -alpha, ply + 1, true, child_hash, key);
                    s_tainted = self.tainted;
                }
                if (score > alpha and score < beta) {
                    score = -self.search(state, depth - 1, -beta, -alpha, ply + 1, true, child_hash, key);
                    s_tainted = self.tainted;
                }
            }
            self.nnUnmake(state, m, u);
            if (score > best) {
                best = score;
                best_key = key;
                best_tainted = s_tainted;
            }
            if (best > alpha) alpha = best;
            if (alpha >= beta) {
                if (quiet) {
                    if (self.killers[ply][0] != key) {
                        self.killers[ply][1] = self.killers[ply][0];
                        self.killers[ply][0] = key;
                    }
                    if (self.opts.hist) {
                        // Bonus to the move that cut, MALUS to every quiet move tried before
                        // it. Rewarding only the winner teaches the table which moves are
                        // good but never which are bad, so a quiet move that keeps getting
                        // searched early and keeps failing holds its slot forever.
                        const bonus = @min(depth * depth * 16 + 32 * depth, HIST_MAX / 4);
                        self.histBump(key, bonus);
                        for (quiets[0..n_quiets]) |qk| {
                            if (qk != key) self.histBump(qk, -bonus);
                        }
                        if (prev_key != 0) self.counter[@intCast(prev_key)] = key;
                    } else {
                        self.history[@intCast(key)] += depth * depth;
                    }
                }
                break;
            }
            if (self.outOfBudget()) break;
        }

        self.tainted = best_tainted;
        if (self.tt_enabled and !best_tainted and !self.outOfBudget()) {
            const flag: u8 = if (best <= alpha_orig) UPPER else if (best >= beta) LOWER else EXACT;
            self.ttStore(hash, depth, toTT(best, ply), flag, best_key);
        }
        return best;
    }

    const RootWindow = struct { score: i32, move: Move, aborted: bool };

    // One root pass over the (already ordered) root move list inside the window [lo, hi].
    // Split out of the iterative-deepening loop so the aspiration window can call it again
    // with a wider window when the guess was wrong. `hi` is a real beta: a move that beats
    // it ends the pass (there is no point proving how much better it is under a window that
    // is about to be reopened). With lo = -INF and hi = INF this is exactly the full-width
    // root PVS pass the search has always done.
    fn rootWindow(self: *Searcher, work: *State, state: *const State, root: *const engine.MoveList, d: i32, lo: i32, hi: i32, root_hash: u64) RootWindow {
        var alpha = lo;
        var best_score: i32 = -INF;
        var local_best = root.items[0];
        var move_count: i32 = 0;
        for (root.slice()) |m| {
            move_count += 1;
            const child_hash = zobrist.hashAfter(root_hash, state, m);
            const key = keyOf(m);
            const u = self.nnMake(work, m);
            var score: i32 = undefined;
            if (move_count == 1) {
                score = -self.search(work, d, -hi, -alpha, 1, true, child_hash, key);
            } else {
                score = -self.search(work, d, -alpha - 1, -alpha, 1, true, child_hash, key);
                if (score > alpha and score < hi) score = -self.search(work, d, -hi, -alpha, 1, true, child_hash, key);
            }
            self.nnUnmake(work, m, u);
            if (self.outOfBudget()) return .{ .score = 0, .move = local_best, .aborted = true };
            if (score > best_score) {
                best_score = score;
                local_best = m;
            }
            if (score > alpha) alpha = score;
            if (alpha >= hi) break; // fail high — the caller reopens the window
        }
        return .{ .score = best_score, .move = local_best, .aborted = false };
    }

    // Choose a move: iterative deepening to `max_depth`, never past `max_ms`
    // (<= 0 means no time limit) and never past `max_nodes` (0 means no node limit — the
    // same "non-positive = unbounded" convention as max_ms). `prev_hashes` are positions
    // already seen in the real game (repetition awareness). Mirrors chooseMoveDetailed.
    pub fn chooseMove(self: *Searcher, state: *const State, max_depth: u32, max_ms: i64, max_nodes: u64, prev_hashes: []const u64) Result {
        return self.chooseMoveExcl(state, max_depth, max_ms, max_nodes, prev_hashes, &.{});
    }

    // As `chooseMove`, but ignores any root move whose `keyOf` (from*64+to) is in
    // `exclude`. Used by gen's `--opening-topk`: re-searching with the best moves so
    // far excluded yields the Nth-best, the same idiom as the JS puzzle miner.
    pub fn chooseMoveExcl(self: *Searcher, state: *const State, max_depth: u32, max_ms: i64, max_nodes: u64, prev_hashes: []const u64, exclude: []const i32) Result {
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
        // No legal moves = terminal. Report the true side-to-move score so a caller
        // that searches/ponders into this position (e.g. the eval bar) pins to the
        // result instead of reading a bare 0 as an even position: -MATE when
        // checkmated, 0 for stalemate. Mirrors the in-tree terminal handling.
        if (root.len == 0) {
            const lost = engine.kingAttacked(&state.board, state.turn);
            return .{ .move = null, .score = if (lost) -MATE else 0, .depth = 0, .nodes = 0 };
        }

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
        @memset(self.counter[0..], 0);
        self.cur_gen = (self.cur_gen % 65535) + 1;
        const root_hash = zobrist.hashOf(state);
        self.rep_path[0] = root_hash;
        self.rep_seen = prev_hashes;
        self.deadline_ns = if (max_ms <= 0 or (self.io == null and !is_wasm))
            std.math.maxInt(i96)
        else
            self.monoNs() + @as(i96, max_ms) * 1_000_000;
        self.nodes = 0;
        // Node cap for this search only, set right beside the deadline and from the same kind
        // of argument, so the two budgets stay one mechanism. `nodes` is reset above, so the
        // cap is per-search (per move), not per game — the way movetime is.
        self.node_cap = if (max_nodes == 0) std.math.maxInt(u64) else max_nodes;

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
            self.orderMoves(root.items[0..root.len], &state.board, 0, keyOf(best_move), 0);
            var best_score: i32 = if (self.minimize) INF else -INF;
            var local_best = root.items[0];
            var aborted = false;
            const d: i32 = @as(i32, @intCast(depth)) - 1;
            if (self.minimize) {
                // Loser mode: every root move needs its TRUE score (so the worst is exact),
                // so search each with a full window — no alpha tightening, no PVS, and no
                // aspiration window (which is an assumption about the BEST move's score).
                for (root.slice()) |m| {
                    const child_hash = zobrist.hashAfter(root_hash, state, m);
                    const u = self.nnMake(&work, m);
                    const score = -self.search(&work, d, -INF, INF, 1, true, child_hash, keyOf(m));
                    self.nnUnmake(&work, m, u);
                    if (self.outOfBudget()) {
                        aborted = true;
                        break;
                    }
                    if (score < best_score) {
                        best_score = score;
                        local_best = m;
                    }
                }
            } else {
                // Aspiration window: the score at depth d is usually close to the score at
                // d-1, so searching [prev-delta, prev+delta] instead of the full window makes
                // every node's window narrower and cuts the tree. The cost is a re-search when
                // the guess is wrong, which is why the window only opens once there is a
                // previous score to guess from and the position is not already a forced mate.
                var lo: i32 = -INF;
                var hi: i32 = INF;
                var delta: i32 = ASP_DELTA;
                if (self.opts.asp and depth >= 4 and completed > 0 and
                    root_score < MATE_THRESH and root_score > -MATE_THRESH)
                {
                    lo = root_score - delta;
                    hi = root_score + delta;
                }
                while (true) {
                    const rr = self.rootWindow(&work, state, &root, d, lo, hi, root_hash);
                    if (rr.aborted) {
                        aborted = true;
                        break;
                    }
                    // Fail low: the true score is below the window, so the move that came
                    // back is not trustworthy — widen downward and keep the previous best.
                    if (rr.score <= lo and lo != -INF) {
                        hi = @divTrunc(lo + hi, 2);
                        lo = @max(-INF, rr.score - delta);
                        delta += @divTrunc(delta, 2) + 5;
                        continue;
                    }
                    // Fail high: a move beat the window, so it IS the new best; widen upward.
                    if (rr.score >= hi and hi != INF) {
                        local_best = rr.move;
                        hi = @min(INF, rr.score + delta);
                        delta += @divTrunc(delta, 2) + 5;
                        continue;
                    }
                    best_score = rr.score;
                    local_best = rr.move;
                    break;
                }
            }
            if (!aborted) {
                best_move = local_best;
                completed = depth;
                root_score = best_score;
                if (self.on_progress) |cb| cb(root_score, depth); // stream the live eval bar
            }
            // Stop once the outcome is forced: a found win (normal) or a found loss (loser mode).
            if (aborted or (if (self.minimize) best_score <= -MATE_THRESH else best_score >= MATE_THRESH)) break;
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
    pub fn chooseMoveNoTT(self: *Searcher, state: *const State, max_depth: u32, max_ms: i64, max_nodes: u64, prev_hashes: []const u64) Result {
        self.tt_enabled = false;
        defer self.tt_enabled = true;
        return self.chooseMoveExcl(state, max_depth, max_ms, max_nodes, prev_hashes, &.{});
    }
};
