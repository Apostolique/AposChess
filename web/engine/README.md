# AposChess engine (Zig)

The production implementation of the engine core (`board` → `engine` → `ai` → `nn`).
One source tree, two targets:

- **Native CLI binaries** the npm tools spawn — `apos-gen` (self-play generation),
  `apos-match` (match runner / gate / rank gauntlet), `apos-bench`, `apos-parity`.
- **`apos.wasm`** — the in-browser search, loaded by the Web Worker (`web/src/aiWorker.js`).

The JS engine in `web/src/` is the readable **reference**: rules and search are
implemented there too, and Zig is validated against it (see Parity). The dataset,
featurize, and training glue stay in JS/Python.

- **Toolchain:** Zig **0.16.0** (pinned — pre-1.0, expect churn on upgrades).

## Files (`src/`)

| File | Mirrors | Contents |
|------|---------|----------|
| `board.zig` | `board.js` | board rep, FEN parse/serialize, `squareName` |
| `engine.zig` | `engine.js` | move generation, `applyMove` (make/unmake, in-place), safety zones, `gameStatus` |
| `zobrist.zig` | (key gen in `ai.js`) | Zobrist keys + incremental hashing, and `isThreefold` (repetition needs game history, so it can't live in `gameStatus`) |
| `eval.zig` | `evalStm` in `ai.js` | handcrafted PST eval (bit-exact vs JS) |
| `nn.zig` | `nn.js` | neural-net eval (KING_BUCKETS = 0; ±1 cp vs JS) |
| `ai.zig` | `ai.js` | search: alpha-beta, PVS, null-move, LMR, qsearch, TT, killers/history, repetition |
| `wasm.zig` | — | wasm32 entry: exports `searchFen`/`allocBytes`/`memory`; movetime/progress via imported `env` callbacks |
| `main_gen.zig` / `main_match.zig` / `main_bench.zig` | — | native CLI entry points |
| `parity.zig` | — | replays the JS oracle and checks every layer |

## Build & verify

Run from `web/engine/`:

```sh
zig build parity                       # all engine layers vs the JS oracle
zig build bench  -Doptimize=ReleaseFast -- --depth=8 [--nn | --weights=PATH]  # reports nps + ns/node
zig build match  -Doptimize=ReleaseFast -- --games=20 --depth=6 [--nn]
zig build gen    -Doptimize=ReleaseFast -- --games=200 --depth=6 --eval=nn
zig build wasm                         # -> zig-out/bin/apos.wasm
```

## Gating a search change (`--search`)

`ai.SearchOpts` is a switch per search refinement above the base alpha-beta —
`rfp`, `fp`, `lmp`, `lmr`, `hist`, `nullr`, `asp` (see `ai.zig` for what each is, and
for the gate numbers that decided which are on). `apos-match` takes `--search-a=SPEC` /
`--search-b=SPEC` and `apos-bench` takes `--search=SPEC`, so **one binary plays its new
search against its own predecessor** and the two sides provably differ in nothing else.
A spec that names features (`rfp,lmr`) enables exactly those; a spec that only subtracts
(`-rfp`) starts from the shipped set; `none` is every switch off (the search before any
of these existed, and what a change is gated against), `all` is every switch on
including the ones that lost, `default` is the shipped set. An unknown name is a hard
error, because a typo would otherwise play the shipped search against itself and report
"no change" — the one answer a search gate must never produce by accident.

`--search-a`/`--search-b` and `--save-games` are mutually exclusive (exit 2). A harvested
record stamps one search era (`se` = `ai.SEARCH_ERA`) for the whole game, so a mixed-search
game would be filed under a player label it never played with, and the rating pool would
then average two engines under one `(engine, depth)` id. **Bump `SEARCH_ERA` (`ai.zig` and
`src/ai.js`, they must match) whenever the shipped search changes what it does** — the
defaults, the margins, or the code they gate. Everything the ladder knows is scoped by it:
`rank:pool --era` rates one era at a time, and the era-2 pin is a measured offset from
era 1 rather than a fresh 1500, so absolute Elo survives the change (`docs/tools.md`).

Shipping era N+1 is three steps beyond the gate itself:

```
# 1. bump SEARCH_ERA in ai.zig and src/ai.js (they must match), then rebuild
# 2. measure the new pin — A is the new search, B is era N's spec, no --save-games
apos-match --eval-a=handcrafted --eval-b=handcrafted --depth=6 --depth-b=6 \
           --search-b=rfp,lmr,nullr,asp --games=400
# 3. add `pin(N) + that Elo` to PIN_ELO_BY_ERA in scripts/depth-ladder.mjs, and
#    archive the old ledger as loop/engine-elo.ladder.era<N>.json
```

The ladder then re-measures itself from era N+1 games as they arrive. Nothing is deleted:
the old games stay training data, the archived ledger keeps rating them.

**A fixed-DEPTH match cannot gate a search change**, and the failure is not subtle:
a selective search visits ~10× fewer nodes at the same nominal depth, so it loses a
depth-paced match by hundreds of Elo while being several times faster to that depth.
Pace the gate by `--nodes` (equal work, deterministic) or `--movetime` (equal time,
but reads machine load into the score). `--nodes` with the `-b` twin also prices an
uneven per-node cost: a refinement that adds a static eval per interior node buys its
tree reduction with nodes/sec, so giving the baseline proportionally more nodes turns
an equal-work match into an equal-time one without leaving the deterministic budget.

`apos-bench --search=SPEC` is the cheap screen that comes first: at a fixed depth it
reports how much smaller the tree got, and (with `--nodes=N`) how much deeper the same
budget reaches. Run it over a spread of real midgame positions rather than the start
position — this variant's opening is unusually closed and reads nothing like the
middlegame the loop actually plays.

From `web/`, `npm run build:wasm` rebuilds + copies `apos.wasm` into `public/` (so the
GitHub Pages deploy needs no Zig toolchain), and `npm run match` / `npm run train:gen`
run the native binaries through their shims.

## Parity contract

Any rule or search change goes in **both** `web/src/*.js` and `web/engine/src/*.zig`.
Verify with the JS oracle:

1. `npm run parity` (from `web/`) writes `web/engine-parity.json` + `web/engine-parity.eval.json`.
2. `zig build parity` (from `web/engine/`) checks the port reproduces them exactly.

`engine-parity.json` is the **frozen** rules contract — the variant rules never change, so
move-gen/perft/Zobrist are a permanent invariant. Only `engine-parity.eval.json` is
regenerated, when the champion net changes.

## Perf notes

- Move gen uses in-place **make/unmake** (no per-node board clone), validated by perft.
  Depth-8 from the start position: nn ~2.6× faster than JS, hc ~1.6×.
- **Incremental NNUE accumulator** (done for *quantized* nets). The first layer is maintained
  as two raw (pre-ReLU) perspective accumulators (`acc_white`/`acc_black` in `ai.zig`), updated
  by a constant ± piece-column delta in `nnMake`/`nnUnmake` instead of re-summed from scratch
  at every leaf. The leaf eval reads the side-to-move one (`nn.evalFromAcc`). The float-drift
  risk that made this dubious before is gone because the arithmetic is now **integer** (exact,
  order-independent — see Quantization below), so incremental == from-scratch bit-for-bit (the
  `ACC_DEBUG` switch in `ai.zig` asserts this every node). Measured **~1.5× nodes/sec** at
  depth 8–9 with the shipped `768→64→32→16→1` net; bigger first layers gain more.
- **Quantization.** A net with `"int": true` (exported via `train.py --quant`; training is
  unchanged, plain ReLU either way) runs an integer forward pass: layer-0 weights/bias at scale
  `QA`, dense weights at `QW` (biases `QW·QA`); activations are plain ReLU at scale `QA`, *not*
  clamped to `[0,QA]` — the clipped form was tried and cost −230 Elo, so `QA=QW=1024` buys a
  faithful-to-the-float export instead. Only the final `tanh` is float, so cp
  values match JS within ±1 like the float path. `nn.zig` and `nn.js` (`compileInt`) keep this
  bit-identical — parity asserts it.
- **Not done**: bitboard move generation (a large rewrite; the parity harness would make it
  safe), and the representational HalfKP feature set (two-perspective *concat* + king-square
  indexing) — deferred, since the prior king-feature experiment regressed.

## Zig 0.16 API notes

Pre-1.0 stdlib churn; the patterns this codebase relies on:

- **Entry point:** `pub fn main(init: std.process.Init)` — gives `init.gpa`, `init.io`, and
  args via `init.minimal.args.toSlice(arena)` (no `argsAlloc`). Otherwise build an `Io` with
  `var t: std.Io.Threaded = .init(gpa, .{}); const io = t.io();`.
- **Filesystem:** `std.Io.Dir` (not `std.fs`) — `std.Io.Dir.cwd()`,
  `readFileAlloc(io, path, gpa, .unlimited)`, `writeFile(io, .{ .sub_path, .data })`.
- **Time:** `std.Io.Clock.now(.awake, io).nanoseconds` (no `std.time.milliTimestamp`/`Timer`).
- **Threads:** `std.Thread.spawn(.{}, fn, args)` works; mutex is `std.Io.Mutex` (takes `io`;
  `lockUncancelable(io)` / `unlock(io)`).
- **Build:** `b.addExecutable(.{ .root_module = b.createModule(.{ ... }) })`.
- **Misc:** array fill via `@splat`; `ArrayList` is unmanaged (`.empty`, `append(alloc, x)`).
