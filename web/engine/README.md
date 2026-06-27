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
| `zobrist.zig` | (key gen in `ai.js`) | Zobrist keys + incremental hashing |
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
zig build bench  -Doptimize=ReleaseFast -- --depth=8 [--nn]
zig build match  -Doptimize=ReleaseFast -- --games=20 --depth=6 [--nn]
zig build gen    -Doptimize=ReleaseFast -- --games=200 --depth=6 --eval=nn
zig build wasm                         # -> zig-out/bin/apos.wasm
```

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
  as two raw (pre-clip) perspective accumulators (`acc_white`/`acc_black` in `ai.zig`), updated
  by a constant ± piece-column delta in `nnMake`/`nnUnmake` instead of re-summed from scratch
  at every leaf. The leaf eval reads the side-to-move one (`nn.evalFromAcc`). The float-drift
  risk that made this dubious before is gone because the arithmetic is now **integer** (exact,
  order-independent — see Quantization below), so incremental == from-scratch bit-for-bit (the
  `ACC_DEBUG` switch in `ai.zig` asserts this every node). Measured **~1.5× nodes/sec** at
  depth 8–9 with the shipped `768→64→32→16→1` net; bigger first layers gain more.
- **Quantization.** A net with `"int": true` (trained via `train.py --quant`, clipped ReLU)
  runs an integer forward pass: layer-0 weights/bias at scale `QA`, dense weights at `QW`
  (biases `QW·QA`); activations are clipped to `[0,QA]`; only the final `tanh` is float, so cp
  values match JS within ±1 like the float path. `nn.zig` and `nn.js` (`compileInt`) keep this
  bit-identical — parity asserts it.
- **Not done**: bitboard move generation (a large rewrite; the parity harness would make it
  safe), and the representational HalfKP feature set (two-perspective *concat* + king-square
  indexing) — deferred, since the prior king-feature experiment regressed.
