# AposChess engine (Zig)

A native + wasm port of the engine core (`board` → `engine` → `ai` → `nn`), built to
speed up the offline self-play tools and the in-browser bot. The JS engine in
`web/src/` stays the reference; this is validated against it, never the other way.

- **Toolchain:** Zig **0.16.0** (pinned — pre-1.0, expect churn on upgrades).
- **Architecture:** one source, two targets — native CLI binaries the npm scripts
  spawn (gen/match/rank), and a `wasm32` module the browser worker loads. The
  dataset/featurize/training glue stays in JS.
- **Safety net:** the JS oracle. Generate it in `web/` with `npm run parity`
  (writes `web/engine-parity.json` + `web/engine-parity.eval.json`), then check the
  port against it here:

  ```sh
  zig build parity                       # all engine layers vs the JS oracle
  zig build bench  -Doptimize=ReleaseFast -- --depth=8 [--nn]
  zig build match  -Doptimize=ReleaseFast -- --games=20 --depth=6 [--nn]
  zig build gen    -Doptimize=ReleaseFast -- --games=200 --depth=4 --eval=nn
  zig build wasm                         # -> zig-out/bin/apos.wasm (browser)
  ```

  `engine-parity.json` is the **frozen** rules contract (rules never change); only
  `engine-parity.eval.json` is regenerated when the champion changes.

## Port status

- [x] `board.zig` — board rep, FEN parse/serialize, `squareName` (FEN round-trip green)
- [x] `engine.zig` — move generation, `applyMove`, safety zones, `gameStatus`
      (legal-move sets + perft + status all green, verified to 45.5M perft nodes)
- [x] `zobrist.zig` — key-gen replicated; hashes + incremental invariant green
- [x] `eval.zig` / `nn.zig` — handcrafted (bit-exact) and nn (±1 cp) vs the eval oracle
- [x] `ai.zig` — search (alpha-beta, PVS, null-move, LMR, qsearch, TT, killers/history,
      repetition). Plays legal self-play games to completion; `zig build bench`/`match`.
- [x] `wasm.zig` — wasm32 target (`apos.wasm`, exports `searchFen`/`allocBytes`/`memory`),
      verified searching from a JS host. The browser half of the two-target architecture.

### Next phase (engine port itself is complete + verified on both targets)

- [x] perf: **make/unmake** (in-place, no per-node board clone; validated by perft).
      Measured vs JS at equal depth (depth-8 from start): **nn 2.64× faster** (4.8s vs
      12.7s), hc 1.64× (24.1s vs 39.6s).
- [ ] perf (deferred, low ROI): incremental nn accumulator. The champion is tiny
      (`768→64→16→8→1`), so a dual-perspective f64 accumulator is only ~1.2–1.3× more,
      at notable complexity / float-drift risk. Revisit only with a wider first layer.
- [ ] perf (bigger lever): bitboard move generation — speeds every node (and perft,
      gen, gate). Large rewrite; the parity harness makes it safe.
- [x] npm wiring — **match**: `npm run match:zig` (shim `scripts/match-zig.mjs` → native
      `apos-match`). Drop-in for `npm run match`: A/B evals+weights, parallel threads,
      SPRT early-stop, the same `--result-file` JSON `train:loop` reads, AND the
      `--save-games` harvest (winner-relative `v`: direct depth-d on its plies, derived
      depth-(d-1) on the loser's — exactly like selfplay.mjs). ~2.4M nps aggregate
      (8 threads) at depth 4.
- [x] npm wiring — **gen**: `npm run gen:zig` (shim `scripts/gen-zig.mjs` → native
      `apos-gen`). Drop-in for `npm run train:gen`: parallel self-play, same
      `{fen,r,g,v,vs}` JSONL appended to the same dataset, `--opening-topk` supported.
      ~2.4x faster than the JS generator. The `vs` provenance hash matches `vtag.mjs`
      bit-for-bit (verified), so refresh-v/rank still recognize the labels.
- [x] **loop integration**: `train:loop`, `train` (pipeline) and `rank` build the engine at
      startup and spawn `apos-gen` (generate) + `apos-match` (gate / gauntlet, with harvest)
      instead of the JS tools. `npm run match` and `npm run train:gen` now run the native
      binaries via the shims. featurize/train/refresh-v stay JS+Python.
- [x] **rank**: `rank-engines.mjs` (Elo ledger / weakest-first refresh glue) now spawns
      `apos-match` per matchup — including its asymmetric `--depth-b` (cheap contenders vs a
      deep stable anchor), `--movetime`, and `--save-games` harvest. The orchestration glue
      stays JS; only the heavy matches went native.
- [x] **JS cleanup**: the replaced tools (`selfplay.mjs`, `matchWorker.mjs`,
      `gen-selfplay.mjs`, `genWorker.mjs`) are deleted. The JS *engine* (`src/*.js`) stays —
      it's still the browser bot (until the wasm worker lands) and the engine for the JS glue
      tools (featurize, refresh-v, puzzles, openings, the parity oracle).
- [x] app wiring: `apos.wasm` in the Web Worker (`src/aiWorker.js`). Full parity with the
      old `ai.js` search — nn weights written into wasm memory (`loadWeights`), **movetime**
      via a JS-imported clock (`env.aposNowMs`), **live eval-bar** streaming
      (`env.aposProgress` after each depth), **ponder**, repetition + opening-exclude passed
      as byte buffers; the worker reconstructs the full variant move from the wasm's
      from/to/promo via its own `legalMoves`. `ai.js` stays only as a fallback if the wasm
      can't load. Build + commit the artifact with `npm run build:wasm` (→ `public/apos.wasm`),
      so the GitHub Pages deploy needs no Zig. Verified in Node: legal play, exact eval parity.
