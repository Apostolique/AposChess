# AposChess neural-net training

The "Neural net" engine option uses a small evaluation network in place of the
handcrafted eval. The search (alpha-beta, TT, quiescence, …) is unchanged — only
the leaf evaluation differs. This folder is the **offline pipeline** that produces
its weights. Run it whenever you want to improve the net; the browser only ever
*consumes* the resulting `web/src/nn-weights.json`.

## One-time setup

```
pip install -r training/requirements.txt
```

(A GPU is optional — the net is tiny and trains fine on CPU.)

## What to run, and when

Everything runs from `web/`. One command does generate → train → build:

```
npm run train -- --games=300 --depth=4 --match=100
```

`--match=100` plays 100 games against the handcrafted engine afterward and prints
the score — that's your "did it get better?" gauge. Just repeat this command to
keep improving; you never pick an epoch count (early stopping handles it).

### Phase 1 — bootstrap (net is still weak)

Run the command above several times. Each run **adds 300 more handcrafted-played
games** to the dataset and **retrains from them**, so the net climbs toward the
handcrafted engine's strength. Watch the match score after each run — it should
trend upward toward ~50% (parity with handcrafted).

### Phase 2 — self-improvement (net ≈ or > handcrafted)

Once the match score reaches roughly 50% or better, let the net teach itself by
generating data from its own play:

```
npm run train -- --games=300 --depth=4 --eval=nn --match=100
```

or run several cycles at once (generation 2+ automatically use the net):

```
npm run train -- --games=300 --depth=4 --generations=5 --match=100
```

Keep watching the match against handcrafted — it stays your fixed yardstick even
though the net is now the teacher.

### Does it just keep improving? (important)

**Not guaranteed.** Each run trains a fresh net and **overwrites**
`web/src/nn-weights.json`; it's usually better, but a run can come out *worse*
(training noise, or net-generated data while the net is still weak). So:

- **Watch the `--match` score.** If it went up, keep going. If it dropped, this run
  regressed.
- **You can always roll back.** `npm run train` backs up the previous net to
  `web/src/nn-weights.bak.json` before overwriting. To restore it:
  ```
  copy src\nn-weights.bak.json src\nn-weights.json
  npm run build
  ```
- If a strength jump makes the old data feel like a drag, start the dataset over
  with `--fresh`.

That's the whole routine. The individual stages (`npm run train:gen`,
`python training/train.py`, `npm run match`) still exist if you want to run one at
a time, but `npm run train` is the normal path.

## How it fits together

- **Feature definition is single-sourced** in `web/src/nn.js` (`featureIndices`):
  12 piece-kinds × 64 squares + a side-to-move bit (769 inputs). The generator
  writes those indices, so the trainer needs no chess logic — it only sees vectors.
- **Target** is the pure game result (+1 / 0 / −1, White's view). Training on who
  actually won — rather than mimicking the handcrafted eval — is what lets the net
  *improve* on it instead of inheriting its blind spots.
- **Network**: sparse input → ReLU hidden (default 64) → scalar, squashed with
  `tanh` and scaled to centipawns. Small enough to recompute at every search leaf.

## Knobs worth trying

- `--games`, `--depth` (generator): more/deeper games = better but slower data.
- `--hidden`, `--lr` (trainer): network capacity and learning rate.
- `--epochs` (max cap) / `--patience` (early-stopping tolerance): normally leave
  these — early stopping picks the real epoch count. Raise `--patience` if it's
  stopping too eagerly; set `--patience 0` to disable and run all `--epochs`.
- `--scale` (trainer): centipawns at tanh saturation (default 600).

## Notes / future work

- `web/src/nn-weights.json` ships as a placeholder until you train; the NN engine
  falls back to a material-only eval so it always plays.
- For real NNUE speed, the first-layer accumulator could be updated incrementally
  through `applyMove` (currently pure-functional); recomputing from scratch is fine
  to start, but watch the fixed-time matches.
- Targets are currently game-result only; blending in a shallow search score
  (Stockfish-style `λ·eval + (1−λ)·result`) is a natural next improvement.
