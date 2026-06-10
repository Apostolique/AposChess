# AposChess neural-net training

The "Neural net" engine option uses a small evaluation network in place of the
handcrafted eval. The search (alpha-beta, TT, quiescence, …) is unchanged — only
the leaf evaluation differs. This folder is the **offline pipeline** that produces
net weights. Trained nets live in the catalog `web/public/nn/` (a `manifest.json`
plus one JSON per net); the web app fetches the one you pick from there at runtime.
(`web/src/nn-weights.json` is just the default net for the Node tools — `train:gen`
and `npm run match` — not what the browser loads.)

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
`npm run train:featurize`, `python training/train.py`, `npm run match`) still exist
if you want to run one at a time, but `npm run train` is the normal path.

## Unattended improvement: the gated loop (`train:loop`)

`npm run train` overwrites the net each cycle and can regress (you watch the score and
roll back by hand). `npm run train:loop` automates the **safe** version — gated
expert-iteration that can run all day and **never regresses**:

```
npm run train:loop -- --fresh --batch=200 --depth=6 --gate-games=400 --elo1=5
```

Each cycle: generate games with the **champion** (deep search → better labels) →
featurize → train a **candidate** (same shape as the champion) → play
**candidate vs champion** as an SPRT → **promote the candidate only if it wins**
(accepts H1). Otherwise the champion is kept. So the champion only ever moves uphill.

- The champion is `web/src/nn-weights.json`. On each promotion it's also published to
  the catalog as **`loop-champion`**, so you can play the current champion in the app
  (rebuild for the production bundle; `npm run dev` serves it live).
- Runs forever until `Ctrl-C` (or pass `--cycles=N`). Per-cycle decisions are printed
  and appended to `training/data/loop/loop.log`.
- `--fresh` starts the dataset over — recommended, so the bootstrap is built from the
  champion's deep-search games rather than the old depth-4 handcrafted set.
- Options: `--batch`, `--depth` (generation), `--gate-games`, `--gate-depth`, `--elo1`
  (promotion bar, Elo over the champion), `--hidden` (candidate shape; default =
  champion's), `--lambda` (TD target mix, below), `--jobs`.

### TD / bootstrap targets (`--lambda`)

By default each position's training target is the **game result** (`±1/0`). With
`--lambda=L` (L < 1) the target becomes `L·result + (1−L)·tanh(v/scale)`, where `v` is
the **search value** of that position recorded at generation time. With `--eval=nn`
generation (what the loop uses) `v` is the *net's own deeper-search* value — so the
candidate's static eval learns to predict what currently costs a depth-`D` search.
That's the NNUE-style trick and the most direct lever against the signal-limited
ceiling, and it stays **unbiased** (it's the net bootstrapping off its own search, not
the handcrafted eval). Try e.g. `--lambda=0.5`:

```
npm run train:loop -- --batch=200 --depth=6 --lambda=0.5 --elo1=5
```

Caveat: only use `--lambda<1` on **nn-generated** data. Handcrafted-played games record
the *handcrafted* search value, so leaning on it there reintroduces the bias the
pure-result target avoids. (Positions without a `v` — random openings, legacy data —
always fall back to the pure result.)

**Honest expectation.** This makes the loop *safe*, not *guaranteed to improve*. The
net is signal-limited (see below), so the gate may rarely fire — that's informative,
not a bug: it's telling you replaying same-strength games isn't adding information.
The lever that makes the gate actually fire is better *labels* (deeper `--depth`,
self-play from a stronger champion), not more cycles. Matches are slow, so expect a
handful of cycles per hour.

## Trying new ideas (recipes)

Every idea follows the same shape: **train a candidate, then compare it head-to-head
against the current best with an SPRT**. A net vs *itself* scores ~50% — that's the
built-in sanity check, and the comparison is far more sensitive than each-vs-handcrafted.

The current best ships as `web/public/nn/balanced-64.json`; use it as the opponent.

### A. Try a different network shape (width/depth)

Architecture only — features are unchanged, so no re-featurize:

```
npm run train:fit -- --hidden=256,64,16 --name=cand      # each comma value = one hidden layer
npm run match -- --eval-a=nn --eval-b=nn \
  --weights-a=public/nn/cand.json --weights-b=public/nn/balanced-64.json --sprt
```

### B. Add a new input feature

Features are defined in **one place**: `featureIndices(board, turn)` in
`web/src/nn.js`. Two edits there:

1. **Emit the new index** in `featureIndices`. Example — an "in check" flag:
   ```js
   import { kingAttacked } from './engine.js';
   // …after the piece-square loop:
   if (kingAttacked(board, turn)) idx.push(PIECE_SQUARE_FEATURES); // index 768
   ```
   For a count (e.g. mobility), push the same index N times — the input layer sums,
   so N pushes = the value N.
2. **Grow `NUM_FEATURES`** to cover the new range (express it with shared constants
   so it can't drift, e.g. `PIECE_SQUARE_FEATURES + 1`).

Then re-derive features (no self-play regen — that's the point of position-primary
data) and train + compare:

```
npm run train:featurize
npm run train:fit -- --name=cand
npm run match -- --eval-a=nn --eval-b=nn \
  --weights-a=public/nn/cand.json --weights-b=public/nn/balanced-64.json --sprt
```

Notes: only `board` + `turn` are available to `featureIndices` (a castling- or
clock-dependent feature needs the signature widened and threaded through `evaluate`);
and it runs at **every search leaf**, so keep it cheap (a flag/bucket is free,
generating moves for a true mobility count costs node speed — check a `--movetime`
match too). The trainer auto-picks up the new input size from the meta sidecar; you
never touch `train.py`.

### C. Compare any two existing nets

```
npm run match -- --eval-a=nn --eval-b=nn \
  --weights-a=public/nn/<A>.json --weights-b=public/nn/<B>.json --depth=4 --sprt
```

## Publishing a net to the web app

`train.py`'s `--name` (via `npm run train:fit` or `npm run train`) writes the net
into the catalog and registers it, so it becomes selectable in the browser:

1. Train with a name (and optional description / default flag):
   ```
   npm run train:fit -- --name=my-net --note="what's different" [--set-default]
   ```
   → writes `web/public/nn/my-net.json` and adds an entry to
   `web/public/nn/manifest.json`.
2. Build (or `npm run dev`): `npm run build`.
3. In the app, set an AI slot's engine to **Neural net** — the **net dropdown**
   lists every catalog entry; pick `my-net`. The worker fetches that file from
   `public/nn/` at runtime.

To remove a net, delete its `web/public/nn/<name>.json` and its `manifest.json`
entry. Keep at least one (the manifest `default`).

## Pooling data from several machines or runs

Self-play scales out: generate on multiple computers, then combine. The records
are independent and order-agnostic, so merging is just concatenation.

1. On each machine, run the same code version (same commit) with a **distinct
   `--seed`** (same seed = identical games = wasted effort), and the **same teacher**
   (both handcrafted, or both the same `nn-weights.json` for `--eval=nn`).
2. Copy every machine's `training/data/*.jsonl` into one `training/data/` folder
   (any filenames).
3. Fold them into one dataset and delete the leftovers (raw `*.jsonl` only — the
   derived `*.features.jsonl` is skipped):
   ```
   npm run train:merge
   ```
4. Featurize the merged raw dataset, then train:
   ```
   npm run train:featurize && npm run train:fit
   ```

## How it fits together

- **Position-primary data.** `npm run train:gen` writes the **raw** dataset
  `selfplay.jsonl` = `{fen, r, g, v}` — board position, result (side-to-move view), game
  id, and the search value `v` of the position (cp, side-to-move-relative; omitted for
  random openings) for TD targets — and nothing net-specific (the generator doesn't
  import `nn.js`). A separate
  step turns positions into net inputs:
  `npm run train:featurize` applies the **current** `featureIndices` and writes
  `selfplay.features.jsonl` = `{f, r, g}`, which the trainer reads. So changing the
  feature set is a quick `featurize` pass, never a self-play regen — and the trainer
  still needs no chess logic (it only sees `f`). Run order: `train:gen` →
  `train:featurize` → `train:fit` (the full `npm run train` does all three).
- **Feature definition is single-sourced** in `web/src/nn.js` (`featureIndices`):
  12 piece-kinds × 64 squares (768 inputs) in canonical side-to-move orientation (no
  separate side-to-move bit — the board is flipped for Black). `featurize` takes each
  position from its `fen`, or for legacy pre-`fen` records reconstructs the canonical
  board from a stored `f` (works for any canonical board+turn feature; castling/move
  counters need `fen`-bearing data). It also writes the input count into a
  `*.meta.json` sidecar that the trainer reads, so `NUM_FEATURES` lives only in
  `nn.js` — change features there and re-featurize; nothing to sync in Python.
- **Target** is the pure game result (+1 / 0 / −1) from the **side-to-move's** view.
  Training on who actually won — rather than mimicking the handcrafted eval — is
  what lets the net *improve* on it instead of inheriting its blind spots.
- **Network**: sparse input → one or more ReLU hidden layers → scalar, squashed
  with `tanh` and scaled to centipawns. `--hidden` sets the hidden widths (default
  `128`; a comma list like `256,32` adds depth). Small enough to recompute at every
  search leaf, but bigger/deeper trades node speed — gate with `npm run match`.

## Knobs worth trying

- `--games`, `--depth` (generator): more/deeper games = better but slower data.
- `--hidden`, `--lr` (trainer): network capacity (one width, or a comma list such
  as `256,32` for a deeper net) and learning rate.
- `--epochs` (max cap) / `--patience` (early-stopping tolerance): normally leave
  these — early stopping picks the real epoch count. Raise `--patience` if it's
  stopping too eagerly; set `--patience 0` to disable and run all `--epochs`.
- `--scale` (trainer): centipawns at tanh saturation (default 600).

## Notes / future work

- If a net's weights fail to load (or none is selected), the NN engine falls back to
  a material-only eval so it always plays. `loadWeights` also refuses a file whose
  input size doesn't match the current `featureIndices` layout (so an old net after a
  feature change can't silently produce garbage) — re-featurize and re-train.
- For real NNUE speed, the first-layer accumulator could be updated incrementally
  through `applyMove` (currently pure-functional); recomputing from scratch is fine
  to start, but watch the fixed-time matches.
- **Strength is currently signal-limited, not capacity-limited.** Adding width,
  depth, or features tends to *overfit* the game-result-only target (measured: a
  64-wide net is at parity with handcrafted; wider and king-relative-feature variants
  both lost Elo). The unbiased lever is better *labels*, not a bigger net: generate
  with deeper search (`--depth`), bootstrap from the net's own play (`--eval=nn`), and
  blend the net's **own search value** into the target via `--lambda` (TD targets,
  above) — that densifies the signal without the handcrafted bias. Blending the
  *handcrafted* eval would reintroduce that bias, so it's deliberately avoided.
