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
- **You can always roll back.** `web/src/nn-weights.json` is committed to git, so the
  previous net is its checked-in version. To restore it:
  ```
  git checkout -- web/src/nn-weights.json
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
npm run train:loop -- --lambda=0.5
```

Each cycle: gather fresh data (by default `--batch=0` — the gate harvest plus the
ranked pool's strong-engine play at depth 8 are the generators; set `--batch=N` to add
a dedicated champion self-play batch at `--depth`, default 8) → featurize (incremental
— only the new games) → train a **candidate** (**warm-started** from its track's
lineage/best or the champion so it fine-tunes in a few epochs; `--cold` restores
from-scratch training) → play **candidate vs champion** as an SPRT → **promote the
candidate only if it wins** (accepts H1). Otherwise the champion is kept. So the
champion only ever moves uphill. Every cycle it also refreshes the weakest value
labels (`--refresh-cycle`, below), and on promotion it can run a bigger refresh
(`--refresh-frac`, below).

**Candidate lineage (sub-threshold gains accumulate).** A mature champion's real
per-cycle gains are often +10-ish Elo — genuinely positive but below what the SPRT can
certify, so cycle after cycle would train a slightly-better candidate and throw it
away, then re-derive roughly the same candidate from the same champion on a dataset
that grew only ~2%. Instead, when the gate is **inconclusive but the candidate scored
≥ 50%**, the candidate is kept as the recipe's **lineage** (per experiment track —
see "Experiment tracks" below) and the next cycle's candidate warm-starts **from it**
rather than from the champion — so those small gains compound across cycles until the
lineage clears the gate in one decided match. The champion stays protected (only an
H1 promotes); a candidate that scores < 50% (or a decided H0) resets the lineage. The
lineage survives a `Ctrl-C` restart, and switching `--hidden` just switches to that
shape's own track (the previous shape's lineage is preserved and resumes later).

The gate's games are themselves **harvested into the dataset** (`--no-harvest` to
disable): up to `--gate-games` per cycle — comparable volume to generation, already
paid for. Each position keeps the search value `v` from the engine that actually
searched it (the mover), tagged with that engine×depth provenance — so the dataset's
own machinery (merge dedup prefers the stronger engine, `v`-refresh relabels the
weakest cohort first) judges label quality, not who won the game. Gate games are
played at `--gate-depth` (default 6); the deep-label anchor is the ranked pool's
strong-engine play at depth 8 (plus `--batch` generation when enabled).

- The champion is `web/src/nn-weights.json`. On each promotion it's also published to
  the catalog under its **own human name** (Ada, Boris, …) and flagged the current
  champion, so you can play it in the app under a real name from the moment it's
  promoted (rebuild for the production bundle; `npm run dev` serves it live).
- Runs forever until `Ctrl-C` (or pass `--cycles=N`). Per-cycle decisions are printed
  and appended to `training/data/loop/loop.log`.
- **`--fresh` deletes the dataset** before the first cycle (irreversible — the raw
  `selfplay.jsonl` is git-ignored, so there's no recovery). On a small post-`--fresh`
  set a candidate has too little signal to beat a champion trained on millions of
  positions (warm-starting softens this — the candidate at least *starts* from the
  champion — but a tiny set still can't demonstrate a +`elo1` gain), so the gate
  rejects everything. Only use `--fresh` intentionally, after backing the dataset up
  outside the repo, or with a `--batch` large enough to rebuild real volume. The
  normal path **omits it** and lets the dataset accumulate (and be maintained — see
  "Dataset maintenance").
- Options: `--batch`, `--depth` (generation), `--gate-games` (default 2000), `--gate-depth`,
  `--elo1` (promotion bar, below), `--hidden` (candidate shape; default = champion's),
  `--lambda` (TD target mix, below), `--cold` (random-init candidates instead of
  warm-starting from the lineage/champion), `--skip-gen` (skip the FIRST cycle's
  generation and gate the dataset as it stands — resume after a Ctrl-C mid-generation
  without regenerating; completed games were already flushed), `--no-harvest` (don't
  save gate games into the dataset), `--refresh-cycle` / `--refresh-cycle-depth`
  (per-cycle value refresh — in ledger mode it drains the weakest label cohort under a
  10-minute budget) and `--refresh-frac` /
  `--refresh-depth` (on-promotion refresh — see "Dataset maintenance"), `--no-refresh`
  (skip both refreshes for the fastest cycles, at the cost of staler `v` targets),
  `--jobs`.

### The promotion gate: `--elo1` vs `--gate-games` (calibrate them together)

The gate is an SPRT testing H0 (champion is ≥ as good) vs H1 (candidate is ≥ `elo1`
Elo better). It **early-stops** when decided, so `--gate-games` is just a cap. The trap
is making `elo1` too small for the cap: SPRT evidence accrues ~`(elo1−elo0)²`, so a
tiny band needs huge samples. With `[0, 5]` over 400 games a candidate must be **≈ +170
Elo** to ever trip H1 — so genuine +40/+60 improvements get rejected and the loop
**stalls** (it keeps training better candidates and discarding them, the champion never
moves, generation never improves). The default is therefore **`elo1=20`** with a
**default cap of 2000 games** (a true +20 candidate clears an 800-game gate only ~⅓ of
the time but ~80% at 2000), which decides a +50-class candidate quickly and gives an
honest +20–30 candidate a realistic shot at reaching the H1 boundary before the cap. As
gains shrink with maturity, **raise `--gate-games`** further rather than lowering `elo1`
— more games is the sound way to detect a smaller edge. (Candidates in
the +10 range stay genuinely undecidable at sane game counts; the **lineage** mechanism
above is what stops those gains from being wasted.)

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
npm run train:loop -- --batch=200 --depth=6 --lambda=0.5
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

### Experiment tracks: try recipes without losing them

Each loop run trains one **recipe** — the knobs that shape the candidate net: `--hidden`
(architecture), `--lambda`, `--quiet-only`, `--float`/`--no-quant` (quantized vs float),
`--scale`/`--lr`/`--wd`, and a free-form `--recipe-extra=k=v,…` for anything else. The loop
keeps a **persistent track per recipe** under `training/data/loop/experiments/<id>/`, so trying
a different architecture (or quiet-games, or any other recipe knob) **doesn't throw away** the
previous recipe's accumulated progress. Run recipe A for a while, switch to B, come back to A
later — the loop finds A's track and **resumes its warm-start lineage and best net
automatically** (same recipe → same track). There's still **one shared champion** (the app's
net and the generator's teacher): every recipe's candidate gates against it, and whichever
recipe produces a winner promotes it — so the tracks are diverse *paths* to one champion, and
the ones that don't win are preserved for later.

```
npm run train:loop -- --hidden=256,32              # a track for this architecture
npm run train:loop -- --hidden=128 --quiet-only    # a different, independent track
npm run train:loop -- --hidden=256,32              # resumes the first track's lineage/best
```

Browse and get suggestions with **`npm run train:experiments`**:

```
npm run train:experiments                 # every track: best Elo, cycles, promotions
npm run train:experiments -- --show=<id>  # one track's recipe + per-cycle history
npm run train:experiments -- --suggest    # "what to try next" only
```

The suggestions are the answer to *"the loop is stalling — what now?"*: it points at
**promising-but-stalled past recipes** worth reviving (they warm-start from their saved best
net, so it's cheap) and **architectures with no track yet**. `npm run train:progress` shows the
same ideas inline once it detects a stall (many cycles since the last promotion). As always,
strength is signal-limited — a bigger net isn't automatically better — so gate every recipe
head-to-head; the registry just makes the exploration non-destructive.

## Dataset maintenance

The dataset is the substrate the loop learns from, so it's worth **maintaining**, not
just appending to. Each game record's result `r` is the actual outcome and **never
goes stale**, but the per-position values `v` are the champion's search values *at
the time they were recorded* — a weaker net's opinion that drifts out of date as the
champion improves. `refresh-v` rewrites `selfplay.jsonl` in place (atomic), so
**re-featurize afterward** (`npm run train:featurize`).

### Refreshing `v` (value iteration) — `refresh-v.mjs`

Recompute `v` with the **current champion**. This is value iteration: re-bootstrap the
TD targets from the improved value function (only the `v` part of
`λ·result + (1−λ)·tanh(v/scale)` moves; `result` is untouched). Note this helps even
**between promotions**: at any given moment most of the dataset's `v` values were
written by *older* champions (or by shallower searches — gate harvests at
`--gate-depth`, the depth-3 backfill), so re-labeling a random slice with the current
champion upgrades targets regardless of whether the champion just changed. Only records
the current champion already labeled at the same depth are true no-ops.

```
# fill v only where it's missing (e.g. legacy/opening records):
node scripts/refresh-v.mjs
# refresh a random 20% of existing v with the current champion (parallel, seeded):
node scripts/refresh-v.mjs --refresh --frac=0.2 --depth=6
```

**Partial (`--frac`) is the right default.** A full refresh of millions of positions is
hours; refreshing a fraction `P` each time amortizes that cost and keeps the *average*
`v` ~`1/P` champions old — a smooth moving average across recent champions instead of a
synchronized all-stale→all-fresh lurch. (Pure-random `P` has a coupon-collector tail; a
future champion-version stamp would enable exact oldest-first coverage.)

**Wired into the loop, two ways.** `train:loop --refresh-cycle=P` (default **1**)
refreshes **every cycle** between generation and featurize, at `--refresh-cycle-depth`
(default = the generation `--depth`). In ledger mode (the loop's default) refresh-v
restricts itself to the **weakest label cohort** under a 10-minute wall-clock budget,
so `P=1` just drains the weakest labels first; the adaptive maintenance budget scales
the effective fraction per cycle. Separately, `--refresh-frac=P` (default 0 = off)
runs a bigger pass **after each promotion** (using the just-promoted champion) at
`--refresh-depth` (default **8**, matching generation — a big fraction at depth 8 is
many hours). Both seed the slice differently each run, so successive refreshes cover
different parts of the set.

### Capping duplicate positions (de-bias) — `featurize --cap=N`

Common positions are wildly over-represented: the **start position occurs once per
game** (tens of thousands of times), so the net sees the opening orders of magnitude
more than any midgame position. `featurize --cap=N` caps how many copies of any single
*training input* (canonical feature set `f`, what the net actually sees) survive, by
**Bernoulli thinning** (keep each copy with probability `cap/count`) — which preserves
the win/draw/loss ratio across the copies rather than collapsing them to one (the same
position carries different `r` across games, and averaging those is how the net learns
its value).

```
npm run train:featurize -- --cap=64
```

It needs a global count, so it forces a full two-pass featurize (no incremental path).
In practice the dataset is mostly unique (the duplication is concentrated in a handful
of opening positions), so this removes only a couple percent by volume — but it cuts the
start position's training weight from tens-of-thousands× down to ~`cap`, which is the
point. Because diversity is high, an aggressive recency window isn't urgent; `v`
staleness is the more pressing lever.

### Future levers (not yet implemented)

- **Resign/adjudication in generation.** Decided positions carry little learning signal
  and the mop-up tail of a won game is redundant; stopping a game once `|v|` is decisive
  for a few plies would both speed generation and stop manufacturing those positions.
- **Position-seeded generation.** Seeding new games from balanced midgame positions
  sampled out of the dataset (instead of random opening plies) reuses the diverse
  positions as launch points and yields current-strength continuations — the sound form
  of "replay/override an old game." Mind the feedback loop (keep anchoring on played-out
  results, don't train purely on the net's own evals).

## Trying new ideas (recipes)

Every idea follows the same shape: **train a candidate, then compare it head-to-head
against the current best with an SPRT**. A net vs *itself* scores ~50% — that's the
built-in sanity check, and the comparison is far more sensitive than each-vs-handcrafted.

The current best is the live `train:loop` champion (`web/src/nn-weights.json`; in the
catalog it's the entry flagged `current` and the manifest `default`, published under its
human name); use it as the opponent. (`balanced-64.json` is the older ~handcrafted-parity
baseline, kept as a fixed yardstick.)

### A. Try a different network shape (width/depth)

Architecture only — features are unchanged, so no re-featurize:

```
npm run train:fit -- --hidden=256,64,16 --name=cand      # each comma value = one hidden layer
npm run match -- --eval-a=nn --eval-b=nn \
  --weights-a=public/nn/cand.json --weights-b=src/nn-weights.json --sprt
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
  --weights-a=public/nn/cand.json --weights-b=src/nn-weights.json --sprt
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

Self-play scales out: generate on multiple computers, then combine. `npm run
train:merge` is a **smart** merge, not a blind concatenation — it identifies records
by game+ply, collapses positions shared across machines to one copy, and keeps the
best-provenance `v` (see `docs/tools.md`).

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

- **Game-primary data.** `npm run train:gen` appends to the **raw** dataset
  `selfplay.jsonl` — one record per **game** (`web/scripts/gameRecord.mjs`): the move
  list, who played, the result, and a per-position search value `v` (cp,
  side-to-move-relative) for TD targets. `v` is recorded for **every** position,
  openings included: the opening move is random (for variety), but the position is
  still searched for its value, so the TD target is uniform across the dataset. A
  separate step replays each game into positions and turns them into net inputs:
  `npm run train:featurize` applies the **current** `featureIndices` and writes
  `selfplay.features.jsonl` = `{f, r, g, v}`, which the trainer reads (`v` carried
  through unchanged; positions that genuinely lack one fall back to the pure result).
  So changing the feature set is a quick `featurize` pass, never a self-play regen —
  and the trainer still needs no chess logic (it only sees `f`). Run order:
  `train:gen` → `train:featurize` → `train:fit` (the full `npm run train` does all
  three).
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

- In the **Node tools**, a net with no weights falls back to a material-only eval so
  it can play (it's also the ranking pool's floor engine); `loadWeights` refuses a
  file whose input size doesn't match the current `featureIndices` layout (so an old
  net after a feature change can't silently produce garbage) — re-featurize and
  re-train. The **browser app** deliberately has no such fallback: a failed weights
  fetch throws, so caller bugs surface instead of degrading silently.
- **Labels first, capacity second.** The unbiased levers are better *labels*: deeper
  search, bootstrapping from the net's own play (`--eval=nn`), and blending the net's
  own search value into the target via `--lambda` (blending the *handcrafted* eval
  would reintroduce its bias, so it's deliberately avoided). Capacity pays only once
  labels are good — early width/feature additions overfit and lost Elo, while a
  larger architecture later won through the gated loop on matured labels. Gate every
  capacity change head-to-head; never judge by validation loss.
