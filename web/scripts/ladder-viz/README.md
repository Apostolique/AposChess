# Ladder visualizer

A local, zero-dependency dashboard for the AposChess strength data. It reads the
**git-ignored** `training/data/` artifacts (the Elo ladder, the pairwise pool, the
harvested games, the self-play dataset, the training tracks) plus the shipped net
catalog, and serves a single-page UI that updates **live** while `train:loop` /
`rank:pool` run.

It is a dev tool only — it lives under `web/scripts/` and is served directly by
Node, so it never enters the Vite build or the GitHub Pages deploy (only
`web/index.html` + `web/public/` ship).

## Run

```
cd web
npm run viz            # → http://localhost:5178
npm run viz -- --port=6000
```

No install step, no bundler: pure `node:http` + `node:fs`. Open browsers refresh
themselves over SSE whenever a watched file changes.

While `rank:pool` runs, a finished game is appended to `ladder-games.jsonl` every
second or two, so those changes only refresh the counters. A view rebuilds when the
ledger, the pool or a training track changes, and the game you have open keeps its
place across the rebuild.

Indexing both game files takes ~4s at 870 MB and happens once at boot, so give it a
moment before the first request.

## Games come from two places

`rank:pool` rates engines from the matchups it schedules itself **plus** a scan of the
whole dataset (`--corpus`), and only the first half gets persisted to `ladder-pool.json`.
The other half is the promotion gates, which `foldGateHarvest` writes straight into
`selfplay.jsonl`. So the dashboard reads both, and merges them the way `fit()` does: the
pool store is authoritative for its own matchups, and the corpus adds only the games
beyond what the store already counts.

It matters most at depth 6, which is where the gate runs (`--gate-depth`). Reading the
pool alone, the whole champion lineage from Kara to Rosa had never met at depth 6. It had:
Rosa vs Quinn is 1648 games. A cell's tooltip breaks out how many of its games came from
gates rather than from the ranker.

Non-promoted gate candidates are left out on purpose. They carry an `elo<N>` strength
label instead of an identity, `rank:pool` keeps them out of the fit, and so they have no
ladder row to hang a heatmap cell on. Their games are skipped at index time, which is most
of `selfplay.jsonl` (~168k of 240k records).

The corpus half is only as fresh as the last `rank:pool` run, though. The cache keys on
the dataset's size and mtime, so the toolbar shows `corpus scan behind dataset` when the
dataset has moved on since.

## Depth is shared, and the URL remembers everything

There is one depth for the whole dashboard. Pick depth 6 on the ladder and the
generations curve and the matchup heatmap are already on depth 6 when you switch to
them, so the numbers you are reading side by side come from the same search. Pick
`all depths` and each view says what that means for it: the ladder shows every engine
at its strongest depth, the heatmap stops filtering.

A pick rebuilds the view, but the dropdown keeps the focus it had, so you can arrow
through the depths and watch the table move under you. The family filter on the matchups
and the two filters on the games do the same.

The view state lives in the URL hash, so a refresh lands you back where you were and a
link is worth pasting:

```
#/ladder?depth=8
#/matchups?depth=6&fam=nn&cross=all
#/expected?depth=4&opp=6&show=margin
#/games?a=nn8@2c9f1a&b=hc8@2&g=<game-id>&ply=42
#/training?depth=8&track=mona
```

The family filter is shared between Matchups and Expected. Those two get read against each other
constantly, so having to re-pick `nn` on every switch got old.

Back and forward work too. The ply is written while you step through a game but not
while a replay is running, so the URL holds the ply you stopped on.

## The heatmap can run past the picked depth

The depth filter used to apply to both axes, so a cell only showed up when the row and the
column sat at the same depth. That hides the first thing a new engine does. The ladder
calibrates a fresh net against itself at depth 6, and the direct-link floor pairs it with a
rated champion, so the day Tara arrived her depth-1 row was blank while carrying 128 games:
28 against Tara d6 and 100 against Quinn d2.

Rows are still the nodes at the picked depth. Columns can now run past them, and the
`Cross-depth` dropdown says how far:

- `for blank rows` (default) — a node whose every game was played at another depth gets its
  opponents added as columns. That is 3 extra columns on top of 23 at depth 1.
- `all opponents` — every off-depth opponent of every row. Complete, and wide: 23 rows by
  115 columns at depth 6, mostly empty, since every node keeps a calibration link to itself
  at depth 6 and to the handcrafted anchors.
- `hide` — same-depth only, the square grid. A row the filter leaves blank gets named in the
  toolbar instead, so it doesn't read as a node that never played.

The extra columns sit past a dashed divider with their labels in italics, and they open the
games on click like any other cell. `all depths` never has any, because there every opponent
is already a row.

New engines aren't the only ones this catches. At depth 4 the handcrafted anchor carries 820
games and Quinn 742, and neither has a same-depth opponent.

## Expected: the same grid with nothing missing

Matchups can only show the pairs that met. Fitting a Bradley-Terry model gives every pair an
expected score whether it met or not, which is the whole reason to fit one, so the Expected tab is
that matrix filled in end to end. Draws count half a point each side (the convention the gate and
`eloFromScore` use), so these are expected **scores**, not win rates: a pair that draws every game
scores 50% with no win in it.

The scores themselves add nothing the Ladder column doesn't already have. A cell is one Elo
subtraction put through `1/(1+10^(-Δ/400))`, so reading the grid is reading a subtraction. Two other
things are on it, and those are the reason it exists.

**±95** is how well the pool pins *that* difference. Cells with no game behind them have their
number dimmed, which at depth 6 is 434 of 506 — most of the grid is inference through third
parties, and without the dimming it looks exactly as solid as a 3000-game measurement. The interval
says how much that costs: Tara d8 has never played the `hc6@2` pin directly, and their gap still
comes out +566 ±72 because the path between them is short and busy.

**Observed − expected** is the residual, on the pairs that did play. The fit already saw those games
and tried to match them, so a residual that survives on a well-played pair is the model failing to
fit, which is what non-transitivity looks like from inside a model that can't represent it. Shading
is by σ (±3 saturates) with the bound `Var ≤ p(1−p)`, which a draw-heavy pair sits well under, so it
understates rather than oversells.

Since nothing here needs a game to exist, the columns are free to come from another depth entirely.
`Opponents` picks: same depth as the rows, one specific depth, or every depth (184 columns).
"How does Ada at depth 8 do against today's champion at depth 4" is a question only this view answers.

### Where the ±95 comes from

`rank:pool` computes the pairwise contrast variance — it is the currency the scheduler spends games
in — but only ever writes out each node's ±95 against the pin (`margin`). Those can't be subtracted
into a pairwise interval: they share the whole path to the anchor, so the difference overstates how
unknown the pair is (`depth-ladder.mjs:794`, and the generations band below says the same).

So the view rebuilds it, which is cheaper than a refit. The Fisher information of a BT model is
`H_ii = prior/4 + Σ_j N_ij·p_ij(1−p_ij)`, `H_ij = −N_ij·p_ij(1−p_ij)`, and that needs only the game
counts (the pool) and the strengths (the ladder's own Elos, since `γ = 10^(Elo/400)`). No MM
iteration is repeated and no rating is recomputed — every Elo on screen is the ledger's. `H` is
scale-free in `γ`, so the pin offset drops out.

It checks itself, because the ledger's `margin` column *is* `1.96·√varDiff(node, pin)`. Across 183
nodes the rebuild reproduces it to a median of 0.008 Elo. The exceptions show up as a
`pool ahead of ledger` badge and are almost always a `rank:pool` running right now: the store it
appends to has games the ledger it last wrote hasn't seen, so those rows come out tighter. A
disagreement that outlives the run would mean the fit used a `--prior` the ledger didn't record,
which is why `rank:pool` now writes `prior` into the ledger.

## Views

- **Ladder** — leaderboard at the shared depth (or each engine's best depth, at
  `all depths`), convergence verdict, current-champion card, latest gate result. Click
  a row to jump to its depth curve.
- **Generations** — champion Elo across `train:loop` generations, plus Elo gained
  per generation. The line carries a band, see below.
- **Depth × Elo** — one line per engine of Elo vs search depth, over the depths
  actually rated. Toggle engines, read Elo-per-ply slopes. Depth is the x-axis here, so
  nothing is filtered — a vertical marker shows where the other views are reading.
- **Matchups** — head-to-head heatmap of every game two engines have played
  (diverging blue↔orange around 50%). Click a cell for the games. Columns can reach past the
  picked depth, see above.
- **Expected** — the same grid with no gaps: what the ledger says every matchup would score,
  played or not, plus the ±95 on each pair and the residual where games exist. See above.
- **Games** — filter by engine / matchup, replay any game on a board with an eval
  graph (white-POV, mate-aware) and a clickable move list. The board is the app's:
  material trays either side, and the app's move / capture / check sounds as you step
  forward (🔊 in the header mutes them, and the mute sticks).
- **Training** — per-experiment tracks: absElo and gate score over cycles.

The depth dropdown is built from the depths the ledger actually contains, and it opens on
the deepest one the champions were rated at. If a run drops the depth you had picked, it
falls back rather than leaving every view empty. `rank:pool` rates every node its store
knows, so a `--depths=1-3` run still puts depths 4-8 on the ladder, those nodes just
aren't gaining games.

## The generations band, and what it does not say

The champion line is drawn with its ±95 around it: a 10% wash in the series hue, the two bounds
as hairlines, the Elo itself the only full-weight mark. Hover gives you the numbers,
`1889` with `1842 – 1936` beside it.

It is worth having because the interval moves by two orders of magnitude across the pool. A
champion with a few thousand games at depth 6 sits at ±35, and the whole 13-generation climb from
Hugo to Tara is +163, so the band is about half the height of the climb. A champion the ranker
hasn't onboarded yet reads ±426, and the band swallows the chart. That is the correct picture.
Sven's depth-1 rating right now is not a measurement of Sven at depth 1, it's the `--prior=1`
virtual half-draw holding up a node whose entire record is 0 points from 42 games against Sven d6.
Take the prior away and the fit sends it to −∞. It fixes itself: onboarding gives every node a
real opponent within a run or two, and Tara's depth-1 band went from ±443 to ±91 over the course
of one afternoon.

The band is each node's ±95 **against the `hc6` pin**, so it answers "how well do we know where
this champion sits on the scale". It does not answer "is generation N+1 above generation N".
Every consecutive pair of champions has overlapping bands at every depth, and that is not a
verdict: two vs-pin intervals share the path to the anchor, so subtracting them overstates the
pairwise uncertainty badly. Two neighbours that played each other directly can have their
difference pinned to ±15 while both still read ±50 vs the pin. The pairwise contrast variance is
what settles the order, `rank:pool` computes it (`depth-ladder.mjs:794`) but doesn't persist it,
so it isn't in the ladder file and isn't on this chart. It is on the **Expected** tab, which
rebuilds it from the pool — that is where to go to ask whether two champions are really in that
order.

For the same reason the Elo-gained bars below the line carry no error bars. A difference of two
banded numbers is not a banded difference.

## The viewer runs the real rules

The game viewer imports `web/src/board.js` and `web/src/engine.js` — the app's own move
generator, served straight out of the Vite tree (they are plain ES modules, so the browser
takes them as they ship; the server only exposes `.js` under `/src/`). Opening a game
replays it through `legalMoves` + `applyMove` once, into one entry per ply.

That is where the check sound comes from: nothing in a record says a move gave check, and
in this variant you cannot work it out with standard-chess logic — knights travel like
rooks, bishops and rooks jump, and a king's safety zone repels enemy jumps. Reimplementing
any of that here would be a second copy of frozen rules waiting to drift. It also means the
board is exact rather than approximate: castling, promotions and the material trays all
come from the same `applyMove` the app uses.

A recorded move the rules reject stops the replay, and the viewer says which ply and which
move. These records are written by the engines themselves, so that would be a real bug —
worth showing rather than papering over. (150 games from each source replay clean today.)

## Data sources (all read-only, re-read per request)

| File | Used for |
|---|---|
| `training/data/loop/engine-elo.ladder.json` | ranking, Elo, depths, convergence |
| `training/data/loop/ladder-pool.json` | pairwise scores the pool played itself |
| `training/data/loop/ladder-corpus-cache.json` | pairwise scores from the dataset scan |
| `training/data/loop/ladder-games.jsonl` | game bodies (byte-indexed, incremental) |
| `training/data/selfplay.jsonl` | game bodies for gate + generation games |
| `training/data/loop/match.json` | latest gate summary (transient) |
| `training/data/loop/experiments/*/` | training tracks (`state.json`, `history.jsonl`) |
| `web/public/nn/manifest.json` + `name-history.json` | name ↔ hash ↔ gen ↔ arch |

Nothing here is written or committed data — if `training/data/loop/` is absent the
UI just shows empty states.
