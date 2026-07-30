# Ladder visualizer

A local, zero-dependency dashboard for the AposChess strength data. It reads the
**git-ignored** `training/data/loop/` artifacts (the Elo ladder, the pairwise
pool, the harvested games, the training tracks) plus the shipped net catalog, and
serves a single-page UI that updates **live** while `train:loop` / `rank:pool` run.

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

## Views

- **Ladder** — leaderboard at a chosen depth (or each engine's best depth),
  convergence verdict, current-champion card, latest gate result. Click a row to
  jump to its depth curve.
- **Generations** — champion Elo across `train:loop` generations, plus Elo gained
  per generation.
- **Depth × Elo** — one line per engine of Elo vs search depth, over the depths
  actually rated. Toggle engines, read Elo-per-ply slopes.
- **Matchups** — head-to-head heatmap of the games actually played in the ranking
  pool (diverging blue↔orange around 50%). Click a cell for the games.
- **Games** — filter by engine / matchup, replay any game on a board with an eval
  graph (white-POV, mate-aware) and a clickable move list.
- **Training** — per-experiment tracks: absElo and gate score over cycles.

Every depth dropdown is built from the depths the ledger actually contains, and each
view opens on the deepest one that has something to show. `rank:pool` rates every node
its store knows, so a `--depths=1-3` run still puts depths 4-8 on the ladder — those
nodes just aren't gaining games.

## Data sources (all read-only, re-read per request)

| File | Used for |
|---|---|
| `training/data/loop/engine-elo.ladder.json` | ranking, Elo, depths, convergence |
| `training/data/loop/ladder-pool.json` | pairwise scores (matchup heatmap) |
| `training/data/loop/ladder-games.jsonl` | game bodies (byte-indexed, incremental) |
| `training/data/loop/match.json` | latest gate summary (transient) |
| `training/data/loop/experiments/*/` | training tracks (`state.json`, `history.jsonl`) |
| `web/public/nn/manifest.json` + `name-history.json` | name ↔ hash ↔ gen ↔ arch |

Nothing here is written or committed data — if `training/data/loop/` is absent the
UI just shows empty states.
