#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2019-2026 Jean-David Moisan
#
# Trainer for the AposChess neural-net evaluation. Reads the self-play data
# produced by web/scripts/gen-selfplay.mjs (JSONL: {"f":[feature indices],"r":
# result,"g":game id}) and fits a small MLP, then writes the weights to
# web/src/nn-weights.json in the exact layout web/src/nn.js expects.
#
# The train/val split is by GAME ("g"), not by position: every position in a game
# shares one label and consecutive positions are nearly identical, so a
# position-level split would put the same game on both sides and make the val loss
# (hence early stopping) optimistic. Records without "g" (pre-migration data) each
# count as their own game, which is conservative — it never leaks across the split.
#
# Network: EmbeddingBag(sum) over the active feature indices implements the sparse
# input->hidden layer (summing a feature's row == our JS forward pass), then a
# manual bias + ReLU + a linear scalar head. The target is the game result from
# the SIDE-TO-MOVE's view in {-1, 0, +1} (matching nn.js's canonical, side-to-move
# feature orientation); we squash the raw output with tanh and fit MSE, so the net
# learns a win-probability-like signal grounded in who actually won.
#
# Usage (run from the repo root or anywhere):
#   pip install -r training/requirements.txt
#   python training/train.py [--data ...] [--epochs N] [--hidden H] ...

import argparse
import json
import os
import sys
import time


def fmt_dur(secs):
    """Format a duration like the Node scripts: '45s', '3m 02s', '1h 04m'."""
    secs = round(secs)
    if secs < 60:
        return f"{secs}s"
    m, s = divmod(secs, 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(THIS_DIR)
# The trainer reads the FEATURIZED data ({f,r,g}), produced from the raw positions
# (selfplay.jsonl) by web/scripts/featurize.mjs (`npm run train:featurize`).
DEFAULT_DATA = os.path.join(THIS_DIR, "data", "selfplay.features.jsonl")
DEFAULT_OUT = os.path.join(REPO, "web", "src", "nn-weights.json")
NN_CATALOG = os.path.join(REPO, "web", "public", "nn")  # named, app-selectable nets


def update_manifest(name, file, arch, note, set_default):
    """Register a net in web/public/nn/manifest.json (read/modify/write)."""
    mpath = os.path.join(NN_CATALOG, "manifest.json")
    man = {"default": name, "nets": []}
    if os.path.exists(mpath):
        with open(mpath) as f:
            man = json.load(f)
    nets = [n for n in man.get("nets", []) if n.get("name") != name]
    nets.append({"name": name, "file": file, "arch": arch, "note": note})
    nets.sort(key=lambda n: n["name"])
    man["nets"] = nets
    # First net, an explicit request, or a dangling default all (re)point the default.
    if set_default or not man.get("default") or \
            not any(n["name"] == man["default"] for n in nets):
        man["default"] = name
    with open(mpath, "w") as f:
        json.dump(man, f, indent=2)
    return mpath

# The EmbeddingBag vocab size (nn.js NUM_FEATURES) is read from the sidecar that
# featurize.mjs writes next to the data (<data>.meta.json), so it's never hand-synced
# across the JS/Python split — nn.js is the single source. This is only the fallback
# for data with no sidecar (hand-made / pre-sidecar): the original plain layout.
DEFAULT_NUM_FEATURES = 12 * 64  # 768


def read_num_features(data_path):
    meta = (data_path[:-len(".jsonl")] if data_path.endswith(".jsonl") else data_path) + ".meta.json"
    if os.path.exists(meta):
        with open(meta) as f:
            return int(json.load(f)["num_features"])
    print(f"  (no {os.path.basename(meta)}; assuming num_features={DEFAULT_NUM_FEATURES})")
    return DEFAULT_NUM_FEATURES


def parse_args():
    p = argparse.ArgumentParser(description="Train the AposChess NN evaluation.")
    p.add_argument("--data", default=DEFAULT_DATA, help="JSONL training data")
    p.add_argument("--out", default=DEFAULT_OUT, help="weights output (JSON for nn.js)")
    p.add_argument("--name", default=None,
                   help="publish to the web net catalog under this name: writes "
                        "web/public/nn/<name>.json and registers it in manifest.json "
                        "(so it's selectable in the app). Overrides --out.")
    p.add_argument("--note", default="", help="description shown in the catalog (with --name)")
    p.add_argument("--set-default", action="store_true",
                   help="make this net the catalog default (with --name)")
    p.add_argument("--hidden", type=str, default="128",
                   help="hidden layer size(s); a comma list adds depth, "
                        "e.g. --hidden=128 or --hidden=256,32")
    p.add_argument("--init", default=None,
                   help="warm-start from an existing weights file (e.g. the current "
                        "champion) instead of random init — converges in far fewer "
                        "epochs on a mostly-unchanged dataset. The file's arch must "
                        "match --hidden/the feature layout, else it is ignored.")
    p.add_argument("--epochs", type=int, default=200,
                   help="max epochs; early stopping usually ends sooner")
    p.add_argument("--patience", type=int, default=8,
                   help="stop after this many epochs with no val improvement "
                        "(0 disables early stopping, runs all --epochs)")
    p.add_argument("--batch", type=int, default=8192)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.0,
                   help="weight decay (AdamW). 0 = none (default, == plain Adam). "
                        "A small value (e.g. 1e-4) regularizes the extra parameters "
                        "of a wider net / richer feature set so they overfit less; "
                        "the first-layer upgrade plan relies on this (docs/"
                        "first-layer-strategy.md).")
    p.add_argument("--scale", type=float, default=600.0,
                   help="centipawns at tanh saturation (written into the weights)")
    p.add_argument("--lambda", dest="lam", type=float, default=1.0,
                   help="TD/bootstrap mix: target = lam*result + (1-lam)*tanh(v/scale), "
                        "where v is the recorded per-position search value. 1.0 = pure "
                        "game result (default). <1 leans on the search value — use only "
                        "with nn-generated data (handcrafted v reintroduces its bias).")
    p.add_argument("--val", type=float, default=0.05, help="validation fraction")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_data(path, np):
    """Read the featurized JSONL and pack it into dense arrays.

    Feature lists are variable-length, so they are packed into one fixed-width
    int32 matrix padded with a dedicated padding index (= num_features, filled in
    by the caller); the model's EmbeddingBag uses padding_idx so the padding
    contributes exactly zero. A whole batch is then a single tensor indexing op
    instead of a pure-Python packing loop — the old per-batch make_batch loop
    dominated CPU training time on a millions-of-rows dataset.
    """
    if not os.path.exists(path):
        sys.exit(f"No training data at {path}. Generate raw positions, then featurize:\n"
                 f"  cd web && npm run train:gen && npm run train:featurize")
    feats, targets, values, games = [], [], [], []
    max_f = 1
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            fr = rec["f"]
            feats.append(fr)
            if len(fr) > max_f:
                max_f = len(fr)
            targets.append(float(rec["r"]))
            v = rec.get("v")  # per-position search value (cp) or absent
            values.append(float("nan") if v is None else float(v))
            # No "g" (pre-migration data) -> give the row its own unique id so it
            # stays a singleton "game"; this can never leak across the split.
            games.append(rec.get("g", f"_nog_{len(games)}"))
    if not targets:
        sys.exit(f"{path} has no samples.")
    return (feats, np.asarray(targets, np.float32), np.asarray(values, np.float32),
            games, max_f)


def warm_start(model, path, arch, np, torch):
    """Initialize the model from an existing weights file (the champion).

    Returns the file's scale on success (the caller adopts it so the squash
    matches the init), or None if the file is missing/placeholder/wrong shape —
    then training proceeds from random init exactly as without --init.
    """
    try:
        with open(path) as f:
            obj = json.load(f)
    except OSError:
        print(f"--init: cannot read {path}; training from scratch.")
        return None
    layers = obj.get("layers")
    if not layers and obj.get("w0"):  # legacy single-hidden-layer layout
        layers = [{"w": obj["w0"], "b": obj["b0"]}, {"w": obj["w1"], "b": obj["b1"]}]
    if not obj.get("arch") or not layers or list(obj["arch"]) != arch:
        print(f"--init: {os.path.basename(path)} arch {obj.get('arch')} != {arch}; "
              "training from scratch.")
        return None
    with torch.no_grad():
        h0 = arch[1]
        model.emb.weight[:arch[0]] = torch.tensor(
            np.asarray(layers[0]["w"], np.float32).reshape(arch[0], h0))
        model.b0.copy_(torch.tensor(np.asarray(layers[0]["b"], np.float32)))
        for i, lin in enumerate(model.lins):
            w = np.asarray(layers[i + 1]["w"], np.float32).reshape(arch[i + 1], arch[i + 2])
            lin.weight.copy_(torch.tensor(w.T))  # [in,out] -> [out,in]
            lin.bias.copy_(torch.tensor(np.asarray(layers[i + 1]["b"], np.float32)))
    print(f"Warm start: initialized from {os.path.basename(path)}.")
    return obj.get("scale")


def main():
    args = parse_args()
    import numpy as np
    import torch
    from torch import nn

    hidden = [int(x) for x in str(args.hidden).split(",") if x.strip()]
    if not hidden:
        sys.exit("--hidden must be one or more positive integers (e.g. 128 or 256,32)")

    num_features = read_num_features(args.data)  # from featurize.mjs sidecar (nn.js)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    feats, targets, values, games, max_f = load_data(args.data, np)
    n = len(targets)

    # Pack the variable-length feature lists into one padded int32 matrix (see
    # load_data); index num_features is the padding row.
    mat = np.full((n, max_f), num_features, dtype=np.int32)
    for i, fr in enumerate(feats):
        mat[i, :len(fr)] = fr
    del feats

    # Split by GAME, not by position (see header): group rows by game id, shuffle
    # the games, then send whole games to val/train so no game straddles the split.
    from collections import defaultdict
    by_game = defaultdict(list)
    for i, gid in enumerate(games):
        by_game[gid].append(i)
    game_ids = list(by_game.keys())
    order = torch.randperm(len(game_ids)).tolist()
    game_ids = [game_ids[i] for i in order]
    n_val_games = max(1, int(len(game_ids) * args.val))
    val_set = set(game_ids[:n_val_games])
    val_idx = torch.tensor(
        [i for gid in game_ids if gid in val_set for i in by_game[gid]], dtype=torch.long)
    train_idx = torch.tensor(
        [i for gid in game_ids if gid not in val_set for i in by_game[gid]], dtype=torch.long)
    n_val = len(val_idx)
    print(f"Loaded {n:,} positions in {len(game_ids):,} games from {args.data}")
    print(f"Split by game: {len(game_ids) - n_val_games:,} train / {n_val_games:,} val games "
          f"({len(train_idx):,} / {n_val:,} positions)")

    class Net(nn.Module):
        # hidden is a list of layer widths. The first (sparse) layer is an
        # EmbeddingBag(sum) over active features == nn.js's column-add; any further
        # widths add dense ReLU layers; a final Linear(.,1) is the scalar head.
        # The vocab has one extra row — the padding index — so a batch is a fixed-
        # width int matrix (padding contributes zero and gets no gradient); the
        # export below drops that row.
        def __init__(self, hidden):
            super().__init__()
            self.emb = nn.EmbeddingBag(num_features + 1, hidden[0], mode="sum",
                                       padding_idx=num_features)
            self.b0 = nn.Parameter(torch.zeros(hidden[0]))
            dims = hidden + [1]
            self.lins = nn.ModuleList(
                [nn.Linear(dims[i], dims[i + 1]) for i in range(len(hidden))])

        def forward(self, rows):
            x = torch.relu(self.emb(rows) + self.b0)
            last = len(self.lins) - 1
            for i, lin in enumerate(self.lins):
                x = lin(x)
                if i < last:
                    x = torch.relu(x)
            return x.squeeze(-1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Net(hidden).to(device)

    # Warm start (--init): begin from the champion's weights so a candidate on a
    # mostly-unchanged dataset fine-tunes in a few epochs instead of relearning
    # everything from scratch. The init file's scale is adopted (the net's output
    # is calibrated to it), overriding --scale if they differ.
    if args.init:
        init_scale = warm_start(model, args.init, [num_features, *hidden, 1], np, torch)
        if init_scale is not None and init_scale != args.scale:
            print(f"Adopting scale {init_scale} from --init (was {args.scale}).")
            args.scale = init_scale

    # TD/bootstrap target: blend the game result with the recorded search value
    # (target = lam*result + (1-lam)*tanh(v/scale)). lam=1 -> pure result (unchanged).
    # Positions without a `v` (random openings / legacy data) always use the result.
    # Computed after --init so the blend uses the adopted scale.
    if args.lam >= 1.0:
        blended = targets
    else:
        has_v = ~np.isnan(values)
        blended = np.where(
            has_v,
            args.lam * targets + (1.0 - args.lam) * np.tanh(values / args.scale),
            targets).astype(np.float32)
        print(f"TD target: lambda={args.lam}, {int(has_v.sum()):,}/{n:,} positions have a search value")
    # Whole-dataset tensors; batches are plain row-indexing (and a device copy
    # when training on GPU — a no-op on CPU).
    mat_t = torch.from_numpy(mat).to(device)
    targets_t = torch.from_numpy(blended).to(device)

    # AdamW so --wd is decoupled weight decay (with wd=0 this is identical to the
    # previous plain Adam). Weight decay is the regularizer the 2026-06 capacity
    # experiments lacked — see docs/first-layer-strategy.md.
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    loss_fn = nn.MSELoss()
    print(f"Training on {device}: {len(train_idx):,} train / {n_val:,} val, "
          f"inputs={num_features}, hidden={hidden}, wd={args.wd}, "
          f"max epochs={args.epochs}, patience={args.patience}")

    def evaluate(idx):
        model.eval()
        with torch.no_grad():
            total = 0.0
            for s in range(0, len(idx), args.batch):
                b = idx[s:s + args.batch].to(device)
                pred = torch.tanh(model(mat_t[b]))
                total += loss_fn(pred, targets_t[b]).item() * len(b)
        return total / len(idx)

    # Early stopping: keep the weights with the lowest validation loss and stop
    # once it hasn't improved for `patience` epochs. This auto-tunes the epoch
    # count — it learns the data fully without overfitting, regardless of how big
    # --epochs is set. The exported net is the *best* one seen, not the last.
    import copy
    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    stale = 0
    t_train = time.time()
    for epoch in range(args.epochs):
        t_epoch = time.time()
        model.train()
        order = train_idx[torch.randperm(len(train_idx))]
        run = 0.0
        for s in range(0, len(order), args.batch):
            b = order[s:s + args.batch].to(device)
            pred = torch.tanh(model(mat_t[b]))
            loss = loss_fn(pred, targets_t[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            run += loss.item() * len(b)
        tr = run / len(order)
        va = evaluate(val_idx)

        improved = va < best_val - 1e-4
        if improved:
            best_val, best_epoch, stale = va, epoch + 1, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        print(f"  epoch {epoch + 1:>3}/{args.epochs}  train {tr:.4f}  val {va:.4f}  "
              f"{time.time() - t_epoch:4.0f}s"
              f"{'  *best' if improved else f'  (no improvement {stale}/{args.patience})'}")
        if args.patience and stale >= args.patience:
            print(f"Early stop: no val improvement for {args.patience} epochs.")
            break

    # Restore and export the best net (lowest val loss), not the final epoch's.
    model.load_state_dict(best_state)
    print(f"Best val {best_val:.4f} at epoch {best_epoch} "
          f"(trained {fmt_dur(time.time() - t_train)}).")

    # Export in nn.js's layout (generic `layers`). Every layer's w is input-major
    # and flattened as w[i*outDim + o]:
    #   layer 0  = EmbeddingBag.weight, already [NUM_FEATURES, h0] = [feature, h].
    #   layers k = nn.Linear, whose weight is [out, in]; transpose to [in, out].
    model.cpu().eval()

    def rnd(a):
        return [round(float(x), 6) for x in a.reshape(-1).tolist()]

    layers = [{
        # Drop the padding row (the extra vocab entry) — nn.js indexes 0..NUM_FEATURES-1.
        "w": rnd(model.emb.weight.detach().numpy()[:num_features]),
        "b": rnd(model.b0.detach().numpy()),
    }]
    for lin in model.lins:
        layers.append({
            "w": rnd(lin.weight.detach().numpy().T),  # [out,in] -> [in,out]
            "b": rnd(lin.bias.detach().numpy()),
        })

    out = {
        "arch": [num_features, *hidden, 1],
        "scale": args.scale,
        "layers": layers,
    }
    # --name publishes into the web catalog (web/public/nn/<name>.json) and registers
    # it in the manifest; otherwise write the plain --out file (the Node-tools default).
    out_path = os.path.join(NN_CATALOG, f"{args.name}.json") if args.name else args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f)
    print(f"Wrote {out_path} ({os.path.getsize(out_path) // 1024} KB). "
          f"Rebuild the web app (or restart dev) to pick it up.")
    if args.name:
        mpath = update_manifest(args.name, f"{args.name}.json",
                                [num_features, *hidden, 1], args.note, args.set_default)
        print(f"Registered '{args.name}' in {mpath}.")


if __name__ == "__main__":
    main()
