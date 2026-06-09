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

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(THIS_DIR)
DEFAULT_DATA = os.path.join(THIS_DIR, "data", "selfplay.jsonl")
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

# Must match nn.js NUM_FEATURES (the EmbeddingBag vocab size). 768 = the plain
# piece-square block (12 piece-kinds × 64 squares). Keep in sync with nn.js if you
# add feature blocks (e.g. king-relative buckets would multiply this).
NUM_FEATURES = 12 * 64  # 768


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
    p.add_argument("--epochs", type=int, default=200,
                   help="max epochs; early stopping usually ends sooner")
    p.add_argument("--patience", type=int, default=8,
                   help="stop after this many epochs with no val improvement "
                        "(0 disables early stopping, runs all --epochs)")
    p.add_argument("--batch", type=int, default=8192)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--scale", type=float, default=600.0,
                   help="centipawns at tanh saturation (written into the weights)")
    p.add_argument("--val", type=float, default=0.05, help="validation fraction")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_data(path):
    if not os.path.exists(path):
        sys.exit(f"No training data at {path}. Generate it first:\n"
                 f"  cd web && npm run train:gen")
    samples, targets, games = [], [], []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            samples.append(rec["f"])
            targets.append(float(rec["r"]))
            # No "g" (pre-migration data) -> give the row its own unique id so it
            # stays a singleton "game"; this can never leak across the split.
            games.append(rec.get("g", f"_nog_{len(games)}"))
    if not targets:
        sys.exit(f"{path} has no samples.")
    return samples, targets, games


def main():
    args = parse_args()
    import numpy as np
    import torch
    from torch import nn

    hidden = [int(x) for x in str(args.hidden).split(",") if x.strip()]
    if not hidden:
        sys.exit("--hidden must be one or more positive integers (e.g. 128 or 256,32)")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    samples, targets, games = load_data(args.data)
    n = len(targets)

    targets_t = torch.tensor(targets, dtype=torch.float32)

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
    print(f"Loaded {n} positions in {len(game_ids)} games from {args.data}")
    print(f"Split by game: {len(game_ids) - n_val_games} train / {n_val_games} val games "
          f"({len(train_idx)} / {n_val} positions)")

    class Net(nn.Module):
        # hidden is a list of layer widths. The first (sparse) layer is an
        # EmbeddingBag(sum) over active features == nn.js's column-add; any further
        # widths add dense ReLU layers; a final Linear(.,1) is the scalar head.
        def __init__(self, hidden):
            super().__init__()
            self.emb = nn.EmbeddingBag(NUM_FEATURES, hidden[0], mode="sum")
            self.b0 = nn.Parameter(torch.zeros(hidden[0]))
            dims = hidden + [1]
            self.lins = nn.ModuleList(
                [nn.Linear(dims[i], dims[i + 1]) for i in range(len(hidden))])

        def forward(self, flat, offs):
            x = torch.relu(self.emb(flat, offs) + self.b0)
            last = len(self.lins) - 1
            for i, lin in enumerate(self.lins):
                x = lin(x)
                if i < last:
                    x = torch.relu(x)
            return x.squeeze(-1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Net(hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    print(f"Training on {device}: {len(train_idx)} train / {n_val} val, "
          f"hidden={hidden}, max epochs={args.epochs}, patience={args.patience}")

    def make_batch(sample_ids):
        # Pack variable-length feature lists into one flat tensor + bag offsets.
        flat, offs = [], []
        pos = 0
        for i in sample_ids.tolist():
            offs.append(pos)
            f = samples[i]
            flat.extend(f)
            pos += len(f)
        flat = torch.tensor(flat, dtype=torch.long, device=device)
        offs = torch.tensor(offs, dtype=torch.long, device=device)
        tgt = targets_t[sample_ids].to(device)
        return flat, offs, tgt

    def evaluate(idx):
        model.eval()
        with torch.no_grad():
            total = 0.0
            for s in range(0, len(idx), args.batch):
                flat, offs, tgt = make_batch(idx[s:s + args.batch])
                pred = torch.tanh(model(flat, offs))
                total += loss_fn(pred, tgt).item() * len(tgt)
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
    for epoch in range(args.epochs):
        model.train()
        order = train_idx[torch.randperm(len(train_idx))]
        run = 0.0
        for s in range(0, len(order), args.batch):
            flat, offs, tgt = make_batch(order[s:s + args.batch])
            pred = torch.tanh(model(flat, offs))
            loss = loss_fn(pred, tgt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            run += loss.item() * len(tgt)
        tr = run / len(order)
        va = evaluate(val_idx)

        improved = va < best_val - 1e-4
        if improved:
            best_val, best_epoch, stale = va, epoch + 1, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        print(f"  epoch {epoch + 1:>3}/{args.epochs}  train {tr:.4f}  val {va:.4f}"
              f"{'  *best' if improved else f'  (no improvement {stale}/{args.patience})'}")
        if args.patience and stale >= args.patience:
            print(f"Early stop: no val improvement for {args.patience} epochs.")
            break

    # Restore and export the best net (lowest val loss), not the final epoch's.
    model.load_state_dict(best_state)
    print(f"Best val {best_val:.4f} at epoch {best_epoch}.")

    # Export in nn.js's layout (generic `layers`). Every layer's w is input-major
    # and flattened as w[i*outDim + o]:
    #   layer 0  = EmbeddingBag.weight, already [NUM_FEATURES, h0] = [feature, h].
    #   layers k = nn.Linear, whose weight is [out, in]; transpose to [in, out].
    model.cpu().eval()

    def rnd(a):
        return [round(float(x), 6) for x in a.reshape(-1).tolist()]

    layers = [{
        "w": rnd(model.emb.weight.detach().numpy()),
        "b": rnd(model.b0.detach().numpy()),
    }]
    for lin in model.lins:
        layers.append({
            "w": rnd(lin.weight.detach().numpy().T),  # [out,in] -> [in,out]
            "b": rnd(lin.bias.detach().numpy()),
        })

    out = {
        "arch": [NUM_FEATURES, *hidden, 1],
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
                                [NUM_FEATURES, *hidden, 1], args.note, args.set_default)
        print(f"Registered '{args.name}' in {mpath}.")


if __name__ == "__main__":
    main()
