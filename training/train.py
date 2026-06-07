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
# White's view in {-1, 0, +1}; we squash the raw output with tanh and fit MSE, so
# the net learns a win-probability-like signal grounded in who actually won.
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

NUM_FEATURES = 12 * 64 + 1  # must match nn.js (769)


def parse_args():
    p = argparse.ArgumentParser(description="Train the AposChess NN evaluation.")
    p.add_argument("--data", default=DEFAULT_DATA, help="JSONL training data")
    p.add_argument("--out", default=DEFAULT_OUT, help="weights output (JSON for nn.js)")
    p.add_argument("--hidden", type=int, default=64, help="hidden layer size")
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
        def __init__(self, h):
            super().__init__()
            self.emb = nn.EmbeddingBag(NUM_FEATURES, h, mode="sum")
            self.b0 = nn.Parameter(torch.zeros(h))
            self.head = nn.Linear(h, 1)

        def forward(self, flat, offs):
            x = self.emb(flat, offs) + self.b0
            x = torch.relu(x)
            return self.head(x).squeeze(-1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Net(args.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    print(f"Training on {device}: {len(train_idx)} train / {n_val} val, "
          f"hidden={args.hidden}, max epochs={args.epochs}, patience={args.patience}")

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

    # Export in nn.js's layout. EmbeddingBag.weight is [NUM_FEATURES, hidden] =
    # exactly the input-major [feature, h] order nn.js reads as w0[f*hidden + h].
    model.cpu().eval()
    w0 = model.emb.weight.detach().numpy().reshape(-1).tolist()
    b0 = model.b0.detach().numpy().reshape(-1).tolist()
    w1 = model.head.weight.detach().numpy().reshape(-1).tolist()
    b1 = model.head.bias.detach().numpy().reshape(-1).tolist()

    out = {
        "arch": [NUM_FEATURES, args.hidden, 1],
        "scale": args.scale,
        "w0": [round(x, 6) for x in w0],
        "b0": [round(x, 6) for x in b0],
        "w1": [round(x, 6) for x in w1],
        "b1": [round(x, 6) for x in b1],
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"Wrote {args.out} ({os.path.getsize(args.out) // 1024} KB). "
          f"Rebuild the web app (or restart dev) to pick it up.")


if __name__ == "__main__":
    main()
