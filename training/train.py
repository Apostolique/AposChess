#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2019-2026 Jean-David Moisan
#
# Trainer for the AposChess neural-net evaluation. Reads the featurized self-play
# data from web/scripts/featurize.mjs (JSONL: {"f":[feature indices],"r":result,
# "g":game id,"v":search value}) and fits a small MLP, then writes the weights to
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
import hashlib
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

# NNUE integer-quantization scales (used only with --quant). Activations are stored at
# fixed-point scale QA (a real activation x -> round(x*QA), plain ReLU, not clamped);
# weights are scaled by QA (accumulator / input layer) or QW (dense layers, biases QW*QA).
# These are stamped into the weights JSON so nn.js / nn.zig reproduce the exact pipeline.
QUANT_QA = 1024  # activation fixed-point scale (~1cp rounding vs float; int8-style 127 was too coarse)
QUANT_QW = 1024  # dense-weight scale


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
                        "epochs on a mostly-unchanged dataset. If the file's arch "
                        "matches --hidden it is copied exactly; if it differs (but the "
                        "feature layout matches) the weights are GRAFTED into the new "
                        "shape (function-preserving widening when every layer grows, a "
                        "lossy sub-block copy when shrinking) — see --graft-noise / "
                        "--no-graft. Ignored only if the feature layout differs.")
    p.add_argument("--graft-noise", type=float, default=0.02,
                   help="when --init grafts across a WIDER shape, the duplicated units "
                        "are function-preserving but start tied (identical gradients, so "
                        "the extra capacity never trains). This adds N(0, graft-noise * "
                        "std) to the new units' outgoing weights to break that symmetry — "
                        "tiny, so the graft stays ~function-preserving. 0 = exact (units "
                        "stay tied until other noise breaks them; use for verification).")
    p.add_argument("--no-graft", action="store_true",
                   help="disable grafting: an --init whose arch differs from --hidden is "
                        "ignored (train from random init), the pre-graft behaviour.")
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
    p.add_argument("--quant", action="store_true",
                   help="export INTEGER (quantized) weights instead of float: layer-0 "
                        "weights/bias at fixed-point scale QA, dense weights at QW (biases "
                        "QW*QA), plain-ReLU activations at scale QA. Bit-exact across JS/Zig "
                        "and the prerequisite for the incremental accumulator; faithfully "
                        "reproduces the float net (rounding only). Training is unchanged — "
                        "an existing float net can also be quantized post-hoc.")
    p.add_argument("--lambda", dest="lam", type=float, default=1.0,
                   help="TD/bootstrap mix: target = lam*result + (1-lam)*tanh(v/scale), "
                        "where v is the recorded per-position search value. 1.0 = pure "
                        "game result (default). <1 leans on the search value — use only "
                        "with nn-generated data (handcrafted v reintroduces its bias).")
    p.add_argument("--val", type=float, default=0.05, help="validation fraction")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-cache", action="store_true",
                   help="don't read or write the packed-array cache beside --data "
                        "(<data>.cache.*): parse the JSONL every run, as before. The "
                        "cache is keyed to the file's length + a hash of its tail, so "
                        "it invalidates itself; this is for debugging it.")
    return p.parse_args()


# --- Packed-array cache -----------------------------------------------------
#
# Parsing the featurized JSONL is the trainer's biggest fixed cost once the
# dataset is large, and it is paid again on every loop cycle even though the
# cycle usually only APPENDED a few MB to the file. Measured on the 3.07 GB /
# 25.7M-position set: ~14 min of the ~25-59 min training step, and a ~18 GB
# memory peak (14.7 GB of that is interpreter objects — a list of 25.7M little
# Python lists — on a 32 GB machine shared with the gate match).
#
# So the packed arrays are cached beside the data in NumPy's binary format.
# Validity mirrors featurize.mjs's own incremental sidecar: the source's byte
# LENGTH at cache time plus a SHA-1 of that prefix's last 64 KB.
#   same length and hash          -> load the cache, parse nothing
#   longer, prefix still hashes    -> parse only the appended tail, extend
#   anything else                  -> full rebuild
# An in-place rewrite (refresh-v), a filter change, or a num_features change
# therefore each fall back to a full pass on their own, with no flag to
# remember. The cache is a derived artifact under the git-ignored
# training/data/ and is safe to delete at any time (--no-cache skips it).
CACHE_VERSION = 1
CACHE_TAIL = 64 * 1024


def _cache_paths(data_path):
    base = data_path[:-len(".jsonl")] if data_path.endswith(".jsonl") else data_path
    return base + ".cache.json", base + ".cache.f.npy", base + ".cache.aux.npz"


def _tail_hash(path, upto):
    """SHA-1 of the last CACHE_TAIL bytes of `path`'s first `upto` bytes."""
    with open(path, "rb") as f:
        f.seek(max(0, upto - CACHE_TAIL))
        return hashlib.sha1(f.read(min(upto, CACHE_TAIL))).hexdigest()


def _parse_jsonl(path, start, np, dtype, pad, codes, next_code):
    """Parse whole JSONL records from byte offset `start` to EOF.

    Feature rows go straight into fixed-width int blocks and each record's Python
    list is dropped immediately. The old list-of-lists cost ~570 B/row in
    interpreter objects (14.7 GB across the current set) and was what made the
    load slow as much as the json.loads itself. Blocks carry their own width and
    are padded up to the global one at the end, so no maximum row length is
    baked in.

    `codes` maps game id -> int code in FIRST-APPEARANCE order and `next_code` is
    the next free one; both are carried in so an incremental extend keeps
    numbering where the cached prefix left off. Returns
    (mat, r, v, g, end_offset, next_code, last_gid, last_code) — the trailing
    pair identifies the last real game seen, which is what an extend needs to
    rejoin a game straddling the cut.
    """
    BLOCK = 1 << 20
    fb, rb, vb, gb, widths = [], [], [], [], []
    cur = cur_r = cur_v = cur_g = None
    w = k = 0
    pos = start
    last_gid = last_code = None

    def flush():
        if cur is not None and k:
            fb.append(cur[:k]); widths.append(w)
            rb.append(cur_r[:k]); vb.append(cur_v[:k]); gb.append(cur_g[:k])

    with open(path, "rb") as f:
        f.seek(start)
        for line in f:  # binary mode: len(line) is the exact byte count consumed
            pos += len(line)
            if not line.strip():
                continue
            rec = json.loads(line)
            fr = rec["f"]
            m = len(fr)
            if cur is None or k == BLOCK:
                flush()
                w = max(m, 1)
                cur = np.full((BLOCK, w), pad, dtype)
                cur_r = np.empty(BLOCK, np.float32)
                cur_v = np.empty(BLOCK, np.float32)
                cur_g = np.empty(BLOCK, np.int32)
                k = 0
            if m > w:  # a longer row than this block was sized for — widen it
                grown = np.full((BLOCK, m), pad, dtype)
                grown[:k, :w] = cur[:k]
                cur, w = grown, m
            cur[k, :m] = fr
            cur_r[k] = rec["r"]
            v = rec.get("v")  # per-position search value (cp) or absent
            cur_v[k] = np.nan if v is None else v
            gid = rec.get("g")
            if gid is None:
                # No "g" (pre-migration data) -> its own singleton "game", so it
                # can never leak across the train/val split.
                cur_g[k] = next_code
                next_code += 1
            else:
                c = codes.get(gid)
                if c is None:
                    c = codes[gid] = next_code
                    next_code += 1
                cur_g[k] = c
                last_gid, last_code = gid, int(c)
            k += 1
    flush()

    if not fb:
        z = np.zeros
        return (z((0, 1), dtype), z(0, np.float32), z(0, np.float32), z(0, np.int32),
                pos, next_code, last_gid, last_code)
    width = max(widths)
    n = sum(len(b) for b in fb)
    mat = np.full((n, width), pad, dtype)
    o = 0
    for i, b in enumerate(fb):
        mat[o:o + len(b), :widths[i]] = b
        o += len(b)
        fb[i] = None  # release each block as it lands, so the peak is ~1 copy
    return (mat, np.concatenate(rb), np.concatenate(vb), np.concatenate(gb),
            pos, next_code, last_gid, last_code)


def load_data(path, np, num_features, use_cache=True):
    """Read the featurized JSONL into packed arrays, via the binary cache above.

    Feature lists are variable-length, so they are packed into one fixed-width
    int matrix padded with a dedicated padding index (= num_features); the
    model's EmbeddingBag uses padding_idx so the padding contributes exactly
    zero. A whole batch is then a single tensor indexing op instead of a
    pure-Python packing loop, which would dominate CPU training time on a
    millions-of-rows dataset. Indices fit in int16 (768 features today), which
    halves both the resident matrix and the per-batch gather; the batch is cast
    to int32 on the way into the EmbeddingBag.

    Returns (mat, targets, values, games, n_games), where `games` is a per-row
    int32 game code in first-appearance order.
    """
    if not os.path.exists(path):
        sys.exit(f"No training data at {path}. Generate raw positions, then featurize:\n"
                 f"  cd web && npm run train:gen && npm run train:featurize")
    dtype = np.int16 if num_features < 32767 else np.int32
    meta_p, f_p, aux_p = _cache_paths(path)
    st = os.stat(path)
    size = st.st_size

    meta = None
    if use_cache and all(os.path.exists(p) for p in (meta_p, f_p, aux_p)):
        try:
            with open(meta_p) as f:
                m = json.load(f)
            if (m.get("version") == CACHE_VERSION and m.get("num_features") == num_features
                    and m.get("dtype") == np.dtype(dtype).name
                    and 0 < m.get("bytes", 0) <= size
                    and m["tail_hash"] == _tail_hash(path, m["bytes"])):
                meta = m
            # An in-place rewrite that happens to land on the same length with the
            # same last 64 KB would pass the check above; the mtime never does.
            # (Only for an exact hit — an append moves the mtime legitimately, and
            # there the prefix hash is what has to hold.)
            if meta and meta["bytes"] == size and meta.get("mtime_ns") != st.st_mtime_ns:
                meta = None
        except (OSError, ValueError, KeyError):
            meta = None  # unreadable/garbage cache -> just rebuild it

    t0 = time.time()
    if meta and meta["bytes"] == size:
        mat = np.load(f_p)
        # np.load on an .npz is lazy and holds the file handle open, which on
        # Windows blocks the os.replace that rewrites the cache — so read the
        # members out and close it.
        with np.load(aux_p) as aux:
            targets, values, games = aux["r"], aux["v"], aux["g"]
        print(f"Feature cache: hit — {len(targets):,} rows, "
              f"{mat.nbytes / 1e9:.2f} GB, loaded in {fmt_dur(time.time() - t0)}.")
        return mat, targets, values, games, meta["n_games"]

    if meta:  # append-only growth: parse just the new tail and extend the cache
        print(f"Feature cache: extending — {(size - meta['bytes']) / 1e6:.1f} MB of new data "
              f"on {meta['bytes'] / 1e9:.2f} GB already packed.")
        # Only the game at the cut can straddle the boundary (featurize appends
        # whole games), so seeding its id is enough to keep a straddling game on
        # one side of the by-game split instead of splitting it in two. Its code
        # is stored rather than assumed to be the highest: rows carrying no "g"
        # take codes of their own, so the last real game need not be the last one
        # numbered.
        codes = ({meta["last_gid"]: meta["last_code"]}
                 if meta.get("last_gid") is not None else {})
        new = _parse_jsonl(path, meta["bytes"], np, dtype, num_features, codes, meta["n_games"])
        mat_n, r_n, v_n, g_n, end, next_code, last_gid, last_code = new
        old_mat = np.load(f_p)
        with np.load(aux_p) as aux:  # closed before the os.replace below (see above)
            old_r, old_v, old_g = aux["r"], aux["v"], aux["g"]
        width = max(old_mat.shape[1], mat_n.shape[1])
        if len(r_n):
            mat = np.full((len(old_mat) + len(mat_n), width), num_features, dtype)
            mat[:len(old_mat), :old_mat.shape[1]] = old_mat
            mat[len(old_mat):, :mat_n.shape[1]] = mat_n
            targets = np.concatenate([old_r, r_n])
            values = np.concatenate([old_v, v_n])
            games = np.concatenate([old_g, g_n])
        else:  # trailing whitespace only — nothing new to add
            mat, targets, values, games = old_mat, old_r, old_v, old_g
        if last_gid is None:  # the tail added no real game — keep the prefix's
            last_gid, last_code = meta.get("last_gid"), meta.get("last_code")
    else:
        print(f"Feature cache: building from {size / 1e9:.2f} GB of JSONL "
              f"(one-time per featurized file; appends after this are incremental).")
        mat, targets, values, games, end, next_code, last_gid, last_code = _parse_jsonl(
            path, 0, np, dtype, num_features, {}, 0)

    if not len(targets):
        sys.exit(f"{path} has no samples.")
    n_games = int(next_code)
    print(f"Packed {len(targets):,} rows in {fmt_dur(time.time() - t0)} "
          f"({mat.nbytes / 1e9:.2f} GB).")

    if use_cache:
        # Write via temp files + replace so an interrupted run can't leave a cache
        # that claims to describe more data than it holds.
        try:
            np.save(f_p + ".tmp.npy", mat)
            np.savez(aux_p + ".tmp.npz", r=targets, v=values, g=games)
            with open(meta_p + ".tmp", "w") as f:
                json.dump({"version": CACHE_VERSION, "num_features": num_features,
                           "dtype": np.dtype(dtype).name, "bytes": end,
                           "tail_hash": _tail_hash(path, end), "rows": len(targets),
                           "mtime_ns": os.stat(path).st_mtime_ns, "n_games": n_games,
                           "last_gid": last_gid, "last_code": last_code}, f)
            os.replace(f_p + ".tmp.npy", f_p)
            os.replace(aux_p + ".tmp.npz", aux_p)
            os.replace(meta_p + ".tmp", meta_p)
            print(f"Feature cache: wrote {os.path.basename(f_p)} — next run skips the parse.")
        except OSError as e:  # out of disk / read-only data dir: not fatal
            print(f"Feature cache: could not write ({e}); continuing without it.")

    return mat, targets, values, games, n_games


def _widen_map(p, c):
    """Map c child units to p parent units for a grafted layer.

    Widen (c >= p): keep the p originals, then round-robin-replicate them into the
    c-p new slots. Shrink (c < p): keep the first c parents (a lossy truncation).
    Returns (sel, count): sel[k] is the parent unit child unit k copies; count[u]
    is how many child units map to parent u (its replication factor, 0 if dropped).
    """
    sel = list(range(p)) + [k % p for k in range(max(0, c - p))] if c >= p else list(range(c))
    count = [0] * p
    for s in sel:
        count[s] += 1
    return sel, count


def _dense(layers, p_arch, np):
    """Parse the (dequantized) JSON layer list into parent float arrays:
    emb [feats, h0], b0 [h0], and per dense layer W [in,out] + b [out]."""
    p_hidden = p_arch[1:-1]
    p_dims = list(p_hidden) + [1]
    emb = np.asarray(layers[0]["w"], np.float32).reshape(p_arch[0], p_hidden[0])
    b0 = np.asarray(layers[0]["b"], np.float32)
    W, B = [], []
    for m in range(len(p_hidden)):
        W.append(np.asarray(layers[m + 1]["w"], np.float32).reshape(p_dims[m], p_dims[m + 1]))
        B.append(np.asarray(layers[m + 1]["b"], np.float32))
    return emb, b0, W, B


def _graft(model, layers, p_arch, c_arch, np, torch, noise, seed):
    """Graft parent weights (float `layers`, parent arch `p_arch`) into `model`
    (child arch `c_arch`), reusing what transfers instead of starting from random.

    Net2WiderNet function-preserving widening: to widen an activation layer, we
    REPLICATE existing units (so their activations are unchanged) and divide each
    replicated unit's OUTGOING weights by its replication count (so the sum into the
    next layer is unchanged) — the child computes the same function as the parent,
    then fine-tunes. Only the input (feature) layer must line up; a same-depth child
    grafts every layer, a different-depth child grafts just the big embedding layer
    (the feature detector — the most expensive and most transferable part) and leaves
    the dense stack at random init. Shrinking a layer truncates it (lossy — no longer
    function-preserving, but still a far better start than random). Returns a short
    mode string for the log. `noise` breaks the tied-duplicate symmetry (see argparse).
    """
    rng = np.random.default_rng(seed)
    feats = c_arch[0]
    p_hidden, c_hidden = p_arch[1:-1], c_arch[1:-1]
    emb, b0, Pw, Pb = _dense(layers, p_arch, np)

    with torch.no_grad():
        # Embedding (layer 0): copy/replicate/truncate its output columns to width c_hidden[0].
        # It is the INCOMING side of activation a[0], so no outgoing division here.
        sel0, _ = _widen_map(p_hidden[0], c_hidden[0])
        model.emb.weight[:feats] = torch.tensor(emb[:, sel0])
        model.b0.copy_(torch.tensor(b0[sel0]))

        if len(p_hidden) != len(c_hidden):
            return f"emb-only graft (depth {len(p_hidden)}->{len(c_hidden)}; dense stack random-init)"

        L = len(c_hidden)
        sels = [_widen_map(p_hidden[m], c_hidden[m])[0] for m in range(L)]
        counts = [np.asarray(_widen_map(p_hidden[m], c_hidden[m])[1], np.float32) for m in range(L)]
        grew = shrank = False
        for m, lin in enumerate(model.lins):
            # Expand the INPUT dim (a[m]) with the outgoing split: divide each source unit's
            # row by how many child units share it, so the copies sum back to the original.
            div = counts[m][sels[m]][:, None]
            M = Pw[m][sels[m], :] / div                    # [c_in, p_out]
            if m < L - 1:                                  # expand OUTPUT dim (a[m+1]) by copy
                M = M[:, sels[m + 1]]                      # [c_in, c_out]
                b = Pb[m][sels[m + 1]]
            else:                                          # scalar head — output width stays 1
                b = Pb[m]
            # Break the duplicate-unit symmetry: perturb the NEW units' outgoing weights only,
            # so activations stay ~unchanged but the copies get distinct gradients and train.
            if noise and c_hidden[m] > p_hidden[m]:
                k0 = p_hidden[m]
                sigma = noise * float(np.std(Pw[m])) or noise
                M = M.copy()
                M[k0:, :] += rng.normal(0, sigma, size=M[k0:, :].shape).astype(np.float32)
            lin.weight.copy_(torch.tensor(M.T))            # [in,out] -> [out,in]
            lin.bias.copy_(torch.tensor(b))
            grew |= c_hidden[m] > p_hidden[m]
            shrank |= c_hidden[m] < p_hidden[m]

    if shrank:
        return "lossy graft (some layers truncated)"
    if grew:
        return f"function-preserving widen ({'exact' if not noise else '+symmetry noise'})"
    return "graft"  # same widths, different arch elsewhere — shouldn't reach here


def warm_start(model, path, arch, np, torch, graft_noise=0.02, allow_graft=True, seed=0):
    """Initialize the model from an existing weights file (the champion).

    Exact copy when the file's arch matches `arch`; otherwise GRAFT it into the new
    shape (unless --no-graft) as long as the feature layout matches. Returns the
    file's scale on success (the caller adopts it so the squash matches the init), or
    None if the file is missing/placeholder/ungraftable — then training proceeds from
    random init exactly as without --init.
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
    if not obj.get("arch") or not layers:
        print(f"--init: {os.path.basename(path)} has no usable arch/layers; training from scratch.")
        return None
    p_arch = list(obj["arch"])
    # An integer (quantized) champion stores int weights at scales QA/QW — dequantize
    # back to float so warm-starting works in either direction (float<->quant), so the
    # loop keeps fine-tuning even after a quantized net becomes champion.
    if obj.get("int"):
        qa = obj["quant"]["qa"]
        qw = obj["quant"]["qw"]
        deq = [{"w": [x / qa for x in layers[0]["w"]], "b": [x / qa for x in layers[0]["b"]]}]
        for L in layers[1:]:
            deq.append({"w": [x / qw for x in L["w"]], "b": [x / (qw * qa) for x in L["b"]]})
        layers = deq

    if p_arch == arch:
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

    # Arch mismatch: graft (unless disabled) as long as the feature layout lines up.
    if not allow_graft:
        print(f"--init: {os.path.basename(path)} arch {p_arch} != {arch} and --no-graft; training from scratch.")
        return None
    if p_arch[0] != arch[0]:
        print(f"--init: {os.path.basename(path)} feature layout {p_arch[0]} != {arch[0]}; "
              "training from scratch.")
        return None
    mode = _graft(model, layers, p_arch, arch, np, torch, graft_noise, seed)
    print(f"Graft warm start from {os.path.basename(path)}: arch {p_arch} -> {arch} — {mode}.")
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

    mat, targets, values, games, n_games = load_data(args.data, np, num_features,
                                                     use_cache=not args.no_cache)
    n = len(targets)

    # Split by GAME, not by position (see header): shuffle the games, then send
    # whole games to val/train so no game straddles the split. `games` is a
    # per-row game code in first-appearance order, so the randperm below shuffles
    # exactly the sequence the old dict-keyed version did. Grouping is a stable
    # argsort (rows ordered by their game's shuffled rank) rather than a
    # 25M-iteration Python loop building a list per game.
    perm = torch.randperm(n_games).numpy()
    n_val_games = max(1, int(n_games * args.val))
    rank = np.empty(n_games, np.int32)
    rank[perm] = np.arange(n_games, dtype=np.int32)
    row_order = np.argsort(rank[games], kind="stable")
    n_val = int(np.bincount(games, minlength=n_games)[perm[:n_val_games]].sum())
    val_idx = torch.from_numpy(row_order[:n_val])
    train_idx = torch.from_numpy(row_order[n_val:])
    print(f"Loaded {n:,} positions in {n_games:,} games from {args.data}")
    print(f"Split by game: {n_games - n_val_games:,} train / {n_val_games:,} val games "
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
        init_scale = warm_start(model, args.init, [num_features, *hidden, 1], np, torch,
                                graft_noise=args.graft_noise, allow_graft=not args.no_graft,
                                seed=args.seed)
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
    # when training on GPU — a no-op on CPU). The feature matrix is int16 (see
    # load_data), so each batch is cast to int32 for the EmbeddingBag — a ~1 MB
    # conversion against half the gather traffic of a full-width matrix.
    mat_t = torch.from_numpy(mat).to(device)
    targets_t = torch.from_numpy(blended).to(device)

    # AdamW so --wd is decoupled weight decay (wd=0 == plain Adam).
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
                pred = torch.tanh(model(mat_t[b].int()))
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
            pred = torch.tanh(model(mat_t[b].int()))
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

    # layer 0's w is EmbeddingBag.weight, already [feature, h]; the padding row (the
    # extra vocab entry) is dropped — nn.js indexes 0..NUM_FEATURES-1. lins are
    # nn.Linear weights [out,in], transposed to input-major [in,out].
    emb_w = model.emb.weight.detach().numpy()[:num_features]
    if args.quant:
        # Integer NNUE export: layer 0 (accumulator) weights+bias at scale QA; dense
        # layers (incl. the scalar head) weights at scale QW, biases at QW*QA (they add
        # to a QW*QA-scaled pre-activation). See nn.js for the matching forward pass.
        def qi(a, s):
            return [int(round(float(x) * s)) for x in a.reshape(-1).tolist()]
        layers = [{"w": qi(emb_w, QUANT_QA), "b": qi(model.b0.detach().numpy(), QUANT_QA)}]
        for lin in model.lins:
            layers.append({
                "w": qi(lin.weight.detach().numpy().T, QUANT_QW),
                "b": qi(lin.bias.detach().numpy(), QUANT_QW * QUANT_QA),
            })
        out = {
            "arch": [num_features, *hidden, 1],
            "scale": args.scale,
            "int": True,
            "quant": {"qa": QUANT_QA, "qw": QUANT_QW},
            "layers": layers,
        }
    else:
        layers = [{"w": rnd(emb_w), "b": rnd(model.b0.detach().numpy())}]
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
