"""Probe saved recognition models with negatives that tell comparison mechanisms apart.

A detector that only uses token counts (a bag of tokens) cannot see swaps or shuffles; a plain sum checksum cannot
see a double substitution whose shifts cancel; a position-by-position comparison sees all of them.

  python experiments/probe_ckpt.py --results results/diag --ckpts ckpt/diag --out results/diag/probe.json
"""

import argparse
import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

from train_gdn2 import REPO, VOCAB, Model, cls_logit, load_split

KINDS = ["sub1", "sub2_random", "sub2_cancel", "swap_adjacent", "swap_far", "shuffle"]


def make_negative(seq, kind, gen):
    """A negative of `seq` [B, N] and a mask of the rows where it differs from `seq`."""
    b, n = seq.shape
    dev, rows = seq.device, torch.arange(b, device=seq.device)
    ri = lambda lo, hi: torch.randint(lo, hi, (b,), device=dev, generator=gen)  # noqa: E731
    neg = seq.clone()
    if kind == "sub1":
        i = ri(0, n)
        neg[rows, i] = (seq[rows, i] + ri(1, VOCAB)) % VOCAB
    elif kind in ("sub2_random", "sub2_cancel"):
        i = ri(0, n)
        j = (i + ri(1, n)) % n
        s = ri(1, VOCAB)
        neg[rows, i] = (seq[rows, i] + s) % VOCAB
        neg[rows, j] = (seq[rows, j] + (VOCAB - s if kind == "sub2_cancel" else ri(1, VOCAB))) % VOCAB  # cancel: sum unchanged
    elif kind in ("swap_adjacent", "swap_far"):
        i = ri(0, n - 1) if kind == "swap_adjacent" else ri(0, n // 2)
        j = i + 1 if kind == "swap_adjacent" else i + n // 2
        neg[rows, i], neg[rows, j] = seq[rows, j], seq[rows, i]
    else:  # shuffle: same token counts, new order
        neg = seq.gather(1, torch.rand(b, n, device=dev, generator=gen).argsort(1))
    return neg, (neg != seq).any(1)


@torch.no_grad()
def probe(model, seq_all, bs, dev, seed=0):
    gen = torch.Generator(device=dev).manual_seed(seed)
    rates = {}
    for kind in KINDS:
        win = tot = 0.0
        for i in range(0, len(seq_all), bs):
            seq = seq_all[i:i + bs].to(dev).long()
            neg, valid = make_negative(seq, kind, gen)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                s = cls_logit(model, torch.cat([seq, seq]), torch.cat([seq, neg])).float().view(2, -1)
            w = (s[0] > s[1]).float() + 0.5 * (s[0] == s[1]).float()
            win += w[valid].sum().item()
            tot += valid.sum().item()
        rates[kind] = win / max(tot, 1)
    return rates


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results/diag")
    p.add_argument("--ckpts", default="ckpt/diag")
    p.add_argument("--samples", type=int, default=5000)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    dev = torch.device("cuda")
    out, data = {}, {}
    for f in sorted(Path(a.results).glob("diag_*.json")):
        r = json.loads(f.read_text())
        ck = Path(a.ckpts) / f"{f.stem}.pt"
        if not ck.exists():
            continue
        g = r["args"]
        if g["subset"] not in data:
            root = Path(snapshot_download(REPO, repo_type="dataset", local_dir=g["data"], allow_patterns=[f"{g['subset']}/*"]))
            data[g["subset"]] = load_split(root, g["subset"], "validation")[0][:a.samples]
        model = Model(g["hidden"], g["layers"], g["heads"], g["head_dim"] or g["hidden"] // g["heads"], g["expand_v"]).to(dev)
        model.load_state_dict(torch.load(ck, map_location=dev))
        model.eval()
        out[f.stem] = {"acc9": r["test"]["recog_acc9"], **probe(model, data[g["subset"]], 1024, dev)}
        print(f.stem, json.dumps({k: round(v, 3) for k, v in out[f.stem].items()}), flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
