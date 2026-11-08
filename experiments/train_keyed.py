"""Keyed recall vs recognition: a small GDN-2 stores N (key, sequence) pairs of length 8, then is queried with a key.

Input: [BOS] k1 s1 ... kN sN [QRY] kq candidate. Two objectives on the same backbone:
  lm  (recall)      next-token prediction of the true sequence after the query key; greedy decoding copies it exactly when the
                    argmax is right at every position. Candidates are also scored by surprise (summed NLL).
  cls (recognition) "candidate is kq's sequence" logit read at the last token; negatives are 50% small edits of the true
                    sequence and 50% sequences stored under other keys (fresh random ones at N = 1), drawn at every step.

Staged curriculum over the number of pairs: train stage_steps steps at N = 1, then at N = 2, 4, ..., each stage on that N's
subset, and test on its test split at the end of the stage. lm runs stop after the first stage whose exact copy is below
--stop_below.

States below d_k = 16 run in fla's 16-wide kernel with the key and query projections masked to their first d_k channels. In
GDN-2's update S <- (I - k (b*k)^T) Diag(exp g) S + k (w*v)^T, a key channel that is always 0 keeps its state row at 0, and
reads only see the live rows; masking the query too keeps its l2 norm over the live channels. So the state is exactly d_k x d_v.

  python experiments/train_keyed.py --objective cls --k 4 --v 4 --out results/keyed/x.json
"""

import argparse
import json
import math
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from fla.modules import RMSNorm
from huggingface_hub import snapshot_download

from train_gdn2 import VOCAB, Block, corrupt, param_groups, provenance

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmark"))
import generate_keyed as kb  # noqa: E402

REPO = "irodkin/recognition-memory-keyed"
L = kb.LENGTH
KEY0 = VOCAB                        # key id i is token KEY0 + i
QRY = KEY0 + kb.KEY_POOL
BOS = QRY + 1
N_TOKENS = BOS + 1


class ChannelMask(nn.Module):
    def __init__(self, live, width):
        super().__init__()
        self.register_buffer("mask", (torch.arange(width) < live).float())

    def forward(self, x):
        return x * self.mask.to(x.dtype)


class KeyedModel(nn.Module):
    def __init__(self, d=32, n_layers=2, k=16, v=16):
        super().__init__()
        width = max(16, k)          # fla's chunk kernel needs a key width of at least 16
        self.embed = nn.Embedding(N_TOKENS, d)
        self.blocks = nn.ModuleList(Block(d, 1, width, v / width, i) for i in range(n_layers))
        if k < width:
            for b in self.blocks:
                b.attn.q_proj = nn.Sequential(b.attn.q_proj, ChannelMask(k, width))
                b.attn.k_proj = nn.Sequential(b.attn.k_proj, ChannelMask(k, width))
        self.norm = RMSNorm(d)
        self.lm_head = nn.Linear(d, VOCAB, bias=False)
        self.cls_head = nn.Linear(d, 1)

    def forward(self, tokens):
        x = self.embed(tokens)
        for b in self.blocks:
            x = b(x)
        return self.norm(x)


@torch.no_grad()
def check_state_size(model, k, dev):
    """The masked key and query channels must be exactly 0 where GDN-2 uses them (after the short convolution and silu)."""
    seen = []
    hooks = [m.register_forward_hook(lambda mod, inp, out: seen.append(out[0] if isinstance(out, tuple) else out))
             for b in model.blocks for m in (b.attn.q_conv1d, b.attn.k_conv1d)]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        model(torch.randint(0, N_TOKENS, (4, 200), device=dev))
    for h in hooks:
        h.remove()
    worst = max(x[..., k:].abs().max().item() for x in seen)
    if worst != 0:
        raise SystemExit(f"masked key channels are not zero (max {worst}): the state would be larger than {k} rows")
    return len(seen)


def make_input(keys, seqs, qkey, cand):
    """keys [B, N], seqs [B, N, L], qkey [B], cand [B, L] -> tokens [B, 1 + N(L+1) + 2 + L]."""
    b = seqs.shape[0]
    pairs = torch.cat([(keys.long() + KEY0)[:, :, None], seqs.long()], 2).reshape(b, -1)
    col = lambda t: torch.full((b, 1), t, dtype=torch.long, device=seqs.device)  # noqa: E731
    return torch.cat([col(BOS), pairs, col(QRY), (qkey.long() + KEY0)[:, None], cand.long()], 1)


def candidate_nll(model, keys, seqs, qkey, cand):
    """Per-token NLL of the candidate after the query key [B, L], and its logits."""
    length = cand.shape[1]
    tokens = make_input(keys, seqs, qkey, cand)
    start = tokens.shape[1] - length                                   # position of cand[0]
    h = model(tokens)
    with torch.autocast("cuda", enabled=False):
        logits = model.lm_head(h[:, start - 1:start - 1 + length].float())
    return F.cross_entropy(logits.transpose(1, 2), cand.long(), reduction="none"), logits


def cls_logit(model, keys, seqs, qkey, cand):
    h = model(make_input(keys, seqs, qkey, cand))[:, -1]
    with torch.autocast("cuda", enabled=False):
        return model.cls_head(h.float()).squeeze(-1)


def load(root, subset, split, n, candidates):
    cols = ["keys", "sequences", "query"] + (["candidates"] if candidates else [])
    out = {c: [] for c in cols}
    for f in sorted((root / subset).glob(f"{split}-*.parquet")):
        t = pq.read_table(f, columns=cols)
        m = t.num_rows
        out["keys"].append(t["keys"].combine_chunks().flatten().to_numpy().reshape(m, n).astype(np.int16))   # pool < 2^15
        out["sequences"].append(t["sequences"].combine_chunks().flatten().flatten().to_numpy().reshape(m, n, L))
        out["query"].append(t["query"].to_numpy().astype(np.int64))
        if candidates:
            out["candidates"].append(t["candidates"].combine_chunks().flatten().flatten().to_numpy().reshape(m, kb.N_EDIT + kb.N_FAR, L))
    return {c: torch.from_numpy(np.concatenate(v)) for c, v in out.items()}


def loss_fn(model, objective, keys, seqs, query, gen):
    b, n, length = seqs.shape
    rows = torch.arange(b, device=seqs.device)
    qkey, true = keys[rows, query], seqs[rows, query]
    if objective == "lm":
        nll, logits = candidate_nll(model, keys, seqs, qkey, true)
        return nll.mean(), (logits.argmax(-1) == true.long()).float().mean()
    rate = torch.full((b,), 1 / (length - 1), device=seqs.device)
    edit = corrupt(true, rate, gen)
    if n > 1:
        other = seqs[rows, (query + torch.randint(1, n, (b,), device=seqs.device, generator=gen)) % n].long()
    else:           # one pair: a fresh random sequence (equal to the true one with probability 16^-8)
        other = torch.randint(0, VOCAB, true.shape, device=seqs.device, generator=gen)
    neg = torch.where((torch.rand(b, device=seqs.device, generator=gen) < 0.5)[:, None], edit, other)
    label = torch.rand(b, device=seqs.device, generator=gen) < 0.5
    logit = cls_logit(model, keys, seqs, qkey, torch.where(label[:, None], true.long(), neg))
    return F.binary_cross_entropy_with_logits(logit, label.float()), ((logit > 0) == label).float().mean()


@torch.no_grad()
def evaluate(model, objective, data, bs, dev):
    """acc9 (ties share the win), pairwise accuracy for edits, other keys and fresh sequences, and, for lm, exact copy and
    token accuracy."""
    model.eval()
    tot, n = {}, 0
    for i in range(0, len(data["query"]), bs):
        keys, seqs, query, cand = (data[c][i:i + bs].to(dev) for c in ("keys", "sequences", "query", "candidates"))
        b, n, length = seqs.shape
        rows = torch.arange(b, device=dev)
        true = seqs[rows, query]
        cands = torch.cat([true[:, None], cand], 1)                                      # [B, 9, L], true first
        rep = lambda x: x.repeat_interleave(9, 0)  # noqa: E731
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if objective == "lm":
                nll, logits = candidate_nll(model, rep(keys), rep(seqs), rep(keys[rows, query]), cands.reshape(b * 9, length))
                score = -nll.float().sum(1).view(b, 9)
                ok = (logits.view(b, 9, length, -1)[:, 0].argmax(-1) == true.long()).float()
                m = {"recall_exact": ok.min(1).values, "recall_token_acc": ok.mean(1), "nll_true": nll.view(b, 9, length)[:, 0].float().mean(1)}
            else:
                score = cls_logit(model, rep(keys), rep(seqs), rep(keys[rows, query]), cands.reshape(b * 9, length)).float().view(b, 9)
                m = {}
        top = score == score.max(1, keepdim=True).values
        win = (score[:, :1] > score[:, 1:]).float() + 0.5 * (score[:, :1] == score[:, 1:]).float()
        e, o = kb.N_EDIT, kb.N_EDIT + kb.n_other(n)
        m.update(recog_acc9=top[:, 0].float() / top.sum(1), pairwise_edit=win[:, :e].mean(1))
        if o > e:
            m["pairwise_other"] = win[:, e:o].mean(1)
        if o < win.shape[1]:
            m["pairwise_fresh"] = win[:, o:].mean(1)
        for k, v in m.items():
            tot[k] = tot.get(k, 0.0) + v.sum().item()
        n += b
    model.train()
    return {k: v / n for k, v in tot.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--objective", choices=["lm", "cls"], required=True)
    p.add_argument("--k", type=int, required=True, help="key dimension of the state (per layer)")
    p.add_argument("--v", type=int, required=True, help="value dimension of the state (per layer)")
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--counts", type=int, nargs="+", default=kb.COUNTS, help="curriculum stages (numbers of pairs N), in order")
    p.add_argument("--stage_steps", type=int, default=2000)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--test_samples", type=int, default=None, help="default: the whole test split")
    p.add_argument("--stop_below", type=float, default=None,
                   help="lm: stop after the first stage whose exact copy is below this (default 0.5 for lm, never for cls)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data", default="data_keyed", help="the generated subsets; downloaded from the Hub when missing")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    stop_below = a.stop_below if a.stop_below is not None else (0.5 if a.objective == "lm" else -1.0)
    prov = provenance(files=(__file__, kb.__file__))
    torch.manual_seed(a.seed)
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(a.seed)

    model = KeyedModel(a.hidden, a.layers, a.k, a.v).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    if a.k < 16:
        print(f"state check: {check_state_size(model, a.k, dev)} masked q/k activations are zero past channel {a.k}", flush=True)
    opt = torch.optim.AdamW(param_groups(model, a.wd), lr=a.lr, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / a.warmup))
    print(f"keyed {a.objective} state {a.k}x{a.v}: {n_params:,} params, stages N = {a.counts}", flush=True)

    out = {"args": vars(a), "n_params": n_params, "stages": [], "stopped": False, "host": socket.gethostname(),
           "gpu": torch.cuda.get_device_name(), **prov}
    t0, step = time.time(), 0
    for n in a.counts:
        subset = kb.subset_name(n)
        root = Path(a.data)
        if not list((root / subset).glob("test-*.parquet")):    # not synced with the code: download it
            root = Path(snapshot_download(REPO, repo_type="dataset", local_dir=a.data, allow_patterns=[f"{subset}/*"]))
        tr = {c: x.to(dev) for c, x in load(root, subset, "train", n, candidates=False).items()}
        te = load(root, subset, "test", n, candidates=True)
        if a.test_samples:
            te = {c: x[:a.test_samples] for c, x in te.items()}
        eval_bs = max(1, min(7000, 2**20 // (9 * (n * (L + 1) + L + 3))))   # ~1M tokens per forward
        perm, pos, run_loss, run_acc, ts = torch.randperm(len(tr["query"]), device=dev, generator=gen), 0, 0.0, 0.0, time.time()
        for _ in range(a.stage_steps):
            if pos + a.bs > len(perm):
                perm, pos = torch.randperm(len(tr["query"]), device=dev, generator=gen), 0
            idx = perm[pos:pos + a.bs]
            pos += a.bs
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, acc = loss_fn(model, a.objective, tr["keys"][idx], tr["sequences"][idx], tr["query"][idx], gen)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            run_loss, run_acc = run_loss + loss.item(), run_acc + acc.item()
        ev = evaluate(model, a.objective, te, eval_bs, dev)
        out["stages"].append({"N": n, "step": step, "train_loss": run_loss / a.stage_steps, "train_acc": run_acc / a.stage_steps,
                              "stage_time": time.time() - ts, **ev})
        print(f"N={n:<5} loss {run_loss / a.stage_steps:.4f} " + " ".join(f"{k} {v:.4f}" for k, v in ev.items())
              + f"  ({time.time() - ts:.0f} s)", flush=True)
        out.update(train_time=time.time() - t0, peak_gb=torch.cuda.max_memory_allocated() / 2**30)
        stop = a.objective == "lm" and ev["recall_exact"] < stop_below
        out["stopped"] = stop
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(a.out + ".tmp")
        tmp.write_text(json.dumps(out, indent=1))      # after every stage, so a stopped or killed run keeps what it has
        os.replace(tmp, a.out)
        del tr
        if stop:
            print(f"stopping: exact copy {ev['recall_exact']:.3f} < {stop_below} at N={n}", flush=True)
            break


if __name__ == "__main__":
    main()
