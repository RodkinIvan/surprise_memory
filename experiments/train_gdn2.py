"""Train a small Gated DeltaNet-2 on the recognition-memory benchmark.

Input: [BOS] sequence [SEP] candidate. Two objectives on the same backbone:
  lm  (recall)      next-token prediction of the candidate when it is a copy of the sequence;
                    candidates are scored by surprise (summed NLL of the candidate tokens).
  cls (recognition) binary "candidate == sequence" logit read at the last token,
                    trained on the sequence itself or one of its corrupted copies (50/50).
Both are evaluated on choosing the true sequence among it and its 8 corrupted copies.
"""

import argparse
import json
import math
import os
import socket
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from fla.layers import GatedDeltaNet2
from fla.modules import GatedMLP, RMSNorm
from huggingface_hub import snapshot_download

VOCAB = 16
SEP, BOS = VOCAB, VOCAB + 1
REPO = "irodkin/recognition-memory"


class Block(nn.Module):
    def __init__(self, d, n_heads, head_dim, expand_v, layer_idx):
        super().__init__()
        self.attn_norm = RMSNorm(d)
        self.attn = GatedDeltaNet2(hidden_size=d, num_heads=n_heads, head_dim=head_dim, expand_v=expand_v,
                                   layer_idx=layer_idx)
        self.mlp_norm = RMSNorm(d)
        self.mlp = GatedMLP(hidden_size=d, hidden_ratio=4)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))[0]
        return x + self.mlp(self.mlp_norm(x))


class Model(nn.Module):
    def __init__(self, d=128, n_layers=2, n_heads=1, head_dim=128, expand_v=1.0):
        super().__init__()
        self.embed = nn.Embedding(VOCAB + 2, d)
        self.blocks = nn.ModuleList(Block(d, n_heads, head_dim, expand_v, i) for i in range(n_layers))
        self.norm = RMSNorm(d)
        self.lm_head = nn.Linear(d, VOCAB, bias=False)
        self.cls_head = nn.Linear(d, 1)

    def forward(self, tokens):
        x = self.embed(tokens)
        for b in self.blocks:
            x = b(x)
        return self.norm(x)


# ---------------------------------------------------------------- data

def load_split(root, subset, split):
    seqs, cors = [], []
    for f in sorted((root / subset).glob(f"{split}-*.parquet")):
        t = pq.read_table(f, columns=["sequence", "corrupted"])
        n = t.num_rows
        seqs.append(t["sequence"].combine_chunks().flatten().to_numpy().reshape(n, -1))
        cors.append(t["corrupted"].combine_chunks().flatten().flatten().to_numpy().reshape(n, 8, -1))
    return torch.from_numpy(np.concatenate(seqs)), torch.from_numpy(np.concatenate(cors))


def make_input(seq, cand):
    b = seq.shape[0]
    bos = torch.full((b, 1), BOS, dtype=torch.long, device=seq.device)
    sep = torch.full((b, 1), SEP, dtype=torch.long, device=seq.device)
    return torch.cat([bos, seq.long(), sep, cand.long()], 1)


def candidate_nll(model, seq, cand):
    """Per-token NLL of the candidate given the sequence: [B, N]."""
    n = seq.shape[1]
    h = model(make_input(seq, cand))
    with torch.autocast("cuda", enabled=False):  # fp32 readout, so scores of near-identical candidates don't tie
        logits = model.lm_head(h[:, n + 1:2 * n + 1].float())  # positions SEP..2N predict cand[0..N-1]
    return F.cross_entropy(logits.transpose(1, 2), cand.long(), reduction="none"), logits


def cls_logit(model, seq, cand):
    h = model(make_input(seq, cand))[:, -1]
    with torch.autocast("cuda", enabled=False):
        return model.cls_head(h.float()).squeeze(-1)


# ---------------------------------------------------------------- train / eval

def corrupt(seq, rate, gen):
    """Change each token with probability rate[b] (and one random token always) by a shift of 1..15."""
    b, n = seq.shape
    mask = torch.rand(b, n, device=seq.device, generator=gen) < rate[:, None]
    mask[torch.arange(b, device=seq.device), torch.randint(0, n, (b,), device=seq.device, generator=gen)] = True
    shift = torch.randint(1, VOCAB, (b, n), device=seq.device, generator=gen)
    return (seq.long() + shift * mask) % VOCAB


def loss_fn(model, objective, seq, cor, gen, p_dense=0.0, crop=None, cls_loss="bce"):
    if objective == "lm":
        nll, logits = candidate_nll(model, seq, seq)
        return nll.mean(), (logits.argmax(-1) == seq.long()).float().mean()
    if cls_loss == "pair":  # half the batch as (true, negative) pairs: as many sequences per step as bce
        seq, cor = seq[:seq.shape[0] // 2], cor[:cor.shape[0] // 2]
    b = seq.shape[0]
    label = torch.rand(b, device=seq.device, generator=gen) < 0.5
    pick = torch.randint(0, 8, (b,), device=seq.device, generator=gen)
    if crop is None:
        neg = cor[torch.arange(b, device=seq.device), pick].long()
    else:  # curriculum: a prefix of the sequence; easy-negative warm-up: a share p_dense of the negatives is heavily corrupted
        seq = seq[:, :crop]
        dense = torch.rand(b, device=seq.device, generator=gen) < p_dense
        rate = torch.where(dense, 0.5 * torch.rand(b, device=seq.device, generator=gen), 1 / max(crop - 1, 1))
        neg = corrupt(seq, rate, gen)
    if cls_loss == "pair":  # rank the true sequence above its negative: softplus(s_neg - s_true)
        logit = cls_logit(model, torch.cat([seq, seq]), torch.cat([seq.long(), neg])).view(2, b)
        margin = logit[0] - logit[1]
        return F.softplus(-margin).mean(), (margin > 0).float().mean()
    cand = torch.where(label[:, None], seq.long(), neg)
    logit = cls_logit(model, seq, cand)
    loss = F.binary_cross_entropy_with_logits(logit, label.float())
    return loss, ((logit > 0) == label).float().mean()


@torch.no_grad()
def evaluate(model, objective, seq_all, cor_all, bs, dev):
    model.eval()
    tot = {}
    n_total = 0
    for i in range(0, len(seq_all), bs):
        seq = seq_all[i:i + bs].to(dev)
        cor = cor_all[i:i + bs].to(dev)
        b, n = seq.shape
        cands = torch.cat([seq[:, None], cor], 1)  # [B, 9, N], index 0 is the true sequence
        flat_seq = seq[:, None].expand(-1, 9, -1).reshape(-1, n)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if objective == "lm":
                nll, logits = candidate_nll(model, flat_seq, cands.reshape(-1, n))
                score = -nll.sum(-1).view(b, 9)
                pred = logits.view(b, 9, n, VOCAB)[:, 0].argmax(-1)
                tok_acc = (pred == seq.long()).float()
                m = {"recall_token_acc": tok_acc.mean(1), "recall_exact": (tok_acc.min(1).values == 1).float(),
                     "nll_true": nll.view(b, 9, n)[:, 0].mean(1)}
            else:
                score = cls_logit(model, flat_seq, cands.reshape(-1, n)).view(b, 9)
                m = {"tpr": (score[:, 0] > 0).float(), "tnr": (score[:, 1:] <= 0).float().mean(1)}
        score = score.float()
        top = score == score.max(1, keepdim=True).values
        m["recog_acc9"] = top[:, 0].float() / top.sum(1)  # ties share the win, so a constant score gives 1/9
        m["recog_pairwise"] = (score[:, :1] > score[:, 1:]).float().mean(1) + \
            0.5 * (score[:, :1] == score[:, 1:]).float().mean(1)
        for k, v in m.items():
            tot[k] = tot.get(k, 0.0) + v.sum().item()
        n_total += b
    model.train()
    out = {k: v / n_total for k, v in tot.items()}
    if objective == "cls":
        out["balanced_acc"] = (out.pop("tpr") + out.pop("tnr")) / 2
    return out


class Probe:
    """Statistics of the GDN-2 gates and of the recognition logit, recorded on the forward passes where `on` is set."""

    def __init__(self, model):
        self.on, self.stats = False, {}
        for i, blk in enumerate(model.blocks):
            at = blk.attn
            at.f_proj.register_forward_hook(self._decay(f"L{i}", at))
            at.b_proj.register_forward_hook(self._sigmoid(f"L{i}_write"))
            at.w_proj.register_forward_hook(self._sigmoid(f"L{i}_wgate"))
        model.cls_head.register_forward_hook(self._logit)

    def _decay(self, name, at):
        def hook(mod, inp, out):  # per-step memory decay exp(g), as in GatedDeltaNet2.forward
            if self.on:
                with torch.no_grad():
                    g = F.softplus(out.float() + at.dt_bias).unflatten(-1, (at.num_heads, at.head_k_dim))
                    d = torch.exp(-at.A_log.float().exp()[:, None] * g)
                    self.stats.update({f"{name}_decay_mean": d.mean().item(), f"{name}_decay_gt0.999": (d > 0.999).float().mean().item(),
                                       f"{name}_decay_lt0.5": (d < 0.5).float().mean().item()})
        return hook

    def _sigmoid(self, name):
        def hook(mod, inp, out):
            if self.on:
                with torch.no_grad():
                    x = out.float().sigmoid()
                    self.stats.update({f"{name}_mean": x.mean().item(), f"{name}_lt0.01": (x < 0.01).float().mean().item(),
                                       f"{name}_gt0.99": (x > 0.99).float().mean().item()})
        return hook

    def _logit(self, mod, inp, out):
        if self.on:
            with torch.no_grad():
                x = out.float()
                self.stats.update({"logit_mean": x.mean().item(), "logit_std": x.std().item(), "logit_absmax": x.abs().max().item(),
                                   "logit_abs_gt10": (x.abs() > 10).float().mean().item()})


def grad_norms(model):
    """Gradient norm per module (before clipping)."""
    parts = {"embed": model.embed, "cls_head": model.cls_head, "lm_head": model.lm_head}
    for i, blk in enumerate(model.blocks):
        parts.update({f"L{i}_attn": blk.attn, f"L{i}_mlp": blk.mlp})
    return {f"grad_{k}": torch.sqrt(sum((p.grad.float() ** 2).sum() for p in m.parameters() if p.grad is not None)
                                    + torch.zeros((), device=next(model.parameters()).device)).item() for k, m in parts.items()}


@torch.no_grad()
def detection_profile(model, objective, seq_all, cor_all, bs, dev):
    """Over corrupted copies with exactly one changed token: how often the true sequence scores higher (ties count 1/2),
    by the changed position, by the shift (new - old mod 16) and by the original token."""
    model.eval()
    n = seq_all.shape[1]
    win = {"pos": torch.zeros(n, device=dev), "shift": torch.zeros(VOCAB, device=dev), "token": torch.zeros(VOCAB, device=dev)}
    cnt = {k: torch.zeros_like(v) for k, v in win.items()}
    for i in range(0, len(seq_all), bs):
        seq, cor = seq_all[i:i + bs].to(dev).long(), cor_all[i:i + bs].to(dev).long()
        b = seq.shape[0]
        cands = torch.cat([seq[:, None], cor], 1).reshape(-1, n)
        flat_seq = seq[:, None].expand(-1, 9, -1).reshape(-1, n)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            score = (-candidate_nll(model, flat_seq, cands)[0].sum(-1) if objective == "lm" else cls_logit(model, flat_seq, cands))
        score = score.float().view(b, 9)
        diff = cor != seq[:, None]                                  # [B, 8, N]
        single = diff.sum(-1) == 1
        w = (score[:, :1] > score[:, 1:]).float() + 0.5 * (score[:, :1] == score[:, 1:]).float()
        pos = diff.float().argmax(-1)                               # the changed position, for single changes
        old = seq.gather(1, pos)
        new = cor.gather(2, pos[..., None]).squeeze(-1)
        for key, idx in (("pos", pos), ("shift", (new - old) % VOCAB), ("token", old)):
            win[key].index_add_(0, idx[single], w[single])
            cnt[key].index_add_(0, idx[single], torch.ones_like(w[single]))
    model.train()
    out = {f"by_{k}": (win[k] / cnt[k].clamp(min=1)).tolist() for k in win}
    out["n_single"] = int(cnt["pos"].sum().item())
    return out


def param_groups(model, wd):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if p.ndim < 2 or getattr(p, "_no_weight_decay", False) or "embed" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subset", required=True)
    p.add_argument("--objective", choices=["lm", "cls"], required=True)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=1)
    p.add_argument("--head_dim", type=int, default=None, help="default: hidden // heads")
    p.add_argument("--expand_v", type=float, default=1.0, help="value dim = head_dim * expand_v")
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--eval_samples", type=int, default=10_000)
    p.add_argument("--test_samples", type=int, default=None, help="default: whole test split")
    p.add_argument("--cls_loss", choices=["bce", "pair"], default="bce",
                   help="cls: bce on one candidate (50%% true) or pairwise ranking of the true sequence above a negative")
    p.add_argument("--cur_min", type=int, default=16, help="cls: shortest prefix of the length curriculum (a power of 2)")
    p.add_argument("--cur_weight", choices=["equal", "length"], default="equal",
                   help="cls: time per curriculum stage: equal, or proportional to the prefix length")
    p.add_argument("--curriculum", type=float, default=0.0,
                   help="cls: fraction of training spent on a curriculum: prefixes of length cur_min, 2*cur_min, ... N (equal time each)"
                        " with negatives corrupted on the fly, the heavily corrupted share decaying 1 -> 0")
    p.add_argument("--diag_every", type=int, default=0, help="log gradient, gate and logit statistics every this many steps")
    p.add_argument("--save", default=None, help="save the final weights here")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data", default="hf_data")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    if a.cur_min < 1 or a.cur_min & (a.cur_min - 1):
        p.error(f"--cur_min must be a power of 2, got {a.cur_min}")

    torch.manual_seed(a.seed)
    dev = torch.device("cuda")
    root = Path(snapshot_download(REPO, repo_type="dataset", local_dir=a.data, allow_patterns=[f"{a.subset}/*"]))
    tr_seq, tr_cor = (x.to(dev) for x in load_split(root, a.subset, "train"))
    va_seq, va_cor = load_split(root, a.subset, "validation")
    te_seq, te_cor = load_split(root, a.subset, "test")
    va_seq, va_cor = va_seq[:a.eval_samples], va_cor[:a.eval_samples]
    if a.test_samples:
        te_seq, te_cor = te_seq[:a.test_samples], te_cor[:a.test_samples]

    model = Model(a.hidden, a.layers, a.heads, a.head_dim or a.hidden // a.heads, a.expand_v).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(param_groups(model, a.wd), lr=a.lr, betas=(0.9, 0.95))
    steps = int(a.epochs * len(tr_seq) / a.bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / a.warmup) *
                                              (0.1 + 0.45 * (1 + math.cos(math.pi * min(1, s / steps)))))
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    # ~1M tokens per eval forward; 9 * eval_bs sequences must stay under CUDA's 65535 grid limit (fla short conv)
    eval_bs = min(4096, max(16, 2**20 // (9 * (2 * tr_seq.shape[1] + 2))))
    n = tr_seq.shape[1]
    lens = [m for m in (2 ** i for i in range(16)) if a.cur_min <= m < n] + [n]  # curriculum lengths cur_min, ..., N
    ends = [sum(lens[:i + 1]) / sum(lens) for i in range(len(lens))]  # --cur_weight length: where each stage ends
    print(f"{a.subset} {a.objective}: {n_params:,} params, {steps} steps", flush=True)

    probe = Probe(model) if a.diag_every else None
    diag, profiles = [], {}
    curve, t0, run_loss, run_acc, k = [], time.time(), 0.0, 0.0, 0
    perm, pos = torch.randperm(len(tr_seq), device=dev, generator=gen), 0
    for step in range(1, steps + 1):
        if pos + a.bs > len(perm):
            perm, pos = torch.randperm(len(tr_seq), device=dev, generator=gen), 0
        idx = perm[pos:pos + a.bs]
        pos += a.bs
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p_dense, crop = 0.0, None
            if a.objective == "cls" and step <= a.curriculum * steps:
                t = step / (a.curriculum * steps)
                stage = (min(int(t * len(lens)), len(lens) - 1) if a.cur_weight == "equal"
                         else next(i for i, e in enumerate(ends) if t < e or i == len(ends) - 1))
                p_dense, crop = 1 - t, lens[stage]
            is_diag = bool(a.diag_every) and (step % a.diag_every == 0 or step == 1)
            if probe:
                probe.on = is_diag
            loss, acc = loss_fn(model, a.objective, tr_seq[idx], tr_cor[idx], gen, p_dense, crop, a.cls_loss)
            if probe:
                probe.on = False
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if is_diag:
            parts = grad_norms(model)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if is_diag:
            diag.append({"step": step, "loss": loss.item(), "acc": acc.item(), "grad_total": gnorm.item(), "p_dense": p_dense,
                         "crop": crop or tr_seq.shape[1], **parts, **probe.stats})
        if a.diag_every and a.objective == "cls" and a.curriculum and step == int(a.curriculum * steps):
            profiles["curriculum_end"] = {"step": step, **detection_profile(model, a.objective, va_seq, va_cor, eval_bs, dev)}
        opt.step()
        sched.step()
        run_loss, run_acc, k = run_loss + loss.item(), run_acc + acc.item(), k + 1
        if step % a.eval_every == 0 or step == steps:
            ev = evaluate(model, a.objective, va_seq, va_cor, eval_bs, dev)
            rec = {"step": step, "train_loss": run_loss / k, "train_acc": run_acc / k,
                   "grad_norm": gnorm.item(), "lr": sched.get_last_lr()[0], "time": time.time() - t0, **ev}
            curve.append(rec)
            run_loss = run_acc = 0.0; k = 0
            print(json.dumps(rec), flush=True)

    test = evaluate(model, a.objective, te_seq, te_cor, eval_bs, dev)
    print("TEST", json.dumps(test), flush=True)
    if a.diag_every:
        profiles["final"] = {"step": steps, **detection_profile(model, a.objective, va_seq, va_cor, eval_bs, dev)}
    if a.save:
        Path(a.save).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), a.save)
    version = json.loads(Path("CODE_VERSION.json").read_text()) if Path("CODE_VERSION.json").exists() else None
    out = {"args": vars(a), "n_params": n_params, "steps": steps, "test": test, "curve": curve,
           "train_time": time.time() - t0, "host": socket.gethostname(),
           "gpu": torch.cuda.get_device_name(), "code_version": version}
    if a.diag_every:
        out.update(diag=diag, profiles=profiles)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(a.out + ".tmp")
    tmp.write_text(json.dumps(out, indent=1))
    os.replace(tmp, a.out)


if __name__ == "__main__":
    main()
