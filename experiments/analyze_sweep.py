"""Collect the state-size sweep into one table: results/<dir>/*.json -> results/<dir>/summary.json.

  python3 experiments/analyze_sweep.py [--dir sweep]
Rows are keyed by the full config (key dim, value dim, N, objective, seed), so nothing collides.
"""

import argparse
import json
import math
from pathlib import Path

KEEP = ["recog_acc9", "recog_pairwise", "recall_token_acc", "recall_exact", "nll_true", "balanced_acc"]


def row(path):
    r = json.loads(path.read_text())
    a = r["args"]
    k = a["head_dim"] or a["hidden"] // a["heads"]
    v = round(k * a.get("expand_v", 1.0))
    curve = r["curve"]
    out = {"name": path.stem, "k": k, "v": v, "state": k * v, "N": int(a["subset"][1:].split("_")[0]),
           "objective": a["objective"], "seed": a["seed"], "steps": r["steps"], "n_params": r["n_params"],
           "train_time": r["train_time"], "host": r["host"],
           **{m: r["test"][m] for m in KEEP if m in r["test"]}}
    # a classifier stuck at ln 2 has learned nothing; flag it instead of reporting it as a result
    if a["objective"] == "cls":
        out["stuck"] = abs(curve[-1]["train_loss"] - math.log(2)) < 0.01
    # first eval step at which validation acc9 reaches 99% of its final value: how fast it learns
    final = curve[-1]["recog_acc9"]
    out["learn_step"] = next((c["step"] for c in curve if c["recog_acc9"] >= 0.99 * final), None)
    out["curve"] = [{"step": c["step"], "acc9": c["recog_acc9"], "loss": c["train_loss"]} for c in curve]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="sweep")
    a = p.parse_args()
    d = Path("results") / a.dir
    rows = sorted((row(f) for f in d.glob("*.json") if f.name != "summary.json"),
                  key=lambda r: (r["state"], r["N"], r["objective"], r["seed"]))
    (d / "summary.json").write_text(json.dumps(rows, indent=1))
    print(f"{'state':>9} {'N':>4} {'obj':>4} {'acc9':>7} {'pair':>7} {'exact':>7} {'tok':>7} {'note'}")
    for r in rows:
        print(f"{r['k']:>4}x{r['v']:<4} {r['N']:>4} {r['objective']:>4} {r['recog_acc9']:7.4f} {r['recog_pairwise']:7.4f}"
              f" {r.get('recall_exact', float('nan')):7.4f} {r.get('recall_token_acc', float('nan')):7.4f}"
              f" {'STUCK' if r.get('stuck') else ''}")
    print(f"{len(rows)} runs -> {d / 'summary.json'}")


if __name__ == "__main__":
    main()
