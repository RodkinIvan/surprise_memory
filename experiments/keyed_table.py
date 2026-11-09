"""Experiment 9 capacity table: the largest N each state reconstructs and recognises, per seed.

Reconstruct: the largest N whose test exact copy is >= 0.5 (lm runs). Recognise: the largest N whose test acc9 is >= a
threshold (cls runs; chance is 1/9), and the lm runs' surprise-scored acc9 at the last N they trained on.

  python experiments/keyed_table.py [--dir results/keyed] > results/keyed/capacity_keyed.md
"""

import argparse
import json
from pathlib import Path

STATES = (4, 8, 16, 32)


def largest(stages, metric, thr):
    """Largest N with metric >= thr (0 if none)."""
    ok = [s["N"] for s in stages if s[metric] >= thr]
    return max(ok) if ok else 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="results/keyed")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = p.parse_args()
    run = lambda k, obj, s: json.loads((Path(a.dir) / f"keyed_k{k}_{obj}_s{s}.json").read_text())  # noqa: E731
    seeds = lambda xs: ", ".join(str(x) for x in xs)  # noqa: E731
    print("| state | floats (2 layers) | reconstruct: exact copy >= 0.5 | recognise (cls): acc9 >= 0.5 | recognise (cls): acc9 >= 0.9 "
          "| lm surprise acc9 at its last N |")
    print("|---|---|---|---|---|---|")
    for k in STATES:
        lm = [run(k, "lm", s) for s in a.seeds]
        cls = [run(k, "cls", s) for s in a.seeds]
        rec = [largest(r["stages"], "recall_exact", 0.5) for r in lm]
        g5 = [largest(r["stages"], "recog_acc9", 0.5) for r in cls]
        g9 = [largest(r["stages"], "recog_acc9", 0.9) for r in cls]
        sur = [f"{r['stages'][-1]['recog_acc9']:.2f} (N={r['stages'][-1]['N']})" for r in lm]
        print(f"| {k}x{k} | {2 * k * k} | {seeds(rec)} | {seeds(g5)} | {seeds(g9)} | {', '.join(sur)} |")
    print("\nPer seed (0, 1, 2). Every lm run stopped at the first N below 0.5 exact copy, so its surprise score is only known up to there.")
    print("\nRecognition (cls) by N, acc9 (pairwise on edits / on other keys), median seed by acc9:\n")
    counts = [s["N"] for s in run(STATES[0], "cls", a.seeds[0])["stages"]]
    print("| state | " + " | ".join(f"N={n}" for n in counts) + " |")
    print("|---|" + "---|" * len(counts))
    for k in STATES:
        cells = []
        for i in range(len(counts)):
            st = sorted((run(k, "cls", s)["stages"][i] for s in a.seeds), key=lambda t: t["recog_acc9"])[len(a.seeds) // 2]
            other = f"{st['pairwise_other']:.2f}" if "pairwise_other" in st else "-"
            cells.append(f"{st['recog_acc9']:.2f} ({st['pairwise_edit']:.2f} / {other})")
        print(f"| {k}x{k} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
