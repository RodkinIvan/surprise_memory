"""Longest sequence each state reconstructs and recognises, from the sweep runs (seeds 0-2 where they exist).

  python3 experiments/capacity_table.py   (reads results/sweep; writes results/sweep/capacity.json)

Reconstruction: `recall_exact`, the share of test sequences where the recall model's argmax is right at every position
given the true prefix. That is the same as greedy decoding copying the whole sequence: if every argmax is right, greedy
decoding never leaves the true prefix. Recognition: acc9 of the recall model's surprise and of the recognition model.
A length counts when the run meets the threshold there; lengths are powers of 2, so the true limit lies between the
reported N and 2N, and ">=256" means the longest length tested.
"""

import json
from pathlib import Path
from statistics import median

STATES = [(16, 4), (16, 8), (16, 16), (32, 32), (64, 64), (128, 128)]
LENGTHS = [4, 8, 16, 32, 64, 128, 256]
SEEDS = (0, 1, 2)
CRITERIA = {"reconstruct": ("lm", "recall_exact", (0.9, 0.5)),
            "surprise": ("lm", "recog_acc9", (0.99, 0.95)),
            "recognition_model": ("cls", "recog_acc9", (0.99, 0.95))}


def values(k, v, n, obj, key):
    out = []
    for s in SEEDS:
        p = Path(f"results/sweep/k{k}v{v}_N{n}_{obj}_s{s}.json")
        if p.exists():
            out.append(json.loads(p.read_text())["test"][key])
    return out


def longest(k, v, obj, key, thr, pick):
    """Largest N whose `pick` (max or median) over runs meets thr, and whether every shorter N meets it too."""
    ok = [n for n in LENGTHS if (x := values(k, v, n, obj, key)) and pick(x) >= thr]
    if not ok:
        return None, True
    return ok[-1], ok == LENGTHS[:len(ok)]


def fmt(n, mono):
    return "-" if n is None else (">=256" if n == 256 else str(n)) + ("" if mono else "*")


def main():
    out = {}
    for name, (obj, key, thrs) in CRITERIA.items():
        for thr in thrs:
            print(f"\n{name} ({key} >= {thr}): largest N, best run / median run   (* a shorter N misses it)")
            for k, v in STATES:
                runs = len(values(k, v, 64, obj, key))
                best, med = longest(k, v, obj, key, thr, max), longest(k, v, obj, key, thr, median)
                out[f"{name}/{thr}/{k}x{v}"] = {"best": best[0], "best_monotone": best[1], "median": med[0], "median_monotone": med[1], "runs": runs}
                print(f"  {k:>3}x{v:<3} {fmt(*best):>6} / {fmt(*med):<6} ({runs} runs at N=64)")
    Path("results/sweep/capacity.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
