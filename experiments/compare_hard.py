"""Experiment 7: keep a share of benchmark-like (1-2 token) negatives from the first step of the easy-negative warm-up.

  python3 experiments/compare_hard.py   (reads results/hard; writes results/hard/summary.json)
Every variant (hard_min 0, 0.25, 0.5) has the same 12 fresh seeds per config. A run "fails" when its test acc9 is below 0.95.
"""

import json
from math import comb
from pathlib import Path

CONFIGS = [(16, 4, 16), (16, 4, 32)]
VARIANTS = (0, 25, 50)
SEEDS = range(200, 212)


def fisher_less(a, n1, b, n2):
    """One-sided Fisher exact test: P(failures in group 1 <= a) given a + b failures in n1 + n2 runs."""
    k, n = a + b, n1 + n2
    return sum(comb(n1, i) * comb(n2, k - i) for i in range(max(0, k - n2), a + 1)) / comb(n, k)


def load(h, k, v, n):
    rs = {}
    for s in SEEDS:
        p = Path(f"results/hard/hard{h}_k{k}v{v}_N{n}_cls_s{s}.json")
        if p.exists():
            r = json.loads(p.read_text())
            prof = r.get("profiles", {})
            single = lambda w: sum(prof[w]["by_pos"]) / len(prof[w]["by_pos"]) if w in prof else None  # noqa: E731
            rs[s] = {"acc9": r["test"]["recog_acc9"], "single_cur_end": single("curriculum_end"), "single_final": single("final")}
    return rs


def main():
    out = {}
    print("test acc9 per run (seeds 200-211), failures = acc9 < 0.95; p: one-sided Fisher test, fewer failures than hard_min 0")
    totals = {h: [0, 0] for h in VARIANTS}
    for k, v, n in CONFIGS:
        cfg = f"{k}x{v}_N{n}"
        out[cfg] = {}
        base = None
        for h in VARIANTS:
            rs = load(h, k, v, n)
            acc = [r["acc9"] for r in rs.values()]
            fails = sum(a < 0.95 for a in acc)
            totals[h][0] += fails
            totals[h][1] += len(acc)
            out[cfg][f"hard{h}"] = {"runs": rs, "n": len(acc), "fails": fails,
                                    "mean": sum(acc) / len(acc) if acc else None}
            if h == 0:
                base = (fails, len(acc))
            p = fisher_less(fails, len(acc), *base) if h and acc and base[1] else None
            print(f"  {cfg:<9} hard_min {h / 100:<4} {len(acc):>2} runs  fail {fails:>2}  mean {out[cfg][f'hard{h}']['mean'] or 0:.3f}"
                  + (f"  p {p:.3f}" if p is not None else "")
                  + "  | " + " ".join(f"{a:.3f}" for a in sorted(acc)))
    print("\n  both configs:")
    out["total"] = {}
    for h in VARIANTS:
        f, m = totals[h]
        p = fisher_less(f, m, *totals[0]) if h and m else None
        out["total"][f"hard{h}"] = {"fails": f, "n": m, "p_vs_0": p}
        print(f"  hard_min {h / 100:<4} fail {f}/{m}" + (f"  p {p:.3f}" if p is not None else ""))
    Path("results/hard/summary.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
