"""Experiment 5: fresh seeds for the baseline and the ranking loss (C), longer training at N=256, reproduction check.

  python3 experiments/compare_exp5.py   (reads results/{exp5,sweep}; writes results/exp5/summary.json)
"""

import json
from pathlib import Path

from stats import fisher_greater, fisher_less

SELECTED = [(16, 4, n) for n in (16, 32, 64, 128, 256)] + [(16, 8, n) for n in (32, 64, 128, 256)] + [(16, 16, n) for n in (32, 64)]
UNSELECTED = [(16, 8, 16), (16, 16, 128), (16, 16, 256)]
SEEDS = (3, 4, 5)


def acc9(path):
    p = Path(path)
    return json.loads(p.read_text())["test"]["recog_acc9"] if p.exists() else None


def stats(x):
    x = [a for a in x if a is not None]
    return {"n": len(x), "mean": sum(x) / len(x), "min": min(x), "max": max(x), "below_0.95": sum(a < 0.95 for a in x),
            "below_0.85": sum(a < 0.85 for a in x), "at_0.99": sum(a >= 0.99 for a in x)} if x else None


def fmt(s):
    return f"{s['mean']:.3f} [{s['min']:.3f}-{s['max']:.3f}]" if s else "-"


def main():
    out = {"repro": {}, "fresh": {}, "long": {}}
    print("reproduction (current code, default flags, seed 0) vs the sweep (archived code 56a994):")
    for k, v, n in [(16, 4, 32), (16, 8, 128)]:
        new, old = acc9(f"results/exp5/repro_k{k}v{v}_N{n}_cls_s0.json"), acc9(f"results/sweep/k{k}v{v}_N{n}_cls_s0.json")
        out["repro"][f"{k}x{v}_N{n}"] = {"rerun": new, "sweep": old}
        print(f"  {k}x{v} N={n}: rerun {new:.4f}  sweep {old:.4f}")

    print(f"\nfresh seeds {SEEDS}: baseline vs C (ranking loss, curriculum from 4), test acc9 mean [min-max]")
    for k, v, n in SELECTED + UNSELECTED:
        cfg = f"{k}x{v}_N{n}"
        row = {name: [acc9(f"results/exp5/e5_{name}_k{k}v{v}_N{n}_cls_s{s}.json") for s in SEEDS] for name in ("base", "C")}
        out["fresh"][cfg] = row
        tag = "" if (k, v, n) in SELECTED else "  (not selected)"
        print(f"  {cfg:<11} base {fmt(stats(row['base'])):<22} C {fmt(stats(row['C'])):<22} "
              f"base {' '.join(f'{a:.3f}' for a in row['base'])} | C {' '.join(f'{a:.3f}' for a in row['C'])}{tag}")
    groups = {"all 14": SELECTED + UNSELECTED, "selected 11": SELECTED, "unselected 3": UNSELECTED,
              "N<=128": [c for c in SELECTED + UNSELECTED if c[2] <= 128], "N=256": [c for c in SELECTED + UNSELECTED if c[2] == 256]}
    out["groups"] = {}
    print("\n  group            version  mean   min    >=0.99  <0.95  <0.85")
    for g, cfgs in groups.items():
        for name in ("base", "C"):
            s = stats([a for k, v, n in cfgs for a in out["fresh"][f"{k}x{v}_N{n}"][name]])
            out["groups"][f"{g}/{name}"] = s
            print(f"  {g:<16} {name:<7}  {s['mean']:.3f}  {s['min']:.3f}  {s['at_0.99']:>3}/{s['n']:<3} {s['below_0.95']:>4}  {s['below_0.85']:>4}")

    out["tests"] = {}
    print("\none-sided Fisher tests on runs below 0.95, C against the baseline (fewer failures):")
    for g in ("N<=128", "all 14"):
        c, base = out["groups"][f"{g}/C"], out["groups"][f"{g}/base"]
        pv = fisher_less(c["below_0.95"], c["n"], base["below_0.95"], base["n"])
        out["tests"][f"C_fewer_failures/{g}"] = pv
        print(f"  {g:<8} C {c['below_0.95']}/{c['n']} vs base {base['below_0.95']}/{base['n']}: p = {pv:.3f}")

    print("\nN=256, seeds 3-5: 10 epochs (equal stages) vs 20 epochs (stage time proportional to prefix length)")
    for k, v in [(16, 4), (16, 8)]:
        row = {name: [acc9(f"results/exp5/e5_{name}_k{k}v{v}_N256_cls_s{s}.json") for s in SEEDS] for name in ("base", "baseL", "C", "CL")}
        out["long"][f"{k}x{v}_N256"] = row
        print(f"  {k}x{v}: " + "  ".join(f"{name} {fmt(stats(x))} ({' '.join(f'{a:.3f}' for a in x)})" for name, x in row.items()))
    short = [a for row in out["long"].values() for name in ("base", "C") for a in row[name]]
    longer = [a for row in out["long"].values() for name in ("baseL", "CL") for a in row[name]]
    f10, f20 = sum(a < 0.95 for a in short), sum(a < 0.95 for a in longer)
    pv = fisher_greater(f20, len(longer), f10, len(short))
    out["tests"]["20_epochs_more_failures"] = pv
    print(f"  pooled runs below 0.95: 20 epochs {f20}/{len(longer)}, 10 epochs {f10}/{len(short)}; one-sided p (more failures) = {pv:.3f}")
    Path("results/exp5/summary.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
