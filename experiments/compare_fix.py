"""Compare fixes for the recognition objective with the sweep's baseline, config by config and seed by seed.

  python3 experiments/compare_fix.py   (reads results/sweep and results/fix; writes results/fix/summary.json)
Baseline: --curriculum 0.5 (prefixes from 16), bce, lr 1e-3.  A: prefixes from 4.  B: A + lr 3e-4.  C: A + pairwise ranking loss.
"""

import json
from pathlib import Path

VARIANTS = {"base": "results/sweep/k{k}v{v}_N{n}_cls_s{s}.json", "A": "results/fix/fixA_k{k}v{v}_N{n}_cls_s{s}.json",
            "B": "results/fix/fixB_k{k}v{v}_N{n}_cls_s{s}.json", "C": "results/fix/fixC_k{k}v{v}_N{n}_cls_s{s}.json"}
CONFIGS = [(16, 4, n) for n in (16, 32, 64, 128, 256)] + [(16, 8, n) for n in (32, 64, 128, 256)] + [(16, 16, n) for n in (32, 64)]
SEEDS = (0, 1, 2)


def acc9(path):
    p = Path(path)
    return json.loads(p.read_text())["test"]["recog_acc9"] if p.exists() else None


def main():
    table = {f"{k}x{v}_N{n}": {name: [acc9(t.format(k=k, v=v, n=n, s=s)) for s in SEEDS] for name, t in VARIANTS.items()}
             for k, v, n in CONFIGS}
    Path("results/fix/summary.json").write_text(json.dumps(table, indent=1))
    print(f"{'config':<14}" + "".join(f"{name:>24}" for name in VARIANTS))
    for cfg, row in table.items():
        cells = []
        for name in VARIANTS:
            x = [a for a in row[name] if a is not None]
            cells.append(f"{sum(x) / len(x):.3f} [{min(x):.3f}-{max(x):.3f}]" if x else "-")
        print(f"{cfg:<14}" + "".join(f"{c:>24}" for c in cells))
    print("\nover all configs and seeds: mean acc9 | min | runs >= 0.99 | runs below the 4-bit checksum (0.85)")
    for name in VARIANTS:
        x = [a for row in table.values() for a in row[name] if a is not None]
        if x:
            print(f"{name:<5} {sum(x) / len(x):.4f} | {min(x):.3f} | {sum(a >= 0.99 for a in x)}/{len(x)} | {sum(a < 0.85 for a in x)}")


if __name__ == "__main__":
    main()
