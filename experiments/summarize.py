"""Print a table of test metrics from results/<dir>/*.json, one row per sequence length."""

import argparse
import json
import re
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dir", nargs="?", default="results/gdn2")
    a = p.parse_args()

    rows = {}
    for f in Path(a.dir).glob("*.json"):
        d = json.loads(f.read_text())
        n = int(re.match(r"N(\d+)_", d["args"]["subset"]).group(1))
        rows.setdefault(n, {})[d["args"]["objective"]] = d

    print(f"{'N':>4} | {'recall (lm): tok acc':>20} {'exact':>7} {'acc9':>7} {'pairwise':>8} | "
          f"{'recognition (cls): acc9':>23} {'pairwise':>8} {'bal acc':>7}")
    for n in sorted(rows):
        lm, cls = rows[n].get("lm"), rows[n].get("cls")
        left = (f"{lm['test']['recall_token_acc']:20.4f} {lm['test']['recall_exact']:7.4f} "
                f"{lm['test']['recog_acc9']:7.4f} {lm['test']['recog_pairwise']:8.4f}") if lm else f"{'-':>45}"
        right = (f"{cls['test']['recog_acc9']:23.4f} {cls['test']['recog_pairwise']:8.4f} "
                 f"{cls['test']['balanced_acc']:7.4f}") if cls else f"{'-':>40}"
        print(f"{n:4d} | {left} | {right}")


if __name__ == "__main__":
    main()
