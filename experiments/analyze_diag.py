"""Collapse diagnosis: compare the logged training statistics of successful and failed recognition runs.

  python3 experiments/analyze_diag.py   (reads results/diag; writes results/diag/summary.json)
A run "fails" when its test acc9 is below 0.95.
"""

import json
from pathlib import Path

KEYS = ["loss", "grad_total", "grad_cls_head", "grad_L0_attn", "grad_L1_attn", "L0_decay_mean", "L1_decay_mean",
        "L0_decay_gt0.999", "L1_decay_gt0.999", "L0_write_mean", "L1_write_mean", "L0_wgate_mean", "L1_wgate_mean",
        "logit_std", "logit_absmax"]
STEPS = [1, 300, 1000, 2000, 4000, 8000, 12000, 17500, 25000, 35100]


def at(diag, step):
    """The diag record nearest to `step`."""
    return min(diag, key=lambda d: abs(d["step"] - step))


def main():
    runs = []
    for f in sorted(Path("results/diag").glob("diag_*.json")):
        r = json.loads(f.read_text())
        runs.append({"name": f.stem, "cfg": f.stem.rsplit("_cls", 1)[0].removeprefix("diag_"), "acc9": r["test"]["recog_acc9"],
                     "diag": r["diag"], "profiles": r["profiles"], "curve": r["curve"]})
    out = {}
    for cfg in sorted({r["cfg"] for r in runs}):
        rs = [r for r in runs if r["cfg"] == cfg]
        ok, bad = [r for r in rs if r["acc9"] >= 0.95], [r for r in rs if r["acc9"] < 0.95]
        print(f"\n== {cfg}: {len(ok)} ok, {len(bad)} failed; acc9 " + " ".join(f"{r['acc9']:.3f}" for r in sorted(rs, key=lambda r: r["acc9"])))
        out[cfg] = {"acc9": {r["name"]: r["acc9"] for r in rs}, "stats": {}}
        for key in KEYS:
            line = []
            for grp, members in (("ok", ok), ("fail", bad)):
                if members:
                    vals = [sum(at(r["diag"], s).get(key, float("nan")) for r in members) / len(members) for s in STEPS]
                    out[cfg]["stats"][f"{key}/{grp}"] = vals
                    line.append(f"{grp:>4}: " + " ".join(f"{v:8.3g}" for v in vals))
            print(f"  {key:<17} " + "\n  " + " " * 17 + " ".join(line) if len(line) == 2 else f"  {key:<17} {line[0] if line else ''}")
        for when in ("curriculum_end", "final"):
            for grp, members in (("ok", ok), ("fail", bad)):
                if members and all(when in r["profiles"] for r in members):
                    pos = [sum(r["profiles"][when]["by_pos"][i] for r in members) / len(members) for i in range(len(members[0]["profiles"][when]["by_pos"]))]
                    shift = [sum(r["profiles"][when]["by_shift"][i] for r in members) / len(members) for i in range(1, 16)]
                    out[cfg][f"profile/{when}/{grp}"] = {"by_pos": pos, "by_shift": shift}
                    print(f"  {when:<15}{grp:>4} pos  " + " ".join(f"{x:.2f}" for x in pos[:16]) + (" ..." if len(pos) > 16 else ""))
                    print(f"  {'':<15}{grp:>4} shift " + " ".join(f"{x:.2f}" for x in shift))
    print("\nstep columns:", STEPS)
    Path("results/diag/summary.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
