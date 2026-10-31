"""Figures for reports/recall_vs_recognition.md from results/{gdn2,sweep_smoke,sweep}.

  conda run -n ai python reports/make_figures.py   ->  reports/figures/*.png
"""

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "figures"
LENGTHS = [4, 8, 16, 32, 64, 128, 256]

SURFACE, GRID = "#fcfcfb", "#e6e5e0"
TEXT, TEXT2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]          # categorical slots 1-3, 7
STATES = [(16, 4), (16, 8), (16, 16), (32, 32), (64, 64), (128, 128)]
ORDINAL = ["#86b6ef", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#184f95", "#0d366b"]  # blue 250..700, one per N

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT2, "axes.titlecolor": TEXT, "axes.titlesize": 11,
    "axes.titleweight": "semibold", "axes.titlelocation": "left", "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False, "xtick.color": TEXT2,
    "ytick.color": TEXT2, "font.size": 9.5, "legend.frameon": False, "legend.labelcolor": TEXT2,
    "lines.linewidth": 2, "lines.solid_capstyle": "round", "lines.markersize": 5,
})


def load(path):
    return json.loads((ROOT / path).read_text())


def grid(n, obj):
    return load(f"results/gdn2/gdn2_h128_L2_N{n}_{obj}_s0.json")


def curve(r, key):
    return [c["step"] for c in r["curve"]], [c[key] for c in r["curve"]]


def chance(ax, y=1 / 9, label="chance, 1 in 9"):
    ax.axhline(y, color=MUTED, lw=1, zorder=1)
    ax.annotate(label, (1, y), xycoords=("axes fraction", "data"), xytext=(-4, 4), textcoords="offset points",
                ha="right", va="bottom", color=MUTED, fontsize=8.5)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


def fig_first_grid():
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
    lm = [grid(n, "lm")["test"] for n in LENGTHS]
    cls = [grid(n, "cls")["test"] for n in LENGTHS]
    for ax in (a, b):
        ax.set_xscale("log", base=2)
        ax.set_xticks(LENGTHS, [str(n) for n in LENGTHS])
        ax.set_xlabel("sequence length N")
    a.plot(LENGTHS, [t["recog_acc9"] for t in lm], "o-", color=SERIES[0], label="recall model, scored by surprise")
    a.plot(LENGTHS, [t["recog_acc9"] for t in cls], "o-", color=SERIES[1], label="recognition model, last-token score")
    chance(a)
    a.set_ylim(-0.04, 1.06)
    a.set_ylabel("test acc9 (true sequence ranked first)")
    a.set_title("a  Picking the true sequence out of 9")
    a.legend(loc="center left", fontsize=8.5)
    b.plot(LENGTHS, [t["recall_exact"] for t in lm], "o-", color=SERIES[0], label="whole sequence copied exactly")
    b.plot(LENGTHS, [t["recall_token_acc"] for t in lm], "o-", color=SERIES[2], label="tokens copied correctly")
    b.set_ylim(0.984, 1.002)
    b.set_ylabel("test accuracy (teacher-forced)")
    b.set_title("b  Recall model: copying the sequence")
    b.legend(loc="lower left", fontsize=8.5)
    save(fig, "fig1_first_grid.png")


def fig_plateau():
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
    for n, col in zip(LENGTHS, ORDINAL):
        s, y = curve(grid(n, "cls"), "train_loss")
        a.plot(s, y, color=col, lw=1.6, label=f"N={n}")
        s, y = curve(grid(n, "lm"), "train_loss")
        b.plot(s, y, color=col, lw=1.6, label=f"N={n}")
    a.axhline(math.log(2), color=MUTED, lw=1, zorder=1)
    a.annotate("ln 2: constant output", (1, math.log(2)), xycoords=("axes fraction", "data"), xytext=(-4, 4),
               textcoords="offset points", ha="right", va="bottom", color=MUTED, fontsize=8.5)
    a.set_ylim(-0.02, 0.78)
    a.set_title("a  Recognition classifier, no curriculum")
    a.set_ylabel("train loss (BCE, mean over 1000 steps)")
    b.set_yscale("log")
    b.set_title("b  Recall model")
    b.set_ylabel("train loss (token NLL, mean over 1000 steps)")
    for ax in (a, b):
        ax.set_xlabel("training step")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
    a.legend(ncol=2, fontsize=8, loc="center right")
    save(fig, "fig2_plateau.png")


def fig_curriculum():
    runs = {
        64: [("none (first grid, 10 epochs)", grid(64, "cls")),
             ("hard-negative curriculum (3 epochs)", load("results/sweep_smoke/smoke_k128v128_N64_cls.json"))],
        128: [("none (first grid, 10 epochs)", grid(128, "cls")),
              ("length curriculum (3 epochs)", load("results/sweep_smoke/smoke2_k128v128_N128_cls.json"))],
        256: [("none (first grid, 10 epochs)", grid(256, "cls")),
              ("hard-negative curriculum (3 epochs)", load("results/sweep_smoke/smoke_k128v128_N256_cls.json")),
              ("length curriculum (3 epochs)", load("results/sweep_smoke/smoke2_k128v128_N256_cls.json"))],
    }
    color = {"none": SERIES[3], "hard": SERIES[1], "length": SERIES[2]}
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True, layout="constrained")
    for ax, (n, rs) in zip(axes, runs.items()):
        for label, r in rs:
            s, y = curve(r, "recog_acc9")
            ax.plot(s, y, "o-", ms=3.5, color=color[label.split()[0].split("-")[0]], label=label)
        chance(ax)
        ax.set_xlim(0, 16000)
        ax.set_xlabel("training step")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
        ax.set_title(f"N={n}, state 128x128")
    axes[0].set_ylabel("validation acc9")
    axes[0].set_ylim(-0.04, 1.06)
    handles = {h.get_label(): h for ax in axes for h in ax.get_lines() if not h.get_label().startswith("_")}
    fig.legend(handles.values(), handles.keys(), loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.12), fontsize=8.5)
    save(fig, "fig3_curriculum.png")


def fig_small_state():
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
    runs = [("recall, 16x4 (1 epoch)", load("results/sweep_smoke/smoke_k16v4_N256_lm.json"), SERIES[0]),
            ("recognition, 16x4 (length curriculum, 3 epochs)", load("results/sweep_smoke/smoke2_k16v4_N256_cls.json"), SERIES[1]),
            ("recognition, 128x128 (length curriculum, 3 epochs)", load("results/sweep_smoke/smoke2_k128v128_N256_cls.json"), SERIES[2])]
    for label, r, col in runs:
        s, y = curve(r, "recog_acc9")
        a.plot(s, y, "o-", ms=3.5, color=col, label=label)
    chance(a)
    a.set_ylim(-0.04, 1.06)
    a.set_xlabel("training step")
    a.set_ylabel("validation acc9")
    a.set_title("a  N=256: acc9 by state size")
    a.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
    r = load("results/sweep_smoke/smoke2_k16v4_N256_cls.json")
    s, y = curve(r, "train_loss")
    steps, cur = r["steps"], r["args"]["curriculum"]
    lens = [16, 32, 64, 128, 256]
    stage = cur * steps / len(lens)
    for i, n in enumerate(lens):
        if i % 2 == 0:
            b.axvspan(i * stage, (i + 1) * stage, color=GRID, alpha=0.5, lw=0)
        b.text((i + 0.5) * stage, 0.05, f"prefix\n{n}", ha="center", va="bottom", color=TEXT2, fontsize=8)
    b.plot(s, y, "o-", ms=3.5, color=SERIES[1])
    b.axhline(math.log(2), color=MUTED, lw=1, zorder=1)
    b.annotate("ln 2", (1, math.log(2)), xycoords=("axes fraction", "data"), xytext=(-4, 4),
               textcoords="offset points", ha="right", va="bottom", color=MUTED, fontsize=8.5)
    b.set_ylim(0, 0.78)
    b.set_xlabel("training step")
    b.set_ylabel("train loss (BCE)")
    b.set_title("b  Recognition 16x4: loss by curriculum stage")
    for ax in (a, b):
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
    save(fig, "fig4_small_state.png")


def sweep():
    rows = json.loads((ROOT / "results/sweep/summary.json").read_text())
    return {(r["k"], r["v"], r["N"], r["objective"]): r for r in rows if r["seed"] == 0}


def seeds():
    """(k, v, N, objective) -> list of test-metric dicts over seeds, for the configurations run with several seeds."""
    out = {}
    for r in json.loads((ROOT / "results/sweep/summary.json").read_text()):
        out.setdefault((r["k"], r["v"], r["N"], r["objective"]), []).append(r)
    return {key: rs for key, rs in out.items() if len(rs) > 1}


def kept_bits(r):
    """Lower bound on what the recall model keeps about the sequence: N * (H(token) - cross-entropy), in bits."""
    return r["N"] * (math.log(16) - r["nll_true"]) / math.log(2)


def fig_sweep():
    d = sweep()
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.4), sharex=True, sharey=True, layout="constrained")
    series = [("recall: whole sequence exact", "lm", "recall_exact", SERIES[3], "--"),
              ("recall: tokens correct", "lm", "recall_token_acc", SERIES[0], "--"),
              ("recognition by surprise of the recall model (acc9)", "lm", "recog_acc9", SERIES[0], "-"),
              ("recognition model, last-token score (acc9)", "cls", "recog_acc9", SERIES[1], "-")]
    for ax, (k, v) in zip(axes.flat, STATES):
        for label, obj, key, col, ls in series:
            ax.plot(LENGTHS, [d[(k, v, n, obj)][key] for n in LENGTHS], ls, marker="o", ms=3.5, color=col, label=label)
        chance(ax, label="chance (acc9)")
        ax.set_xscale("log", base=2)
        ax.set_xticks(LENGTHS, [str(n) for n in LENGTHS])
        ax.set_ylim(-0.04, 1.06)
        ax.set_title(f"state {k}x{v} = {k * v:,} per layer")
    for ax in axes[1]:
        ax.set_xlabel("sequence length N")
    for ax in axes[:, 0]:
        ax.set_ylabel("test accuracy")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h[:4], l[:4], loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.09), fontsize=8.5)
    save(fig, "fig5_sweep.png")


def fig_bits():
    d = sweep()
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    for (k, v), col in zip(STATES, ORDINAL[1:]):
        bits = [kept_bits(d[(k, v, n, "lm")]) for n in LENGTHS]
        a.plot(LENGTHS, bits, "o-", ms=3.5, color=col, label=f"{k}x{v}")
    a.plot(LENGTHS, [4 * n for n in LENGTHS], color=MUTED, lw=1, zorder=1)
    a.annotate("all 4N bits of the sequence", (16, 64), xytext=(-8, 4), textcoords="offset points", ha="right", va="bottom",
               rotation=33, rotation_mode="anchor", color=MUTED, fontsize=8.5)
    a.set_xscale("log", base=2)
    a.set_yscale("log", base=2)
    a.set_xticks(LENGTHS, [str(n) for n in LENGTHS])
    a.set_yticks([16, 64, 256, 1024], ["16", "64", "256", "1024"])
    a.set_xlabel("sequence length N")
    a.set_ylabel("bits kept (lower bound)")
    a.set_title("a  What the recall model keeps of the sequence")
    a.legend(title="state", fontsize=8, title_fontsize=8, loc="upper left")
    for obj, key, col, label in [("lm", "recog_acc9", SERIES[0], "surprise of the recall model"),
                                 ("cls", "recog_acc9", SERIES[1], "recognition model, same state and N")]:
        xs, ys = [], []
        for k, v in STATES:
            for n in LENGTHS:
                r = d[(k, v, n, "lm")]
                frac = kept_bits(r) / (4 * n)
                if frac < 0.995:                       # only the configurations where recall is lossy
                    xs.append(frac)
                    ys.append(d[(k, v, n, obj)][key])
        b.scatter(xs, ys, s=36, color=col, edgecolor=SURFACE, linewidth=1.5, label=label, zorder=3)
    chance(b)
    b.set_xlim(0, 1.02)
    b.set_ylim(-0.04, 1.06)
    b.set_xlabel("share of the sequence's bits kept by the recall model")
    b.set_ylabel("test acc9")
    b.set_title("b  Recognition when recall is lossy")
    b.legend(fontsize=8, loc="lower right")
    save(fig, "fig6_bits.png")


def fig_seeds():
    d = seeds()
    ns = LENGTHS[2:]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True, layout="constrained")
    series = [("recall: whole sequence exact", "lm", "recall_exact", SERIES[3], "--"),
              ("recognition by surprise of the recall model (acc9)", "lm", "recog_acc9", SERIES[0], "-"),
              ("recognition model, last-token score (acc9)", "cls", "recog_acc9", SERIES[1], "-")]
    for ax, (k, v) in zip(axes, STATES[:3]):
        for label, obj, key, col, ls in series:
            vals = [[r[key] for r in d[(k, v, n, obj)]] for n in ns]
            mean = [sum(x) / len(x) for x in vals]
            ax.fill_between(ns, [min(x) for x in vals], [max(x) for x in vals], color=col, alpha=0.12, lw=0)
            ax.plot(ns, mean, ls, marker="o", ms=3.5, color=col, label=label)
            for n, x in zip(ns, vals):                       # each seed as a small dot
                ax.scatter([n] * len(x), x, s=9, color=col, alpha=0.6, lw=0, zorder=3)
        chance(ax, label="chance (acc9)")
        ax.set_xscale("log", base=2)
        ax.set_xticks(ns, [str(n) for n in ns])
        ax.set_ylim(-0.04, 1.06)
        ax.set_xlabel("sequence length N")
        ax.set_title(f"state {k}x{v}, 3 seeds")
    axes[0].set_ylabel("test accuracy")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.1), fontsize=8.5)
    save(fig, "fig7_seeds.png")


def fig_fix():
    t = json.loads((ROOT / "results/fix/summary.json").read_text())
    variants = [("base", "baseline: prefixes from 16, bce", SERIES[3]), ("A", "A: prefixes from 4", SERIES[2]),
                ("B", "B: A + lr 3e-4", SERIES[1]), ("C", "C: A + ranking loss", SERIES[0])]
    fig, ax = plt.subplots(figsize=(11, 3.8), layout="constrained")
    cfgs = list(t)
    for j, (key, label, col) in enumerate(variants):
        xs, ys = [], []
        for i, c in enumerate(cfgs):
            for a in t[c][key]:
                if a is not None:
                    xs.append(i + (j - 1.5) * 0.18)
                    ys.append(a)
        ax.scatter(xs, ys, s=30, color=col, edgecolor=SURFACE, linewidth=1.2, label=label, zorder=3)
    chance(ax)
    ax.axhline(0.852, color=MUTED, lw=1, zorder=1)
    ax.annotate("4-bit checksum", (1, 0.852), xycoords=("axes fraction", "data"), xytext=(-4, -12), textcoords="offset points",
                ha="right", color=MUTED, fontsize=8.5)
    ax.set_xticks(range(len(cfgs)), [c.replace("_N", "\nN=") for c in cfgs])
    ax.grid(axis="x", visible=False)
    ax.set_ylim(-0.04, 1.06)
    ax.set_ylabel("test acc9, one dot per seed")
    ax.set_title("Recognition model: training fixes on the configurations where a seed fell short")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=4, fontsize=8.5)
    save(fig, "fig8_fix.png")


if __name__ == "__main__":
    fig_first_grid()
    fig_plateau()
    fig_curriculum()
    fig_small_state()
    fig_sweep()
    fig_bits()
    fig_seeds()
    fig_fix()
