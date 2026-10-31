"""Poster-sized figures for posters/neurips_poster.tex, from results/{gdn2,sweep_smoke}.

  conda run -n ai python posters/make_poster_figures.py   ->  posters/figures/*.pdf

Figures are drawn at their printed size (inches) so fonts come out at poster scale.
"""

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "posters" / "figures"
LENGTHS = [4, 8, 16, 32, 64, 128, 256]
W = 14.6  # printed width of a figure in the poster's wide columns, inches

SURFACE, GRID = "#ffffff", "#e6e5e0"
TEXT, TEXT2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
RECALL, RECOG, CURR, NONE = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
RECALL_L, RECOG_L = "#a9cbf3", "#f6c1a8"  # same hues, light: full-size (128x128) state
ORDINAL = ["#86b6ef", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "pdf.fonttype": 42, "font.size": 24, "axes.titlesize": 26, "axes.labelsize": 24,
    "xtick.labelsize": 21, "ytick.labelsize": 21, "legend.fontsize": 21,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT2, "axes.titlecolor": TEXT,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 14,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1.2, "axes.linewidth": 1.4,
    "axes.spines.top": False, "axes.spines.right": False, "xtick.color": TEXT2, "ytick.color": TEXT2,
    "xtick.major.size": 0, "ytick.major.size": 0, "xtick.major.pad": 8, "ytick.major.pad": 8,
    "legend.frameon": False, "legend.labelcolor": TEXT2,
    "lines.linewidth": 4, "lines.solid_capstyle": "round", "lines.markersize": 10,
})
KSTEPS = FuncFormatter(lambda v, _: f"{v / 1000:.0f}k")


def load(path):
    return json.loads((ROOT / path).read_text())


def grid(n, obj):
    return load(f"results/gdn2/gdn2_h128_L2_N{n}_{obj}_s0.json")


def smoke(name):
    return load(f"results/sweep_smoke/{name}.json")


def curve(r, key):
    return [c["step"] for c in r["curve"]], [c[key] for c in r["curve"]]


def hline(ax, y, label, **kw):
    ax.axhline(y, color=MUTED, lw=2, ls=(0, (4, 3)), zorder=1)
    ax.annotate(label, (1, y), xycoords=("axes fraction", "data"), xytext=(-6, 6), textcoords="offset points",
                ha="right", va="bottom", color=MUTED, fontsize=19, **kw)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print("wrote", OUT / name)


def fig_full_state():
    """Result 1: at 128x128 recall is trivial, the plain classifier never leaves ln 2 for N >= 64."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 5.6), layout="constrained")
    lm = [grid(n, "lm")["test"] for n in LENGTHS]
    cls = [grid(n, "cls")["test"] for n in LENGTHS]
    a.set_xscale("log", base=2)
    a.set_xticks(LENGTHS, [str(n) for n in LENGTHS])
    a.minorticks_off()
    a.plot(LENGTHS, [t["recog_pairwise"] for t in lm], "o-", color=RECALL, label="recall model, surprise")
    a.plot(LENGTHS, [t["recog_pairwise"] for t in cls], "o-", color=RECOG, label="recognition classifier")
    hline(a, 0.5, "chance")
    a.set_ylim(0.3, 1.06)
    a.set_xlabel("sequence length N")
    a.set_ylabel("test pairwise accuracy")
    a.set_title("a  True vs corrupted copy")
    a.legend(loc="lower left", borderaxespad=0.2)
    for n, col in zip(LENGTHS, ORDINAL):
        s, y = curve(grid(n, "cls"), "train_loss")
        b.plot(s, y, color=col, lw=3.2, label=f"{n}")
    hline(b, math.log(2), "ln 2 = constant output")
    b.set_ylim(-0.02, 0.8)
    b.set_xlabel("training step")
    b.set_ylabel("classifier train loss")
    b.xaxis.set_major_formatter(KSTEPS)
    b.set_title("b  Classifier training, no curriculum")
    b.legend(title="N", title_fontsize=20, ncol=2, loc="center right", bbox_to_anchor=(1, 0.5),
             columnspacing=1, handlelength=1.4)
    save(fig, "full_state.pdf")


def fig_curriculum():
    """Result 2: a length curriculum lets the classifier train at N = 128, 256."""
    runs = {
        64: [("none", grid(64, "cls")), ("hard", smoke("smoke_k128v128_N64_cls"))],
        128: [("none", grid(128, "cls")), ("length", smoke("smoke2_k128v128_N128_cls"))],
        256: [("none", grid(256, "cls")), ("hard", smoke("smoke_k128v128_N256_cls")),
              ("length", smoke("smoke2_k128v128_N256_cls"))],
    }
    style = {"none": (NONE, "no curriculum"), "hard": (RECOG, "easy negatives first"),
             "length": (CURR, "length curriculum (prefixes 16 … N)")}
    fig, axes = plt.subplots(1, 3, figsize=(W, 5.0), sharey=True, layout="constrained")
    for ax, (n, rs) in zip(axes, runs.items()):
        for key, r in rs:
            s, y = curve(r, "recog_acc9")
            col, label = style[key]
            ax.plot(s, y, "o-", ms=7, lw=3.5, color=col, label=label)
        hline(ax, 1 / 9, "chance")
        ax.set_xlim(0, 11500)
        ax.set_xticks([0, 5000, 10000])
        ax.set_xlabel("training step")
        ax.xaxis.set_major_formatter(KSTEPS)
        ax.set_title(f"N = {n}")
    axes[0].set_ylabel("validation acc9")
    axes[0].set_ylim(-0.04, 1.08)
    handles = {h.get_label(): h for ax in axes for h in ax.get_lines() if not h.get_label().startswith("_")}
    order = ["no curriculum", "easy negatives first", "length curriculum (prefixes 16 … N)"]
    fig.legend([handles[k] for k in order], order, loc="outside upper center", ncol=3, columnspacing=1.6)
    save(fig, "curriculum.pdf")


def fig_small_state_curves():
    """16x4 state, N=256: the classifier escapes in the full-length curriculum stage, which the 3-epoch pilot cut short."""
    runs = [("classifier, 10 epochs", sweep(16, 4, 256, "cls"), RECOG, "-"),
            ("classifier, 3 epochs (pilot)", smoke("smoke2_k16v4_N256_cls"), RECOG_L, "--"),
            ("surprise, 10 epochs", sweep(16, 4, 256, "lm"), RECALL, "-")]
    fig, ax = plt.subplots(figsize=(W, 4.1), layout="constrained")
    lens = [16, 32, 64, 128, 256]
    stage = runs[0][1]["args"]["curriculum"] * runs[0][1]["steps"] / len(lens)
    for i, n in enumerate(lens):  # curriculum stages of the 10-epoch runs
        if i % 2 == 0:
            ax.axvspan(i * stage, (i + 1) * stage, color=GRID, alpha=0.6, lw=0)
        ax.text((i + 0.5) * stage, 1.02, f"prefix {n}", ha="center", va="top", color=TEXT2, fontsize=17)
    ax.text((len(lens) * stage + runs[0][1]["steps"]) / 2, 1.02, "full length", ha="center", va="top",
            color=TEXT2, fontsize=17)
    for label, r, col, ls in runs:
        s_, y = curve(r, "recog_acc9")
        ax.plot(s_, y, "o", ls=ls, ms=7, lw=3.5, color=col, label=label)
    hline(ax, 1 / 9, "chance")
    ax.set_ylim(-0.04, 1.06)
    ax.set_xlim(0, runs[0][1]["steps"] + 500)
    ax.set_xlabel("training step")
    ax.set_ylabel("validation acc9")
    ax.xaxis.set_major_formatter(KSTEPS)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.9), handlelength=1.8)
    save(fig, "small_state_curves.pdf")


STATES = [(16, 4), (16, 8), (16, 16), (32, 32), (64, 64), (128, 128)]


def sweep(k, v, n, obj):
    p = ROOT / f"results/sweep/k{k}v{v}_N{n}_{obj}_s0.json"
    return json.loads(p.read_text()) if p.exists() else None


def fig_sweep(lengths=(64, 128, 256)):
    """Result 3: test scores against state size, one panel per N; missing runs are skipped."""
    fig, axes = plt.subplots(1, len(lengths), figsize=(W, 5.1), sharey=True, layout="constrained")
    x = list(range(len(STATES)))  # evenly spaced: each step is 2-4x the state
    for ax, n in zip(axes, lengths):
        lm = [sweep(k, v, n, "lm") for k, v in STATES]
        cls = [sweep(k, v, n, "cls") for k, v in STATES]

        def pts(rs, key):
            return zip(*[(xi, r["test"][key]) for xi, r in zip(x, rs) if r]) if any(rs) else ([], [])

        ax.plot(*pts(lm, "recall_token_acc"), "s:", color=RECALL_L, lw=3, ms=9, label="recall: tokens copied")
        ax.plot(*pts(lm, "recall_exact"), "^:", color=MUTED, lw=3, ms=9, label="recall: exact copy")
        ax.plot(*pts(lm, "recog_acc9"), "o-", color=RECALL, label="recognition by surprise")
        ax.plot(*pts(cls, "recog_acc9"), "o-", color=RECOG, label="recognition classifier")
        hline(ax, 1 / 9, "chance")
        ax.set_xticks(x, [f"{k}×{v}" for k, v in STATES], rotation=45, ha="right", rotation_mode="anchor",
                      fontsize=18)
        ax.set_xlim(-0.3, len(STATES) - 0.7)
        ax.set_xlabel("state $d_k × d_v$ per layer")
        ax.set_title(f"N = {n}")
    axes[0].set_ylabel("test acc9 / accuracy")
    axes[0].set_ylim(-0.04, 1.08)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=2, columnspacing=1.6)
    save(fig, "sweep.pdf")


if __name__ == "__main__":
    fig_full_state()
    fig_curriculum()
    fig_sweep()
    fig_small_state_curves()
