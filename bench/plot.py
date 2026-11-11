"""Draw docs/qsa-vs-dense.png from bench/sweep.sh's results.

    python3 bench/plot.py [results_dir] [output.png]     # needs matplotlib
"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

RESULTS = sys.argv[1] if len(sys.argv) > 1 else "docs/results"
OUT = sys.argv[2] if len(sys.argv) > 2 else "docs/qsa-vs-dense.png"

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
SERIES = {  # colour, line style, marker, marker face
    "qsa": ("QSA", "#2a78d6", "-", "o", "#2a78d6"),
    "dense": ("Dense", "#eb6834", (0, (5, 3)), "s", SURFACE),
}


def load(mode, kind):
    rows = [json.loads(l) for l in open(f"{RESULTS}/{mode}-{kind}.jsonl") if l.startswith("{")]
    key = "prefill_tok_s" if kind == "prefill" else "decode_tok_s"
    xs = [r["prompt_tokens"] for r in rows]
    ys = [sum(r[key]) / len(r[key]) if isinstance(r[key], list) else r[key] for r in rows]
    return xs, ys


plt.rcParams.update({"font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=SURFACE)
for ax, kind, title in zip(axes, ("prefill", "decode"), ("Prefill (cold)", "Decode (MTP 3)")):
    ax.set_facecolor(SURFACE)
    top = 0
    for mode, (label, colour, style, marker, face) in SERIES.items():
        xs, ys = load(mode, kind)
        top = max(top, max(ys))
        ax.plot(xs, ys, linestyle=style, color=colour, linewidth=2, marker=marker, markersize=8,
                markerfacecolor=face, markeredgewidth=2, label=label, zorder=3)
        # Direct label: the value at the deepest context.
        ax.annotate(f"{ys[-1]:,.0f}", (xs[-1], ys[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=10, color=INK)
    ax.set_title(title, loc="left", fontsize=14, pad=12)
    ax.set_ylim(0, top * 1.15)
    ax.set_xlim(-4000, 262144 * 1.08)
    ax.set_xticks([0, 65536, 131072, 196608, 262144])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1024:.0f}K"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_xlabel("Context depth (tokens)")
    ax.set_ylabel("tokens/s")
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="lower left")
fig.text(0.99, 0.01, "Qwen3.8-27B-QSA · vLLM 0.31.0 · 2× Arc Pro B70 · TP 2 · FP8 weights · greedy",
         ha="right", va="bottom", fontsize=9, color=MUTED)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(OUT, dpi=150, facecolor=SURFACE)
print(OUT)
