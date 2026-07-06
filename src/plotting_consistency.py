"""Figure generation for the embedding-consistency analysis (Figure 7)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy import stats

from config import FIGURES_DIR

C_REAL_TR, C_REAL_TE = "#1B4F72", "#2E86C1"
C_SHUF_TR, C_SHUF_TE = "#7F8C8D", "#BDC3C7"
EDGE, EDGE_S = "#0A2846", "#4D5656"
DOT_REAL, DOT_SHUF = "#0D2137", "#3D3D3D"


def plot_consistency_figure(results: dict, out_path: str | Path = None, seed: int = 42):
    """`results` maps a config label (e.g. "Propofol\\n(All states)") to
    the dict returned by `src.consistency.consistency_for_config`.
    """
    out_path = out_path or FIGURES_DIR / "fig7_consistency.png"
    rng = np.random.default_rng(seed)
    jitter = 0.07
    positions = [0, 1, 2.5, 3.5]
    keys = ["cons_tr", "cons_te", "cons_tr_s", "cons_te_s"]
    colors = [C_REAL_TR, C_REAL_TE, C_SHUF_TR, C_SHUF_TE]
    edges = [EDGE, EDGE, EDGE_S, EDGE_S]
    dot_cols = [DOT_REAL, DOT_REAL, DOT_SHUF, DOT_SHUF]
    xtick_labels = ["Train", "Test", "Train", "Test"]

    n = len(results)
    fig = plt.figure(figsize=(4 * n, 6))
    outer = gridspec.GridSpec(1, n, figure=fig, wspace=0.4)
    axes = [fig.add_subplot(outer[i]) for i in range(n)]

    for ax, (cfg_name, fd) in zip(axes, results.items()):
        groups = [np.array(fd[k]) for k in keys]
        for pos, grp, fc, ec, dc in zip(positions, groups, colors, edges, dot_cols):
            m = grp.mean()
            se = grp.std(ddof=1) / np.sqrt(len(grp)) if len(grp) > 1 else 0.0
            ax.bar(pos, m, width=0.68, color=fc, alpha=0.9, linewidth=0.9, edgecolor=ec, zorder=2)
            ax.errorbar(pos, m, yerr=se, fmt="none", color="#111111", capsize=4.5, linewidth=1.3, zorder=4)
            jx = pos + rng.uniform(-jitter, jitter, len(grp))
            ax.scatter(jx, grp, s=28, color=dc, alpha=0.7, zorder=5)

        real_te, shuf_te = np.array(fd["cons_te"]), np.array(fd["cons_te_s"])
        if len(real_te) > 1:
            _, p = stats.ttest_rel(real_te, shuf_te)
            stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            ymax = max(real_te.max(), shuf_te.max()) + 0.07
            ax.annotate("", xy=(3.5, ymax), xytext=(1, ymax),
                        arrowprops=dict(arrowstyle="-", color="#333333", lw=0.9))
            ax.text(2.25, ymax + 0.03, stars, ha="center", fontsize=9, fontweight="bold")

        ax.set_xlim(-0.6, 4.1)
        ax.set_ylim(-0.15, 1.15)
        ax.set_xticks(positions)
        ax.set_xticklabels(xtick_labels, fontsize=9)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_title(cfg_name.replace("\n", " "), fontsize=10, fontweight="bold")
        ax.set_ylabel("Consistency Score")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    legend_elements = [
        Patch(facecolor=C_REAL_TR, edgecolor=EDGE, label="True labels — Train"),
        Patch(facecolor=C_REAL_TE, edgecolor=EDGE, label="True labels — Test"),
        Patch(facecolor=C_SHUF_TR, edgecolor=EDGE_S, label="Shuffled labels — Train"),
        Patch(facecolor=C_SHUF_TE, edgecolor=EDGE_S, label="Shuffled labels — Test"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#333333", markersize=6, label="Fold score"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=5, fontsize=8.5, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("CEBRA Embedding Consistency", fontsize=13.5, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")
