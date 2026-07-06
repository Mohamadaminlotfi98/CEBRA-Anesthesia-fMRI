"""
Figure-generation helpers for the decoding results (Figures 4-6).

These functions take the output of `src.decoding.run_decoding` for two
conditions (e.g. propofol / sevoflurane) and reproduce the aggregated
ROC, confusion-matrix, and precision-recall panels shown in the paper.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics import roc_curve, precision_recall_curve

from config import CLASS_NAMES, FIGURES_DIR

ALL_CLASSES = [1, 2, 3, 4]
TICK_LABELS = [CLASS_NAMES[c] for c in ALL_CLASSES]


def plot_roc_curves(results_by_condition: dict, out_path: str | Path = None):
    """results_by_condition: {"Propofol": (decoding_result, color), "Sevoflurane": (...)}"""
    out_path = out_path or FIGURES_DIR / "fig4_roc_curves.png"
    fig, axes = plt.subplots(1, len(results_by_condition), figsize=(6 * len(results_by_condition), 5), dpi=300)
    if len(results_by_condition) == 1:
        axes = [axes]
    base_fpr = np.linspace(0, 1, 300)

    for ax, (label, (res, color)) in zip(axes, results_by_condition.items()):
        for c in ALL_CLASSES:
            tprs_c = []
            for fold in res["folds"]:
                if c not in fold["classes"]:
                    continue
                ci = list(fold["classes"]).index(c)
                y_bin = (fold["true"] == c).astype(int)
                fpr, tpr, _ = roc_curve(y_bin, fold["prob"][:, ci])
                tprs_c.append(np.interp(base_fpr, fpr, tpr))
            if not tprs_c:
                continue
            mean_tpr = np.mean(tprs_c, axis=0)
            mean_tpr[0] = 0.0
            std_tpr = np.std(tprs_c, axis=0)
            mean_auc = np.mean([f["aucs"][c] for f in res["folds"] if c in f["aucs"]])
            ax.plot(base_fpr, mean_tpr, lw=2.2, label=f"{CLASS_NAMES[c]} (AUC={mean_auc:.2f})")
            ax.fill_between(base_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, alpha=0.15)

        ax.plot([0, 1], [0, 1], color="#cccccc", lw=1.0, ls=":")
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.04)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(label, fontweight="bold", color=color)
        ax.legend(loc="lower right", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Aggregated One-vs-Rest ROC Curves", fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_confusion_matrices(results_by_condition: dict, out_path: str | Path = None):
    out_path = out_path or FIGURES_DIR / "fig5_confusion_matrices.png"
    fig, axes = plt.subplots(1, len(results_by_condition), figsize=(5 * len(results_by_condition), 4.2), dpi=300)
    if len(results_by_condition) == 1:
        axes = [axes]

    for ax, (label, (res, color)) in zip(axes, results_by_condition.items()):
        cm = res["cm"]
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cmap = mcolors.LinearSegmentedColormap.from_list("c", ["#ffffff", color], N=256)
        ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for i in range(len(ALL_CLASSES)):
            for j in range(len(ALL_CLASSES)):
                txt_col = "white" if cm_norm[i, j] > 0.55 else "#222222"
                ax.text(j, i, f"{cm_norm[i, j]:.2f}\n(n={cm[i, j]})", ha="center", va="center",
                        fontsize=8.5, fontweight="bold", color=txt_col)
        ax.set_xticks(range(len(ALL_CLASSES)))
        ax.set_yticks(range(len(ALL_CLASSES)))
        ax.set_xticklabels(TICK_LABELS, rotation=20, ha="right")
        ax.set_yticklabels(TICK_LABELS)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(label, fontweight="bold")

    fig.suptitle("Aggregated Confusion Matrices", fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_precision_recall_curves(results_by_condition: dict, out_path: str | Path = None):
    out_path = out_path or FIGURES_DIR / "fig6_precision_recall.png"
    fig, axes = plt.subplots(1, len(results_by_condition), figsize=(6 * len(results_by_condition), 5), dpi=300)
    if len(results_by_condition) == 1:
        axes = [axes]
    base_recall = np.linspace(0, 1, 300)

    for ax, (label, (res, color)) in zip(axes, results_by_condition.items()):
        for c in ALL_CLASSES:
            precs_c = []
            for fold in res["folds"]:
                if c not in fold["classes"]:
                    continue
                ci = list(fold["classes"]).index(c)
                y_bin = (fold["true"] == c).astype(int)
                prec, rec, _ = precision_recall_curve(y_bin, fold["prob"][:, ci])
                idx = np.argsort(rec)
                precs_c.append(np.interp(base_recall, rec[idx], prec[idx]))
            if not precs_c:
                continue
            mean_prec = np.mean(precs_c, axis=0)
            std_prec = np.std(precs_c, axis=0)
            mean_ap = np.mean([f["aps"][c] for f in res["folds"] if c in f["aps"]])
            ax.plot(base_recall, mean_prec, lw=2.2, label=f"{CLASS_NAMES[c]} (AP={mean_ap:.2f})")
            ax.fill_between(base_recall, mean_prec - std_prec, mean_prec + std_prec, alpha=0.15)

        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.04)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(label, fontweight="bold", color=color)
        ax.legend(loc="lower left", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Precision-Recall Curves", fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")
