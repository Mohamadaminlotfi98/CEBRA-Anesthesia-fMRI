"""Figure generation for regional attribution brain maps (Figure 9)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from nilearn import plotting

from config import FIGURES_DIR
from src.attribution import METHODS, METHOD_LABELS, build_power_map


def plot_brain_on_ax(nifti_img, ax, title, fontsize=11):
    plotting.plot_stat_map(
        nifti_img, display_mode="ortho", cmap="coolwarm", vmin=-1, vmax=1,
        symmetric_cbar=True, colorbar=True, title=None, axes=ax,
        annotate=True, draw_cross=False, cut_coords=(0, 0, 0),
    )
    ax.set_title(title, fontsize=fontsize, fontweight="bold", pad=6)


def plot_attribution_maps(imp_train_all, imp_test_all, atlas_data, region_indices, affine,
                           condition_label: str, mean_acc: float, std_acc: float,
                           out_path: str | Path = None):
    """Reproduces Figure 9: a 2 (train/test) x 3 (methods) grid of brain
    maps for one condition (e.g. propofol or sevoflurane)."""
    out_path = out_path or FIGURES_DIR / f"fig9_attribution_{condition_label.lower()}.png"
    n_rows, n_cols = 2, len(METHODS)
    fig = plt.figure(figsize=(7.5 * n_cols, 6.5 * n_rows + 1.8), dpi=200)
    outer = gridspec.GridSpec(n_rows + 1, 1, figure=fig, hspace=0.08, height_ratios=[0.06] + [1.0] * n_rows)

    header_ax = fig.add_subplot(outer[0])
    header_ax.axis("off")
    header_ax.text(0.5, 0.5,
                    f"{condition_label} Average Feature Importance Maps Across Folds "
                    f"(Mean Accuracy: {mean_acc:.3f} ± {std_acc:.3f})",
                    transform=header_ax.transAxes, ha="center", va="center",
                    fontsize=16, fontweight="bold")

    row_labels = ["Train", "Test"]
    row_data = [imp_train_all, imp_test_all]

    for row_i in range(n_rows):
        inner = gridspec.GridSpecFromSubplotSpec(1, n_cols, subplot_spec=outer[row_i + 1], wspace=0.04)
        for col_i, method in enumerate(METHODS):
            ax = fig.add_subplot(inner[col_i])
            nifti_img = build_power_map(row_data[row_i][method], atlas_data, region_indices, affine)
            plot_brain_on_ax(nifti_img, ax, f"{row_labels[row_i]} — {METHOD_LABELS[method]}")
            if col_i == 0:
                ax.annotate(row_labels[row_i], xy=(-0.06, 0.5), xycoords="axes fraction",
                            fontsize=13, fontweight="bold", va="center", ha="right", rotation=90)

    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out_path}")
