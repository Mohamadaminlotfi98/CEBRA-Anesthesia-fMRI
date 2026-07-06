"""
3D latent-space trajectory visualization (Results: "Latent Space
decoding", Figure 3; and the combined propofol+sevoflurane latent space
figure).

For visualization only, embeddings are Procrustes-aligned across
subjects and PCA-compressed to 3D, as described in the paper -- this
does not change the classifier's decision boundary, only how the
learned geometry is displayed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import orthogonal_procrustes
from scipy.stats import chi2
from sklearn.decomposition import PCA

from config import CLASS_NAMES, CLASS_COLORS, FIGURES_DIR


def procrustes_align(reference: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Aligns `target` onto `reference` using an orthogonal Procrustes
    transform (rotation/reflection only, no scaling), so that subject
    trajectories can be shown in a common coordinate frame."""
    R, _ = orthogonal_procrustes(target, reference)
    return target @ R


def compute_centroids(emb: np.ndarray, labels: np.ndarray, order=(1, 2, 3, 4)) -> np.ndarray:
    return np.array([emb[labels == c].mean(axis=0) for c in order])


def ellipsoid_surface(mean: np.ndarray, cov: np.ndarray, confidence: float = 0.95, n: int = 40):
    """Returns (x, y, z) surface coordinates of a confidence ellipsoid,
    used to visualize the spread of each state cluster."""
    chi2_val = chi2.ppf(confidence, df=3)
    vals, vecs = np.linalg.eigh(cov)
    radii = np.sqrt(np.maximum(vals * chi2_val, 0))
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    pts = np.stack([x, y, z], axis=-1) @ vecs.T
    return mean[0] + pts[..., 0], mean[1] + pts[..., 1], mean[2] + pts[..., 2]


def plot_fold_trajectory(ax, emb_3d: np.ndarray, labels: np.ndarray, order=(1, 2, 3, 4)):
    """Draws point clouds, confidence ellipsoids, and a centroid-to-centroid
    trajectory for one fold's test-set embeddings, projected to 3D."""
    for cls in order:
        mask = labels == cls
        if not mask.any():
            continue
        color = CLASS_COLORS[cls]
        pts = emb_3d[mask]
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=2, alpha=0.3, color=color)
        if mask.sum() >= 4:
            ex, ey, ez = ellipsoid_surface(pts.mean(axis=0), np.cov(pts.T))
            ax.plot_surface(ex, ey, ez, alpha=0.08, color=color, linewidth=0)

    centroids = compute_centroids(emb_3d, labels, order)
    ax.plot(centroids[:, 0], centroids[:, 1], centroids[:, 2], "-o",
            color="#222222", linewidth=1.6, markersize=4, zorder=10)
    for c, cls in zip(centroids, order):
        ax.scatter(*c, s=100, color=CLASS_COLORS[cls], edgecolors="black",
                   linewidths=1.5, zorder=12, label=CLASS_NAMES[cls])


def plot_latent_space_by_fold(cv_data: dict, condition_label: str, out_path: str | Path = None):
    """Reproduces Figure 3: one 3D panel per cross-validation fold,
    showing the test-set latent trajectory after Procrustes alignment
    and PCA compression to 3 dimensions."""
    out_path = out_path or FIGURES_DIR / f"fig3_latent_space_{condition_label.lower()}.png"
    folds = cv_data["folds"]
    n_folds = len(folds)

    fig = plt.figure(figsize=(6 * n_folds, 5.5), dpi=200)
    reference = None

    for i, fold in enumerate(folds):
        emb, labels = fold["test"]["embeddings"], fold["test"]["labels"]
        if reference is None:
            reference = emb
            aligned = emb
        else:
            n = min(len(reference), len(emb))
            aligned = procrustes_align(reference[:n], emb[:n])
            if len(emb) > n:
                aligned = np.vstack([aligned, emb[n:]])

        pca = PCA(n_components=3)
        emb_3d = pca.fit_transform(aligned)

        ax = fig.add_subplot(1, n_folds, i + 1, projection="3d")
        plot_fold_trajectory(ax, emb_3d, labels)
        ax.set_title(f"Fold {i + 1}")
        ax.view_init(elev=25, azim=45)

    fig.suptitle(f"{condition_label}: Latent Space Geometry Across Folds", fontweight="bold")
    handles, hlabels = ax.get_legend_handles_labels()
    fig.legend(handles, hlabels, loc="lower center", ncol=4)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")
