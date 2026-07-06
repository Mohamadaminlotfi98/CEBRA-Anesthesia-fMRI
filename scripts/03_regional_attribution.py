#!/usr/bin/env python
"""
Step 3: Regional attribution analysis (Figure 9).

For each dataset, runs subject-wise CV, computes Jacobian /
Neuron-Gradient / Feature-Ablation importance maps per fold, averages
them, and projects them onto the AAL v3v2 atlas.

Usage
-----
    python scripts/03_regional_attribution.py --dataset propofol
    python scripts/03_regional_attribution.py --dataset sevoflurane
"""

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from config import RESULTS_DIR
from src.data_loading import load_propofol, load_sevoflurane, data_handler
from src.attribution import load_aal_atlas, run_attribution_cv
from src.plotting_attribution import plot_attribution_maps

TRAIN_CLASSES = [1, 2, 3, 4]
LOADERS = {"propofol": load_propofol, "sevoflurane": load_sevoflurane}
LABELS = {"propofol": "Propofol", "sevoflurane": "Sevoflurane"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["propofol", "sevoflurane"], required=True)
    args = parser.parse_args()

    data, labels, subs = LOADERS[args.dataset]()
    mask = np.isin(labels, TRAIN_CLASSES)
    X, y, subject_ids = data_handler(data[mask], labels[mask], subs[mask])

    atlas_img, atlas_data, region_indices = load_aal_atlas()

    imp_train, imp_test, mean_acc, std_acc = run_attribution_cv(X, y, subject_ids)

    with open(RESULTS_DIR / f"attribution_{args.dataset}.pkl", "wb") as f:
        pickle.dump({"train": imp_train, "test": imp_test, "acc": (mean_acc, std_acc)}, f)

    plot_attribution_maps(
        imp_train, imp_test, atlas_data, region_indices, atlas_img.affine,
        condition_label=LABELS[args.dataset], mean_acc=mean_acc, std_acc=std_acc,
    )


if __name__ == "__main__":
    main()
