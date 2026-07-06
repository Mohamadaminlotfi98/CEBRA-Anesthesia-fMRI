#!/usr/bin/env python
"""
Step 1: Train CEBRA embeddings and decode anesthesia state.

Reproduces:
    Figure 3 (latent space geometry across folds)
    Figure 4 (ROC curves)
    Figure 5 (confusion matrices)
    Figure 6 (precision-recall curves)

Usage
-----
    python scripts/01_train_and_decode.py --dataset propofol
    python scripts/01_train_and_decode.py --dataset sevoflurane
    python scripts/01_train_and_decode.py --dataset both   # also saves combined fig4-6
"""

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import RESULTS_DIR
from src.data_loading import load_and_flatten
from src.decoding import cross_validate_embeddings, run_decoding
from src.plotting_decoding import plot_roc_curves, plot_confusion_matrices, plot_precision_recall_curves
from src.viz_latent_space import plot_latent_space_by_fold

COLORS = {"propofol": "#2166ac", "sevoflurane": "#b2182b"}
LABELS = {"propofol": "Propofol", "sevoflurane": "Sevoflurane"}


def run_one(dataset: str):
    print(f"\n=== {LABELS[dataset]} ===")
    X, y, subject_ids = load_and_flatten(dataset)
    print(f"Loaded {X.shape[0]} timepoints, {X.shape[1]} ROIs, "
          f"{len(set(subject_ids))} subjects")

    print("Running subject-wise cross-validation and training CEBRA...")
    cv_data = cross_validate_embeddings(X, y, subject_ids)

    with open(RESULTS_DIR / f"cv_embeddings_{dataset}.pkl", "wb") as f:
        pickle.dump(cv_data, f)

    plot_latent_space_by_fold(cv_data, LABELS[dataset])

    print("Decoding latent embeddings...")
    decoding_result = run_decoding(cv_data)
    with open(RESULTS_DIR / f"decoding_result_{dataset}.pkl", "wb") as f:
        pickle.dump(decoding_result, f)

    return decoding_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["propofol", "sevoflurane", "both"], default="both")
    args = parser.parse_args()

    results = {}
    datasets = ["propofol", "sevoflurane"] if args.dataset == "both" else [args.dataset]
    for ds in datasets:
        results[ds] = run_one(ds)

    if args.dataset == "both":
        by_condition = {LABELS[ds]: (results[ds], COLORS[ds]) for ds in datasets}
        plot_roc_curves(by_condition)
        plot_confusion_matrices(by_condition)
        plot_precision_recall_curves(by_condition)


if __name__ == "__main__":
    main()
