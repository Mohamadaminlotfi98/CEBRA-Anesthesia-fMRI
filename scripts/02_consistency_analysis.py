#!/usr/bin/env python
"""
Step 2: Empirical identifiability / consistency analysis (Figure 7).

Trains a ConsistencyEnsemble of CEBRA models under true vs. shuffled
labels for four configurations (Propofol / Sevoflurane x Awake-vs-Deep
/ All-states), and plots the resulting consistency scores.

Usage
-----
    python scripts/02_consistency_analysis.py
"""

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from config import RESULTS_DIR
from src.data_loading import load_propofol, load_sevoflurane, data_handler
from src.consistency import consistency_for_config
from src.plotting_consistency import plot_consistency_figure

CONFIGS = {
    "Propofol\n(Awake vs High)": {"classes": [1, 4], "loader": load_propofol},
    "Propofol\n(All classes)": {"classes": [1, 2, 3, 4], "loader": load_propofol},
    "Sevoflurane\n(Awake vs High)": {"classes": [1, 4], "loader": load_sevoflurane},
    "Sevoflurane\n(All classes)": {"classes": [1, 2, 3, 4], "loader": load_sevoflurane},
}


def main():
    results = {}
    for cfg_name, cfg in CONFIGS.items():
        clean_name = cfg_name.replace("\n", " ")
        print(f"\n=== {clean_name} ===")
        data, labels, subs = cfg["loader"]()
        mask = np.isin(labels, cfg["classes"])
        X, y, subject_ids = data_handler(data[mask], labels[mask], subs[mask])
        results[cfg_name] = consistency_for_config(X, y, subject_ids)

    with open(RESULTS_DIR / "consistency_results.pkl", "wb") as f:
        pickle.dump(results, f)

    plot_consistency_figure(results)


if __name__ == "__main__":
    main()
