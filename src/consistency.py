"""
Empirical identifiability / embedding-consistency analysis (Results:
"Latent Space decoding", Figure 7).

For each configuration (dataset x class-subset), trains a
ConsistencyEnsemble of CEBRA models with different random seeds under
both true labels and shuffled labels (control), and reports the
pairwise, affine-aligned consistency score on held-in (train) and
held-out (test) subjects via GroupKFold.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupKFold

from config import N_CV_FOLDS
from src.decoding import make_consistency_ensemble


def consistency_for_config(X: np.ndarray, y: np.ndarray, subject_ids: np.ndarray,
                            n_splits: int = N_CV_FOLDS, ensemble_random_states: int = 5,
                            rng: np.random.Generator | None = None):
    """Returns dict with per-fold consistency scores for true and
    shuffled labels, on train and test splits:
        {"cons_tr": [...], "cons_te": [...], "cons_tr_s": [...], "cons_te_s": [...]}
    """
    rng = rng or np.random.default_rng(0)
    gkf = GroupKFold(n_splits=n_splits)
    out = {"cons_tr": [], "cons_te": [], "cons_tr_s": [], "cons_te_s": []}

    for fold_idx, (tr, te) in enumerate(gkf.split(X, y, groups=subject_ids), 1):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]

        ens = make_consistency_ensemble(ensemble_random_states)
        ens.fit(X_tr, y_tr.astype(int))
        out["cons_tr"].append(ens.score(X_tr))
        out["cons_te"].append(ens.score(X_te))

        ens_shuffled = make_consistency_ensemble(ensemble_random_states)
        y_tr_shuffled = rng.permutation(y_tr)
        ens_shuffled.fit(X_tr, y_tr_shuffled.astype(int))
        out["cons_tr_s"].append(ens_shuffled.score(X_tr))
        out["cons_te_s"].append(ens_shuffled.score(X_te))

        print(f"  fold {fold_idx}: train={out['cons_tr'][-1]:.3f} "
              f"test={out['cons_te'][-1]:.3f}  "
              f"(shuffled train={out['cons_tr_s'][-1]:.3f}, "
              f"test={out['cons_te_s'][-1]:.3f})")

    return out
