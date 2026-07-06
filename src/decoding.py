"""
Subject-wise cross-validated decoding of anesthesia state from CEBRA
latent embeddings (Methods: "Network Architecture and Training
Dynamics"; Results: "Latent Space decoding").

Produces the fold-wise embeddings plus everything needed to build:
    - Figure 3  (3D latent trajectories per fold)
    - Figure 4  (aggregated one-vs-rest ROC curves)
    - Figure 5  (aggregated confusion matrices)
    - Figure 6  (precision-recall curves)

Subjects are never split within a fold: GroupKFold guarantees
{s_train} ∩ {s_test} = ∅ (see Eq. in Results), so the classifier is
always evaluated on entirely unseen subjects.
"""

from __future__ import annotations

import numpy as np
import ituna
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

from config import N_CV_FOLDS, RANDOM_SEED
from src.cebra_model import CEBRAWrapper

ALL_CLASSES = [1, 2, 3, 4]


def make_consistency_ensemble(random_states: int = 2) -> "ituna.ConsistencyEnsemble":
    """CEBRA wrapped in ituna's ConsistencyEnsemble: trains several random
    seeds and aligns them under the affine indeterminacy class, improving
    robustness to stochastic-optimization variability (see Eq. 3, Methods
    on empirical identifiability)."""
    return ituna.ConsistencyEnsemble(
        estimator=CEBRAWrapper(),
        consistency_transform=ituna.metrics.PairwiseConsistency(
            indeterminacy=ituna.metrics.Affine(), symmetric=True, include_diagonal=False
        ),
        random_states=random_states,
    )


def cross_validate_embeddings(
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    n_splits: int = N_CV_FOLDS,
    ensemble_random_states: int = 2,
):
    """Runs subject-wise GroupKFold CV: for every fold, fits a CEBRA
    consistency ensemble on the training subjects, transforms both train
    and test data, and returns per-fold embeddings + labels.

    Returns
    -------
    dict with key "folds": list of
        {"train": {"embeddings":..., "labels":...},
         "test":  {"embeddings":..., "labels":...}}
    """
    gkf = GroupKFold(n_splits=n_splits)
    folds = []
    for fold_idx, (tr, te) in enumerate(gkf.split(X, y, groups=subject_ids)):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]

        ens = make_consistency_ensemble(ensemble_random_states)
        ens.fit(X_tr, y_tr.astype(int))

        folds.append(
            {
                "train": {"embeddings": ens.transform(X_tr), "labels": y_tr},
                "test": {"embeddings": ens.transform(X_te), "labels": y_te},
            }
        )
        print(f"  fold {fold_idx + 1}/{n_splits} done "
              f"(train n={len(tr)}, test n={len(te)})")
    return {"folds": folds}


def run_decoding(cv_data: dict, train_classes=ALL_CLASSES):
    """Fits a logistic-regression decoder per fold on the CEBRA
    embeddings and aggregates ROC / PR / confusion-matrix statistics
    across folds (Results: "Latent Space decoding", Figures 4-6).
    """
    fold_res = []
    cm_all = np.zeros((len(ALL_CLASSES), len(ALL_CLASSES)), dtype=int)
    dropped = 0

    for fold in cv_data["folds"]:
        X_tr, y_tr = fold["train"]["embeddings"], fold["train"]["labels"]
        X_te, y_te = fold["test"]["embeddings"], fold["test"]["labels"]

        if not all(c in y_tr for c in train_classes) or not all(c in y_te for c in train_classes):
            dropped += 1
            continue

        scaler = StandardScaler().fit(X_tr)
        X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

        clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs", multi_class="multinomial")
        clf.fit(X_tr_s, y_tr)
        pred = clf.predict(X_te_s)
        prob = clf.predict_proba(X_te_s)

        fold_aucs, fold_aps = {}, {}
        for i, c in enumerate(clf.classes_):
            y_bin = (y_te == c).astype(int)
            fpr, tpr, _ = roc_curve(y_bin, prob[:, i])
            fold_aucs[c] = auc(fpr, tpr)
            fold_aps[c] = average_precision_score(y_bin, prob[:, i])

        fold_res.append(
            {
                "acc": accuracy_score(y_te, pred),
                "aucs": fold_aucs,
                "aps": fold_aps,
                "pred": pred,
                "true": y_te,
                "prob": prob,
                "classes": clf.classes_,
            }
        )
        cm_all += confusion_matrix(y_te, pred, labels=ALL_CLASSES)

    print(f"Folds used: {len(fold_res)}  |  Dropped (missing class): {dropped}")
    accs = [f["acc"] for f in fold_res]
    print(f"Mean accuracy: {np.mean(accs):.3f} ± {np.std(accs):.3f}")
    return {"folds": fold_res, "cm": cm_all}
