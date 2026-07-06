"""
Regional attribution analysis (Methods: spatial-attribution paragraph;
Results: "Regional Decoding", Figures 8-9).

Quantifies, for a trained CEBRA encoder, which of the 116 AAL regions
contribute most to the learned latent geometry, using three
complementary methods:
    - Jacobian-based attribution   (local sensitivity, Eq. 3)
    - Neuron Gradient               (latent-unit sensitivity, Eq. 4)
    - Feature Ablation               (global perturbation importance, Eq. 5)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import nibabel as nib
import torch
from cebra.attribution import FeatureAblationMethod, NeuronGradientMethod, JFMethodBased

from config import AAL_ATLAS_DIR

METHODS = ["jacobian", "neuron_gradient", "feature_ablation"]
METHOD_LABELS = {
    "jacobian": "Jacobian",
    "neuron_gradient": "Neuron Gradient",
    "feature_ablation": "Feature Ablation",
}


def load_aal_atlas(aal_dir: str | None = None):
    """Loads the AAL v3v2 atlas volume and its region index table."""
    base = aal_dir or AAL_ATLAS_DIR
    atlas_img = nib.load(f"{base}/atlas/AAL.nii")
    atlas_data = atlas_img.get_fdata().astype(int)
    labels_df = pd.read_csv(f"{base}/ROI_MNI_V4.txt", sep=r"\s+", header=None)
    region_indices = labels_df[2].astype(int).values
    return atlas_img, atlas_data, region_indices


def get_feature_importance(cebra_wrapper, data: np.ndarray, method: str,
                            seed: int = 42, num_samples: int | None = None) -> np.ndarray:
    """Computes a min-max normalized, region-wise importance vector for
    one attribution method, given a fitted `CEBRAWrapper` (or a fitted
    ituna ensemble member -- both expose `.model_`).

    Parameters
    ----------
    cebra_wrapper : fitted src.cebra_model.CEBRAWrapper
    data          : np.ndarray, (n_samples, N_ROIs) input windows
    method        : one of "jacobian", "neuron_gradient", "feature_ablation"
    """
    internal_model = cebra_wrapper.model_.model_ if hasattr(cebra_wrapper.model_, "model_") else cebra_wrapper.model_
    output_dimension = cebra_wrapper.output_dimension
    device = next(internal_model.parameters()).device
    input_tensor = torch.from_numpy(data).float().to(device)
    input_tensor.requires_grad_(True)

    if method == "jacobian":
        attr = JFMethodBased(model=internal_model, input_data=input_tensor,
                              output_dimension=output_dimension, num_samples=num_samples, seed=seed)
        maps = attr.compute_attribution_map()
        raw = np.mean(np.abs(maps["jf-convabs"]), axis=(0, 1))

    elif method == "neuron_gradient":
        attr = NeuronGradientMethod(model=internal_model, input_data=input_tensor,
                                     output_dimension=output_dimension, num_samples=num_samples, seed=seed)
        maps = attr.compute_attribution_map(attribute_to_neuron_input=False)
        raw = np.mean(np.abs(maps["neuron-gradient-convabs"]), axis=(0, 1))

    elif method == "feature_ablation":
        attr = FeatureAblationMethod(model=internal_model, input_data=input_tensor,
                                      output_dimension=output_dimension, num_samples=num_samples, seed=seed)
        maps = attr.compute_attribution_map(baselines=None, feature_mask=None,
                                             perturbations_per_eval=1, attribute_to_neuron_input=False)
        raw = np.mean(np.abs(maps["feature-ablation-convabs"]), axis=(0, 1))

    else:
        raise ValueError(f"Unknown attribution method '{method}'")

    del input_tensor
    torch.cuda.empty_cache()
    return raw / raw.max()


def build_power_map(activation: np.ndarray, atlas_data: np.ndarray,
                     region_indices: np.ndarray, affine) -> "nib.Nifti1Image":
    """Projects a region-wise importance vector back onto the AAL volume,
    rescaled to [-1, 1] for a symmetric diverging colormap."""
    scaled = 2 * (activation - activation.min()) / (activation.max() - activation.min() + 1e-12) - 1
    power_map = np.zeros_like(atlas_data, dtype=float)
    for i, region in enumerate(region_indices):
        power_map[atlas_data == region] = scaled[i]
    return nib.Nifti1Image(power_map, affine)


def run_attribution_cv(X: np.ndarray, y: np.ndarray, subject_ids: np.ndarray,
                        n_splits: int = 5, ensemble_random_states: int = 5):
    """Runs subject-wise GroupKFold CV, and for each fold selects the
    ensemble member with the best held-out logistic-regression accuracy,
    then computes all three attribution maps on both train and test data.
    Importance maps are averaged across folds.

    Returns
    -------
    imp_train_all, imp_test_all : dict[str, np.ndarray] keyed by method name
    mean_acc, std_acc : float
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from src.decoding import make_consistency_ensemble

    n_regions = X.shape[1]
    imp_train_all = {m: np.zeros(n_regions) for m in METHODS}
    imp_test_all = {m: np.zeros(n_regions) for m in METHODS}
    accs = []

    gkf = GroupKFold(n_splits=n_splits)
    n_folds = gkf.get_n_splits(X, y, groups=subject_ids)

    for fold_idx, (tr, te) in enumerate(gkf.split(X, y, groups=subject_ids)):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]

        ens = make_consistency_ensemble(ensemble_random_states)
        ens.fit(X_tr, y_tr.astype(int))

        best_acc, best_idx = -1.0, 0
        for i, est in enumerate(ens.estimators_):
            emb_tr, emb_te = est.transform(X_tr), est.transform(X_te)
            acc_i = LogisticRegression(max_iter=1000).fit(emb_tr, y_tr).score(emb_te, y_te)
            if acc_i > best_acc:
                best_acc, best_idx = acc_i, i

        best_model = ens.estimators_[best_idx]
        acc = LogisticRegression(max_iter=1000).fit(
            best_model.transform(X_tr), y_tr
        ).score(best_model.transform(X_te), y_te)
        accs.append(acc)
        print(f"Fold {fold_idx + 1}/{n_folds} | best_idx={best_idx} | acc={acc:.3f}")

        for method in METHODS:
            imp_train_all[method] += get_feature_importance(best_model, X_tr, method, seed=fold_idx)
            imp_test_all[method] += get_feature_importance(best_model, X_te, method, seed=fold_idx)

    for method in METHODS:
        imp_train_all[method] /= n_folds
        imp_test_all[method] /= n_folds

    mean_acc, std_acc = float(np.mean(accs)), float(np.std(accs))
    print(f"Final decoding accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    return imp_train_all, imp_test_all, mean_acc, std_acc
