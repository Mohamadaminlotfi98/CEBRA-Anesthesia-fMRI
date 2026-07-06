"""
Data loading and reshaping utilities.

These functions expect that fMRI preprocessing (motion correction,
spatial normalization, AAL v3v2 parcellation, nuisance regression,
z-scoring -- see Methods: "fMRI Preprocessing and ROI Time Series
Extraction") has already been performed upstream (e.g. with fMRIPrep /
CONN / SPM), producing a per-subject ROI-by-time matrix for every run.

The raw fMRI data used in the paper are not publicly available
(institutional data-sharing restrictions). This module only defines the
*shape contract* the rest of the pipeline expects, so you can point it
at your own ROI-extracted data. See data/README.md for details.
"""

from __future__ import annotations

import os
import numpy as np
import torch

from config import PROPOFOL_DIR, SEVOFLURANE_FILE, SEVOFLURANE_SESSION_TO_LABEL


def load_propofol(propofol_dir: str | None = None):
    """Load propofol ROI time series.

    Expects three files in `propofol_dir`:
        propofol_flickerdata_aal.pt  -> (n_runs, N_ROIs, T) float tensor
        propofol_run_ids.pt          -> (n_runs,) int labels in {1,2,3,4}
        propofol_subject_ids.pt      -> (n_runs,) int subject ids

    Returns
    -------
    data   : np.ndarray, shape (n_runs, N_ROIs, T)
    labels : np.ndarray, shape (n_runs,)
    subs   : np.ndarray, shape (n_runs,)
    """
    base = propofol_dir or PROPOFOL_DIR
    data = torch.load(os.path.join(base, "propofol_flickerdata_aal.pt"), weights_only=False).numpy()
    labels = torch.load(os.path.join(base, "propofol_run_ids.pt"), weights_only=False).numpy()
    subs = torch.load(os.path.join(base, "propofol_subject_ids.pt"), weights_only=False).numpy()
    return data, labels, subs


def load_sevoflurane(sevoflurane_file: str | None = None):
    """Load sevoflurane ROI time series.

    Expects a single .npy file of shape (n_sessions, n_subjects, N_ROIs, T).
    Session order is mapped to harmonized state labels via
    `SEVOFLURANE_SESSION_TO_LABEL` (see Methods / Data section of the paper).

    Returns
    -------
    data   : np.ndarray, shape (n_runs, N_ROIs, T)
    labels : np.ndarray, shape (n_runs,)
    subs   : np.ndarray, shape (n_runs,)
    """
    path = sevoflurane_file or SEVOFLURANE_FILE
    raw = np.load(path)
    data = raw.transpose(1, 0, 2, 3).reshape(-1, raw.shape[2], raw.shape[3])
    n_sess = raw.shape[0]
    n_subj = data.shape[0] // n_sess
    labels = np.vectorize(SEVOFLURANE_SESSION_TO_LABEL.get)(np.repeat(np.arange(n_sess), n_subj))
    subs = np.tile(np.arange(n_subj), n_sess)
    return data, labels, subs


def data_handler(data: np.ndarray, labels: np.ndarray, subs: np.ndarray):
    """Flatten (n_runs, N_ROIs, T) run-wise arrays into per-timepoint arrays.

    Returns
    -------
    X : np.ndarray, shape (sum(T_i), N_ROIs)
    y : np.ndarray, shape (sum(T_i),)   -- state label per timepoint
    s : np.ndarray, shape (sum(T_i),)   -- subject id per timepoint
    """
    wd, wl, ws = [], [], []
    for i in range(data.shape[0]):
        T = data.shape[2]
        wd.append(data[i].T)
        wl.append(np.repeat(labels[i], T))
        ws.append(np.repeat(subs[i], T))
    return np.concatenate(wd), np.concatenate(wl), np.concatenate(ws)


def load_and_flatten(dataset: str):
    """Convenience wrapper: load a dataset by name and flatten it.

    Parameters
    ----------
    dataset : "propofol" or "sevoflurane"
    """
    if dataset == "propofol":
        data, labels, subs = load_propofol()
    elif dataset == "sevoflurane":
        data, labels, subs = load_sevoflurane()
    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Use 'propofol' or 'sevoflurane'.")
    return data_handler(data, labels, subs)
