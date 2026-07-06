"""
Central configuration for the CEBRA anesthesia-fMRI pipeline.

All machine-specific / dataset-specific paths are read from environment
variables, with local defaults under `data/`. Set them before running a
script, e.g.:

    export CEBRA_PROPOFOL_DIR=/path/to/propofol_roi
    export CEBRA_SEVOFLURANE_FILE=/path/to/sevoflurane/data.npy

See data/README.md for the full list and expected file formats.
"""

import os
from pathlib import Path

# ----------------------------------------------------------------------
# Project root / output locations
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
FIGURES_DIR = Path(os.environ.get("CEBRA_FIGURES_DIR", PROJECT_ROOT / "figures"))
RESULTS_DIR = Path(os.environ.get("CEBRA_RESULTS_DIR", PROJECT_ROOT / "results"))
CACHE_DIR = Path(os.environ.get("CEBRA_CACHE_DIR", PROJECT_ROOT / ".cache"))

for _d in (FIGURES_DIR, RESULTS_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# Private, dataset-specific paths (NEVER hardcode real paths here)
# ----------------------------------------------------------------------
# Propofol dataset directory. Must contain:
#   propofol_flickerdata_aal.pt   -> shape (n_runs, N_ROIs, T)
#   propofol_run_ids.pt           -> shape (n_runs,), state label per run
#   propofol_subject_ids.pt       -> shape (n_runs,), subject id per run
PROPOFOL_DIR = os.environ.get("CEBRA_PROPOFOL_DIR", "data/propofol")

# Sevoflurane dataset file: shape (n_sessions, n_subjects, N_ROIs, T)
SEVOFLURANE_FILE = os.environ.get(
    "CEBRA_SEVOFLURANE_FILE", "data/sevoflurane/data.npy"
)

# AAL v3v2 atlas directory. Must contain:
#   atlas/AAL.nii
#   ROI_MNI_V4.txt
AAL_ATLAS_DIR = os.environ.get("CEBRA_AAL_ATLAS_DIR", "data/atlas/aal_SPM12/aal_for_SPM12/aal")

# ----------------------------------------------------------------------
# Label conventions used throughout the paper
# ----------------------------------------------------------------------
CLASS_NAMES = {1: "Awake", 2: "Low", 3: "Moderate", 4: "Deep"}
CLASS_COLORS = {1: "#4dac26", 2: "#f1b6da", 3: "#d01c8b", 4: "#313695"}

# Sevoflurane raw session order -> harmonized state label (see Methods, Data section)
SEVOFLURANE_SESSION_TO_LABEL = {0: 1, 1: 4, 2: 3, 3: 2, 4: 1}

# ----------------------------------------------------------------------
# CEBRA / training hyperparameters (Methods: Network Architecture section)
# ----------------------------------------------------------------------
CEBRA_PARAMS = dict(
    model_architecture="offset10-model-mse",
    output_dimension=3,
    batch_size=512,
    learning_rate=0.0005,
    max_iterations=5000,
    conditional="discrete",
    distance="euclidean",
    num_hidden_units=512,
    time_offsets=10,
    temperature=0.1,
    device="cuda_if_available",
    verbose=False,
)

N_CV_FOLDS = 5
RANDOM_SEED = 42
