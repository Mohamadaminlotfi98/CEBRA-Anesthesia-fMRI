# CEBRA-Anesthesia: Latent Trajectories of Propofol- and Sevoflurane-Induced Unconsciousness

CEBRA (contrastive representation learning) applied to raw resting-state
fMRI signals under propofol and sevoflurane anesthesia. Instantaneous
fMRI frames are mapped to a 3D latent space that tracks anesthetic
depth in a subject-generalizable way: propofol traces a continuous,
overlapping trajectory through the state space, while sevoflurane
traces a discrete, stepwise one.

> **Data availability.** The fMRI datasets analyzed here are **not**
> included in this repository (institutional / third-party data-sharing
> restrictions). This repository contains only the analysis code. See
> [`data/README.md`](data/README.md) for the expected data format.

## Repository structure

```
.
├── config.py                    # all paths / hyperparameters (no private info)
├── src/
│   ├── data_loading.py          # load + reshape ROI time series
│   ├── cebra_model.py           # sklearn-style CEBRA wrapper (Methods d,e)
│   ├── decoding.py               # subject-wise CV + logistic-regression decoding
│   ├── plotting_decoding.py     # Figures 4-6 (ROC / confusion / PR curves)
│   ├── consistency.py            # embedding consistency via ituna (Fig. 7)
│   ├── plotting_consistency.py  # Figure 7
│   ├── attribution.py            # Jacobian / Neuron-Gradient / Feature-Ablation (Figs 8-9)
│   ├── plotting_attribution.py  # Figure 9 brain maps
│   └── viz_latent_space.py      # Procrustes + PCA 3D trajectory plots (Fig. 3)
├── scripts/
│   ├── 01_train_and_decode.py       # → Figures 3, 4, 5, 6
│   ├── 02_consistency_analysis.py   # → Figure 7
│   └── 03_regional_attribution.py   # → Figure 9
├── data/README.md                # data format + how to point code at your data
├── environment.yml / requirements.txt
└── figures/                      # outputs land here (git-ignored)
```

This mirrors the paper's Methods/Results structure:

| Paper section | Code |
|---|---|
| fMRI Preprocessing & ROI extraction | performed upstream (motion correction, spatial normalization, AAL parcellation, nuisance regression); see `data/README.md` |
| Label-Guided Manifold Learning via CEBRA | `src/cebra_model.py` |
| Latent space decoding (Figs 3–6) | `src/decoding.py`, `src/viz_latent_space.py`, `src/plotting_decoding.py` |
| Embedding consistency / iTuna (Fig. 7) | `src/consistency.py`, `src/plotting_consistency.py` |
| Regional attribution (Figs 8–9) | `src/attribution.py`, `src/plotting_attribution.py` |

## Installation

```bash
git clone https://github.com/Mohamadaminlotfi98/CEBRA-Anesthesia-fMRI.git
cd CEBRA-Anesthesia-fMRI

# option A: conda
conda env create -f environment.yml
conda activate cebra-anesthesia

# option B: pip
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install git+https://github.com/dynamical-inference/ituna.git
```

## Configuring your data paths

Either drop your ROI-extracted files into `data/` following the layout
in [`data/README.md`](data/README.md), or export environment variables:

```bash
export CEBRA_PROPOFOL_DIR=/path/to/propofol_roi
export CEBRA_SEVOFLURANE_FILE=/path/to/sevoflurane/data.npy
export CEBRA_AAL_ATLAS_DIR=/path/to/aal_SPM12/aal_for_SPM12/aal
```

## Running the analysis / generating figures

```bash
# 1. Train CEBRA embeddings, decode anesthesia state, produce Figures 3-6
python scripts/01_train_and_decode.py --dataset both

# 2. Embedding-consistency analysis (Figure 7)
python scripts/02_consistency_analysis.py

# 3. Regional attribution brain maps (Figure 9; run per dataset)
python scripts/03_regional_attribution.py --dataset propofol
python scripts/03_regional_attribution.py --dataset sevoflurane
```

All figures are written to `figures/`, and intermediate results
(cross-validated embeddings, decoding stats, attribution maps) are
cached as `.pkl` files in `results/` so you don't need to retrain CEBRA
every time you tweak a plot.

Training is GPU-accelerated if available (`device="cuda_if_available"`
in `config.CEBRA_PARAMS`); on CPU, 5-fold CV with the default 5000
iterations per fold will be considerably slower.

## Reproducibility notes

- CEBRA hyperparameters exactly match the Methods section
  (`offset10-model-mse`, 3D output, batch size 512, lr 5e-4, 5000
  iterations, temperature 0.1, 512 hidden units) and live in one place:
  `config.CEBRA_PARAMS`.
- All cross-validation is subject-wise (`GroupKFold`), so no timepoints
  from a held-out subject ever appear in training.
- The `ConsistencyEnsemble` (ituna) trains multiple random seeds per
  fold and aligns them under an affine transform before averaging /
  scoring, to make the reported embeddings robust to stochastic
  optimization.

## License

Code is released under the MIT License (see `LICENSE`). The underlying
fMRI datasets are not covered by this license and are not distributed
with this repository.
