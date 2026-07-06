# Data

**The fMRI data used in this study are not included in this repository
and cannot be publicly redistributed**, due to institutional data-sharing
and participant-consent restrictions (propofol dataset: University
Hospital of the Technical University of Munich; sevoflurane dataset:
previously published third-party study, re-analyzed here — see the
paper's Data section and references [4], [28] for provenance).

This folder documents the expected file formats. Files placed here are
git-ignored and never committed.

## Expected layout

```
data/
├── propofol/
│   ├── propofol_flickerdata_aal.pt   # torch tensor, shape (n_runs, 116, T)
│   ├── propofol_run_ids.pt           # torch tensor, shape (n_runs,), values in {1,2,3,4}
│   └── propofol_subject_ids.pt       # torch tensor, shape (n_runs,), integer subject id
├── sevoflurane/
│   └── data.npy                      # shape (n_sessions=5, n_subjects, 116, T)
└── atlas/
    └── aal_SPM12/aal_for_SPM12/aal/
        ├── atlas/AAL.nii             # AAL v3v2 atlas volume
        └── ROI_MNI_V4.txt            # region index / name lookup table
```

State label convention (see `config.py::CLASS_NAMES`):

| Label | Meaning |
|---|---|
| 1 | Awake |
| 2 | Low (light sedation / concentration) |
| 3 | Moderate |
| 4 | Deep |

For the sevoflurane `.npy` file, sessions are stored in acquisition
order and are remapped to the labels above via
`config.SEVOFLURANE_SESSION_TO_LABEL`.

## Getting your own ROI time series into this format

If you have your own preprocessed resting-state fMRI data (motion
correction, spatial normalization, nuisance regression already applied
— see the paper's Methods, "fMRI Preprocessing and ROI Time Series
Extraction"), you need to:

1. Parcellate each subject's 4D BOLD volume with the **AAL v3v2** atlas
   (116 regions), spatially averaging BOLD signal within each region at
   every TR.
2. Z-score each region's time series.
3. Stack runs into a `(n_runs, 116, T)` array and save subject/label
   metadata alongside it, matching the layout above.

## Configuring paths

Point the code at your local data location either via environment
variables:

```bash
export CEBRA_PROPOFOL_DIR=/path/to/data/propofol
export CEBRA_SEVOFLURANE_FILE=/path/to/data/sevoflurane/data.npy
export CEBRA_AAL_ATLAS_DIR=/path/to/data/atlas/aal_SPM12/aal_for_SPM12/aal
```

or by placing the files directly under `data/` as shown above (the
defaults in `config.py` already point here).
