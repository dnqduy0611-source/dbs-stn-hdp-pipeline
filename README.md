# DBS-STN tractography & electrode–tract analysis pipeline

Research code from a clinical study of subthalamic nucleus deep brain stimulation
(STN-DBS) for Parkinson's disease, using a Medtronic SenSight directional lead (B33005).
The pipeline goes from pre-operative diffusion MRI and post-operative CT to
patient-specific fibre tracts, electrode contact positions, stimulation-volume
models, and statistics against clinical outcome (MDS-UPDRS III).

> Code only. No patient images, tractograms, coordinates or clinical scores are
> included. Subjects are referred to by pseudonymous codes (`sub-01` … `sub-10`).
> This is research software, not a validated clinical tool.

## What it does

```text
DWI (multi-/single-shell) + T1/T2/SWI          post-op CT
        │                                          │
  preprocessing (MRtrix3, FSL, ANTs, Synb0)   electrode localisation
        │                                     (Lead-DBS / DiODe, CT axis QC)
  FOD (MSMT-CSD / SS3T) → ACT → 10M streamlines + SIFT2
        │                                          │
  tract extraction: HDP · CST · medial lemniscus · DRTT
        │                                          │
        └──────── contact-to-tract distances ──────┘
                           │
         OSS-DBS volume of tissue activated / pathway activation (NEURON)
                           │
            statistics vs. clinical improvement (UPDRS III)
```

## Repository layout

| folder | contents |
|---|---|
| `scripts/` | numbered shell pipeline (`00`–`07`), single-shell variants, fine-grid 5TT, CT/DiODe electrode utilities, OSS-DBS VTA/PAM drivers |
| `tools/cst_ml/` | config-driven toolkit that builds the corticospinal tract and medial lemniscus for any case from one `cases/<case>.conf` file, with QC gates |
| `tractography/` | streamline filters: tilted midsagittal-plane fit, midline-crossing pruning, length/tortuosity pruning, physiological HDP constraints |
| `analysis/` | electrode–tract distance measures, cluster bootstrap and block-permutation tests, measurement-robustness and specificity tests |
| `deid/` | two-pass de-identification gate for imaging archives (binary-safe `.tck`/`.mif` header rewriting) |
| `configs/`, `docs/` | example configuration and workflow notes |

## Design choices

- **Every measurement in native patient space.** MNI space is only used to carry
  atlas ROIs into the patient's T1; distances are never measured in template space.
- **QC gates return numbers.** Each stage has a gate (registration landmark error,
  ROI volume/side checks, NaN checks before optimisation, an anatomical
  anterior–posterior check of the CST against the red nucleus). Steps that need a
  human decision are not auto-approved.
- **Thresholds are measured per case.** Fixed absolute cut-offs (e.g. a depth limit
  in mm) failed when moved between patients with different fields of view, so
  filters anchor to each patient's own anatomy instead.
- **Clustered statistics.** Two hemispheres per patient are not independent, so
  inference uses patient-level cluster bootstrap and block permutation rather than
  naive per-contact p-values.

## Requirements

WSL/Linux with MRtrix3 (3.0.x), FSL, ANTs; Python 3 with `nibabel`, `numpy`,
`scipy`, `pandas` (see `requirements.txt`). Lead-DBS/DiODe (MATLAB), OSS-DBS,
NEURON and 3D Slicer are needed for the corresponding stages.

```bash
bash scripts/00_check_environment.sh
bash scripts/01_prepare_patient.sh patient_example "$(pwd)"

# CST/ML toolkit for one case
cp tools/cst_ml/cases/example.conf tools/cst_ml/cases/sub-01.conf   # then edit paths
export DBS_ROOT=/path/to/project
bash tools/cst_ml/fit_msp.sh sub-01
bash tools/cst_ml/build_cst_ml_rois.sh sub-01
python3 tools/cst_ml/qc_cst_ml_rois.py <roi_dir> sub-01
bash tools/cst_ml/build_cst.sh sub-01
bash tools/cst_ml/build_ml.sh sub-01
```

Some scripts were written for one specific case and still contain example paths;
pass paths as arguments or edit the constants before running. Many in-code
comments are in Vietnamese (without diacritics); they record why each step and
threshold was chosen.

## Data safety

Never commit patient directories, DICOM/NIfTI, tractograms, electrode
coordinates, screenshots, logs or simulation outputs. The identity map
(`PATIENT_ID_MAP.csv`) is listed in `.gitignore` and must stay offline.
See `docs/data_backup.md`.

## License

MIT, see `LICENSE`.
