# Workflow Plan

## Goal

```text
DWI + T1/T2/SWI + post-op CT
-> MRtrix/FSL/ANTs preprocessing in WSL
-> Lead-DBS atlas/electrode/VTA
-> MRtrix HDP/CST/DRTT tractography
-> Lead-DBS/Slicer QC
-> streamline + overlap/distance metrics
```

## Phase 1 - Project Template

Create one isolated workspace per patient:

```text
raw/
derivatives/
qc/
logs/
```

The raw folder stores untouched inputs. All generated files go into derivatives, QC outputs, or logs.

## Phase 2 - Input QC

Check:

- DWI NIfTI, bvec, bval.
- T1/T2/SWI pre-op.
- Post-op CT/MRI for lead reconstruction.
- b-values and number of diffusion volumes.
- Whether reverse phase encoding B0 exists.

## Phase 3 - DWI Preprocessing

Use MRtrix + FSL:

```bash
mrconvert
dwidenoise
mrdegibbs
dwifslpreproc
dwibiascorrect ants
dwi2mask
dwiextract + mrmath for b0_mean
```

Manual QC after this phase:

- b0 distortion.
- brain mask.
- FA/b0 anatomical plausibility.

## Phase 4 - FOD/CSD

Use:

```bash
dwi2response dhollander
dwi2fod msmt_csd
mtnormalise
```

## Phase 5 - Lead-DBS

Use Lead-DBS for:

- Pre-op MRI and post-op CT coregistration.
- Electrode reconstruction.
- MNI normalization.
- DISTAL, Middlebrooks, Treu, Meola atlas references.
- VTA/e-field from real stimulation settings.

## Phase 6 - ROI

Manual/QC-heavy:

- STN motor/dorsolateral ROI.
- M1/SMA/premotor ROI.
- CST ROI: PLIC and cerebral peduncle.
- Exclude masks when tract output is anatomically implausible.

## Phase 7 - Tractography

Standard patient-specific whole-brain tractography:

```bash
scripts/05_prepare_act.sh <patient_dir>
scripts/06_run_wholebrain_tractography.sh <patient_dir>
scripts/06b_run_sift2.sh <patient_dir>
```

This produces:

- ACT 5TT and GMWMI seed.
- Whole-brain iFOD2 ACT tractogram, normally `wholebrain_10000000.tck`.
- SIFT2 weights in `derivatives/sift2`.

ROI tracts should be filtered from the matching whole-brain tractogram. When SIFT2 exists, pass the matching weights through `tckedit` using `-tck_weights_in` and `-tck_weights_out`.

Still later / downstream:

- tract activation and e-field overlap.

## Phase 8 - Metrics

Report:

- raw streamline count;
- SIFT2 weighted sum when available, preferred over raw count for quantitative comparison;
- VTA/e-field overlap;
- active contact to tract distance;
- parameter log for reproducibility.
