# DBS-STN HDP Pipeline

De-identified backup of a DBS-STN diffusion MRI, tractography, Lead-DBS,
OSS-DBS, VTA/PAM, and QC workflow.

## Repository scope

This repository stores algorithms, scripts, example configuration, and
workflow documentation only. It intentionally excludes all patient data,
clinical identifiers, generated images, tractograms, simulation outputs, and
tool installations.

```text
configs/   De-identified configuration templates
docs/      Workflow and data-backup documentation
patients/  Empty placeholder; real patient workspaces are ignored
scripts/   MRtrix/FSL/ANTs, Lead-DBS, OSS-DBS, VTA/PAM, and QC utilities
```

## Quick start

The shell pipeline is intended for WSL/Linux with MRtrix3, FSL, and ANTs.
Python utilities additionally use packages listed in `requirements.txt`.
Lead-DBS, OSS-DBS, MATLAB, Slicer, and NEURON are external dependencies for
the relevant stages.

```bash
git clone <private-repository-url>
cd dbs-stn-hdp-pipeline
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

bash scripts/00_check_environment.sh
bash scripts/01_prepare_patient.sh patient_example "$(pwd)"
bash scripts/02_check_inputs.sh patients/patient_example
```

Copy `configs/patient.example.yaml` into the generated patient workspace and
adjust paths and acquisition metadata locally.

## Data safety

Never commit real patient directories or clinical data. This includes DICOM,
NIfTI, tractography, electrode coordinates, screenshots, logs, metrics, and
OSS-DBS/Lead-DBS outputs. See `docs/data_backup.md` for the recommended
encrypted data-backup layout.

Several specialized scripts retain de-identified example paths under
`D:\dbs_workspace`. Override their command-line path arguments or adapt the
constants to the local environment before execution.

## Main workflow

```text
DWI + anatomical MRI + post-op imaging
-> MRtrix/FSL/ANTs preprocessing
-> FOD and ACT tractography
-> Lead-DBS electrode reconstruction
-> HDP/CST/DRTT extraction
-> OSS-DBS VTA/PAM simulation
-> Slicer and quantitative QC
```

This repository is a research workflow backup, not a validated clinical
device. Outputs require expert review before interpretation.
