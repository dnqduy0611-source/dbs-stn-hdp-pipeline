#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <patient_id> [pipeline_root]"
  echo "Example: $0 patient_example /mnt/d/dbs_workspace/dbs_stn_hdp_pipeline"
  exit 2
fi

patient_id="$1"
pipeline_root="${2:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
patient_dir="$pipeline_root/patients/$patient_id"

mkdir -p \
  "$patient_dir/raw/dwi" \
  "$patient_dir/raw/anat" \
  "$patient_dir/raw/ct" \
  "$patient_dir/derivatives/dwi" \
  "$patient_dir/derivatives/fod" \
  "$patient_dir/derivatives/registration" \
  "$patient_dir/derivatives/rois" \
  "$patient_dir/derivatives/tracts" \
  "$patient_dir/derivatives/lead_dbs" \
  "$patient_dir/derivatives/metrics" \
  "$patient_dir/qc/screenshots" \
  "$patient_dir/qc/reports" \
  "$patient_dir/logs"

cat > "$patient_dir/README.md" <<EOF
# Patient Workspace - $patient_id

## Raw Inputs

Place untouched input files here:

\`\`\`text
raw/dwi/dwi.nii.gz
raw/dwi/dwi.bvec
raw/dwi/dwi.bval
raw/anat/t1.nii.gz
raw/anat/t2_or_swi.nii.gz
raw/ct/postop_ct.nii.gz
\`\`\`

## Generated Outputs

- \`derivatives/dwi\`: DWI preprocessing outputs.
- \`derivatives/fod\`: response functions and FOD files.
- \`derivatives/registration\`: transforms and resampled images.
- \`derivatives/rois\`: ROI masks in patient/DWI space.
- \`derivatives/tracts\`: TCK/VTK tract files.
- \`derivatives/lead_dbs\`: exports from Lead-DBS.
- \`derivatives/metrics\`: CSV/JSON tables.
- \`qc\`: screenshots and QC reports.
- \`logs\`: command logs.

## Manual QC Status

- [ ] Raw DWI checked.
- [ ] b0/mask checked.
- [ ] DWI to T1/T2 registration checked.
- [ ] Lead-DBS electrode reconstruction checked.
- [ ] Atlas warp checked.
- [ ] CST checked.
- [ ] HDP checked.
EOF

if [[ ! -f "$patient_dir/config.yaml" ]]; then
  sed "s|patient_id: patient_example|patient_id: $patient_id|; s|base_dir: .*|base_dir: $patient_dir|" \
    "$pipeline_root/configs/patient.example.yaml" > "$patient_dir/config.yaml"
fi

echo "Prepared patient workspace:"
echo "$patient_dir"
