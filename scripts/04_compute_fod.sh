#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <patient_dir>"
  echo "Example: $0 /mnt/d/dbs_workspace/dbs_stn_hdp_pipeline/patients/patient_example"
  exit 2
fi

patient_dir="$(cd "$1" && pwd)"
config="$patient_dir/config.yaml"

if [[ ! -f "$config" ]]; then
  echo "Missing config: $config" >&2
  exit 1
fi

yaml_value() {
  local key_path="$1"
  python3 - "$config" "$key_path" <<'PY'
from pathlib import Path
import sys

config = Path(sys.argv[1]).read_text().splitlines()
keys = sys.argv[2].split(".")
stack = []
values = {}

for raw in config:
    if not raw.strip() or raw.lstrip().startswith("#"):
        continue
    indent = len(raw) - len(raw.lstrip(" "))
    line = raw.strip()
    if ":" not in line or line.startswith("- "):
        continue
    key, value = line.split(":", 1)
    key = key.strip()
    value = value.strip()
    level = indent // 2
    stack = stack[:level]
    stack.append(key)
    full = ".".join(stack)
    if value:
        values[full] = value.strip("'\"")

print(values.get(".".join(keys), ""))
PY
}

patient_id="$(yaml_value patient_id)"
dwi_dir="$patient_dir/derivatives/dwi"
fod_dir="$patient_dir/derivatives/fod"
qc_dir="$patient_dir/qc/reports"
log_dir="$patient_dir/logs"

mkdir -p "$fod_dir" "$qc_dir" "$log_dir"

log="$log_dir/04_compute_fod.log"
report="$qc_dir/04_compute_fod_qc.md"

exec > >(tee "$log") 2>&1

echo "== FOD / MSMT-CSD for $patient_id =="
echo "Date: $(date -Is)"
echo "Patient dir: $patient_dir"
echo "DWI dir: $dwi_dir"
echo "FOD dir: $fod_dir"
echo

dwi="$dwi_dir/dwi_bias.mif"
mask="$dwi_dir/mask.mif"

for f in "$dwi" "$mask"; do
  if [[ ! -f "$f" ]]; then
    echo "Missing required input: $f" >&2
    exit 1
  fi
done

run() {
  echo
  echo "+ $*"
  "$@"
}

cd "$fod_dir"

run dwi2response dhollander "$dwi" wm_response.txt gm_response.txt csf_response.txt \
  -mask "$mask" \
  -voxels response_voxels.mif \
  -force

run dwi2fod msmt_csd "$dwi" \
  wm_response.txt wmfod.mif \
  gm_response.txt gm.mif \
  csf_response.txt csf.mif \
  -mask "$mask" \
  -force

run mtnormalise \
  wmfod.mif wmfod_norm.mif \
  gm.mif gm_norm.mif \
  csf.mif csf_norm.mif \
  -mask "$mask" \
  -force

run fod2dec wmfod_norm.mif wmfod_dec.mif \
  -mask "$mask" \
  -force

run mrconvert response_voxels.mif response_voxels.nii.gz -force
run mrconvert gm_norm.mif gm_norm.nii.gz -force
run mrconvert csf_norm.mif csf_norm.nii.gz -force
run mrconvert wmfod_dec.mif wmfod_dec.nii.gz -force

cat > "$report" <<EOF
# FOD / MSMT-CSD QC - $patient_id

Generated: $(date -Is)

## Inputs

- Bias-corrected DWI: \`$dwi\`
- Brain mask: \`$mask\`

## Outputs

| Output | Path |
|---|---|
| WM response | \`$fod_dir/wm_response.txt\` |
| GM response | \`$fod_dir/gm_response.txt\` |
| CSF response | \`$fod_dir/csf_response.txt\` |
| Response voxels | \`$fod_dir/response_voxels.nii.gz\` |
| WM FOD | \`$fod_dir/wmfod.mif\` |
| Normalized WM FOD | \`$fod_dir/wmfod_norm.mif\` |
| Normalized GM map | \`$fod_dir/gm_norm.nii.gz\` |
| Normalized CSF map | \`$fod_dir/csf_norm.nii.gz\` |
| FOD direction-encoded color preview | \`$fod_dir/wmfod_dec.nii.gz\` |

## Manual QC Checklist

- [ ] Inspect response voxel locations; WM voxels should be in coherent white matter.
- [ ] Inspect \`wmfod_dec.nii.gz\` for plausible corpus callosum/internal capsule directions.
- [ ] Confirm no obvious left-right or anterior-posterior direction issue.
- [ ] If FOD QC is acceptable, proceed to whole-brain tractography.

EOF

echo
echo "FOD computation complete."
echo "Report: $report"
