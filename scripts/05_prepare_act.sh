#!/usr/bin/env bash
set -euo pipefail

export FSLDIR="${FSLDIR:-/home/devold/fsl}"
if [[ -f "$FSLDIR/etc/fslconf/fsl.sh" ]]; then
  # Required when launched from a non-interactive WSL shell.
  # shellcheck disable=SC1091
  source "$FSLDIR/etc/fslconf/fsl.sh"
fi
export PATH="/home/devold/fsl/share/fsl/bin:$PATH"

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
t1="$(yaml_value anat.t1)"
dwi_dir="$patient_dir/derivatives/dwi"
reg_dir="$patient_dir/derivatives/registration"
qc_dir="$patient_dir/qc/reports"
log_dir="$patient_dir/logs"

mkdir -p "$reg_dir" "$qc_dir" "$log_dir"

log="$log_dir/05_prepare_act.log"
report="$qc_dir/05_prepare_act_qc.md"

exec > >(tee "$log") 2>&1

echo "== ACT / 5TT preparation for $patient_id =="
echo "Date: $(date -Is)"
echo "Patient dir: $patient_dir"
echo "Registration dir: $reg_dir"
echo

b0="$dwi_dir/b0_mean.nii.gz"
dwi="$dwi_dir/dwi_bias.mif"
dwi_mask="$dwi_dir/mask.nii.gz"

for f in "$t1" "$b0" "$dwi"; do
  if [[ -z "$f" || "$f" == "null" || ! -f "$f" ]]; then
    echo "Missing required input: $f" >&2
    exit 1
  fi
done

run() {
  echo
  echo "+ $*"
  "$@"
}

cd "$reg_dir"

run 5ttgen fsl "$t1" 5tt_t1.mif \
  -force

run mrmath 5tt_t1.mif sum 5tt_t1_sum.mif \
  -axis 3 \
  -force

run mrcalc 5tt_t1_sum.mif 0.05 -gt 5tt_t1_mask.mif \
  -force

run mrtransform 5tt_t1_mask.mif t1_brainmask_from5tt.nii.gz \
  -template "$t1" \
  -interp nearest \
  -force

run mrcalc "$t1" t1_brainmask_from5tt.nii.gz -mult t1_brain_from5tt.nii.gz \
  -force

run mrcalc "$b0" "$dwi_mask" -mult b0_brain.nii.gz \
  -force

run flirt \
  -in t1_brain_from5tt.nii.gz \
  -ref b0_brain.nii.gz \
  -dof 6 \
  -cost normmi \
  -searchrx -90 90 \
  -searchry -90 90 \
  -searchrz -90 90 \
  -omat t12dwi_fsl.mat \
  -out t1_brain_in_dwi.nii.gz

run flirt \
  -in "$b0" \
  -ref "$t1" \
  -dof 6 \
  -cost normmi \
  -omat dwi2t1_fsl.mat \
  -out b0_in_t1.nii.gz

run transformconvert t12dwi_fsl.mat t1_brain_from5tt.nii.gz b0_brain.nii.gz flirt_import t12dwi_mrtrix.txt \
  -force

run mrtransform 5tt_t1.mif 5tt_dwi.mif \
  -linear t12dwi_mrtrix.txt \
  -template "$dwi" \
  -interp linear \
  -force

run 5tt2gmwmi 5tt_dwi.mif gmwmi_seed.mif \
  -force

run mrconvert 5tt_dwi.mif 5tt_dwi.nii.gz -force
run mrconvert gmwmi_seed.mif gmwmi_seed.nii.gz -force
run 5tt2vis 5tt_dwi.mif 5tt_dwi_vis.mif -force
run mrconvert 5tt_dwi_vis.mif 5tt_dwi_vis.nii.gz -force

cat > "$report" <<EOF
# ACT / 5TT QC - $patient_id

Generated: $(date -Is)

## Inputs

- T1: \`$t1\`
- Mean b0: \`$b0\`
- DWI: \`$dwi\`

## Outputs

| Output | Path |
|---|---|
| 5TT in T1 space | \`$reg_dir/5tt_t1.mif\` |
| T1 brain mask from 5TT | \`$reg_dir/t1_brainmask_from5tt.nii.gz\` |
| T1 brain registered to DWI | \`$reg_dir/t1_brain_in_dwi.nii.gz\` |
| Mean b0 registered to T1 | \`$reg_dir/b0_in_t1.nii.gz\` |
| FSL T1-to-DWI matrix | \`$reg_dir/t12dwi_fsl.mat\` |
| MRtrix T1-to-DWI transform | \`$reg_dir/t12dwi_mrtrix.txt\` |
| 5TT in DWI space | \`$reg_dir/5tt_dwi.nii.gz\` |
| 5TT visual QC volume | \`$reg_dir/5tt_dwi_vis.nii.gz\` |
| GM-WM interface seed | \`$reg_dir/gmwmi_seed.nii.gz\` |

## Manual QC Checklist

- [ ] Inspect \`t1_brain_in_dwi.nii.gz\` over \`b0_mean.nii.gz\`; cortex, ventricles, and brainstem should align acceptably.
- [ ] Inspect \`5tt_dwi.nii.gz\` over \`b0_mean.nii.gz\`; tissue boundaries should roughly match DWI.
- [ ] Inspect \`5tt_dwi_vis.nii.gz\` over \`fa.nii.gz\`; 5TT should cover cortex and brainstem in DWI space.
- [ ] Inspect \`gmwmi_seed.nii.gz\`; seed should follow cortical and deep GM-WM interface, not sit outside brain.
- [ ] If QC is acceptable, proceed to ACT whole-brain tractography.

EOF

echo
echo "ACT preparation complete."
echo "Report: $report"
