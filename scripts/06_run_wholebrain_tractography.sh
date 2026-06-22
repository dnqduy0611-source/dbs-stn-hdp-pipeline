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
wholebrain_count="$(yaml_value tractography.wholebrain_count)"
cutoff="$(yaml_value tractography.cutoff)"
minlength="$(yaml_value tractography.minlength)"
maxlength="$(yaml_value tractography.maxlength)"

wholebrain_count="${wholebrain_count:-10000000}"
cutoff="${cutoff:-0.06}"
minlength="${minlength:-10}"
maxlength="${maxlength:-200}"

fod_dir="$patient_dir/derivatives/fod"
reg_dir="$patient_dir/derivatives/registration"
tract_dir="$patient_dir/derivatives/tracts"
qc_dir="$patient_dir/qc/reports"
log_dir="$patient_dir/logs"

mkdir -p "$tract_dir" "$qc_dir" "$log_dir"

log="$log_dir/06_run_wholebrain_tractography.log"
report="$qc_dir/06_wholebrain_tractography_qc.md"

exec > >(tee "$log") 2>&1

echo "== ACT whole-brain tractography for $patient_id =="
echo "Date: $(date -Is)"
echo "Patient dir: $patient_dir"
echo "Tract dir: $tract_dir"
echo "Streamline target: $wholebrain_count"
echo

wmfod="$fod_dir/wmfod_norm.mif"
act_5tt="$reg_dir/5tt_dwi.mif"
gmwmi="$reg_dir/gmwmi_seed.mif"

for f in "$wmfod" "$act_5tt" "$gmwmi"; do
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

cd "$tract_dir"

run tckgen "$wmfod" wholebrain_${wholebrain_count}.tck \
  -algorithm iFOD2 \
  -act "$act_5tt" \
  -backtrack \
  -crop_at_gmwmi \
  -seed_gmwmi "$gmwmi" \
  -select "$wholebrain_count" \
  -cutoff "$cutoff" \
  -minlength "$minlength" \
  -maxlength "$maxlength" \
  -force

run tckinfo wholebrain_${wholebrain_count}.tck
tckinfo wholebrain_${wholebrain_count}.tck > wholebrain_${wholebrain_count}_tckinfo.txt

cat > "$report" <<EOF
# ACT Whole-Brain Tractography QC - $patient_id

Generated: $(date -Is)

## Inputs

- WM FOD: \`$wmfod\`
- ACT 5TT: \`$act_5tt\`
- GMWMI seed: \`$gmwmi\`

## Parameters

- Algorithm: \`iFOD2\`
- Target streamlines: \`$wholebrain_count\`
- Cutoff: \`$cutoff\`
- Min length: \`$minlength\`
- Max length: \`$maxlength\`
- ACT: enabled
- Backtrack: enabled
- Crop at GMWMI: enabled

## Outputs

| Output | Path |
|---|---|
| Whole-brain tractogram | \`$tract_dir/wholebrain_${wholebrain_count}.tck\` |
| TCK info | \`$tract_dir/wholebrain_${wholebrain_count}_tckinfo.txt\` |

## Manual QC Checklist

- [ ] Open tractogram in MRView if possible, or convert subsets for Slicer visualization.
- [ ] Confirm streamlines fill major WM pathways without extensive outside-brain tracks.
- [ ] Check CST/internal capsule and corpus callosum plausibility before HDP ROI filtering.
- [ ] Do not interpret raw streamline count as biological connectivity before SIFT/SIFT2 or ROI metrics.

EOF

echo
echo "Whole-brain tractography complete."
echo "Report: $report"
