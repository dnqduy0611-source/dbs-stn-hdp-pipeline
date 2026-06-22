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
wholebrain_count="${wholebrain_count:-10000000}"

fod_dir="$patient_dir/derivatives/fod"
reg_dir="$patient_dir/derivatives/registration"
tract_dir="$patient_dir/derivatives/tracts"
sift2_dir="$patient_dir/derivatives/sift2"
qc_dir="$patient_dir/qc/reports"
log_dir="$patient_dir/logs"

mkdir -p "$sift2_dir" "$qc_dir" "$log_dir"

log="$log_dir/06b_run_sift2.log"
report="$qc_dir/06b_sift2_qc.md"

exec > >(tee "$log") 2>&1

echo "== SIFT2 whole-brain filtering for $patient_id =="
echo "Date: $(date -Is)"
echo "Patient dir: $patient_dir"
echo "SIFT2 dir: $sift2_dir"
echo "Whole-brain target: $wholebrain_count"
echo

wholebrain="$tract_dir/wholebrain_${wholebrain_count}.tck"
wmfod="$fod_dir/wmfod_norm.mif"
act_5tt="$reg_dir/5tt_dwi.mif"
weights="$sift2_dir/wholebrain_${wholebrain_count}_sift2_weights.txt"
mu="$sift2_dir/wholebrain_${wholebrain_count}_sift2_mu.txt"
coeffs="$sift2_dir/wholebrain_${wholebrain_count}_sift2_coeffs.txt"

for f in "$wholebrain" "$wmfod" "$act_5tt"; do
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

run tcksift2 "$wholebrain" "$wmfod" "$weights" \
  -act "$act_5tt" \
  -out_mu "$mu" \
  -out_coeffs "$coeffs" \
  -force

weight_count="$(grep -v '^#' "$weights" | wc -w | tr -d ' ')"
track_count="$(tckinfo "$wholebrain" | awk -F: '/^    count:/ {gsub(/ /, "", $2); print $2}')"

echo
echo "Weight count: $weight_count"
echo "Track count: $track_count"

if [[ "$weight_count" != "$track_count" ]]; then
  echo "SIFT2 weight count does not match tract count." >&2
  exit 1
fi

cat > "$report" <<EOF
# SIFT2 QC - $patient_id

Generated: $(date -Is)

## Inputs

- Whole-brain tractogram: \`$wholebrain\`
- WM FOD: \`$wmfod\`
- ACT 5TT: \`$act_5tt\`

## Outputs

| Output | Path |
|---|---|
| SIFT2 weights | \`$weights\` |
| SIFT2 mu | \`$mu\` |
| SIFT2 coefficients | \`$coeffs\` |
| Log | \`$log\` |

## Verification

- Track count from \`tckinfo\`: \`$track_count\`
- SIFT2 weight values: \`$weight_count\`
- Status: weight count matches tract count.

## Notes

- Use these weights when filtering from the matching whole-brain tractogram.
- For ROI-derived tracts, pass \`-tck_weights_in\` with this weights file and \`-tck_weights_out\` for the filtered output.
- Report weighted sums for quantitative comparison; keep raw streamline counts as QC only.

EOF

echo
echo "SIFT2 complete."
echo "Report: $report"
