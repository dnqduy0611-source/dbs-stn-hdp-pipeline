#!/usr/bin/env bash
set -euo pipefail

export FSLDIR="${FSLDIR:-/home/devold/fsl}"
if [[ -f "$FSLDIR/etc/fslconf/fsl.sh" ]]; then
  # Required when the script is launched by a non-interactive WSL shell.
  # shellcheck disable=SC1091
  source "$FSLDIR/etc/fslconf/fsl.sh"
fi
export PATH="/home/devold/fsl/share/fsl/bin:/home/devold/ants_env/bin:$PATH"

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
dwi_nifti="$(yaml_value dwi.nifti)"
dwi_bvec="$(yaml_value dwi.bvec)"
dwi_bval="$(yaml_value dwi.bval)"
dwi_json="$(yaml_value dwi.sidecar_json)"
pe_dir="$(yaml_value dwi.pe_dir)"
rpe_mode="$(yaml_value dwi.rpe_mode)"
reverse_phase_b0="$(yaml_value dwi.reverse_phase_b0)"
readout_time="$(yaml_value dwi.readout_time)"

if [[ -z "$readout_time" && -n "$dwi_json" && "$dwi_json" != "null" && -f "$dwi_json" ]]; then
  readout_time="$(python3 - "$dwi_json" <<'PY'
from pathlib import Path
import json
import sys

value = json.loads(Path(sys.argv[1]).read_text()).get("TotalReadoutTime", "")
print(value if value is not None else "")
PY
)"
fi

readout_args=()
if [[ -n "$readout_time" && "$readout_time" != "null" ]]; then
  readout_args=(-readout_time "$readout_time")
fi

out_dir="$patient_dir/derivatives/dwi"
qc_dir="$patient_dir/qc/reports"
log_dir="$patient_dir/logs"
scratch_root="${DBS_STN_HDP_SCRATCH:-/home/devold/dbs_stn_hdp_scratch}"
scratch_dir="$scratch_root/${patient_id}_tmp"
mkdir -p "$out_dir" "$qc_dir" "$log_dir"

log="$log_dir/03_preprocess_dwi.log"
report="$qc_dir/03_preprocess_dwi_qc.md"

exec > >(tee "$log") 2>&1

echo "== DWI preprocessing for $patient_id =="
echo "Date: $(date -Is)"
echo "Patient dir: $patient_dir"
echo "Output dir: $out_dir"
echo "Scratch dir: $scratch_dir"
echo

for f in "$dwi_nifti" "$dwi_bvec" "$dwi_bval"; do
  if [[ ! -f "$f" ]]; then
    echo "Missing required input: $f" >&2
    exit 1
  fi
done

echo "Inputs:"
echo "  DWI:  $dwi_nifti"
echo "  BVEC: $dwi_bvec"
echo "  BVAL: $dwi_bval"
echo "  JSON: $dwi_json"
echo "  PE:   $pe_dir"
echo "  RPE:  $rpe_mode"
echo "  TRT:  ${readout_time:-not set}"
echo

run() {
  echo
  echo "+ $*"
  "$@"
}

cd "$out_dir"

if [[ -d "$scratch_dir" ]]; then
  timestamp="$(date +%Y%m%d_%H%M%S)"
  mv "$scratch_dir" "${scratch_dir}.incomplete_${timestamp}"
fi
mkdir -p "$scratch_dir"
export TMPDIR="$scratch_dir"

run mrconvert "$dwi_nifti" dwi_raw.mif \
  -fslgrad "$dwi_bvec" "$dwi_bval" \
  -json_export dwi_raw_mrtrix.json \
  -force

run dwidenoise dwi_raw.mif dwi_den.mif \
  -noise noise.mif \
  -force

run mrdegibbs dwi_den.mif dwi_den_unr.mif \
  -force

case "$rpe_mode" in
  none)
    run dwifslpreproc dwi_den_unr.mif dwi_preproc.mif \
      -rpe_none \
      -pe_dir "$pe_dir" \
      "${readout_args[@]}" \
      -eddy_options " --slm=linear --nthr=8" \
      -force
    ;;
  pair)
    if [[ -z "$reverse_phase_b0" || "$reverse_phase_b0" == "null" || ! -f "$reverse_phase_b0" ]]; then
      echo "rpe_mode=pair requires dwi.reverse_phase_b0 to point to an existing file." >&2
      exit 1
    fi
    run mrconvert "$reverse_phase_b0" reverse_phase_b0.mif -force
    run dwiextract dwi_den_unr.mif same_phase_b0.mif \
      -bzero \
      -force
    run mrcat same_phase_b0.mif reverse_phase_b0.mif se_epi_pair.mif \
      -axis 3 \
      -force
    run dwifslpreproc dwi_den_unr.mif dwi_preproc.mif \
      -rpe_pair \
      -se_epi se_epi_pair.mif \
      -pe_dir "$pe_dir" \
      "${readout_args[@]}" \
      -eddy_options " --slm=linear --nthr=8" \
      -force
    ;;
  all)
    run dwifslpreproc dwi_den_unr.mif dwi_preproc.mif \
      -rpe_all \
      -pe_dir "$pe_dir" \
      "${readout_args[@]}" \
      -eddy_options " --slm=linear --nthr=8" \
      -force
    ;;
  *)
    echo "Unsupported rpe_mode: $rpe_mode. Use none, pair, or all." >&2
    exit 1
    ;;
esac

run dwibiascorrect ants dwi_preproc.mif dwi_bias.mif \
  -bias bias.mif \
  -force

run dwi2mask dwi_bias.mif mask.mif \
  -force

run dwiextract dwi_bias.mif b0_vols.mif \
  -bzero \
  -force

run mrmath b0_vols.mif mean b0_mean.mif \
  -axis 3 \
  -force

run dwi2tensor dwi_bias.mif dt.mif \
  -mask mask.mif \
  -force

run tensor2metric dt.mif \
  -fa fa.mif \
  -adc md.mif \
  -vector ev.mif \
  -force

run mrconvert b0_mean.mif b0_mean.nii.gz -force
run mrconvert mask.mif mask.nii.gz -force
run mrconvert fa.mif fa.nii.gz -force
run mrconvert md.mif md.nii.gz -force
run mrconvert ev.mif ev.nii.gz -force

cat > "$report" <<EOF
# DWI Preprocessing QC - $patient_id

Generated: $(date -Is)

## Inputs

- DWI: \`$dwi_nifti\`
- BVEC: \`$dwi_bvec\`
- BVAL: \`$dwi_bval\`
- JSON: \`$dwi_json\`
- Phase encoding direction: \`$pe_dir\`
- Reverse phase mode: \`$rpe_mode\`
- Total readout time: \`${readout_time:-not set}\`
- Eddy options: \`--slm=linear --nthr=8\`
- Scratch directory: \`$scratch_dir\`

## Outputs

| Output | Path |
|---|---|
| Raw MIF | \`$out_dir/dwi_raw.mif\` |
| Denoised DWI | \`$out_dir/dwi_den.mif\` |
| Degibbs DWI | \`$out_dir/dwi_den_unr.mif\` |
| Preprocessed DWI | \`$out_dir/dwi_preproc.mif\` |
| Bias-corrected DWI | \`$out_dir/dwi_bias.mif\` |
| Brain mask | \`$out_dir/mask.nii.gz\` |
| Mean b0 | \`$out_dir/b0_mean.nii.gz\` |
| FA | \`$out_dir/fa.nii.gz\` |
| MD | \`$out_dir/md.nii.gz\` |
| Principal eigenvector | \`$out_dir/ev.nii.gz\` |

## Manual QC Checklist

- [ ] Inspect \`b0_mean.nii.gz\` for EPI distortion near STN/midbrain.
- [ ] Inspect \`mask.nii.gz\` for under/over-masking.
- [ ] Inspect \`fa.nii.gz\` for plausible corpus callosum/internal capsule contrast.
- [ ] Confirm no left/right orientation issue before registration.
- [ ] Confirm phase-encoding correction result is acceptable for this case.

## Notes

This run used \`dwifslpreproc\` with \`rpe_mode=$rpe_mode\`, \`readout_time=${readout_time:-not set}\`, and \`--nthr=8\`.

EOF

echo
echo "DWI preprocessing complete."
echo "Report: $report"
