#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "Usage: $0 <patient_dir> [rpe_none|synb0] [--check-only]" >&2
  exit 2
fi

patient_dir="$(cd "$1" && pwd)"
input_mode="${2:-rpe_none}"
action="${3:-run}"
if [[ "$action" != "run" && "$action" != "--check-only" ]]; then
  echo "Unsupported action: $action (expected --check-only or omit it to run)" >&2
  exit 2
fi
config="$patient_dir/config.yaml"
if [[ ! -f "$config" ]]; then
  echo "Missing config: $config" >&2
  exit 1
fi
cd "$patient_dir"

yaml_value() {
  local key_path="$1"
  python3 - "$config" "$key_path" <<'PY'
from pathlib import Path
import sys

keys = sys.argv[2].split(".")
stack = []
values = {}
for raw in Path(sys.argv[1]).read_text().splitlines():
    if not raw.strip() or raw.lstrip().startswith("#"):
        continue
    indent = len(raw) - len(raw.lstrip(" "))
    line = raw.strip()
    if ":" not in line or line.startswith("- "):
        continue
    key, value = line.split(":", 1)
    stack = stack[: indent // 2]
    stack.append(key.strip())
    if value.strip():
        values[".".join(stack)] = value.strip().strip("'\"")
print(values.get(".".join(keys), ""))
PY
}

patient_id="$(yaml_value patient_id)"
tissue_root="${DBS_MRTRIX3TISSUE_ROOT:-/home/devold/opt/MRtrix3Tissue-5.2.9}"
export PATH="/home/devold/opt/python-shim:$tissue_root/bin:/usr/bin:/bin"

for command in \
  dwi2response ss3t_csd_beta1 mtnormalise fod2dec sh2peaks \
  mrconvert mrcalc mrstats mrinfo dwiextract dwi2fod \
  shconv sh2amp mrcat; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "Missing MRtrix3Tissue command: $command" >&2
    exit 1
  fi
done
for command in shconv sh2amp mrcat; do
  if ! "$command" -version 2>&1 | grep -q "3Tissue_v5.2.9"; then
    echo "Incompatible $command; expected MRtrix3Tissue 5.2.9 at $tissue_root/bin" >&2
    exit 1
  fi
done

version_root="$patient_dir/derivatives/single_shell_v1"
case "$input_mode" in
  rpe_none)
    input_branch="preproc_rpe_none"
    output_branch="fod_ss3t_v1"
    artifact_tag="ss3t_v1"
    branch_limitation="The input used rpe_none preprocessing and therefore has no susceptibility-distortion correction."
    ;;
  synb0)
    input_branch="preproc_synb0"
    output_branch="fod_ss3t_synb0"
    artifact_tag="ss3t_synb0"
    branch_limitation="Susceptibility-distortion correction used a five-fold Synb0 synthetic reverse-phase b0 rather than a measured reverse-phase b0."
    ;;
  *)
    echo "Unsupported input mode: $input_mode (expected rpe_none or synb0)" >&2
    exit 2
    ;;
esac

input_dir="$version_root/$input_branch"
fod_dir="$version_root/$output_branch"
qc_dir="$patient_dir/qc/reports"
log_dir="$patient_dir/logs"
scratch_parent="/home/devold/dbs_stn_hdp_scratch_v1/${patient_id}_${artifact_tag}_$(date +%Y%m%d_%H%M%S)"

dwi="$input_dir/dwi_bias.mif"
mask="$input_dir/mask.mif"
for path in "$dwi" "$mask" "$input_dir/STATUS.txt"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing completed $input_branch prerequisite: $path" >&2
    exit 1
  fi
done
if ! grep -qx 'COMPLETE' "$input_dir/STATUS.txt"; then
  echo "Prerequisite is not marked COMPLETE: $input_dir/STATUS.txt" >&2
  exit 1
fi

read -r dwi_x dwi_y dwi_z dwi_volumes <<< "$(mrinfo "$dwi" -size)"
read -r mask_x mask_y mask_z mask_volumes <<< "$(mrinfo "$mask" -size)"
if [[ "$dwi_x $dwi_y $dwi_z" != "$mask_x $mask_y $mask_z" ]]; then
  echo "DWI/mask geometry mismatch: DWI=$dwi_x,$dwi_y,$dwi_z mask=$mask_x,$mask_y,$mask_z" >&2
  exit 1
fi
if [[ "$(mrinfo "$dwi" -shell_bvalues | tr -s '[:space:]' ' ' | sed 's/^ //;s/ $//')" != "0 1000" ]]; then
  echo "Expected a b0+b1000 single-shell DWI: $dwi" >&2
  exit 1
fi

if [[ "$input_mode" == "synb0" ]]; then
  synb0_outputs="$input_dir/OUTPUTS"
  fold_files=("$synb0_outputs"/b0_u_lin_atlas_2_5_FOLD_*.nii.gz)
  if [[ "${#fold_files[@]}" -ne 5 ]]; then
    echo "Refusing Synb0 FOD: expected 5 fold files, found ${#fold_files[@]} in $synb0_outputs" >&2
    exit 1
  fi
  for fold in 1 2 3 4 5; do
    if [[ ! -s "$synb0_outputs/b0_u_lin_atlas_2_5_FOLD_${fold}.nii.gz" ]]; then
      echo "Refusing Synb0 FOD: missing or empty fold $fold" >&2
      exit 1
    fi
  done
  merged="$synb0_outputs/b0_u_lin_atlas_2_5_merged.nii.gz"
  if [[ ! -s "$merged" ]]; then
    echo "Refusing Synb0 FOD: missing merged ensemble: $merged" >&2
    exit 1
  fi
  merged_volumes="$(mrinfo "$merged" -size | awk '{print $4}')"
  if [[ "$merged_volumes" != "5" ]]; then
    echo "Refusing Synb0 FOD: merged ensemble has $merged_volumes volumes, expected 5" >&2
    exit 1
  fi
fi

if [[ -e "$fod_dir/STATUS.txt" || -e "$fod_dir/wmfod_lmax8.mif" ]]; then
  echo "Refusing to overwrite an existing SS3T-CSD run: $fod_dir" >&2
  exit 1
fi

if [[ "$action" == "--check-only" ]]; then
  echo "SS3T preflight PASS"
  echo "Input mode: $input_mode"
  echo "Input DWI: $dwi ($dwi_x x $dwi_y x $dwi_z x $dwi_volumes)"
  echo "Input mask: $mask"
  if [[ "$input_mode" == "synb0" ]]; then
    echo "Synb0 ensemble: 5 fold files; merged dim4=$merged_volumes"
  fi
  echo "Output (not created): $fod_dir"
  echo "Official tractography FOD after completion: $fod_dir/wmfod_norm_lmax6.mif"
  exit 0
fi

mkdir -p "$fod_dir" "$qc_dir" "$log_dir" "$scratch_parent"
log="$log_dir/04_single_shell_v1_${artifact_tag}.log"
report="$qc_dir/04_single_shell_v1_${artifact_tag}_qc.md"
exec > >(tee -a "$log") 2>&1

run() {
  echo
  printf '+'
  printf ' %q' "$@"
  echo
  "$@"
}

echo "== Single-shell SS3T-CSD FOD: $patient_id =="
echo "Date: $(date -Is)"
echo "MRtrix3Tissue root: $tissue_root"
echo "Input mode: $input_mode"
echo "Input DWI: $dwi"
echo "Output: $fod_dir"
echo "SS3T-required lmax: b0=0, b1000=8"
echo "Locked official tractography output: post-fit lmax=6 truncation"
echo "Sensitivity/provenance output: fitted lmax=8"
echo "Attempt started: $(date -Is)"

cd "$fod_dir"

response_outputs=(
  wm_response.txt
  gm_response.txt
  csf_response.txt
  response_voxels.mif
)
response_output_count=0
for path in "${response_outputs[@]}"; do
  [[ -f "$path" ]] && ((response_output_count += 1))
done
if [[ "$response_output_count" -eq "${#response_outputs[@]}" ]]; then
  echo "Reusing complete dhollander response outputs from the previous attempt."
elif [[ "$response_output_count" -ne 0 ]]; then
  echo "Partial response output set detected; refusing an ambiguous restart." >&2
  exit 1
else
  run dwi2response dhollander "$dwi" \
    wm_response.txt gm_response.txt csf_response.txt \
    -mask "$mask" \
    -voxels response_voxels.mif \
    -lmax 0,8 \
    -scratch "$scratch_parent"
fi

run ss3t_csd_beta1 "$dwi" \
  wm_response.txt wmfod_lmax8.mif \
  gm_response.txt gm.mif \
  csf_response.txt csf.mif \
  -mask "$mask" \
  -nthreads 8 \
  -scratch "$scratch_parent"

run mtnormalise \
  wmfod_lmax8.mif wmfod_norm.mif \
  gm.mif gm_norm.mif \
  csf.mif csf_norm.mif \
  -mask "$mask"

run fod2dec wmfod_norm.mif wmfod_dec.mif -mask "$mask"
run sh2peaks wmfod_norm.mif wmfod_peaks.mif -mask "$mask" -num 3
run mrconvert wmfod_lmax8.mif wmfod_lmax6.mif -coord 3 0:27
run mrconvert wmfod_norm.mif wmfod_norm_lmax6.mif -coord 3 0:27
run fod2dec wmfod_norm_lmax6.mif wmfod_lmax6_dec.mif -mask "$mask"
run sh2peaks wmfod_norm_lmax6.mif wmfod_lmax6_peaks.mif -mask "$mask" -num 3
run mrconvert wmfod_norm.mif wmfod_l0.mif -coord 3 0

# Dhollander writes a 4D binary image in CSF, GM, single-fibre WM order.
run mrconvert response_voxels.mif response_csf_voxels.mif -coord 3 0
run mrconvert response_voxels.mif response_gm_voxels.mif -coord 3 1
run mrconvert response_voxels.mif response_wm_voxels.mif -coord 3 2
run mrcalc wmfod_l0.mif 0 -lt wmfod_l0_negative.mif -datatype bit

for name in \
  response_voxels \
  response_wm_voxels \
  response_gm_voxels \
  response_csf_voxels \
  wmfod_l0 \
  wmfod_l0_negative \
  gm_norm \
  csf_norm \
  wmfod_dec \
  wmfod_lmax6_dec \
  wmfod_lmax6_peaks; do
  run mrconvert "$name.mif" "$name.nii.gz"
done

mask_count="$(mrstats "$mask" -ignorezero -output count | tr -d '[:space:]')"
wm_response_count="$(mrstats response_wm_voxels.mif -ignorezero -output count | tr -d '[:space:]')"
gm_response_count="$(mrstats response_gm_voxels.mif -ignorezero -output count | tr -d '[:space:]')"
csf_response_count="$(mrstats response_csf_voxels.mif -ignorezero -output count | tr -d '[:space:]')"
negative_l0_count="$(mrstats wmfod_l0_negative.mif -ignorezero -output count | tr -d '[:space:]')"
negative_l0_percent="$(awk -v n="$negative_l0_count" -v d="$mask_count" \
  'BEGIN { if (d > 0) printf "%.6f", 100*n/d; else print "nan" }')"
fod_coefficients="$(mrinfo wmfod_norm.mif -size | awk '{print $4}')"
fod_lmax6_coefficients="$(mrinfo wmfod_norm_lmax6.mif -size | awk '{print $4}')"
peak_components="$(mrinfo wmfod_peaks.mif -size | awk '{print $4}')"
lmax6_peak_components="$(mrinfo wmfod_lmax6_peaks.mif -size | awk '{print $4}')"
wm_l0_mean="$(mrstats wmfod_l0.mif -mask "$mask" -output mean | tr -d '[:space:]')"
gm_mean="$(mrstats gm_norm.mif -mask "$mask" -output mean | tr -d '[:space:]')"
csf_mean="$(mrstats csf_norm.mif -mask "$mask" -output mean | tr -d '[:space:]')"

cat > "$report" <<EOF
# ${patient_id} — single-shell SS3T-CSD FOD QC

Generated: $(date -Is)

## Locked model

- Input branch: \`single_shell_v1/$input_branch\`
- Acquisition: \`b0 + b1000\` (single-shell)
- Model: \`SS3T-CSD beta1\`
- MRtrix3Tissue: \`5.2.9\`
- Response estimation: \`dhollander\`
- SS3T harmonic order: \`lmax=8\` for b1000 and \`lmax=0\` for b0
- Official tractography output: post-fit truncation to \`lmax=6\`
- Sensitivity output: full fitted \`lmax=8\`
- Normalisation: multi-tissue \`mtnormalise\`

The model choice remains locked from the sub-09 validation. MRtrix3Tissue
SS3T beta1 does not support a direct lmax-6 fit. The tested three-tissue
single-shell MSMT-CSD route produced a degenerate GM compartment and failed
multi-tissue normalisation; the tested two-tissue lmax-6 route inflated
WM-like signal in GM by 3.20-fold, including the STN/RN region. Neither
rejected route is revived for this patient.

## Response estimation

- WM response voxels: \`$wm_response_count\`
- GM response voxels: \`$gm_response_count\`
- CSF response voxels: \`$csf_response_count\`

## FOD checks

- Brain-mask voxels: \`$mask_count\`
- Full fitted WM FOD coefficients: \`$fod_coefficients\` (expected 45 for lmax 8)
- Official tractography WM FOD coefficients: \`$fod_lmax6_coefficients\` (expected 28 for lmax 6)
- Full fitted peak components: \`$peak_components\` (expected 9)
- Official lmax-6 peak components: \`$lmax6_peak_components\` (expected 9)
- Voxels with negative normalised WM l0: \`$negative_l0_count\`
- Negative WM l0 fraction: \`$negative_l0_percent%\`
- Mean normalised WM l0: \`$wm_l0_mean\`
- Mean normalised GM signal: \`$gm_mean\`
- Mean normalised CSF signal: \`$csf_mean\`

## Outputs

- \`$fod_dir/wmfod_norm_lmax6.mif\`: official WM FOD for tractography
- \`$fod_dir/wmfod_norm.mif\`: full lmax-8 sensitivity FOD
- \`$fod_dir/gm_norm.mif\`: normalised GM-like compartment
- \`$fod_dir/csf_norm.mif\`: normalised CSF-like compartment
- \`$fod_dir/wmfod_dec.nii.gz\`: direction-encoded FOD preview
- \`$fod_dir/wmfod_lmax6_dec.nii.gz\`: official lmax-6 FOD preview
- \`$fod_dir/wmfod_peaks.mif\`: three peak vectors per voxel for QC
- \`$fod_dir/response_voxels.nii.gz\`: response-function voxel labels

## Interpretation gate

- [ ] WM response voxels are located in coherent deep white matter.
- [ ] GM and CSF response voxels occupy plausible tissue locations.
- [ ] DEC orientation is anatomically coherent in corpus callosum and internal capsules.
- [ ] No gross left-right or anterior-posterior direction inversion is present.
- [ ] FODs remain interpretable around the midbrain/STN despite residual EPI distortion.

## QC conclusion

\`PENDING_MANUAL_REVIEW\` immediately after computation. Change this conclusion
to \`PASS\` only after the response-voxel and DEC interpretation gates above
have been reviewed. For the Synb0 branch, record a plain \`PASS\` and document
the synthetic-b0 limitation separately.

## Limitations

- $branch_limitation
- This is a single-shell SS3T-CSD result and must not be treated as
  acquisition-equivalent to multi-shell MSMT-CSD.
- The acquisition has 30 independent diffusion directions (60 diffusion
  volumes with repeated directions), lower angular information than the
  60-direction cohort acquisitions.

MRtrix3Tissue 5.2.9 SS3T beta1 explicitly requires and internally fits
\`lmax=8\`; the official lmax-6 image is therefore produced by removing the
\`l=8\` coefficients after the standard SS3T fit, rather than by modifying the
published beta1 algorithm. For this patient, the post-fit lmax-6 image is the
locked official tractography FOD; the full lmax-8 image is retained only for
sensitivity and provenance.
EOF

cat > "$fod_dir/STATUS.txt" <<EOF
COMPLETE

Pipeline: single_shell_v1 / $output_branch
Completed: $(date -Is)
Input: $input_branch/dwi_bias.mif
Model: SS3T-CSD beta1
MRtrix3Tissue: 5.2.9
lmax: 8 fitted; 6 post-fit official tractography output
Official tractography FOD: wmfod_norm_lmax6.mif
Sensitivity FOD: wmfod_norm.mif
QC: PENDING_MANUAL_REVIEW
QC report: ../../../qc/reports/04_single_shell_v1_${artifact_tag}_qc.md
EOF

case "$scratch_parent" in
  /home/devold/dbs_stn_hdp_scratch_v1/"${patient_id}"_"${artifact_tag}"_*)
    rm -rf -- "$scratch_parent"
    echo "Removed completed SS3T scratch: $scratch_parent"
    ;;
  *)
    echo "SS3T scratch path is outside the managed pattern; retaining: $scratch_parent" >&2
    ;;
esac

echo
echo "SS3T-CSD FOD complete."
echo "Report: $report"
