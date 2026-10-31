#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <patient_dir>" >&2
  exit 2
fi

patient_dir="$(cd "$1" && pwd)"
config="$patient_dir/config.yaml"
if [[ ! -f "$config" ]]; then
  echo "Missing config: $config" >&2
  exit 1
fi
# Make all MRtrix temporary-header operations independent of the Windows
# process working directory used to launch WSL.
cd "$patient_dir"

mrtrix_root="${DBS_MRTRIX_ROOT:-/home/devold/opt/mrtrix3-3.0.8}"
if [[ ! -x "$mrtrix_root/bin/mrconvert" ]]; then
  echo "Pinned MRtrix installation is unavailable: $mrtrix_root" >&2
  exit 1
fi
export PATH="/home/devold/opt/python-shim:$mrtrix_root/bin:/home/devold/fsl/share/fsl/bin:/home/devold/ants_env/bin:$PATH"
export FSLDIR="${FSLDIR:-/home/devold/fsl}"
if [[ -f "$FSLDIR/etc/fslconf/fsl.sh" ]]; then
  # shellcheck disable=SC1091
  source "$FSLDIR/etc/fslconf/fsl.sh"
fi

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
    level = indent // 2
    stack = stack[:level]
    stack.append(key.strip())
    if value.strip():
        values[".".join(stack)] = value.strip().strip("'\"")
print(values.get(".".join(keys), ""))
PY
}

patient_id="$(yaml_value patient_id)"
dwi_nifti="$(yaml_value dwi.nifti)"
dwi_bvec="$(yaml_value dwi.bvec)"
dwi_bval="$(yaml_value dwi.bval)"
dwi_json="$(yaml_value dwi.sidecar_json)"
pe_dir="$(yaml_value dwi.pe_dir)"
readout_time="$(yaml_value dwi.readout_time)"

for path in "$dwi_nifti" "$dwi_bvec" "$dwi_bval" "$dwi_json"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing required input: $path" >&2
    exit 1
  fi
done
if [[ -z "$pe_dir" || -z "$readout_time" ]]; then
  echo "dwi.pe_dir and dwi.readout_time must be explicit in config.yaml" >&2
  exit 1
fi

version_root="$patient_dir/derivatives/single_shell_v1"
common_dir="$version_root/preproc_common"
branch_dir="$version_root/preproc_rpe_none"
provenance_dir="$version_root/provenance"
qc_dir="$patient_dir/qc/reports"
log_dir="$patient_dir/logs"
scratch_root="${DBS_STN_HDP_SCRATCH:-/home/devold/dbs_stn_hdp_scratch_v1}"
continue_scratch="${DBS_DWIFSL_CONTINUE_SCRATCH:-}"
scratch_dir="$scratch_root/${patient_id}_rpe_none_$(date +%Y%m%d_%H%M%S)"

if [[ -e "$branch_dir/dwi_preproc.mif" ]]; then
  echo "Refusing to overwrite an existing single_shell_v1 corrected branch." >&2
  exit 1
fi

mkdir -p "$common_dir" "$branch_dir/eddy_qc" "$provenance_dir" "$qc_dir" "$log_dir"
if [[ -n "$continue_scratch" ]]; then
  if [[ ! -d "$continue_scratch" || ! -f "$continue_scratch/eddy_in.nii" ]]; then
    echo "Invalid dwifslpreproc continuation scratch: $continue_scratch" >&2
    exit 1
  fi
  scratch_dir="$continue_scratch"
else
  mkdir -p "$scratch_dir"
fi
export TMPDIR="$scratch_dir"

log="$log_dir/03_single_shell_v1_rpe_none.log"
report="$qc_dir/03_single_shell_v1_rpe_none_qc.md"
exec > >(tee -a "$log") 2>&1

run() {
  echo
  printf '+'
  printf ' %q' "$@"
  echo
  "$@"
}

echo "== Single-shell v1 preprocessing: $patient_id =="
echo "Date: $(date -Is)"
echo "MRtrix root: $mrtrix_root"
echo "PE direction: $pe_dir"
echo "Total readout time: $readout_time s"
echo "Output root: $version_root"

common_complete=1
for path in \
  "$common_dir/dwi_raw.mif" \
  "$common_dir/dwi_denoised.mif" \
  "$common_dir/noise.mif" \
  "$common_dir/denoise_residual.mif" \
  "$common_dir/dwi_den_degibbs.mif" \
  "$common_dir/degibbs_residual.mif" \
  "$common_dir/b0_raw_mean.mif" \
  "$common_dir/b0_denoised_mean.mif" \
  "$common_dir/b0_degibbs_mean.mif"; do
  [[ -f "$path" ]] || common_complete=0
done

if (( common_complete )); then
  echo "Reusing verified common denoise/degibbs outputs from the interrupted run."
else
  if find "$common_dir" -mindepth 1 -maxdepth 1 -type f | grep -q .; then
    echo "Common preprocessing is incomplete; refusing to mix partial outputs." >&2
    exit 1
  fi

  run mrconvert "$dwi_nifti" "$common_dir/dwi_raw.mif" \
    -fslgrad "$dwi_bvec" "$dwi_bval" \
    -json_import "$dwi_json"

  run mrinfo "$common_dir/dwi_raw.mif" \
    -export_grad_fsl "$provenance_dir/imported.bvec" "$provenance_dir/imported.bval"
  run mrinfo "$common_dir/dwi_raw.mif" -size
  run mrinfo "$common_dir/dwi_raw.mif" -spacing
  run mrinfo "$common_dir/dwi_raw.mif" -shell_bvalues

  run dwidenoise "$common_dir/dwi_raw.mif" "$common_dir/dwi_denoised.mif" \
    -noise "$common_dir/noise.mif"
  run mrcalc "$common_dir/dwi_raw.mif" "$common_dir/dwi_denoised.mif" \
    -subtract "$common_dir/denoise_residual.mif"

  # PartialFourier=0.75 violates the ideal full-Fourier assumption of
  # mrdegibbs. Keep both the input and residual so this step can be rejected at
  # the visual QC gate without repeating denoising.
  run mrdegibbs "$common_dir/dwi_denoised.mif" "$common_dir/dwi_den_degibbs.mif"
  run mrcalc "$common_dir/dwi_denoised.mif" "$common_dir/dwi_den_degibbs.mif" \
    -subtract "$common_dir/degibbs_residual.mif"

  run dwiextract "$common_dir/dwi_raw.mif" "$common_dir/b0_raw_volumes.mif" -bzero
  run mrmath "$common_dir/b0_raw_volumes.mif" mean \
    "$common_dir/b0_raw_mean.mif" -axis 3
  run dwiextract "$common_dir/dwi_denoised.mif" "$common_dir/b0_denoised_volumes.mif" -bzero
  run mrmath "$common_dir/b0_denoised_volumes.mif" mean \
    "$common_dir/b0_denoised_mean.mif" -axis 3
  run dwiextract "$common_dir/dwi_den_degibbs.mif" "$common_dir/b0_degibbs_volumes.mif" -bzero
  run mrmath "$common_dir/b0_degibbs_volumes.mif" mean \
    "$common_dir/b0_degibbs_mean.mif" -axis 3
fi

dwifsl_args=(
  "$common_dir/dwi_den_degibbs.mif"
  "$branch_dir/dwi_preproc.mif"
  -rpe_none
  -pe_dir "$pe_dir"
  -readout_time "$readout_time"
  -eddy_options " --repol --slm=linear --nthr=8 --cnr_maps --residuals"
  -eddyqc_all "$branch_dir/eddy_qc"
)
if [[ -n "$continue_scratch" ]]; then
  echo "Continuing dwifslpreproc from: $continue_scratch (after eddy_in.nii)"
  dwifsl_args+=( -nocleanup -continue "$continue_scratch" eddy_in.nii )
else
  # Keep a native-WSL scratch directory if the process is interrupted; this
  # both improves I/O and permits a deterministic -continue recovery.
  dwifsl_args+=( -nocleanup -scratch "$scratch_dir" )
fi
run dwifslpreproc "${dwifsl_args[@]}"

run mrinfo "$branch_dir/dwi_preproc.mif" \
  -export_grad_fsl "$branch_dir/dwi_preproc.bvec" "$branch_dir/dwi_preproc.bval"

run dwibiascorrect ants "$branch_dir/dwi_preproc.mif" "$branch_dir/dwi_bias.mif" \
  -bias "$branch_dir/bias.mif"
run dwi2mask "$branch_dir/dwi_bias.mif" "$branch_dir/mask.mif"
run dwiextract "$branch_dir/dwi_bias.mif" "$branch_dir/b0_volumes.mif" -bzero
run mrmath "$branch_dir/b0_volumes.mif" mean "$branch_dir/b0_mean.mif" -axis 3

run dwi2tensor "$branch_dir/dwi_bias.mif" "$branch_dir/dt.mif" \
  -mask "$branch_dir/mask.mif"
run tensor2metric "$branch_dir/dt.mif" \
  -mask "$branch_dir/mask.mif" \
  -fa "$branch_dir/fa.mif" \
  -adc "$branch_dir/md.mif" \
  -vector "$branch_dir/ev.mif"
run tensor2metric "$branch_dir/dt.mif" \
  -mask "$branch_dir/mask.mif" \
  -value "$branch_dir/eigenvalues.mif" \
  -num 1,2,3
run mrcalc "$branch_dir/eigenvalues.mif" 0 -lt \
  "$branch_dir/negative_eigenvalue_components.mif" -datatype bit
run mrmath "$branch_dir/negative_eigenvalue_components.mif" max \
  "$branch_dir/negative_eigenvalue_voxels.mif" -axis 3 -datatype bit

mask_count="$(mrstats "$branch_dir/mask.mif" -ignorezero -output count | tr -d '[:space:]')"
negative_count="$(mrstats "$branch_dir/negative_eigenvalue_voxels.mif" -ignorezero -output count | tr -d '[:space:]')"
negative_percent="$(awk -v n="$negative_count" -v d="$mask_count" \
  'BEGIN { if (d > 0) printf "%.4f", 100*n/d; else print "nan" }')"

for name in b0_mean mask fa md eigenvalues negative_eigenvalue_voxels; do
  run mrconvert "$branch_dir/$name.mif" "$branch_dir/$name.nii.gz"
done
for name in b0_raw_mean b0_denoised_mean b0_degibbs_mean; do
  run mrconvert "$common_dir/$name.mif" "$common_dir/$name.nii.gz"
done

cat > "$report" <<EOF
# ${patient_id} — single-shell v1 preprocessing QC

Generated: $(date -Is)

## Locked parameters

- MRtrix: \`$("$mrtrix_root/bin/mrconvert" -version 2>&1 | head -n 1)\`
- Phase encoding: \`$pe_dir\`
- Total readout time: \`$readout_time s\`
- Eddy: \`--repol --slm=linear --nthr=8 --cnr_maps --residuals\`
- Susceptibility branch: \`rpe_none\` (motion/eddy-current correction only)
- Partial Fourier: \`0.75\`; degibbs remains provisional until visual QC.

## Outputs

- Common preprocessing: \`$common_dir\`
- Corrected branch: \`$branch_dir\`
- Eddy QC: \`$branch_dir/eddy_qc\`
- Full log: \`$log\`

## Tensor QC

- Brain-mask voxels: \`$mask_count\`
- Voxels with at least one negative fitted eigenvalue: \`$negative_count\`
- Negative-eigenvalue fraction: \`$negative_percent%\`

## Required visual gate

- [ ] Compare \`b0_denoised_mean.nii.gz\` and \`b0_degibbs_mean.nii.gz\`.
- [ ] Reject degibbs if partial-Fourier-associated edge artifacts increase.
- [ ] Check b0 for residual EPI distortion at frontal lobes, temporal lobes and midbrain/STN.
- [ ] Check mask coverage of brainstem and cerebellum.
- [ ] Check FA orientation and internal-capsule/corpus-callosum contrast.
- [ ] Review all eddy outlier and motion outputs.

This branch cannot correct susceptibility distortion because no reverse-PE
image or field map was acquired. It is the control branch for later comparison
with Synb0-DisCo.
EOF

echo
echo "Completed rpe_none branch."
echo "Report: $report"

if [[ -z "$continue_scratch" && -d "$scratch_dir" ]]; then
  case "$scratch_dir" in
    "$scratch_root"/"${patient_id}"_rpe_none_*)
      rm -rf -- "$scratch_dir"
      echo "Removed completed single-shell scratch: $scratch_dir"
      ;;
    *)
      echo "Scratch path is outside the managed pattern; retaining: $scratch_dir" >&2
      ;;
  esac
fi
