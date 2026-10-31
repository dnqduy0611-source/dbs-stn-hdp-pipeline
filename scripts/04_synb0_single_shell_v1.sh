#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "Usage: $0 <patient_dir> [prepare|run] [freesurfer_license]" >&2
  exit 2
fi

patient_dir="$(cd "$1" && pwd)"
mode="${2:-prepare}"
license_path="${3:-${SYNB0_FREESURFER_LICENSE:-}}"
config="$patient_dir/config.yaml"
mrtrix_root="${DBS_MRTRIX_ROOT:-/home/devold/opt/mrtrix3-3.0.8}"
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
    stack = stack[: indent // 2]
    stack.append(key.strip())
    if value.strip():
        values[".".join(stack)] = value.strip().strip("'\"")
print(values.get(".".join(keys), ""))
PY
}

patient_id="$(yaml_value patient_id)"
pe_dir="$(yaml_value dwi.pe_dir)"
readout_time="$(yaml_value dwi.readout_time)"
t1="$(yaml_value anat.t1)"
case "$pe_dir" in
  i)  pe_vector="1 0 0" ;;
  i-) pe_vector="-1 0 0" ;;
  j)  pe_vector="0 1 0" ;;
  j-) pe_vector="0 -1 0" ;;
  k)  pe_vector="0 0 1" ;;
  k-) pe_vector="0 0 -1" ;;
  *) echo "Unsupported phase encoding direction for Synb0: $pe_dir" >&2; exit 1 ;;
esac

root="$patient_dir/derivatives/single_shell_v1"
common="$root/preproc_common"
branch="$root/preproc_synb0"
inputs="$branch/INPUTS"
outputs="$branch/OUTPUTS"
qc_dir="$branch/eddy_qc"
log="$patient_dir/logs/04_single_shell_v1_synb0.log"
report="$patient_dir/qc/reports/04_single_shell_v1_synb0_qc.md"

for path in "$common/b0_degibbs_mean.nii.gz" "$common/dwi_den_degibbs.mif" "$t1"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing prerequisite: $path" >&2
    exit 1
  fi
done

mkdir -p "$inputs" "$outputs"
if [[ ! -e "$inputs/b0.nii.gz" ]]; then
  mrconvert "$common/b0_degibbs_mean.nii.gz" "$inputs/b0.nii.gz"
fi
if [[ ! -e "$inputs/T1.nii.gz" ]]; then
  cp -- "$t1" "$inputs/T1.nii.gz"
fi
printf '%s %s\n%s 0.000000\n' \
  "$pe_vector" "$readout_time" "$pe_vector" > "$inputs/acqparams.txt"
printf '%s %s\n' \
  "$pe_vector" "$readout_time" > "$inputs/eddy_acqparams.txt"

if [[ "$mode" == "prepare" ]]; then
  cat > "$branch/STATUS.txt" <<EOF
Prepared Synb0-DisCo inputs.
Phase encoding: $pe_dir ($pe_vector)
Total readout time: $readout_time s
Execution pending a valid FreeSurfer license and at least 13 GB container memory.
EOF
  echo "Prepared Synb0 inputs: $inputs"
  exit 0
fi
if [[ "$mode" != "run" ]]; then
  echo "Mode must be prepare or run." >&2
  exit 2
fi
if [[ -z "$license_path" || ! -s "$license_path" ]]; then
  echo "A valid FreeSurfer license is required for Synb0-DisCo v3.1." >&2
  exit 1
fi
if [[ -e "$branch/dwi_preproc.mif" ]]; then
  echo "Refusing to overwrite existing Synb0 preprocessing." >&2
  exit 1
fi

memory_kib="$(awk '/MemTotal:/ {print $2}' /proc/meminfo)"
if (( memory_kib < 13 * 1024 * 1024 )) && [[ "${SYNB0_ALLOW_LOW_MEMORY:-0}" != "1" ]]; then
  echo "Synb0 requires >13 GB RAM; WSL exposes only $((memory_kib / 1024 / 1024)) GB." >&2
  exit 1
fi
exec > >(tee -a "$log") 2>&1
echo
echo "== Synb0 continuation attempt: $(date -Is) =="
synb0_outputs=(
  "$outputs/b0_all_topup.nii.gz" \
  "$outputs/topup_fieldcoef.nii.gz" \
  "$outputs/topup_movpar.txt"
)
synb0_complete=1
for path in "${synb0_outputs[@]}"; do
  [[ -s "$path" ]] || synb0_complete=0
done

# Ensemble check. Synb0's own pipeline.sh averages with `fslmerge -t ... FOLD_*`,
# which silently merges whatever fold files happen to exist, then prints
# FINISHED!!! with exit code 0. sub-09 2026-08-03 lost folds 1/4/5 to
# SIGKILL and the branch ran to completion on a 2-fold field with no warning.
# Presence of the three TOPUP files is therefore NOT sufficient evidence of a
# complete run - the fold count is.
synb0_expected_folds=5
synb0_fold_count="$(find "$outputs" -maxdepth 1 -name 'b0_u_lin_atlas_2_5_FOLD_*.nii.gz' -size +0 | wc -l)"
if (( synb0_complete )) && (( synb0_fold_count != synb0_expected_folds )); then
  echo "Synb0 TOPUP outputs exist but the ensemble is incomplete:" >&2
  echo "  folds on disk: $synb0_fold_count / $synb0_expected_folds" >&2
  echo "  the distortion field was averaged over $synb0_fold_count fold(s) only." >&2
  echo "  Refusing to reuse. Back up $outputs, then rerun the missing folds." >&2
  exit 1
fi

if (( synb0_complete )); then
  echo "Reusing complete Synb0-DisCo TOPUP outputs from: $outputs ($synb0_fold_count/$synb0_expected_folds folds)"
else
  docker image inspect leonyichencai/synb0-disco:v3.1 >/dev/null
  docker run --rm \
    -v "$inputs:/INPUTS" \
    -v "$outputs:/OUTPUTS" \
    -v "$license_path:/extra/freesurfer/license.txt:ro" \
    --user "$(id -u):$(id -g)" \
    leonyichencai/synb0-disco:v3.1
fi

for path in "${synb0_outputs[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "Synb0 output is incomplete: $path" >&2
    exit 1
  fi
done

# Re-check the ensemble after the container has run. `FINISHED!!!` and exit 0
# do not imply all folds survived - see the comment on synb0_expected_folds.
synb0_fold_count="$(find "$outputs" -maxdepth 1 -name 'b0_u_lin_atlas_2_5_FOLD_*.nii.gz' -size +0 | wc -l)"
synb0_merged_vols="$(fslval "$outputs/b0_u_lin_atlas_2_5_merged.nii.gz" dim4 2>/dev/null | tr -d '[:space:]')"
echo "Synb0 ensemble: $synb0_fold_count/$synb0_expected_folds folds, merged dim4=$synb0_merged_vols"
if (( synb0_fold_count != synb0_expected_folds )) || [[ "$synb0_merged_vols" != "$synb0_expected_folds" ]]; then
  echo "Synb0 ensemble is incomplete - the distortion field is degraded." >&2
  echo "  folds on disk: $synb0_fold_count / $synb0_expected_folds" >&2
  echo "  merged dim4  : $synb0_merged_vols (expected $synb0_expected_folds)" >&2
  echo "  Check for SIGKILL in the log, then rerun the missing folds." >&2
  exit 1
fi

eddy_input_complete=1
for path in \
  "$branch/eddy_input.nii.gz" \
  "$branch/input.bvec" \
  "$branch/input.bval" \
  "$branch/dwi.json"; do
  [[ -s "$path" ]] || eddy_input_complete=0
done
if (( eddy_input_complete )); then
  echo "Reusing prepared Eddy input and gradients from: $branch"
else
  mrconvert "$common/dwi_den_degibbs.mif" "$branch/eddy_input.nii.gz" \
    -export_grad_fsl "$branch/input.bvec" "$branch/input.bval" \
    -json_export "$branch/dwi.json" \
    -force
fi

mask_complete=1
for path in \
  "$branch/b0_topup_real.nii.gz" \
  "$branch/b0_topup_real_brain.nii.gz" \
  "$branch/b0_topup_real_brain_mask.nii.gz"; do
  [[ -s "$path" ]] || mask_complete=0
done
if (( mask_complete )); then
  echo "Reusing prepared TOPUP b0 brain mask from: $branch"
else
  fslroi "$outputs/b0_all_topup.nii.gz" "$branch/b0_topup_real.nii.gz" 0 1
  bet "$branch/b0_topup_real.nii.gz" "$branch/b0_topup_real_brain.nii.gz" -m -f 0.2
fi

volume_count="$(mrinfo "$common/dwi_den_degibbs.mif" -size | awk '{print $4}')"
python3 - "$volume_count" "$branch/index.txt" <<'PY'
from pathlib import Path
import sys

count = int(sys.argv[1])
Path(sys.argv[2]).write_text(" ".join(["1"] * count) + "\n")
PY

eddy_complete=1
for path in \
  "$branch/eddy_corrected.nii.gz" \
  "$branch/eddy_corrected.eddy_rotated_bvecs" \
  "$branch/eddy_corrected.eddy_parameters" \
  "$branch/eddy_corrected.eddy_cnr_maps.nii.gz" \
  "$branch/eddy_corrected.eddy_residuals.nii.gz"; do
  [[ -s "$path" ]] || eddy_complete=0
done
if (( eddy_complete )); then
  echo "Reusing complete Eddy outputs from: $branch/eddy_corrected"
else
  eddy_cpu \
    --imain="$branch/eddy_input.nii.gz" \
    --mask="$branch/b0_topup_real_brain_mask.nii.gz" \
    --acqp="$inputs/eddy_acqparams.txt" \
    --index="$branch/index.txt" \
    --bvecs="$branch/input.bvec" \
    --bvals="$branch/input.bval" \
    --topup="$outputs/topup" \
    --out="$branch/eddy_corrected" \
    --repol \
    --slm=linear \
    --nthr=8 \
    --cnr_maps \
    --residuals
fi

if command -v eddy_quad >/dev/null 2>&1; then
  if [[ -d "$qc_dir" ]]; then
    if find "$qc_dir" -mindepth 1 -print -quit | grep -q .; then
      qc_backup="${qc_dir}.pre_retry_$(date +%Y%m%d_%H%M%S)"
      mv -- "$qc_dir" "$qc_backup"
      echo "Preserved previous Eddy QC directory at: $qc_backup"
    else
      rmdir -- "$qc_dir"
    fi
  fi
  eddy_quad "$branch/eddy_corrected" \
    -idx "$branch/index.txt" \
    -par "$inputs/eddy_acqparams.txt" \
    -m "$branch/b0_topup_real_brain_mask.nii.gz" \
    -b "$branch/input.bval" \
    -g "$branch/input.bvec" \
    -o "$qc_dir"
fi

mrconvert "$branch/eddy_corrected.nii.gz" "$branch/dwi_preproc.mif" \
  -fslgrad "$branch/eddy_corrected.eddy_rotated_bvecs" "$branch/input.bval" \
  -json_import "$branch/dwi.json"
mrinfo "$branch/dwi_preproc.mif" \
  -export_grad_fsl "$branch/dwi_preproc.bvec" "$branch/dwi_preproc.bval"
dwibiascorrect ants "$branch/dwi_preproc.mif" "$branch/dwi_bias.mif" \
  -bias "$branch/bias.mif"
dwi2mask "$branch/dwi_bias.mif" "$branch/mask.mif"
dwiextract "$branch/dwi_bias.mif" "$branch/b0_volumes.mif" -bzero
mrmath "$branch/b0_volumes.mif" mean "$branch/b0_mean.mif" -axis 3
dwi2tensor "$branch/dwi_bias.mif" "$branch/dt.mif" -mask "$branch/mask.mif"
tensor2metric "$branch/dt.mif" -mask "$branch/mask.mif" \
  -fa "$branch/fa.mif" -adc "$branch/md.mif" -vector "$branch/ev.mif"

# Degenerate-tensor gate, same as 03_preprocess_single_shell_v1.sh.  This branch
# was missing it while the rpe_none control had it, which meant the version we
# were about to call official had less QC than the one it superseded.
tensor2metric "$branch/dt.mif" -mask "$branch/mask.mif" \
  -value "$branch/eigenvalues.mif" -num 1,2,3
mrcalc "$branch/eigenvalues.mif" 0 -lt \
  "$branch/negative_eigenvalue_components.mif" -datatype bit
mrmath "$branch/negative_eigenvalue_components.mif" max \
  "$branch/negative_eigenvalue_voxels.mif" -axis 3 -datatype bit

mask_count="$(mrstats "$branch/mask.mif" -ignorezero -output count | tr -d '[:space:]')"
negative_count="$(mrstats "$branch/negative_eigenvalue_voxels.mif" -ignorezero -output count | tr -d '[:space:]')"
negative_percent="$(awk -v n="$negative_count" -v d="$mask_count" \
  'BEGIN { if (d > 0) printf "%.4f", 100*n/d; else print "nan" }')"

for name in b0_mean mask fa md eigenvalues negative_eigenvalue_voxels; do
  mrconvert "$branch/$name.mif" "$branch/$name.nii.gz"
done

cat > "$report" <<EOF
# ${patient_id} — Synb0-DisCo preprocessing QC

Generated: $(date -Is)

- Synb0 image: \`leonyichencai/synb0-disco:v3.1\`
- Phase encoding: \`$pe_dir\`
- Real b0 readout: \`$readout_time s\`
- Synthetic b0 readout: \`0 s\`
- Eddy: CPU, \`--repol --slm=linear --nthr=8 --cnr_maps --residuals\`

## Tensor fit

- Voxels in mask: \`$mask_count\`
- Voxels with at least one negative fitted eigenvalue: \`$negative_count\`
- Negative-eigenvalue fraction: \`$negative_percent%\`

## Manual gate

- [ ] Compare Synb0 b0 geometry against T1 at frontal/temporal lobes and midbrain.
- [ ] Reject the branch if folds, over-expansion or focal over-correction appear.
- [ ] Compare against the versioned rpe_none control before selecting the primary branch.
EOF

cat > "$branch/STATUS.txt" <<EOF
COMPLETE

Pipeline: single_shell_v1 / preproc_synb0
Completed: $(date -Is)
Synb0 outputs: OUTPUTS
Eddy acquisition parameters: INPUTS/eddy_acqparams.txt
EOF

echo "Completed Synb0 branch: $branch"
