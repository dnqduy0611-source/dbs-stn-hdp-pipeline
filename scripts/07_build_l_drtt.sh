#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <patient_dir>"
  echo "Example: $0 /mnt/d/dbs_workspace/dbs_stn_hdp_pipeline/patients/patient_example"
  exit 2
fi

patient_dir="$(cd "$1" && pwd)"
patient_id="$(basename "$patient_dir")"

roi_dir="$patient_dir/derivatives/rois"
tract_dir="$patient_dir/derivatives/tracts"
dwi_dir="$patient_dir/derivatives/dwi"
qc_dir="$patient_dir/qc/reports"
log_dir="$patient_dir/logs"

mkdir -p "$roi_dir" "$tract_dir" "$qc_dir" "$log_dir"

log="$log_dir/07_build_l_drtt.log"
report="$qc_dir/07_build_l_drtt_qc.md"

exec > >(tee "$log") 2>&1

echo "== Build L_DRTT for $patient_id =="
echo "Date: $(date -Is)"
echo "Patient dir: $patient_dir"
echo

seg="$roi_dir/L_DRTT.seg.nrrd"
b0="$dwi_dir/b0_mean.nii.gz"
wholebrain="$tract_dir/wholebrain_ACT_fixed_1M.tck"
out_tck="$tract_dir/L_DRTT_current.tck"
out_vtk="$tract_dir/L_DRTT_current.vtk"
out_info="$tract_dir/L_DRTT_current_tckinfo.txt"

for f in "$seg" "$b0" "$wholebrain"; do
  if [[ ! -f "$f" ]]; then
    echo "Missing required input: $f" >&2
    exit 1
  fi
done

win_patient_dir="$(wslpath -w "$patient_dir")"

echo "Converting Slicer segmentation to DWI-space masks..."
include_r_dn="$roi_dir/R_dentate_nucleus_dwi.nii.gz"
include_l_rn="$roi_dir/L_RN_dwi.nii.gz"
include_l_pg="$roi_dir/L_PG_dwi.nii.gz"

if [[ ! -f "$include_r_dn" || ! -f "$include_l_rn" || ! -f "$include_l_pg" ]]; then
  win_script="$(wslpath -w "$(dirname "$0")/convert_slicer_seg_to_dwi_masks.py")"
  if command -v powershell.exe >/dev/null 2>&1; then
    powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "python '$win_script' '$win_patient_dir'"
  else
    echo "Missing DWI-space ROI masks and cannot call Windows Python from this shell." >&2
    echo "Run on Windows first:" >&2
    echo "python $win_script $win_patient_dir" >&2
    exit 1
  fi
else
  echo "DWI-space ROI masks already present; reusing existing masks."
fi

for f in "$include_r_dn" "$include_l_rn" "$include_l_pg"; do
  if [[ ! -f "$f" ]]; then
    echo "Missing converted ROI mask: $f" >&2
    exit 1
  fi
done

echo
echo "ROI voxel checks:"
mrstats "$include_r_dn"
mrstats "$include_l_rn"
mrstats "$include_l_pg"

echo
echo "Filtering ACT wholebrain tractogram..."
tckedit "$wholebrain" "$out_tck" \
  -include "$include_r_dn" \
  -include "$include_l_rn" \
  -include "$include_l_pg" \
  -force

tckconvert "$out_tck" "$out_vtk" -force
tckinfo "$out_tck" | tee "$out_info"

count="$(awk '/count:/ {print $2; exit}' "$out_info")"

cat > "$report" <<EOF
# L_DRTT Tract QC - $patient_id

Generated: $(date -Is)

## Inputs

- Source tractogram: \`$wholebrain\`
- Source segmentation: \`$seg\`
- Reference geometry: \`$b0\`

## ROI Masks

| Segment | Mask |
|---|---|
| R dentate nucleus | \`$include_r_dn\` |
| L RN | \`$include_l_rn\` |
| L PG | \`$include_l_pg\` |

## Filtering

\`\`\`bash
tckedit "$wholebrain" "$out_tck" \\
  -include "$include_r_dn" \\
  -include "$include_l_rn" \\
  -include "$include_l_pg" \\
  -force
\`\`\`

## Outputs

| Output | Path |
|---|---|
| L_DRTT tractogram | \`$out_tck\` |
| Slicer VTK | \`$out_vtk\` |
| TCK info | \`$out_info\` |

## Result

- Streamline count: \`$count\`

## Manual QC Checklist

- [ ] Open \`L_DRTT_current.vtk\` in Slicer with FA/b0/T2.
- [ ] Confirm the tract passes through the intended dentate nucleus, red nucleus, and precentral gyrus ROIs.
- [ ] Confirm there is no gross CST contamination or implausible extracerebral track.
- [ ] If count is low or trajectory is implausible, inspect each ROI mask in DWI space and consider a larger wholebrain tractogram or refined waypoint ROIs.

EOF

echo
echo "L_DRTT build complete."
echo "Streamline count: $count"
echo "Output: $out_tck"
echo "Slicer VTK: $out_vtk"
echo "Report: $report"
