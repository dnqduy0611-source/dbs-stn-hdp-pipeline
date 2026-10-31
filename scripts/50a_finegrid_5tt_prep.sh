#!/bin/bash
# Phase 0: build a fine (DWI 1mm) 5TT, patch STN+Julich-dentate sGM at 1mm
# (no coarse-grid inflation), and generate a matching GMWMI. Shared by both
# staged runs (A: lmax8, B: lmax6). Nothing existing is overwritten.
set -euo pipefail
export FSLDIR="${FSLDIR:-/home/devold/fsl}"
[ -f "$FSLDIR/etc/fslconf/fsl.sh" ] && source "$FSLDIR/etc/fslconf/fsl.sh"
export PATH="$FSLDIR/share/fsl/bin:$PATH"

# Truoc day dong nay hardcode duong dan cua sub-06: chay lai cho ca khac
# ma quen sua se GHI DE len du lieu cua sub-06 ma khong bao gi. Nay bat buoc khai bao:
#   PATIENT_DIR=/mnt/d/.../patients/<Ten> bash scripts/50a_finegrid_5tt_prep.sh
P="${PATIENT_DIR:?Phai dat PATIENT_DIR (duong dan thu muc benh nhan)}"
D="$P/derivatives"
OUT="$D/pilot_finegrid_5tt_staged_1m"
PREP="$OUT/prep"
mkdir -p "$PREP" "$OUT/filtered" "$OUT/logs"

FA1MM="$D/dwi/fa_1mm_ref.nii.gz"
TT_T1="$D/registration/5tt_t1.mif"
XFM="$D/registration/t12dwi_mrtrix.txt"
LSTN="$D/pilot_revised_bilateral_stn_julich_1m/rois/L_STN_revised_dwi_1mm_tract.nii.gz"
RSTN="$D/pilot_revised_bilateral_stn_julich_1m/rois/R_STN_revised_dwi_1mm_tract.nii.gz"
JUL="$D/qc_drtt_julich_dentate/Julich_R_dentate_in_DWI_1mm.nii.gz"

echo "== 1. Resample T1 5TT -> DWI 1mm grid =="
mrtransform "$TT_T1" -linear "$XFM" -template "$FA1MM" -interp linear \
  "$PREP/5tt_dwi_1mm_raw.mif" -force -quiet
echo "== 1b. Renormalize (clamp negatives, force per-voxel tissue sum to 1) =="
mrcalc "$PREP/5tt_dwi_1mm_raw.mif" 0.0 -max "$PREP/_c.mif" -force -quiet
mrmath "$PREP/_c.mif" sum -axis 3 "$PREP/_s.mif" -force -quiet
# where brain sum>0 divide by sum, else divide by 1 (keeps background 0)
mrcalc "$PREP/_s.mif" 0 -gt "$PREP/_s.mif" 1 -if "$PREP/_snz.mif" -force -quiet
mrcalc "$PREP/_c.mif" "$PREP/_snz.mif" -divide "$PREP/5tt_dwi_1mm.mif" -force -quiet
rm -f "$PREP/_c.mif" "$PREP/_s.mif" "$PREP/_snz.mif"
5ttcheck "$PREP/5tt_dwi_1mm.mif" -quiet && echo "   base 5tt OK" || echo "WARN: 5ttcheck still flags base 5tt"

echo "== 2. Build 1mm sGM patch = L_STN + R_STN + Julich dentate =="
mrcalc "$LSTN" 0 -gt "$RSTN" 0 -gt -add "$JUL" 0 -gt -add 0 -gt \
  "$PREP/sgm_patch_1mm.nii.gz" -datatype bit -force -quiet
nL=$(mrstats "$LSTN" -ignorezero -output count -quiet | tr -d '[:space:]')
nR=$(mrstats "$RSTN" -ignorezero -output count -quiet | tr -d '[:space:]')
nJ=$(mrstats "$JUL"  -ignorezero -output count -quiet | tr -d '[:space:]')
nP=$(mrstats "$PREP/sgm_patch_1mm.nii.gz" -ignorezero -output count -quiet | tr -d '[:space:]')
echo "   L_STN=$nL R_STN=$nR Julich=$nJ  union_patch=$nP voxels (1mm^3 each)"
echo "   -> STN patch volume ~ $((nL + nR)) mm3 (anatomical STN ~667 mm3; coarse was 2895 mm3)"

echo "== 3. Patch sGM on the fine 5TT + verify =="
5ttedit "$PREP/5tt_dwi_1mm.mif" "$PREP/5tt_dwi_1mm_patched.mif" \
  -sgm "$PREP/sgm_patch_1mm.nii.gz" -force -quiet
5ttcheck "$PREP/5tt_dwi_1mm_patched.mif" -quiet || echo "WARN: 5ttcheck flagged patched 5tt"
mrconvert "$PREP/5tt_dwi_1mm_patched.mif" -coord 3 1 -axes 0,1,2 \
  "$PREP/sgm_channel_1mm.nii.gz" -force -quiet
nS=$(mrstats "$PREP/sgm_channel_1mm.nii.gz" -ignorezero -output count -quiet | tr -d '[:space:]')
echo "   sGM channel nonzero voxels after patch = $nS"

echo "== 4. GMWMI seed from fine patched 5TT =="
5tt2gmwmi "$PREP/5tt_dwi_1mm_patched.mif" "$PREP/gmwmi_1mm.mif" -force -quiet
nG=$(mrstats "$PREP/gmwmi_1mm.mif" -ignorezero -output count -quiet | tr -d '[:space:]')
echo "   GMWMI nonzero voxels = $nG"

echo "PHASE0_DONE"
