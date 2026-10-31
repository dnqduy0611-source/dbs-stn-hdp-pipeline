#!/usr/bin/env bash
# Dua mot ROI tu khong gian T1 sang luoi DWI 1mm — DUNG CACH.
#
# Usage: warp_roi_t1_to_dwi_1mm.sh <roi_t1.nii.gz> <out_dwi_1mm.nii.gz> <patient_dir>
#        [t12dwi.mat] [t1_in.nii.gz] [b0_brain.nii.gz] [fa_1mm_ref.nii.gz]
#
# TAI SAO PHAI CO SCRIPT NAY
# Cach viet pho bien trong repo:
#     flirt -in ROI_T1 -ref fa_1mm_ref -applyxfm -init t12dwi_fsl.mat
# la SAI. Ma tran FSL gan chat voi cap luoi (in, ref) da sinh ra no; t12dwi_fsl.mat
# sinh voi ref = luoi b0 (2.46/2.5 mm), con fa_1mm_ref la luoi 1 mm. Dung sai -ref
# sinh ra TINH TIEN THUAN bang nua hieu voxel, im lang, khong bao loi.
# Da do: sub-01 1.28 mm, sub-06 1.53 mm, sub-03 1.53 mm — xoay 0.000 do, giong het o moi ROI.
#
# Sua bang cach doi -ref thanh b0_brain roi mrtransform len 1mm thi DUNG KHUNG nhung
# ROI nho bi nat vi phai di qua luoi 2.5 mm (STN 67 -> 89 voxel, centroid xe 2.7 mm).
#
# CACH LAM O DAY: bo han flirt o buoc lay mau. Doi ma tran sang world bang
# transformconvert (dung dung cap luoi da sinh ra no), roi mrtransform mot lan tu
# T1 thang len luoi 1mm. Chi mot lan lay mau, khong qua luoi tho, va — quan trong
# nhat — DUNG Y CHUOI ma 5tt_dwi di qua, nen ROI va 5TT khop nhau theo cau tao.
set -euo pipefail
export FSLDIR="${FSLDIR:-/home/devold/fsl}"
[ -f "$FSLDIR/etc/fslconf/fsl.sh" ] && source "$FSLDIR/etc/fslconf/fsl.sh"
export PATH="$FSLDIR/share/fsl/bin:$PATH"

if [ $# -lt 3 ]; then
  sed -n '2,26p' "$0" >&2
  exit 2
fi
src="$1"; out="$2"; P="$(cd "$3" && pwd)"
REG="$P/derivatives/registration"; DW="$P/derivatives/dwi"
MAT="${4:-$REG/t12dwi_fsl.mat}"
T1IN="${5:-$REG/t1_brain_from5tt.nii.gz}"
B0B="${6:-$REG/b0_brain.nii.gz}"
FA1="${7:-$DW/fa_1mm_ref.nii.gz}"

for f in "$src" "$MAT" "$T1IN" "$B0B" "$FA1"; do
  [ -f "$f" ] || { echo "Thieu: $f" >&2; exit 1; }
done

# ROI nguon phai nam tren dung luoi $T1IN, neu khong transformconvert se sai khung
python3 - "$src" "$T1IN" <<'PY'
import sys, numpy as np, nibabel as nib
a, b = nib.load(sys.argv[1]), nib.load(sys.argv[2])
if not (a.shape[:3] == b.shape[:3]
        and np.allclose(a.header.get_zooms()[:3], b.header.get_zooms()[:3], atol=1e-4)
        and np.allclose(a.affine, b.affine, atol=1e-3)):
    sys.exit(f"LOI: ROI {sys.argv[1]} khong cung luoi voi {sys.argv[2]}. "
             "Ma tran FSL se bi ap sai khung.")
PY

tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT

transformconvert "$MAT" "$T1IN" "$B0B" flirt_import "$tmp/world.txt" -force -quiet
mrtransform "$src" -linear "$tmp/world.txt" -template "$FA1" -interp nearest \
    "$tmp/roi.mif" -force -quiet
mrcalc "$tmp/roi.mif" 0.5 -gt "$tmp/roi_bit.mif" -datatype bit -force -quiet
# Preserve the template's storage orientation as well as its physical extent.
# Without this final conversion MRtrix may write +1,2,3 while fa_1mm_ref uses
# -1,2,3. The images still occupy the same world grid, but their NIfTI affine
# and voxel-array indexing differ, which defeats strict common-grid QC.
ref_strides="$(mrinfo "$FA1" -strides -quiet | tr '\n' ' ' | xargs)"
mrconvert "$tmp/roi_bit.mif" "$out" -datatype bit -strides "$ref_strides" \
    -force -quiet

n=$(mrstats "$out" -output count -mask "$out" 2>/dev/null || echo "?")
echo "  $(basename "$src") -> $(basename "$out")   ($n voxel)"
