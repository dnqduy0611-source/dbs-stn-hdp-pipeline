#!/bin/bash
# Regenerate WM FOD at lmax 6 (30 unique directions cannot support lmax 8).
# Reuses existing Dhollander responses. Does NOT overwrite the lmax 8 FOD.
set -euo pipefail
# Truoc day hardcode duong dan cua sub-06 — xem ghi chu o 50a.
P="${PATIENT_DIR:?Phai dat PATIENT_DIR (duong dan thu muc benh nhan)}"
D="$P/derivatives"
DWI="$D/dwi/dwi_bias.mif"
MASK="$D/dwi/mask.mif"
F="$D/fod"

dwi2fod msmt_csd "$DWI" \
  "$F/wm_response.txt"  "$F/wmfod_lmax6.mif" \
  "$F/gm_response.txt"  "$F/gm_lmax6.mif" \
  "$F/csf_response.txt" "$F/csf_lmax6.mif" \
  -mask "$MASK" -lmax 6,0,0 -force -quiet
mtnormalise \
  "$F/wmfod_lmax6.mif"  "$F/wmfod_norm_lmax6.mif" \
  "$F/gm_lmax6.mif"     "$F/gm_norm_lmax6.mif" \
  "$F/csf_lmax6.mif"    "$F/csf_norm_lmax6.mif" \
  -mask "$MASK" -force -quiet
echo -n "wmfod_norm_lmax6 SH volumes (expect 28 for lmax6): "
mrinfo "$F/wmfod_norm_lmax6.mif" -size | awk '{print $4}'
echo "FOD_LMAX6_DONE"
