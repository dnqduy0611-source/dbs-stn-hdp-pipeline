#!/bin/bash
# Manual gate 2 of qc/reports/04_single_shell_v1_synb0_qc.md - "reject the branch
# if folds, over-expansion or focal over-correction appear" - measured instead of
# eyeballed.
#
# A 1-D susceptibility displacement d(y) folds the image exactly where the
# Jacobian 1 + dd/dy goes non-positive: two source voxels land on one target
# voxel and the tissue there is destroyed, not merely moved.  Values far above 1
# are over-expansion (signal smeared thin), values near 0 are pile-up.
#
# topup wrote only the spline coefficients, and fnirtfileutils refuses to read a
# topup coefficient file (different intent from an fnirt one).  So re-run topup
# with the parameters recorded in b0_all.topup_log, unchanged, adding --fout and
# --jacout.  Everything goes to a scratch prefix; nothing official is touched.
#
# The re-run is only trustworthy if it reproduces the official field, so the last
# step compares the re-run's corrected b0 against the one the container produced.
# If those differ, the Jacobian below describes a different field and must not be
# used.
set -euo pipefail
unset PATH
export FSLDIR=/home/devold/fsl/share/fsl
export FSLOUTPUTTYPE=NIFTI_GZ
export PATH="$FSLDIR/bin:/usr/local/bin:/usr/bin:/bin"

D=/mnt/d/ossdbs_env/dbs_stn_hdp_pipeline/patients/sub-09/derivatives/single_shell_v1
O=$D/preproc_synb0/OUTPUTS
I=$D/preproc_synb0/INPUTS
W=/mnt/d/ossdbs_env/dbs_stn_hdp_pipeline/patients/sub-09/qc/synb0_gates
S=$W/_topup_rerun
mkdir -p "$S"

echo "FSL: $(cat "$FSLDIR/etc/fslversion" 2>/dev/null || echo '?')"
echo "acqparams:"; cat "$I/acqparams.txt"

echo
echo "=== chay lai topup, tham so y nguyen theo b0_all.topup_log, them --fout --jacout ==="
topup --imain="$O/b0_all.nii.gz" --datain="$I/acqparams.txt" \
  --out="$S/topup" --iout="$S/b0_all_topup.nii.gz" \
  --fout="$S/field_hz.nii.gz" --jacout="$S/jac" \
  --warpres=20,16,14,12,10,6,4 \
  --subsamp=2,2,2,2,2,1,1 \
  --fwhm=8,6,4,3,3,2,1 \
  --miter=5,5,5,5,5,15,15 \
  --lambda=0.005,0.001,0.0001,1.5e-05,5e-06,5e-07,5e-08 \
  --ssqlambda=1 --estmov=1,1,1,1,1,0,0 --minmet=0,0,0,0,0,1,1 \
  --regmod=bending_energy --splineorder=3 --numprec=double \
  --interp=spline --scale=1 --regrid=1

echo
echo "=== KIEM: chay lai co ra dung truong cu khong ==="
fslroi "$S/b0_all_topup.nii.gz" "$S/b0_new.nii.gz" 0 1
fslroi "$O/b0_all_topup.nii.gz" "$S/b0_old.nii.gz" 0 1
fslmaths "$S/b0_new.nii.gz" -sub "$S/b0_old.nii.gz" "$S/b0_diff.nii.gz"
old_mean=$(fslstats "$S/b0_old.nii.gz" -k "$W/brain_mask.nii.gz" -M)
d_abs=$(fslmaths "$S/b0_diff.nii.gz" -abs "$S/b0_absdiff.nii.gz"; \
        fslstats "$S/b0_absdiff.nii.gz" -k "$W/brain_mask.nii.gz" -M)
awk -v a="$d_abs" -v m="$old_mean" \
  'BEGIN{printf "  |chenh| trung binh trong nao = %.4f  = %.3f%% cua cuong do trung binh %.2f\n", a, 100*a/m, m}'

echo
echo "=== JACOBIAN trong nao (fold khi <= 0) ==="
jac=$(ls "$S"/jac_*.nii.gz | head -1)
echo "  file: $(basename "$jac")"
echo -n "  min / max / trung binh / do lech : "
fslstats "$jac" -k "$W/brain_mask.nii.gz" -R -M -S
echo -n "  p1 / p50 / p99                   : "
fslstats "$jac" -k "$W/brain_mask.nii.gz" -P 1 -P 50 -P 99
tot=$(fslstats "$W/brain_mask.nii.gz" -V | awk '{print $1}')
for spec in "<=0     nep gap:-u 0.000001" "<0.5  don manh:-u 0.5" ">2.0 gian manh:-l 2.0"; do
  lab=${spec%%:*}; opt=${spec##*:}
  n=$(fslstats "$jac" -k "$W/brain_mask.nii.gz" $opt -V | awk '{print $1}')
  awk -v l="$lab" -v n="$n" -v t="$tot" \
    'BEGIN{printf "  Jacobian %-16s %7d / %7d voxel = %.4f%%\n", l, n, t, 100*n/t}'
done

echo
echo "=== TRUONG (Hz) va do dich chuyen (mm) trong nao ==="
ro=$(awk 'NR==1{print $4}' "$I/acqparams.txt")
echo -n "  truong Hz  min/max/tb/dolech : "
fslstats "$S/field_hz.nii.gz" -k "$W/brain_mask.nii.gz" -R -M -S
echo -n "  |truong| p99 (Hz)            : "
fslmaths "$S/field_hz.nii.gz" -abs "$S/field_abs.nii.gz"
fslstats "$S/field_abs.nii.gz" -k "$W/brain_mask.nii.gz" -P 99
awk -v ro="$ro" 'BEGIN{printf "  readout tu acqparams = %s s  => 1 Hz = %.4f mm dich (voxel PE 1.7111 mm)\n", ro, ro*1.7111}'
