#!/bin/bash
# 10M candidate on the fine branch: fine 1mm patched 5TT + fine GMWMI + lmax6
# FOD, fixed RandomSeed 4815, SIFT2, and the FULL 48g filter set. Writes to a
# NEW directory; 48g is left untouched.
set -euo pipefail
export FSLDIR="${FSLDIR:-/home/devold/fsl}"
[ -f "$FSLDIR/etc/fslconf/fsl.sh" ] && source "$FSLDIR/etc/fslconf/fsl.sh"
export PATH="$FSLDIR/share/fsl/bin:$PATH"

# Truoc day hardcode duong dan cua sub-06 — xem ghi chu o 50a.
P="${PATIENT_DIR:?Phai dat PATIENT_DIR (duong dan thu muc benh nhan)}"
D="$P/derivatives"
PREP="$D/pilot_finegrid_5tt_staged_1m/prep"
OUT="$D/wholebrain_finegrid_lmax6_10m"
TRACT="$OUT/filtered"
mkdir -p "$TRACT" "$P/logs"
log="$P/logs/50d_finegrid_lmax6_10m.log"
exec > >(tee "$log") 2>&1

FOD="$D/fod/wmfod_norm_lmax6.mif"
FIVETT="$PREP/5tt_dwi_1mm_patched.mif"
GMWMI="$PREP/gmwmi_1mm.mif"
select_n=10000000
random_seed=4815

# --- filtering ROIs (identical to 48g) ---
l_stn="$D/pilot_revised_bilateral_stn_julich_1m/rois/L_STN_revised_dwi_1mm_tract.nii.gz"
r_stn="$D/pilot_revised_bilateral_stn_julich_1m/rois/R_STN_revised_dwi_1mm_tract.nii.gz"
l_pg="$D/rois/hdp_left_highres_stn_endpoint/HarvardOxford_Left_Precentral_Gyrus_in_DWI_1mm_tract.nii.gz"
r_pg="$D/rois/hdp_right_atlas_pg_highres_stn_waypoint/HarvardOxford_Right_Precentral_Gyrus_in_DWI_1mm_tract.nii.gz"
l_hdp_exclude="$D/rois/hdp_left_stn_dil1_endpoint/Exclude_HDP_manual_DWI.nii.gz"
r_hdp_exclude="$D/rois/hdp_right_atlas_pg_highres_stn_waypoint_exclude/R_HDP_exclude_manual_DWI.nii.gz"
julich="$D/qc_drtt_julich_dentate/Julich_R_dentate_in_DWI_1mm.nii.gz"
manual_dentate="$D/rois/drtt_left_atlas_pg_highres_waypoint/R_Dentate_nucleus_dwi_1mm_tract.nii.gz"
scp="$D/rois/drtt_left_targeted_scp/SCP_bilateral_dwi_1mm_tract.nii.gz"
rn="$D/qc_drtt_middlebrooks_rn_dilation/L_RN_manual_volume_preserved_DWI_1mm.nii.gz"
drtt_exclude="$D/rois/drtt_left_atlas_pg_highres_waypoint/L_DRTT_exclude_DWI_1mm.nii.gz"

WB="$OUT/wholebrain_${select_n}_finegrid_lmax6.tck"
WEIGHTS="$OUT/wholebrain_${select_n}_finegrid_lmax6_sift2_weights.txt"

echo "== tckgen ${select_n} (fine 5TT + fine GMWMI + lmax6, RandomSeed $random_seed) =="
tckgen "$FOD" "$WB" \
  -algorithm iFOD2 -act "$FIVETT" -backtrack -crop_at_gmwmi \
  -seed_gmwmi "$GMWMI" -select "$select_n" -cutoff 0.06 \
  -minlength 10 -maxlength 200 -config RandomSeed "$random_seed" \
  -force 2>&1 | tee "$OUT/tckgen_10m.log" | tail -2

echo "== SIFT2 =="
tcksift2 "$WB" "$FOD" "$WEIGHTS" -force -quiet
wb_n="$(tckstats "$WB" -output count -quiet | tr -d '[:space:]')"
w_n="$(awk '!/^#/ {c += NF} END {print c+0}' "$WEIGHTS")"
echo "   wholebrain=$wb_n  sift2_weights=$w_n"
[[ "$wb_n" -ne "$w_n" ]] && { echo "wholebrain/weights mismatch" >&2; exit 1; }

results="$OUT/filter_results.tsv"
printf "tract\tdefinition\tstreamlines\tweight_count\tsift2_sum\ttck\tvtk\n" > "$results"
filter_from() {
  local in_tck="$1" in_w="$2" tag="$3" def="$4"; shift 4
  local o_tck="$TRACT/${tag}.tck" o_w="$TRACT/${tag}_sift2_weights.txt" o_vtk="$TRACT/${tag}.vtk"
  tckedit "$in_tck" "$o_tck" "$@" -tck_weights_in "$in_w" -tck_weights_out "$o_w" -force -quiet
  local n wn wsum
  n="$(tckstats "$o_tck" -output count -quiet | tr -d '[:space:]')"
  wn="$(awk '!/^#/ {c += NF} END {print c+0}' "$o_w")"
  wsum="$(awk '!/^#/ {for(i=1;i<=NF;i++) s+=$i} END {printf "%.8f", s+0}' "$o_w")"
  [[ "$n" -ne "$wn" ]] && { echo "weight mismatch $tag: $n vs $wn" >&2; exit 1; }
  local vtk="none"; if [[ "$n" -gt 0 ]]; then tckconvert "$o_tck" "$o_vtk" -force -quiet; vtk="$o_vtk"; else rm -f "$o_vtk"; fi
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$tag" "$def" "$n" "$wn" "$wsum" "$o_tck" "$vtk" >> "$results"
}

echo "== filter HDP (full 48g set) =="
filter_from "$WB" "$WEIGHTS" l_hdp_revised_endpoints "L_STN + left PG both endpoints" -include "$l_stn" -include "$l_pg" -ends_only
filter_from "$WB" "$WEIGHTS" l_hdp_revised_endpoints_exclude "L_STN + left PG endpoints + manual exclude" -include "$l_stn" -include "$l_pg" -exclude "$l_hdp_exclude" -ends_only
filter_from "$WB" "$WEIGHTS" l_hdp_revised_waypoints_exclude "L_STN + left PG waypoints + manual exclude" -include "$l_stn" -include "$l_pg" -exclude "$l_hdp_exclude"
filter_from "$WB" "$WEIGHTS" r_hdp_revised_endpoints "R_STN + right PG both endpoints" -include "$r_stn" -include "$r_pg" -ends_only
filter_from "$WB" "$WEIGHTS" r_hdp_revised_endpoints_exclude "R_STN + right PG endpoints + manual exclude" -include "$r_stn" -include "$r_pg" -exclude "$r_hdp_exclude" -ends_only
filter_from "$WB" "$WEIGHTS" r_hdp_revised_waypoints_exclude "R_STN + right PG waypoints + manual exclude" -include "$r_stn" -include "$r_pg" -exclude "$r_hdp_exclude"

echo "== filter DRTT (full 48g set) =="
filter_from "$WB" "$WEIGHTS" drtt_julich_pg_endpoints "Julich right dentate + left PG both endpoints" -include "$julich" -include "$l_pg" -ends_only
filter_from "$WB" "$WEIGHTS" drtt_manual_pg_endpoints "Manual right dentate + left PG both endpoints" -include "$manual_dentate" -include "$l_pg" -ends_only
filter_from "$TRACT/drtt_julich_pg_endpoints.tck" "$TRACT/drtt_julich_pg_endpoints_sift2_weights.txt" drtt_julich_pg_endpoints_scp_rn_exclude "Julich-PG endpoints + SCP + RN + exclude" -include "$scp" -include "$rn" -exclude "$drtt_exclude"
filter_from "$TRACT/drtt_manual_pg_endpoints.tck" "$TRACT/drtt_manual_pg_endpoints_sift2_weights.txt" drtt_manual_pg_endpoints_scp_rn_exclude "Manual-PG endpoints + SCP + RN + exclude" -include "$scp" -include "$rn" -exclude "$drtt_exclude"

echo "FINEGRID_10M_DONE"
column -t -s$'\t' "$results" | cut -c1-95
