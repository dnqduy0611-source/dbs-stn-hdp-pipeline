#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <patient_dir>"
  echo "Example: $0 /mnt/d/ossdbs_env/dbs_stn_hdp_pipeline/patients/sub-09"
  exit 2
fi

patient_dir="$(cd "$1" && pwd)"
config="$patient_dir/config.yaml"

if [[ ! -f "$config" ]]; then
  echo "Missing config: $config" >&2
  exit 1
fi

report_dir="$patient_dir/qc/reports"
log_dir="$patient_dir/logs"
mkdir -p "$report_dir" "$log_dir"

report="$report_dir/02_input_qc.md"
json_report="$report_dir/02_input_qc.json"
log="$log_dir/02_check_inputs.log"

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
t1="$(yaml_value anat.t1)"
t2="$(yaml_value anat.t2)"
swi="$(yaml_value anat.swi)"
postop_ct="$(yaml_value ct.postop_ct)"
pe_dir="$(yaml_value dwi.pe_dir)"
rpe_mode="$(yaml_value dwi.rpe_mode)"

exists_status() {
  local path="$1"
  if [[ -n "$path" && -f "$path" ]]; then
    echo "present"
  else
    echo "missing"
  fi
}

mrinfo_field() {
  local path="$1"
  local flag="$2"
  if [[ -n "$path" && -f "$path" ]]; then
    mrinfo "$path" "$flag" 2>/dev/null | tr '\n' ' ' | sed 's/[[:space:]]*$//'
  else
    echo ""
  fi
}

analyze_gradients() {
  python3 - "$dwi_bval" "$dwi_bvec" <<'PY'
from pathlib import Path
import json
import math
import sys

bval_path = Path(sys.argv[1])
bvec_path = Path(sys.argv[2])
out = {
    "bval_present": bval_path.is_file(),
    "bvec_present": bvec_path.is_file(),
    "n_bvals": None,
    "n_bvecs": None,
    "shells": [],
    "nonzero_volume_count": None,
    "unique_signed_vector_count": None,
    "unique_axis_count": None,
    "direction_repetition_counts": [],
    "repeated_block_pattern": None,
    "b0_count": None,
    "bvec_rows": None,
    "warnings": [],
}

if bval_path.is_file():
    vals = [float(x) for x in bval_path.read_text().split()]
    out["n_bvals"] = len(vals)
    rounded = [int(round(v / 100.0) * 100) for v in vals]
    shells = sorted(set(rounded))
    out["shells"] = shells
    out["b0_count"] = sum(1 for v in vals if v < 50)
    out["nonzero_volume_count"] = sum(1 for v in vals if v >= 50)
else:
    vals = []
    out["warnings"].append("Missing bval file.")

if bvec_path.is_file():
    rows = [[float(x) for x in line.split()] for line in bvec_path.read_text().splitlines() if line.strip()]
    out["bvec_rows"] = len(rows)
    if rows:
        lengths = [len(r) for r in rows]
        out["n_bvecs"] = lengths[0] if len(set(lengths)) == 1 else None
        if len(rows) != 3:
            out["warnings"].append(f"Expected 3 bvec rows, found {len(rows)}.")
        if len(set(lengths)) != 1:
            out["warnings"].append(f"bvec rows have inconsistent lengths: {lengths}.")
        if vals and len(rows) == 3 and lengths[0] != len(vals):
            out["warnings"].append(f"bval count ({len(vals)}) != bvec count ({lengths[0]}).")

        if vals and len(rows) == 3 and len(set(lengths)) == 1 and lengths[0] == len(vals):
            vectors = []
            for index, bvalue in enumerate(vals):
                if bvalue < 50:
                    continue
                vector = [rows[axis][index] for axis in range(3)]
                norm = math.sqrt(sum(component * component for component in vector))
                if norm < 1e-8:
                    out["warnings"].append(
                        f"Non-b0 volume {index} has a zero-length gradient vector."
                    )
                    continue
                vectors.append(tuple(component / norm for component in vector))

            def cluster_directions(items, antipodal=False, tolerance=1e-5):
                representatives = []
                counts = []
                assignments = []
                for vector in items:
                    match = None
                    for candidate_index, candidate in enumerate(representatives):
                        dot = sum(a * b for a, b in zip(vector, candidate))
                        if antipodal:
                            dot = abs(dot)
                        if 1.0 - dot <= tolerance:
                            match = candidate_index
                            break
                    if match is None:
                        representatives.append(vector)
                        counts.append(1)
                        assignments.append(len(representatives) - 1)
                    else:
                        counts[match] += 1
                        assignments.append(match)
                return representatives, counts, assignments

            signed_reps, signed_counts, signed_assignments = cluster_directions(vectors)
            axis_reps, _, _ = cluster_directions(vectors, antipodal=True)
            out["unique_signed_vector_count"] = len(signed_reps)
            out["unique_axis_count"] = len(axis_reps)
            out["direction_repetition_counts"] = sorted(signed_counts)

            half = len(signed_assignments) // 2
            if (
                len(signed_assignments) % 2 == 0
                and signed_assignments[:half] == signed_assignments[half:]
            ):
                out["repeated_block_pattern"] = (
                    f"two identical ordered blocks of {half} non-b0 vectors"
                )
else:
    out["warnings"].append("Missing bvec file.")

if vals and out["nonzero_volume_count"] is not None:
    if out["unique_axis_count"] is not None and out["unique_axis_count"] < 30:
        out["warnings"].append("Fewer than 30 non-b0 diffusion directions; tractography will be weak.")
    if len([s for s in out["shells"] if s >= 50]) < 2:
        out["warnings"].append(
            "Single-shell DWI detected: standard 3-tissue MSMT-CSD is not identifiable. "
            "Use a 2-tissue model or MRtrix3Tissue SS3T-CSD."
        )

print(json.dumps(out, indent=2))
PY
}

gradient_json="$(analyze_gradients)"

{
  echo "== Input QC for $patient_id =="
  echo "Date: $(date -Is)"
  echo "Patient dir: $patient_dir"
  echo
  echo "Files:"
  printf "  DWI NIfTI:   %-8s %s\n" "$(exists_status "$dwi_nifti")" "$dwi_nifti"
  printf "  DWI bvec:    %-8s %s\n" "$(exists_status "$dwi_bvec")" "$dwi_bvec"
  printf "  DWI bval:    %-8s %s\n" "$(exists_status "$dwi_bval")" "$dwi_bval"
  printf "  DWI JSON:    %-8s %s\n" "$(exists_status "$dwi_json")" "$dwi_json"
  printf "  T1:          %-8s %s\n" "$(exists_status "$t1")" "$t1"
  printf "  T2:          %-8s %s\n" "$(exists_status "$t2")" "$t2"
  printf "  SWI:         %-8s %s\n" "$(exists_status "$swi")" "$swi"
  printf "  Post-op CT:  %-8s %s\n" "$(exists_status "$postop_ct")" "$postop_ct"
  echo
  echo "DWI config:"
  echo "  pe_dir:   $pe_dir"
  echo "  rpe_mode: $rpe_mode"
  echo
  echo "Image headers:"
  echo "  DWI size:    $(mrinfo_field "$dwi_nifti" -size)"
  echo "  DWI spacing: $(mrinfo_field "$dwi_nifti" -spacing)"
  echo "  T1 size:     $(mrinfo_field "$t1" -size)"
  echo "  T1 spacing:  $(mrinfo_field "$t1" -spacing)"
  echo "  T2 size:     $(mrinfo_field "$t2" -size)"
  echo "  T2 spacing:  $(mrinfo_field "$t2" -spacing)"
  echo "  SWI size:    $(mrinfo_field "$swi" -size)"
  echo "  SWI spacing: $(mrinfo_field "$swi" -spacing)"
  echo "  CT size:     $(mrinfo_field "$postop_ct" -size)"
  echo "  CT spacing:  $(mrinfo_field "$postop_ct" -spacing)"
  echo
  echo "Gradient analysis:"
  echo "$gradient_json"
} | tee "$log"

python3 - "$json_report" "$patient_id" "$patient_dir" "$gradient_json" <<'PY'
from pathlib import Path
import json
import sys

out = Path(sys.argv[1])
payload = {
    "patient_id": sys.argv[2],
    "patient_dir": sys.argv[3],
    "gradient_analysis": json.loads(sys.argv[4]),
}
out.write_text(json.dumps(payload, indent=2))
PY

cat > "$report" <<EOF
# Input QC - $patient_id

Generated: $(date -Is)

## Files

| Input | Status | Path |
|---|---|---|
| DWI NIfTI | $(exists_status "$dwi_nifti") | \`$dwi_nifti\` |
| DWI bvec | $(exists_status "$dwi_bvec") | \`$dwi_bvec\` |
| DWI bval | $(exists_status "$dwi_bval") | \`$dwi_bval\` |
| DWI JSON | $(exists_status "$dwi_json") | \`$dwi_json\` |
| T1 | $(exists_status "$t1") | \`$t1\` |
| T2 | $(exists_status "$t2") | \`$t2\` |
| SWI | $(exists_status "$swi") | \`$swi\` |
| Post-op CT | $(exists_status "$postop_ct") | \`$postop_ct\` |

## DWI Configuration

- Phase encoding direction: \`$pe_dir\`
- Reverse phase mode: \`$rpe_mode\`

## Image Headers

| Image | Size | Spacing |
|---|---|---|
| DWI | \`$(mrinfo_field "$dwi_nifti" -size)\` | \`$(mrinfo_field "$dwi_nifti" -spacing)\` |
| T1 | \`$(mrinfo_field "$t1" -size)\` | \`$(mrinfo_field "$t1" -spacing)\` |
| T2 | \`$(mrinfo_field "$t2" -size)\` | \`$(mrinfo_field "$t2" -spacing)\` |
| SWI | \`$(mrinfo_field "$swi" -size)\` | \`$(mrinfo_field "$swi" -spacing)\` |
| Post-op CT | \`$(mrinfo_field "$postop_ct" -size)\` | \`$(mrinfo_field "$postop_ct" -spacing)\` |

## Gradient Analysis

\`\`\`json
$gradient_json
\`\`\`

## Manual QC Needed Before Preprocessing

- [ ] Confirm phase-encoding direction from DICOM/sidecar.
- [ ] Confirm whether reverse phase B0 exists.
- [ ] Check DWI orientation and bvec/bval pairing.
- [ ] Check T1/T2/SWI/post-op CT are from the same patient and correct session.

EOF

echo
echo "Wrote:"
echo "  $report"
echo "  $json_report"
echo "  $log"
