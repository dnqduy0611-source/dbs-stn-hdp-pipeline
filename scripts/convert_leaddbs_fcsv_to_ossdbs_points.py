#!/usr/bin/env python3
"""
Convert Lead-DBS/Slicer electrode fiducials from MNI RAS to spaces used by OSS-DBS.

This script keeps electrode points as floating-point coordinates throughout.
It does not rasterize points into a labelmap.

Expected input for PatientExample:
  - .fcsv points are Slicer RAS coordinates in MNI152NLin2009bAsym space.
  - Lead-DBS ANTs displacement field is stored on the MNI grid and its vector
    components follow ITK/LPS convention, so vector x/y are flipped before
    adding to RAS points.
  - MRtrix t12dwi_mrtrix.txt uses MRtrix reverse convention for image reslicing;
    for point coordinates from T1/anchorNative to DWI, use the inverse matrix.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates


ROOT = Path(r"D:\dbs_workspace")
DEFAULT_FCSV = ROOT / "lead_dbs_work" / "PatientExampleLeadDBS6" / "derivatives" / "leaddbs" / "sub-PatientExample" / "reconstruction" / "sub-PatientExample_desc-electrodefiducials.fcsv"
DEFAULT_MNI_TO_ANCHOR_FIELD = ROOT / "lead_dbs_work" / "PatientExampleLeadDBS6" / "derivatives" / "leaddbs" / "sub-PatientExample" / "normalization" / "transformations" / "sub-PatientExample_from-anchorNative_to-MNI152NLin2009bAsym_desc-ants.nii.gz"
DEFAULT_T12DWI = ROOT / "dbs_stn_hdp_pipeline" / "patients" / "patient_example" / "derivatives" / "registration" / "t12dwi_mrtrix.txt"
DEFAULT_OUTPUT = ROOT / "dbs_stn_hdp_pipeline" / "patients" / "patient_example" / "derivatives" / "lead_dbs" / "sub-PatientExample_electrodefiducials_spaces"


def read_fcsv(path: Path) -> list[dict]:
    points = []
    with path.open(newline="") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            row = next(csv.reader([line]))
            points.append(
                {
                    "label": row[11],
                    "mni_ras": np.array([float(row[1]), float(row[2]), float(row[3])], dtype=float),
                }
            )
    points.sort(key=lambda item: int(item["label"][1:]))
    return points


def sample_vector_field(field_path: Path, ras_points: np.ndarray) -> np.ndarray:
    img = nib.load(str(field_path))
    data = np.asanyarray(img.dataobj)
    if data.ndim == 5:
        data = data[:, :, :, 0, :]
    if data.shape[-1] != 3:
        raise ValueError(f"Expected a 3-vector displacement field, got shape {data.shape}")

    ijk = (np.linalg.inv(img.affine) @ np.c_[ras_points, np.ones(len(ras_points))].T).T[:, :3]
    sampled = np.vstack(
        [
            map_coordinates(data[..., axis], [ijk[:, 0], ijk[:, 1], ijk[:, 2]], order=1, mode="nearest")
            for axis in range(3)
        ]
    ).T
    return sampled


def mni_ras_to_anchor_ras(mni_ras: np.ndarray, field_path: Path) -> np.ndarray:
    itk_lps_displacement = sample_vector_field(field_path, mni_ras)
    ras_displacement = itk_lps_displacement.copy()
    ras_displacement[:, 0] *= -1
    ras_displacement[:, 1] *= -1
    return mni_ras + ras_displacement


def apply_affine(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return (matrix @ np.c_[points, np.ones(len(points))].T).T[:, :3]


def electrode_summary(labels: list[str], points: np.ndarray) -> dict:
    axis = points[-1] - points[0]
    axis_norm = float(np.linalg.norm(axis))
    direction = axis / axis_norm if axis_norm else axis
    spacings = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return {
        "labels": labels,
        "tip_or_first_point": points[0].tolist(),
        "last_point": points[-1].tolist(),
        "axis_first_to_last_unit": direction.tolist(),
        "axis_length_mm": axis_norm,
        "mean_neighbor_spacing_mm": float(np.mean(spacings)),
        "neighbor_spacings_mm": spacings.tolist(),
    }


def rows_for_space(labels: Iterable[str], mni: np.ndarray, anchor: np.ndarray, dwi: np.ndarray) -> list[dict]:
    rows = []
    for label, p_mni, p_anchor, p_dwi in zip(labels, mni, anchor, dwi):
        rows.append(
            {
                "label": label,
                "mni_ras_x": p_mni[0],
                "mni_ras_y": p_mni[1],
                "mni_ras_z": p_mni[2],
                "anchorNative_ras_x": p_anchor[0],
                "anchorNative_ras_y": p_anchor[1],
                "anchorNative_ras_z": p_anchor[2],
                "dwi_ras_x": p_dwi[0],
                "dwi_ras_y": p_dwi[1],
                "dwi_ras_z": p_dwi[2],
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fcsv", type=Path, default=DEFAULT_FCSV)
    parser.add_argument("--mni-to-anchor-field", type=Path, default=DEFAULT_MNI_TO_ANCHOR_FIELD)
    parser.add_argument("--t12dwi-mrtrix", type=Path, default=DEFAULT_T12DWI)
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    points = read_fcsv(args.fcsv)
    labels = [p["label"] for p in points]
    mni_ras = np.vstack([p["mni_ras"] for p in points])

    anchor_ras = mni_ras_to_anchor_ras(mni_ras, args.mni_to_anchor_field)

    # MRtrix -linear matrices use reverse convention for image reslicing:
    # the matrix maps template(DWI) -> moving(T1). For point coordinates in
    # moving(T1/anchorNative) going to template(DWI), use inverse(matrix).
    t12dwi_reverse = np.loadtxt(args.t12dwi_mrtrix)
    anchor_to_dwi_points = np.linalg.inv(t12dwi_reverse)
    dwi_ras = apply_affine(anchor_ras, anchor_to_dwi_points)

    rows = rows_for_space(labels, mni_ras, anchor_ras, dwi_ras)
    csv_path = args.out_prefix.with_suffix(".csv")
    json_path = args.out_prefix.with_suffix(".json")
    write_csv(csv_path, rows)

    left = slice(0, 8)
    right = slice(8, 16)
    summary = {
        "inputs": {
            "fcsv": str(args.fcsv),
            "mni_to_anchor_field": str(args.mni_to_anchor_field),
            "t12dwi_mrtrix": str(args.t12dwi_mrtrix),
        },
        "coordinate_conventions": {
            "fcsv": "MNI152NLin2009bAsym RAS mm",
            "ants_field_grid": "MNI152NLin2009bAsym physical grid",
            "ants_field_vector": "ITK/LPS displacement, converted to RAS vector before addition",
            "anchorNative": "RAS mm",
            "dwi": "RAS mm, using inverse of MRtrix t12dwi matrix for point mapping",
        },
        "electrodes": {
            "E0_E7": {
                "mni_ras": electrode_summary(labels[left], mni_ras[left]),
                "anchorNative_ras": electrode_summary(labels[left], anchor_ras[left]),
                "dwi_ras": electrode_summary(labels[left], dwi_ras[left]),
            },
            "E8_E15": {
                "mni_ras": electrode_summary(labels[right], mni_ras[right]),
                "anchorNative_ras": electrode_summary(labels[right], anchor_ras[right]),
                "dwi_ras": electrode_summary(labels[right], dwi_ras[right]),
            },
        },
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {json_path}")
    print("DWI RAS summaries:")
    for key, value in summary["electrodes"].items():
        dwi = value["dwi_ras"]
        print(f"  {key}: tip={np.round(dwi['tip_or_first_point'], 3).tolist()} direction={np.round(dwi['axis_first_to_last_unit'], 4).tolist()}")


if __name__ == "__main__":
    main()
