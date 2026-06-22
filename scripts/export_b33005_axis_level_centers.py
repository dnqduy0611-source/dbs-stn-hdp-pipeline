#!/usr/bin/env python3
"""
Derive Medtronic SenSight B33005 axial contact-level centers from electrode axes.

Important distinction:
  - Input E0-E7 and E8-E15 are axis-defining fiducials, not physical contacts.
  - Output contains four axial level centers per lead: K0, K1_level, K2_level, K3.
  - Directional segment centers K1A/B/C and K2A/B/C are not generated because
    rotation around the shaft is unknown without a reliable orientation marker.

By default, the first fiducial of each lead (E0 for right, E8 for left) is used
as the distal anchor. B33005 axial level center distances are then placed along
the fitted axis at 1.65, 3.65, 5.65, and 7.65 mm from that anchor.

If the first fiducial is not the distal tip/reference point, pass a
distal-anchor-offset-mm value after determining the correct offset visually.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(r"D:\dbs_workspace")
DEFAULT_INPUT = (
    ROOT
    / "dbs_stn_hdp_pipeline"
    / "patients"
    / "patient_example"
    / "derivatives"
    / "lead_dbs"
    / "sub-PatientExample_electrodefiducials_spaces.csv"
)
DEFAULT_OUTPUT = (
    ROOT
    / "dbs_stn_hdp_pipeline"
    / "patients"
    / "patient_example"
    / "derivatives"
    / "lead_dbs"
    / "sub-PatientExample_B33005_axis_level_centers.csv"
)

B33005_LEVELS = [
    ("K0", 0, 1.65),
    ("K1_level", 1, 3.65),
    ("K2_level", 2, 5.65),
    ("K3", 3, 7.65),
]


def read_fiducials(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda row: int(row["label"][1:]))
    return rows


def xyz(row: dict, prefix: str) -> np.ndarray:
    return np.array(
        [float(row[f"{prefix}_x"]), float(row[f"{prefix}_y"]), float(row[f"{prefix}_z"])],
        dtype=float,
    )


def fit_axis(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - center, full_matrices=False)
    direction = vh[0]
    if np.dot(points[-1] - points[0], direction) < 0:
        direction *= -1
    direction = direction / np.linalg.norm(direction)
    residuals = point_to_axis_distance(points, center, direction)
    return center, direction, residuals


def point_to_axis_distance(points: np.ndarray, axis_point: np.ndarray, axis_direction: np.ndarray) -> np.ndarray:
    rel = points - axis_point
    projection = np.outer(rel @ axis_direction, axis_direction)
    perpendicular = rel - projection
    return np.linalg.norm(perpendicular, axis=1)


def build_levels_for_space(
    rows: list[dict],
    prefix: str,
    distal_anchor_offset_mm: float,
) -> tuple[dict[str, np.ndarray], dict]:
    points = np.vstack([xyz(row, prefix) for row in rows])
    _, direction, residuals = fit_axis(points)
    anchor = points[0] + direction * distal_anchor_offset_mm
    levels = {
        name: anchor + direction * axial_distance_mm
        for name, _, axial_distance_mm in B33005_LEVELS
    }
    projected_span_mm = float(np.dot(points[-1] - points[0], direction))
    qc = {
        "axis_direction": direction.tolist(),
        "distal_anchor": anchor.tolist(),
        "distal_anchor_source_fiducial": rows[0]["label"],
        "distal_anchor_offset_mm": distal_anchor_offset_mm,
        "fiducial_projected_span_mm": projected_span_mm,
        "axis_fit_rms_residual_mm": float(np.sqrt(np.mean(residuals**2))),
        "axis_fit_max_residual_mm": float(np.max(residuals)),
    }
    return levels, qc


def build_outputs(rows: list[dict], distal_anchor_offset_mm: float) -> tuple[list[dict], dict]:
    lead_specs = [
        ("right", "R", rows[0:8], "E0-E7"),
        ("left", "L", rows[8:16], "E8-E15"),
    ]

    output_rows: list[dict] = []
    qc = {
        "electrode_model": "Medtronic SenSight B33005 / B3300542 / B3300542M",
        "coordinate_system_primary": "DWI/native RAS mm",
        "method": "Fit a straight electrode axis from manually adjusted Slicer fiducials, then place four B33005 axial level centers at Lead-DBS axial distances from the distal anchor.",
        "b33005_axial_level_distances_from_distal_anchor_mm": {
            name: distance for name, _, distance in B33005_LEVELS
        },
        "directional_segments": "not generated; A/B/C segment rotation around shaft is unknown without orientation marker reconstruction",
        "assumption": "First fiducial of each lead is treated as the distal anchor unless distal_anchor_offset_mm is changed.",
        "lead_qc": {},
    }

    for lead, side, lead_rows, source_range in lead_specs:
        if len(lead_rows) != 8:
            raise ValueError(f"Expected 8 fiducials for {lead}, got {len(lead_rows)}")

        mni_levels, mni_qc = build_levels_for_space(lead_rows, "mni_ras", distal_anchor_offset_mm)
        anchor_levels, anchor_qc = build_levels_for_space(lead_rows, "anchorNative_ras", distal_anchor_offset_mm)
        dwi_levels, dwi_qc = build_levels_for_space(lead_rows, "dwi_ras", distal_anchor_offset_mm)
        qc["lead_qc"][lead] = {
            "side": side,
            "source_fiducials": source_range,
            "mni_ras": mni_qc,
            "anchorNative_ras": anchor_qc,
            "dwi_ras": dwi_qc,
        }

        for name, level_index, axial_distance_mm in B33005_LEVELS:
            mni = mni_levels[name]
            anchor = anchor_levels[name]
            dwi = dwi_levels[name]
            output_rows.append(
                {
                    "lead": lead,
                    "side": side,
                    "level_index": level_index,
                    "level_name": f"{name} ({side})",
                    "electrode_model": "Medtronic B33005",
                    "source_fiducials": source_range,
                    "distal_anchor_fiducial": lead_rows[0]["label"],
                    "axial_distance_from_distal_anchor_mm": axial_distance_mm,
                    "coordinate_system": "DWI/native RAS mm",
                    "mni_ras_x": mni[0],
                    "mni_ras_y": mni[1],
                    "mni_ras_z": mni[2],
                    "anchorNative_ras_x": anchor[0],
                    "anchorNative_ras_y": anchor[1],
                    "anchorNative_ras_z": anchor[2],
                    "dwi_ras_x": dwi[0],
                    "dwi_ras_y": dwi[1],
                    "dwi_ras_z": dwi[2],
                    "segment_orientation_status": "unknown_not_segmented",
                }
            )

    return output_rows, qc


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_fcsv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        f.write("# Markups fiducial file version = 5.6\n")
        f.write("# CoordinateSystem = RAS\n")
        f.write("# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID\n")
        for idx, row in enumerate(rows):
            f.write(
                f"vtkMRMLMarkupsFiducialNode_{idx},"
                f"{row['dwi_ras_x']},{row['dwi_ras_y']},{row['dwi_ras_z']},"
                f"0,0,0,1,1,1,0,{row['level_name']},axis-derived B33005 level center,\n"
            )


def write_axis_vtk(path: Path, qc: dict, space: str = "dwi_ras") -> None:
    points = []
    lines = []
    for lead_name in ["right", "left"]:
        lead_qc = qc["lead_qc"][lead_name][space]
        anchor = np.array(lead_qc["distal_anchor"], dtype=float)
        direction = np.array(lead_qc["axis_direction"], dtype=float)
        p0 = anchor - direction * 2.0
        p1 = anchor + direction * 10.0
        base = len(points)
        points.extend([p0, p1])
        lines.append((base, base + 1))

    with path.open("w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("B33005 axis-derived level center axes\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        f.write(f"POINTS {len(points)} float\n")
        for point in points:
            f.write(f"{point[0]} {point[1]} {point[2]}\n")
        f.write(f"LINES {len(lines)} {len(lines) * 3}\n")
        for a, b in lines:
            f.write(f"2 {a} {b}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--distal-anchor-offset-mm",
        type=float,
        default=0.0,
        help="Offset along fitted distal-to-proximal axis from first fiducial to true distal anchor/tip.",
    )
    args = parser.parse_args()

    rows = read_fiducials(args.input)
    if len(rows) < 16:
        raise ValueError(f"Expected at least E0-E15 in {args.input}, got {len(rows)} rows")

    level_rows, qc = build_outputs(rows, args.distal_anchor_offset_mm)
    write_csv(args.output, level_rows)
    qc_path = args.output.with_suffix(".qc.json")
    qc_path.write_text(json.dumps(qc, indent=2), encoding="utf-8")

    fcsv_path = args.output.with_suffix(".dwi.fcsv")
    axis_vtk_path = args.output.with_suffix(".dwi_axis.vtk")
    write_fcsv(fcsv_path, level_rows)
    write_axis_vtk(axis_vtk_path, qc)

    print(f"Wrote: {args.output}")
    print(f"Wrote: {qc_path}")
    print(f"Wrote: {fcsv_path}")
    print(f"Wrote: {axis_vtk_path}")
    for lead_name, lead_qc in qc["lead_qc"].items():
        dwi = lead_qc["dwi_ras"]
        print(
            f"{lead_name}: DWI axis RMS residual={dwi['axis_fit_rms_residual_mm']:.3f} mm, "
            f"max={dwi['axis_fit_max_residual_mm']:.3f} mm, "
            f"span={dwi['fiducial_projected_span_mm']:.3f} mm"
        )


if __name__ == "__main__":
    main()
