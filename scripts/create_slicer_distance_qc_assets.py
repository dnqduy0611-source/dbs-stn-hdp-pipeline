#!/usr/bin/env python3
"""
Create Slicer-readable QC assets for electrode-to-tract distance review.

Outputs are in DWI/native RAS:
  - electrode fiducials (.fcsv)
  - lead axis polylines (.vtk)
  - nearest-point connector lines for selected streamlines (.vtk)
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from measure_electrode_tract_distances import DEFAULT_ELECTRODES, DEFAULT_OUTPUT_PREFIX, fit_axis, read_electrodes, read_streamlines


ROOT = Path(r"D:\dbs_workspace")
DEFAULT_DISTANCE_CSV = ROOT / "dbs_stn_hdp_pipeline" / "patients" / "patient_example" / "derivatives" / "lead_dbs" / "distance_qc" / "sub-PatientExample_official_tract_distances.per_streamline.csv"
DEFAULT_QC_DIR = ROOT / "dbs_stn_hdp_pipeline" / "patients" / "patient_example" / "derivatives" / "lead_dbs" / "distance_qc" / "slicer_qc"
DEFAULT_TRACTS = {
    "L_CST_official": ROOT / "dbs_stn_hdp_pipeline" / "patients" / "patient_example" / "derivatives" / "tracts" / "official" / "L_CST_current_official.tck",
    "L_DRTT_official": ROOT / "dbs_stn_hdp_pipeline" / "patients" / "patient_example" / "derivatives" / "tracts" / "official" / "L_DRTT_iFOD2_strictEndpoint_nodil_to_LPG_official_max160.tck",
    "L_HDP_official_max80": ROOT / "dbs_stn_hdp_pipeline" / "patients" / "patient_example" / "derivatives" / "tracts" / "official" / "L_HDP_refined_official_max80.vtk",
}


def write_fcsv(path: Path, labels: list[str], points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        f.write("# Markups fiducial file version = 5.10\n")
        f.write("# CoordinateSystem = RAS\n")
        f.write("# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID\n")
        for i, (label, p) in enumerate(zip(labels, points)):
            f.write(
                f"vtkMRMLMarkupsFiducialNode_{i},{p[0]},{p[1]},{p[2]},0,0,0,1,1,1,0,{label},,\n"
            )


def write_polyline_vtk(path: Path, lines: list[np.ndarray], title: str = "polyline") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.vstack(lines) if lines else np.zeros((0, 3))
    total_line_ints = sum(len(line) + 1 for line in lines)
    with path.open("w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write(f"{title}\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        f.write(f"POINTS {len(points)} float\n")
        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")
        f.write(f"LINES {len(lines)} {total_line_ints}\n")
        cursor = 0
        for line in lines:
            indices = list(range(cursor, cursor + len(line)))
            cursor += len(line)
            f.write(f"{len(indices)} {' '.join(str(i) for i in indices)}\n")


def read_distance_rows(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["streamline_index"] = int(row["streamline_index"])
        row["min_to_any_electrode_point_mm"] = float(row["min_to_any_electrode_point_mm"])
        row["min_to_lead_axis_mm"] = float(row["min_to_lead_axis_mm"])
    return rows


def closest_connector(streamline: np.ndarray, electrode_points: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(streamline[:, None, :] - electrode_points[None, :, :], axis=2)
    i, j = np.unravel_index(np.argmin(d), d.shape)
    return np.vstack([streamline[i], electrode_points[j]])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--electrodes", type=Path, default=DEFAULT_ELECTRODES)
    parser.add_argument("--distance-csv", type=Path, default=DEFAULT_DISTANCE_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_QC_DIR)
    parser.add_argument("--n-closest", type=int, default=10)
    args = parser.parse_args()

    electrodes = read_electrodes(args.electrodes)
    left_points = electrodes["E8_E15"]
    write_fcsv(args.out_dir / "sub-PatientExample_left_electrode_E8_E15_dwi.fcsv", [f"E{i}" for i in range(8, 16)], left_points)

    axis_point, axis_dir = fit_axis(left_points)
    t = np.linspace(-8, 16, 2)
    axis_line = axis_point + np.outer(t, axis_dir)
    write_polyline_vtk(args.out_dir / "sub-PatientExample_left_lead_axis_dwi.vtk", [axis_line], "left lead axis")

    rows = read_distance_rows(args.distance_csv)
    for tract_name, tract_path in DEFAULT_TRACTS.items():
        tract_rows = [row for row in rows if row["tract"] == tract_name]
        tract_rows.sort(key=lambda row: row["min_to_any_electrode_point_mm"])
        chosen = tract_rows[: args.n_closest]
        streamlines = read_streamlines(tract_path)
        connectors = [closest_connector(streamlines[row["streamline_index"]], left_points) for row in chosen]
        write_polyline_vtk(args.out_dir / f"{tract_name}_closest_{args.n_closest}_connectors_to_electrode_points.vtk", connectors, f"{tract_name} connectors")

    print(f"Wrote Slicer QC assets to: {args.out_dir}")


if __name__ == "__main__":
    main()
