#!/usr/bin/env python3
"""
Contact-level DBS electrode to tract distance metrics in DWI/native RAS space.

This script complements measure_electrode_tract_distances.py. It reports one row
per contact/fiducial and tract, using definitions that are explicit enough for
methods/QC reporting:

  - tract edge, exact: minimum Euclidean distance from contact center to any
    streamline sample point.
  - tract centerline: minimum distance from contact center to an approximate
    tract centerline made by resampling streamlines and averaging them.
  - distance map: Euclidean distance transform of a voxelized tract on the
    reference DWI/FA grid, sampled at each contact center.

Inputs must all be in the same DWI/native RAS coordinate system.
"""

from __future__ import annotations

import argparse
import csv
import json
import struct
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage


ROOT = Path(r"D:\dbs_workspace")
DEFAULT_ELECTRODES = (
    ROOT
    / "dbs_stn_hdp_pipeline"
    / "patients"
    / "patient_example"
    / "derivatives"
    / "lead_dbs"
    / "sub-PatientExample_B33005_contacts.csv"
)
DEFAULT_REFERENCE = ROOT / "exam_nifti" / "dti_fa_computed.nii.gz"
DEFAULT_OUTPUT_PREFIX = (
    ROOT
    / "dbs_stn_hdp_pipeline"
    / "patients"
    / "patient_example"
    / "derivatives"
    / "lead_dbs"
    / "distance_qc"
    / "sub-PatientExample_contact_tract_metrics"
)
DEFAULT_TRACTS = [
    (
        "L_CST_official",
        ROOT
        / "dbs_stn_hdp_pipeline"
        / "patients"
        / "patient_example"
        / "derivatives"
        / "tracts"
        / "official"
        / "L_CST_current_official.tck",
    ),
    (
        "L_DRTT_official",
        ROOT
        / "dbs_stn_hdp_pipeline"
        / "patients"
        / "patient_example"
        / "derivatives"
        / "tracts"
        / "official"
        / "L_DRTT_iFOD2_strictEndpoint_nodil_to_LPG_official_max160.tck",
    ),
    (
        "L_HDP_official_max80",
        ROOT
        / "dbs_stn_hdp_pipeline"
        / "patients"
        / "patient_example"
        / "derivatives"
        / "tracts"
        / "official"
        / "L_HDP_refined_official_max80.vtk",
    ),
]


def read_contacts(path: Path, labels: list[str] | None, side: str | None) -> list[dict]:
    wanted = set(labels) if labels else None
    contacts = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            row_label = row.get("label") or row.get("source_fiducial") or row.get("contact_name")
            contact_name = row.get("contact_name", row_label)
            row_side = row.get("side")
            searchable = {row_label, contact_name, contact_name.replace(" ", ""), row.get("source_fiducial", "")}
            if side and row_side and row_side.upper() != side.upper():
                continue
            if wanted and not (wanted & searchable):
                continue
            contacts.append(
                {
                    "label": contact_name,
                    "source_fiducial": row.get("source_fiducial", row_label),
                    "lead": row.get("lead", ""),
                    "side": row.get("side", ""),
                    "dwi_ras": np.array(
                        [
                            float(row["dwi_ras_x"]),
                            float(row["dwi_ras_y"]),
                            float(row["dwi_ras_z"]),
                        ],
                        dtype=float,
                    ),
                }
            )
    def source_index(item: dict) -> int:
        source = item["source_fiducial"]
        if source.startswith("E") and source[1:].isdigit():
            return int(source[1:])
        return 10_000

    contacts.sort(key=lambda item: (item["side"], item["lead"], source_index(item), item["label"]))
    if wanted:
        found = {item["label"] for item in contacts} | {item["source_fiducial"] for item in contacts}
        missing = wanted - found
        if missing:
            raise ValueError(f"Missing requested contact labels in {path}: {sorted(missing)}")
    return contacts


def read_tck(path: Path) -> list[np.ndarray]:
    streamlines: list[np.ndarray] = []
    current: list[list[float]] = []
    with path.open("rb") as f:
        header = {}
        while True:
            line = f.readline().decode("latin-1").strip()
            if line == "END":
                break
            if ":" in line:
                key, value = line.split(":", 1)
                header[key.strip()] = value.strip()
        offset = int(header.get("file", ". 0").split()[-1])
        f.seek(offset)
        while True:
            data = f.read(12)
            if len(data) < 12:
                break
            x, y, z = struct.unpack("<fff", data)
            point = np.array([x, y, z], dtype=float)
            if np.isinf(point).any():
                if current:
                    streamlines.append(np.array(current, dtype=float))
                break
            if np.isnan(point).any():
                if current:
                    streamlines.append(np.array(current, dtype=float))
                    current = []
                continue
            current.append([x, y, z])
    return streamlines


def read_vtk_polydata_ascii(path: Path) -> list[np.ndarray]:
    lines = path.read_text().splitlines()
    if len(lines) < 4 or "ASCII" not in lines[:4] or "DATASET POLYDATA" not in lines[:6]:
        raise ValueError(f"Only ASCII VTK POLYDATA is supported: {path}")

    point_idx = next(i for i, line in enumerate(lines) if line.startswith("POINTS "))
    parts = lines[point_idx].split()
    n_points = int(parts[1])
    coords = []
    i = point_idx + 1
    while len(coords) < n_points * 3:
        coords.extend(float(x) for x in lines[i].split())
        i += 1
    points = np.array(coords, dtype=float).reshape(n_points, 3)

    line_idx = next(j for j, line in enumerate(lines) if line.startswith("LINES "))
    line_parts = lines[line_idx].split()
    n_lines = int(line_parts[1])
    values = []
    i = line_idx + 1
    while len(values) < int(line_parts[2]):
        values.extend(int(x) for x in lines[i].split())
        i += 1

    streamlines = []
    cursor = 0
    for _ in range(n_lines):
        count = values[cursor]
        cursor += 1
        indices = values[cursor : cursor + count]
        cursor += count
        streamlines.append(points[indices])
    return streamlines


def read_streamlines(path: Path) -> list[np.ndarray]:
    if path.suffix.lower() == ".tck":
        return read_tck(path)
    if path.suffix.lower() == ".vtk":
        return read_vtk_polydata_ascii(path)
    raise ValueError(f"Unsupported tract format: {path}")


def cumulative_lengths(points: np.ndarray) -> np.ndarray:
    if len(points) == 1:
        return np.array([0.0])
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(segment_lengths)])


def resample_streamline(points: np.ndarray, n_points: int) -> np.ndarray | None:
    if len(points) < 2:
        return None
    distances = cumulative_lengths(points)
    total = distances[-1]
    if total <= 0:
        return None
    targets = np.linspace(0.0, total, n_points)
    out = np.empty((n_points, 3), dtype=float)
    for dim in range(3):
        out[:, dim] = np.interp(targets, distances, points[:, dim])
    return out


def make_centerline(streamlines: list[np.ndarray], n_points: int) -> np.ndarray:
    resampled = []
    reference = None
    for streamline in streamlines:
        sampled = resample_streamline(streamline, n_points)
        if sampled is None:
            continue
        if reference is None:
            reference = sampled
        else:
            same = np.linalg.norm(sampled[0] - reference[0]) + np.linalg.norm(sampled[-1] - reference[-1])
            flipped = np.linalg.norm(sampled[-1] - reference[0]) + np.linalg.norm(sampled[0] - reference[-1])
            if flipped < same:
                sampled = sampled[::-1]
        resampled.append(sampled)
    if not resampled:
        raise ValueError("Could not make tract centerline from empty/degenerate streamlines")
    return np.mean(np.stack(resampled, axis=0), axis=0)


def voxelize_points(points_ras: np.ndarray, reference_img: nib.Nifti1Image) -> np.ndarray:
    inv_affine = np.linalg.inv(reference_img.affine)
    hom = np.c_[points_ras, np.ones(len(points_ras))]
    ijk_float = (inv_affine @ hom.T).T[:, :3]
    ijk = np.rint(ijk_float).astype(int)
    shape = reference_img.shape[:3]
    inside = np.all((ijk >= 0) & (ijk < np.array(shape)), axis=1)
    mask = np.zeros(shape, dtype=bool)
    if np.any(inside):
        valid = ijk[inside]
        mask[valid[:, 0], valid[:, 1], valid[:, 2]] = True
    return mask


def sample_distance_map(distance_map: np.ndarray, reference_img: nib.Nifti1Image, point_ras: np.ndarray) -> float:
    inv_affine = np.linalg.inv(reference_img.affine)
    ijk = (inv_affine @ np.r_[point_ras, 1.0])[:3]
    value = ndimage.map_coordinates(distance_map, ijk[:, None], order=1, mode="nearest")[0]
    return float(value)


def write_tract_mask(path: Path, mask: np.ndarray, reference_img: nib.Nifti1Image) -> None:
    img = nib.Nifti1Image(mask.astype(np.uint8), reference_img.affine, reference_img.header)
    img.set_data_dtype(np.uint8)
    nib.save(img, str(path))


def measure_tract_contacts(
    name: str,
    path: Path,
    contacts: list[dict],
    reference_img: nib.Nifti1Image,
    centerline_points: int,
    contact_radius_mm: float,
    mask_dir: Path,
) -> tuple[list[dict], dict]:
    streamlines = read_streamlines(path)
    if not streamlines:
        raise ValueError(f"No streamlines loaded from {path}")

    tract_points = np.vstack([s for s in streamlines if len(s) > 0])
    centerline = make_centerline(streamlines, centerline_points)
    mask = voxelize_points(tract_points, reference_img)
    if not np.any(mask):
        raise ValueError(f"Tract {name} has no voxels inside reference image grid")

    zooms = reference_img.header.get_zooms()[:3]
    distance_map = ndimage.distance_transform_edt(~mask, sampling=zooms)
    mask_path = mask_dir / f"{name}_dwi_tract_point_mask.nii.gz"
    write_tract_mask(mask_path, mask, reference_img)

    rows = []
    for contact in contacts:
        point = contact["dwi_ras"]
        edge_exact = float(np.min(np.linalg.norm(tract_points - point, axis=1)))
        centerline_distance = float(np.min(np.linalg.norm(centerline - point, axis=1)))
        distance_map_mm = sample_distance_map(distance_map, reference_img, point)
        rows.append(
            {
                "tract": name,
                "tract_path": str(path),
                "contact_label": contact["label"],
                "source_fiducial": contact["source_fiducial"],
                "lead": contact["lead"],
                "side": contact["side"],
                "contact_dwi_ras_x": point[0],
                "contact_dwi_ras_y": point[1],
                "contact_dwi_ras_z": point[2],
                "distance_to_tract_edge_exact_pointcloud_mm": edge_exact,
                "distance_to_tract_edge_distance_map_mm": distance_map_mm,
                "distance_to_tract_centerline_mm": centerline_distance,
                "contact_surface_to_tract_edge_exact_mm": max(edge_exact - contact_radius_mm, 0.0),
                "contact_radius_mm_used": contact_radius_mm,
                "n_streamlines": len(streamlines),
                "n_tract_sample_points": len(tract_points),
                "tract_mask_voxels": int(np.sum(mask)),
                "tract_mask_path": str(mask_path),
            }
        )

    summary = {
        "tract": name,
        "path": str(path),
        "n_streamlines": len(streamlines),
        "n_tract_sample_points": int(len(tract_points)),
        "tract_mask_voxels": int(np.sum(mask)),
        "tract_mask_path": str(mask_path),
        "centerline_points": centerline_points,
    }
    return rows, summary


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--electrodes", type=Path, default=DEFAULT_ELECTRODES)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--contacts", nargs="+", default=None)
    parser.add_argument("--side", choices=["L", "R", "l", "r"], default="L")
    parser.add_argument("--centerline-points", type=int, default=101)
    parser.add_argument(
        "--contact-radius-mm",
        type=float,
        default=0.0,
        help="Optional electrode/contact radius to subtract from center-to-tract edge distance. Default 0 keeps pure center distances.",
    )
    parser.add_argument(
        "--tract",
        action="append",
        nargs=2,
        metavar=("NAME", "PATH"),
        help="Tract name and .tck/.vtk path. Can be repeated.",
    )
    args = parser.parse_args()

    contacts = read_contacts(args.electrodes, args.contacts, args.side)
    if not contacts:
        raise ValueError(f"No contacts selected from {args.electrodes}")
    reference_img = nib.load(str(args.reference))
    tract_specs = args.tract or [(name, str(path)) for name, path in DEFAULT_TRACTS]

    mask_dir = args.out_prefix.parent / "distance_maps"
    mask_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    summaries = []
    for name, path_str in tract_specs:
        rows, summary = measure_tract_contacts(
            name=name,
            path=Path(path_str),
            contacts=contacts,
            reference_img=reference_img,
            centerline_points=args.centerline_points,
            contact_radius_mm=args.contact_radius_mm,
            mask_dir=mask_dir,
        )
        all_rows.extend(rows)
        summaries.append(summary)

    csv_path = args.out_prefix.with_suffix(".per_contact.csv")
    json_path = args.out_prefix.with_suffix(".summary.json")
    write_csv(csv_path, all_rows)
    json_path.write_text(
        json.dumps(
            {
                "electrodes": str(args.electrodes),
                "reference": str(args.reference),
                "contacts": args.contacts,
                "side": args.side,
                "contact_radius_mm": args.contact_radius_mm,
                "coordinate_system": "DWI/native RAS mm",
                "metric_definitions": {
                    "distance_to_tract_edge_exact_pointcloud_mm": "Minimum Euclidean distance from contact center to any tract streamline sample point.",
                    "distance_to_tract_edge_distance_map_mm": "Euclidean distance transform of voxelized tract point mask on the reference DWI/FA grid, sampled at contact center.",
                    "distance_to_tract_centerline_mm": "Minimum Euclidean distance from contact center to an approximate mean tract centerline made by resampling streamlines.",
                    "contact_surface_to_tract_edge_exact_mm": "Exact point-cloud edge distance minus optional contact radius, floored at zero.",
                },
                "tracts": summaries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {json_path}")
    for row in all_rows:
        print(
            f"{row['tract']} {row['contact_label']}: "
            f"edge={row['distance_to_tract_edge_exact_pointcloud_mm']:.2f} mm, "
            f"map={row['distance_to_tract_edge_distance_map_mm']:.2f} mm, "
            f"centerline={row['distance_to_tract_centerline_mm']:.2f} mm"
        )


if __name__ == "__main__":
    main()
