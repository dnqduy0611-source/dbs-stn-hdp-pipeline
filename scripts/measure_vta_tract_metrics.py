#!/usr/bin/env python3
"""
Measure VTA-to-tract proximity in DWI/native RAS space.

This complements contact-to-tract distance metrics. Inputs are expected to be in
the same physical coordinate system:

  - VTA NIfTI mask generated from OSS-DBS E-field thresholding
  - MRtrix .tck or ASCII VTK POLYDATA streamlines in DWI/native RAS mm

Reported metrics are intentionally explicit:

  - min_tract_sample_to_vta_center_mm: exact Euclidean distance from all
    streamline sample points to nearest active VTA voxel center.
  - min_tract_sample_to_vta_distance_map_mm: VTA grid distance-transform
    sampled at streamline points, with zero for points inside the VTA mask.
  - n_streamlines_intersect_vta: streamline count with at least one sample point
    falling inside the VTA mask.
  - n_tract_sample_points_in_vta: sample point count inside the VTA mask.

These are QC/reporting metrics, not a replacement for axon-model PAM.
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
from scipy.spatial import cKDTree


ROOT = Path(r"D:\dbs_workspace")
PATIENT_DIR = ROOT / "dbs_stn_hdp_pipeline" / "patients" / "patient_example"
TRACT_DIR = PATIENT_DIR / "derivatives" / "tracts" / "official"
VTA_PATH = (
    PATIENT_DIR
    / "derivatives"
    / "ossdbs"
    / "vta_postprocess_left_LK1A_current_1mA"
    / "L_K1A_current_1mA_VTA_threshold_200Vm.nii.gz"
)
DEFAULT_OUTPUT_PREFIX = (
    PATIENT_DIR
    / "derivatives"
    / "lead_dbs"
    / "distance_qc"
    / "L_K1A_current_1mA_VTA200_tract_metrics"
)
DEFAULT_TRACTS = [
    ("L_CST_official", TRACT_DIR / "L_CST_current_official.tck"),
    (
        "L_DRTT_official",
        TRACT_DIR / "L_DRTT_iFOD2_strictEndpoint_nodil_to_LPG_official_max160.tck",
    ),
    ("L_HDP_official_max80", TRACT_DIR / "L_HDP_refined_official_max80.vtk"),
]


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
            raw = f.read(12)
            if len(raw) < 12:
                break
            x, y, z = struct.unpack("<fff", raw)
            point = np.array([x, y, z], dtype=float)
            if np.isinf(point).any():
                if current:
                    streamlines.append(np.asarray(current, dtype=float))
                break
            if np.isnan(point).any():
                if current:
                    streamlines.append(np.asarray(current, dtype=float))
                    current = []
            else:
                current.append([x, y, z])
    return streamlines


def read_vtk_polydata_ascii(path: Path) -> list[np.ndarray]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not any(line.strip() == "ASCII" for line in lines[:6]):
        raise ValueError(f"Only ASCII VTK POLYDATA is supported: {path}")
    if not any(line.strip() == "DATASET POLYDATA" for line in lines[:8]):
        raise ValueError(f"Only ASCII VTK POLYDATA is supported: {path}")

    point_idx = next(i for i, line in enumerate(lines) if line.startswith("POINTS "))
    parts = lines[point_idx].split()
    n_points = int(parts[1])
    coords: list[float] = []
    i = point_idx + 1
    while len(coords) < n_points * 3:
        coords.extend(float(x) for x in lines[i].split())
        i += 1
    points = np.asarray(coords, dtype=float).reshape(n_points, 3)

    line_idx = next(i for i, line in enumerate(lines) if line.startswith("LINES "))
    line_parts = lines[line_idx].split()
    n_lines = int(line_parts[1])
    total_values = int(line_parts[2])
    values: list[int] = []
    i = line_idx + 1
    while len(values) < total_values:
        values.extend(int(x) for x in lines[i].split())
        i += 1

    streamlines: list[np.ndarray] = []
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


def voxel_centers_ras(mask: np.ndarray, affine: np.ndarray) -> np.ndarray:
    ijk = np.argwhere(mask)
    hom = np.c_[ijk, np.ones(len(ijk))]
    return (affine @ hom.T).T[:, :3]


def points_to_ijk(points_ras: np.ndarray, affine: np.ndarray) -> np.ndarray:
    inv = np.linalg.inv(affine)
    hom = np.c_[points_ras, np.ones(len(points_ras))]
    return (inv @ hom.T).T[:, :3]


def points_inside_mask(points_ras: np.ndarray, mask: np.ndarray, affine: np.ndarray) -> np.ndarray:
    ijk = np.rint(points_to_ijk(points_ras, affine)).astype(int)
    shape = np.asarray(mask.shape)
    in_bounds = np.all((ijk >= 0) & (ijk < shape), axis=1)
    inside = np.zeros(len(points_ras), dtype=bool)
    valid = ijk[in_bounds]
    inside[in_bounds] = mask[valid[:, 0], valid[:, 1], valid[:, 2]]
    return inside


def sample_distance_map(points_ras: np.ndarray, distance_map: np.ndarray, affine: np.ndarray) -> np.ndarray:
    ijk = points_to_ijk(points_ras, affine)
    return ndimage.map_coordinates(distance_map, ijk.T, order=1, mode="nearest")


def summarize(values: np.ndarray) -> dict:
    if len(values) == 0:
        return {
            "min_mm": None,
            "p05_mm": None,
            "median_mm": None,
            "p95_mm": None,
            "max_mm": None,
            "mean_mm": None,
        }
    percentiles = np.percentile(values, [0, 5, 50, 95, 100])
    return {
        "min_mm": float(percentiles[0]),
        "p05_mm": float(percentiles[1]),
        "median_mm": float(percentiles[2]),
        "p95_mm": float(percentiles[3]),
        "max_mm": float(percentiles[4]),
        "mean_mm": float(np.mean(values)),
    }


def measure_tract(
    tract_name: str,
    tract_path: Path,
    vta_mask: np.ndarray,
    vta_affine: np.ndarray,
    vta_tree: cKDTree,
    distance_map: np.ndarray,
) -> tuple[list[dict], dict]:
    streamlines = read_streamlines(tract_path)
    rows: list[dict] = []
    all_points = np.vstack([s for s in streamlines if len(s) > 0])
    all_inside = points_inside_mask(all_points, vta_mask, vta_affine)
    all_center_distances, _ = vta_tree.query(all_points, k=1)
    all_map_distances = sample_distance_map(all_points, distance_map, vta_affine)
    all_map_distances[all_inside] = 0.0

    cursor = 0
    intersect_count = 0
    min_center_by_streamline = []
    min_map_by_streamline = []
    points_in_vta_by_streamline = []
    for idx, streamline in enumerate(streamlines):
        n_points = len(streamline)
        inside = all_inside[cursor : cursor + n_points]
        center_distances = all_center_distances[cursor : cursor + n_points]
        map_distances = all_map_distances[cursor : cursor + n_points]
        cursor += n_points

        intersects = bool(np.any(inside))
        if intersects:
            intersect_count += 1
        n_inside = int(np.sum(inside))
        min_center = float(np.min(center_distances))
        min_map = float(np.min(map_distances))
        min_center_by_streamline.append(min_center)
        min_map_by_streamline.append(min_map)
        points_in_vta_by_streamline.append(n_inside)

        rows.append(
            {
                "tract": tract_name,
                "streamline_index": idx,
                "n_points": n_points,
                "intersects_vta": int(intersects),
                "n_sample_points_in_vta": n_inside,
                "min_sample_to_vta_center_mm": min_center,
                "min_sample_to_vta_distance_map_mm": min_map,
            }
        )

    summary = {
        "tract": tract_name,
        "tract_path": str(tract_path),
        "n_streamlines": len(streamlines),
        "n_tract_sample_points": int(len(all_points)),
        "n_streamlines_intersect_vta": int(intersect_count),
        "pct_streamlines_intersect_vta": float(100.0 * intersect_count / len(streamlines)),
        "n_tract_sample_points_in_vta": int(np.sum(all_inside)),
        "pct_tract_sample_points_in_vta": float(100.0 * np.sum(all_inside) / len(all_points)),
        "min_tract_sample_to_vta_center_mm": float(np.min(all_center_distances)),
        "min_tract_sample_to_vta_distance_map_mm": float(np.min(all_map_distances)),
        "per_streamline_min_center_distance": summarize(np.asarray(min_center_by_streamline, dtype=float)),
        "per_streamline_min_distance_map": summarize(np.asarray(min_map_by_streamline, dtype=float)),
        "streamlines_within_1mm_of_vta": int(np.sum(np.asarray(min_map_by_streamline) <= 1.0)),
        "streamlines_within_2mm_of_vta": int(np.sum(np.asarray(min_map_by_streamline) <= 2.0)),
        "streamlines_within_3mm_of_vta": int(np.sum(np.asarray(min_map_by_streamline) <= 3.0)),
    }
    return rows, summary


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vta", type=Path, default=VTA_PATH)
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument(
        "--tract",
        action="append",
        nargs=2,
        metavar=("NAME", "PATH"),
        help="Tract name and .tck/.vtk path. Can be repeated.",
    )
    args = parser.parse_args()

    vta_img = nib.load(str(args.vta))
    vta_data = vta_img.get_fdata()
    vta_mask = vta_data > 0
    if not np.any(vta_mask):
        raise ValueError(f"VTA mask is empty: {args.vta}")

    zooms = vta_img.header.get_zooms()[:3]
    distance_map = ndimage.distance_transform_edt(~vta_mask, sampling=zooms)
    centers = voxel_centers_ras(vta_mask, vta_img.affine)
    vta_tree = cKDTree(centers)

    tract_specs = args.tract or [(name, str(path)) for name, path in DEFAULT_TRACTS]
    all_rows: list[dict] = []
    summaries: list[dict] = []
    for name, path_str in tract_specs:
        rows, summary = measure_tract(
            name,
            Path(path_str),
            vta_mask,
            vta_img.affine,
            vta_tree,
            distance_map,
        )
        all_rows.extend(rows)
        summaries.append(summary)

    csv_path = args.out_prefix.with_suffix(".per_streamline.csv")
    json_path = args.out_prefix.with_suffix(".summary.json")
    write_csv(csv_path, all_rows)
    payload = {
        "vta": str(args.vta),
        "coordinate_system": "DWI/native RAS mm",
        "vta_voxels": int(np.sum(vta_mask)),
        "vta_voxel_size_mm": [float(z) for z in zooms],
        "vta_volume_mm3": float(np.sum(vta_mask) * np.prod(zooms)),
        "metric_definitions": {
            "min_tract_sample_to_vta_center_mm": "Minimum Euclidean distance from tract sample points to nearest active VTA voxel center.",
            "min_tract_sample_to_vta_distance_map_mm": "VTA binary distance transform sampled at tract sample points; zero if a point is inside the VTA mask.",
            "n_streamlines_intersect_vta": "Number of streamlines with at least one sample point inside the VTA mask.",
            "pct_streamlines_intersect_vta": "n_streamlines_intersect_vta / n_streamlines * 100.",
        },
        "tracts": summaries,
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {json_path}")
    for summary in summaries:
        print(
            f"{summary['tract']}: min_map={summary['min_tract_sample_to_vta_distance_map_mm']:.3f} mm, "
            f"min_center={summary['min_tract_sample_to_vta_center_mm']:.3f} mm, "
            f"intersect={summary['n_streamlines_intersect_vta']}/{summary['n_streamlines']} "
            f"({summary['pct_streamlines_intersect_vta']:.1f}%), "
            f"points_in_vta={summary['n_tract_sample_points_in_vta']}/{summary['n_tract_sample_points']}"
        )


if __name__ == "__main__":
    main()
