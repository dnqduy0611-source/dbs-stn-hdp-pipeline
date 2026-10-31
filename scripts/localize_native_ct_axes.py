#!/usr/bin/env python3
"""Provisional bilateral DBS lead-axis localization in an original CT grid.

This script is deliberately model-independent: it exports only the two metal
track axes and their detected distal/proximal endpoints. It does not infer
contact centers or directional rotation. All outputs require visual QC.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage


def largest_component(mask: np.ndarray) -> np.ndarray:
    labels, count = ndimage.label(mask)
    if count == 0:
        raise RuntimeError("CT foreground mask is empty")
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(np.argmax(sizes))


def robust_line_fit(
    points: np.ndarray,
    rng: np.random.Generator,
    iterations: int = 8000,
    radius_mm: float = 1.8,
) -> dict:
    if len(points) < 30:
        raise RuntimeError(f"Too few metal candidates ({len(points)})")

    best = None
    for _ in range(iterations):
        i, j = rng.integers(0, len(points), 2)
        vector = points[j] - points[i]
        span = float(np.linalg.norm(vector))
        if span < 35:
            continue
        unit = vector / span
        if abs(float(unit[2])) < 0.45:
            continue
        distances = np.linalg.norm(np.cross(points - points[i], unit), axis=1)
        inliers = distances < radius_mm
        if int(inliers.sum()) < 20:
            continue

        cloud = points[inliers]
        center = cloud.mean(axis=0)
        _, _, vh = np.linalg.svd(cloud - center, full_matrices=False)
        axis = vh[0]
        if axis[2] < 0:
            axis = -axis
        projection = (cloud - center) @ axis
        length = float(np.percentile(projection, 99) - np.percentile(projection, 1))
        score = int(inliers.sum()) * min(length, 100.0)
        if best is None or score > best[0]:
            best = (score, center, axis, inliers)

    if best is None:
        raise RuntimeError("No plausible superior-inferior metal track found")

    _, center, axis, inliers = best
    cloud = points[inliers]
    projection = (cloud - center) @ axis
    lo, hi = np.percentile(projection, [1, 99])
    tip = center + axis * lo
    entry = center + axis * hi
    residual = np.linalg.norm(np.cross(cloud - center, axis), axis=1)
    return {
        "tip": tip,
        "entry": entry,
        "axis": axis,
        "length_mm": float(hi - lo),
        "rms_residual_mm": float(np.sqrt(np.mean(residual**2))),
        "p95_residual_mm": float(np.percentile(residual, 95)),
        "inlier_count": int(inliers.sum()),
    }


def as_uint8(array: np.ndarray) -> np.ndarray:
    array = np.clip(array, 0, 5000) / 5000.0
    return np.asarray(np.rint(array * 255), dtype=np.uint8)


def qc_panel(ct: np.ndarray, affine: np.ndarray, fits: dict, drop_axis: int) -> Image.Image:
    inverse = np.linalg.inv(affine)
    # The overlay below converts data coordinates to image coordinates with
    # `height - 1 - y`, so the raster has to be flipped to match; without the
    # flip the lines land mirrored about the horizontal mid-line.
    mip = np.max(ct, axis=drop_axis).T[::-1]
    image = Image.fromarray(as_uint8(mip), mode="L").convert("RGB")
    draw = ImageDraw.Draw(image)
    colors = {"right": (255, 55, 45), "left": (0, 220, 255)}
    for side, fit in fits.items():
        sample = np.linspace(fit["tip"], fit["entry"], 100)
        vox = nib.affines.apply_affine(inverse, sample)
        if drop_axis == 2:
            xy = vox[:, [0, 1]]
        elif drop_axis == 1:
            xy = vox[:, [0, 2]]
        else:
            xy = vox[:, [1, 2]]
        pts = [(float(x), float(image.height - 1 - y)) for x, y in xy]
        draw.line(pts, fill=colors[side], width=3)
        x, y = pts[0]
        draw.line((x - 5, y - 5, x + 5, y + 5), fill=colors[side], width=2)
        draw.line((x - 5, y + 5, x + 5, y - 5), fill=colors[side], width=2)
    return image


def write_qc(path: Path, ct: np.ndarray, affine: np.ndarray, fits: dict) -> None:
    panels = [qc_panel(ct, affine, fits, axis) for axis in (2, 1, 0)]
    height = max(panel.height for panel in panels)
    width = sum(panel.width for panel in panels)
    canvas = Image.new("RGB", (width, height), "black")
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width
    canvas.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ct", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--threshold", type=float, default=3000.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ct_img = nib.load(str(args.ct))
    ct = np.asarray(ct_img.dataobj, dtype=np.float32)
    foreground = largest_component(ct > -500)
    foreground_voxels = np.column_stack(np.where(foreground))[::100]
    foreground_world = nib.affines.apply_affine(ct_img.affine, foreground_voxels)
    center = np.median(foreground_world, axis=0)

    metal_voxels = np.column_stack(np.where(ct > args.threshold))
    metal_world = nib.affines.apply_affine(ct_img.affine, metal_voxels)
    candidate = (
        (np.abs(metal_world[:, 0] - center[0]) < 40)
        & (metal_world[:, 2] > center[2] - 30)
        & (metal_world[:, 2] < center[2] + 110)
        & (metal_world[:, 1] < center[1] + 60)
    )
    points = metal_world[candidate]

    rng = np.random.default_rng(20260807)
    fits = {}
    for side, side_mask in (
        ("right", points[:, 0] > center[0] + 1),
        ("left", points[:, 0] < center[0] - 1),
    ):
        fits[side] = robust_line_fit(points[side_mask], rng)

    geometry_pass = all(
        fit["length_mm"] >= 50 and fit["rms_residual_mm"] <= 1.5
        for fit in fits.values()
    )
    report = {
        "subject": args.subject,
        "input_ct": str(args.ct.resolve()),
        "coordinate_system": "original CT RAS mm",
        "voxel_size_mm": [float(v) for v in ct_img.header.get_zooms()[:3]],
        "metal_threshold_hu": args.threshold,
        "status": "provisional_needs_visual_qc",
        "geometry_gate_pass": bool(geometry_pass),
        "contact_centers": "not_estimated",
        "directional_rotation": "not_estimated",
        "axes": {
            side: {
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in fit.items()
            }
            for side, fit in fits.items()
        },
    }
    json_path = args.output_dir / "electrode_axes_native_ct_provisional.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    csv_path = args.output_dir / "electrode_axes_native_ct_provisional.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "side", "tip_x", "tip_y", "tip_z", "entry_x", "entry_y", "entry_z",
            "axis_x", "axis_y", "axis_z", "length_mm", "rms_residual_mm",
            "p95_residual_mm", "inlier_count", "status",
        ])
        for side, fit in fits.items():
            writer.writerow([
                side, *fit["tip"], *fit["entry"], *fit["axis"], fit["length_mm"],
                fit["rms_residual_mm"], fit["p95_residual_mm"], fit["inlier_count"],
                "provisional_needs_visual_qc",
            ])

    write_qc(args.output_dir / "electrode_axes_native_ct_qc.png", ct, ct_img.affine, fits)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
