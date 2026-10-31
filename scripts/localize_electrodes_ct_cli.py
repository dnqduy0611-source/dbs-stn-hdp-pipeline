#!/usr/bin/env python3
"""Provisional, non-interactive DBS lead-axis localization from post-op CT.

The input CT must already be rigidly resampled into the pre-op T1 grid.  The
script detects the two bright, approximately superior-inferior metal tracks,
fits a robust 3-D line on each side, and exports provisional distal anchors,
shaft entry points, and B33005 axial level centers.  Segment orientation is
not estimated.  Every output remains pending expert visual QC.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import ndimage


DEFAULT_SUBJECTS = (
    "sub-03",
    "sub-05",
    "sub-06",
    "sub-09",
)
B33005_LEVEL_MM = (1.65, 3.65, 5.65, 7.65)


def largest_component(mask: np.ndarray) -> np.ndarray:
    labels, count = ndimage.label(mask)
    if count == 0:
        raise RuntimeError("T1 foreground mask is empty")
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(np.argmax(sizes))


def brain_center_world(t1_img: nib.Nifti1Image, t1: np.ndarray) -> np.ndarray:
    # The cohort T1 volumes have a low non-zero background.  A fixed intensity
    # floor of 50 separates head tissue reliably in all four prepared inputs.
    mask = largest_component(t1 > 50)
    ijk = np.column_stack(np.where(mask))[::50]
    if len(ijk) < 100:
        raise RuntimeError("Too few voxels in T1 foreground mask")
    world = nib.affines.apply_affine(t1_img.affine, ijk)
    return np.median(world, axis=0)


def robust_line_fit(
    points: np.ndarray,
    rng: np.random.Generator,
    iterations: int = 6000,
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


def normalized_mi(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    aa = a[mask]
    bb = b[mask]
    if len(aa) > 500_000:
        step = int(math.ceil(len(aa) / 500_000))
        aa, bb = aa[::step], bb[::step]
    ar = np.percentile(aa, [0.5, 99.5])
    br = np.percentile(bb, [0.5, 99.5])
    keep = (aa >= ar[0]) & (aa <= ar[1]) & (bb >= br[0]) & (bb <= br[1])
    hist, _, _ = np.histogram2d(aa[keep], bb[keep], bins=64)
    prob = hist / max(hist.sum(), 1)
    px = prob.sum(axis=1)
    py = prob.sum(axis=0)
    hx = -np.sum(px[px > 0] * np.log(px[px > 0]))
    hy = -np.sum(py[py > 0] * np.log(py[py > 0]))
    hxy = -np.sum(prob[prob > 0] * np.log(prob[prob > 0]))
    return float((hx + hy) / hxy) if hxy > 0 else float("nan")


def registration_metric(t1: np.ndarray, ct: np.ndarray) -> dict:
    mask = (t1 > 50) & np.isfinite(t1) & np.isfinite(ct)
    observed = normalized_mi(t1, ct, mask)
    shifted = {}
    for axis, label in enumerate(("LR", "AP", "SI")):
        for sign in (-1, 1):
            delta = [0, 0, 0]
            delta[axis] = sign * 5
            moved = ndimage.shift(ct, shift=delta, order=0, mode="constant", cval=0)
            shifted[f"{label}_{sign * 5:+d}vox"] = normalized_mi(t1, moved, mask)
    best_shifted = max(shifted.values())
    return {
        "normalized_mutual_information": observed,
        "shifted_controls": shifted,
        "margin_over_best_5voxel_shift": observed - best_shifted,
        "automated_gate": "pass" if observed >= best_shifted else "review",
    }


def write_fcsv(path: Path, rows: list[tuple[str, np.ndarray]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("# Markups fiducial file version = 4.11\n")
        handle.write("# CoordinateSystem = RAS\n")
        handle.write(
            "# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID\n"
        )
        writer = csv.writer(handle)
        for index, (label, point) in enumerate(rows, 1):
            writer.writerow(
                [
                    f"vtkMRMLMarkupsFiducialNode_{index}",
                    *[f"{value:.6f}" for value in point],
                    "0",
                    "0",
                    "0",
                    "1",
                    "1",
                    "1",
                    "0",
                    label,
                    "provisional_needs_visual_qc",
                    "",
                ]
            )


def make_qc_figure(
    path: Path,
    subject: str,
    t1: np.ndarray,
    ct: np.ndarray,
    affine: np.ndarray,
    fits: dict,
) -> None:
    inv = np.linalg.inv(affine)
    colors = {"right": "#ff3b30", "left": "#00d8ff"}
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
    projections = ((2, "Axial MIP"), (1, "Coronal MIP"), (0, "Sagittal MIP"))
    for col, (drop_axis, title) in enumerate(projections):
        mip = np.max(np.clip(ct, 0, 5000), axis=drop_axis).T
        axes[0, col].imshow(mip, cmap="gray", origin="lower", vmin=0, vmax=5000)
        axes[0, col].set_title(title)
        for side, fit in fits.items():
            sample = np.linspace(fit["tip"], fit["entry"], 100)
            vox = nib.affines.apply_affine(inv, sample)
            if drop_axis == 2:
                x, y = vox[:, 0], vox[:, 1]
                tipxy = nib.affines.apply_affine(inv, fit["tip"])[[0, 1]]
            elif drop_axis == 1:
                x, y = vox[:, 0], vox[:, 2]
                tipxy = nib.affines.apply_affine(inv, fit["tip"])[[0, 2]]
            else:
                x, y = vox[:, 1], vox[:, 2]
                tipxy = nib.affines.apply_affine(inv, fit["tip"])[[1, 2]]
            axes[0, col].plot(x, y, color=colors[side], linewidth=1.8)
            axes[0, col].scatter(*tipxy, color=colors[side], s=24, marker="x")
        axes[0, col].axis("off")

    mean_tip = np.mean([fit["tip"] for fit in fits.values()], axis=0)
    center_vox = np.rint(nib.affines.apply_affine(inv, mean_tip)).astype(int)
    center_vox = np.clip(center_vox, 0, np.array(t1.shape) - 1)
    slices = (
        (t1[:, :, center_vox[2]].T, ct[:, :, center_vox[2]].T, "Axial at lead tips"),
        (t1[:, center_vox[1], :].T, ct[:, center_vox[1], :].T, "Coronal at lead tips"),
        (t1[center_vox[0], :, :].T, ct[center_vox[0], :, :].T, "Sagittal near lead tips"),
    )
    for col, (t1_slice, ct_slice, title) in enumerate(slices):
        lo, hi = np.percentile(t1_slice[np.isfinite(t1_slice)], [2, 99])
        axes[1, col].imshow(t1_slice, cmap="gray", origin="lower", vmin=lo, vmax=hi)
        metal = np.ma.masked_where(ct_slice < 1000, np.clip(ct_slice, 1000, 5000))
        axes[1, col].imshow(metal, cmap="autumn", origin="lower", alpha=0.55, vmin=1000, vmax=5000)
        axes[1, col].set_title(title)
        axes[1, col].axis("off")

    fig.suptitle(
        f"{subject}: provisional CT lead axes (red=right, cyan=left) — visual QC required",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def process_subject(dataset: Path, subject: str, threshold_hu: float) -> dict:
    t1_path = (
        dataset
        / "rawdata"
        / f"sub-{subject}"
        / "ses-preop"
        / "anat"
        / f"sub-{subject}_ses-preop_acq-mprage_T1w.nii.gz"
    )
    coreg_dir = dataset / "derivatives" / "native_localization" / f"sub-{subject}" / "coreg"
    ct_path = coreg_dir / "ct_to_t1_Warped.nii.gz"
    out_dir = dataset / "derivatives" / "native_localization" / f"sub-{subject}" / "localization"
    out_dir.mkdir(parents=True, exist_ok=True)

    t1_img = nib.load(t1_path)
    ct_img = nib.load(ct_path)
    t1 = np.asarray(t1_img.dataobj, dtype=np.float32)
    ct = np.asarray(ct_img.dataobj, dtype=np.float32)
    if t1.shape != ct.shape or not np.allclose(t1_img.affine, ct_img.affine, atol=1e-3):
        raise RuntimeError("Registered CT and T1 do not share the same grid")

    center = brain_center_world(t1_img, t1)
    voxels = np.column_stack(np.where(ct > threshold_hu))
    points = nib.affines.apply_affine(ct_img.affine, voxels)
    candidate = (
        (np.abs(points[:, 0] - center[0]) < 40)
        & (points[:, 2] > center[2] - 30)
        & (points[:, 2] < center[2] + 110)
        & (points[:, 1] < center[1] + 60)
    )
    points = points[candidate]

    rng = np.random.default_rng(20260807)
    fits = {}
    for side, side_mask in (
        ("right", points[:, 0] > center[0] + 1),
        ("left", points[:, 0] < center[0] - 1),
    ):
        fits[side] = robust_line_fit(points[side_mask], rng)

    tip_z_delta = abs(float(fits["right"]["tip"][2] - fits["left"]["tip"][2]))
    geometry_gate = all(
        fit["length_mm"] >= 50 and fit["rms_residual_mm"] <= 1.5 for fit in fits.values()
    ) and tip_z_delta <= 12
    reg_metric = registration_metric(t1, ct)

    axis_csv = out_dir / "electrode_axes_provisional.csv"
    with axis_csv.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "side",
            "coordinate_system",
            "tip_x",
            "tip_y",
            "tip_z",
            "entry_x",
            "entry_y",
            "entry_z",
            "axis_x",
            "axis_y",
            "axis_z",
            "length_mm",
            "rms_residual_mm",
            "p95_residual_mm",
            "inlier_count",
            "status",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for side, fit in fits.items():
            writer.writerow(
                {
                    "side": side,
                    "coordinate_system": "T1 native RAS mm",
                    **{f"tip_{k}": f"{v:.6f}" for k, v in zip("xyz", fit["tip"])},
                    **{f"entry_{k}": f"{v:.6f}" for k, v in zip("xyz", fit["entry"])},
                    **{f"axis_{k}": f"{v:.9f}" for k, v in zip("xyz", fit["axis"])},
                    "length_mm": f"{fit['length_mm']:.3f}",
                    "rms_residual_mm": f"{fit['rms_residual_mm']:.3f}",
                    "p95_residual_mm": f"{fit['p95_residual_mm']:.3f}",
                    "inlier_count": fit["inlier_count"],
                    "status": "provisional_needs_visual_qc",
                }
            )

    level_rows = []
    level_fcsv = []
    for side, fit in fits.items():
        for level, distance in enumerate(B33005_LEVEL_MM):
            point = fit["tip"] + fit["axis"] * distance
            level_rows.append((side, level, distance, point))
            level_fcsv.append((f"{side[0].upper()}_K{level}_provisional", point))
    levels_csv = out_dir / "b33005_axial_levels_provisional.csv"
    with levels_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "side",
                "level",
                "distance_from_detected_distal_anchor_mm",
                "coordinate_system",
                "x",
                "y",
                "z",
                "status",
            ]
        )
        for side, level, distance, point in level_rows:
            writer.writerow(
                [
                    side,
                    f"K{level}",
                    distance,
                    "T1 native RAS mm",
                    *[f"{value:.6f}" for value in point],
                    "provisional_needs_visual_qc",
                ]
            )

    write_fcsv(
        out_dir / "electrode_tip_entry_provisional.fcsv",
        [
            (f"{side[0].upper()}_tip_provisional", fit["tip"])
            for side, fit in fits.items()
        ]
        + [
            (f"{side[0].upper()}_entry_provisional", fit["entry"])
            for side, fit in fits.items()
        ],
    )
    write_fcsv(out_dir / "b33005_axial_levels_provisional.fcsv", level_fcsv)
    make_qc_figure(out_dir / "electrode_localization_qc.png", subject, t1, ct, ct_img.affine, fits)

    report = {
        "subject": subject,
        "created_by": "localize_electrodes_ct_cli.py",
        "inputs": {"preop_t1": str(t1_path), "registered_postop_ct": str(ct_path)},
        "coordinate_system": "T1 native RAS mm",
        "metal_threshold": threshold_hu,
        "electrode_model_assumption": "Medtronic SenSight B33005",
        "segment_orientation": "not_estimated",
        "status": "provisional_needs_visual_qc",
        "registration_metric": reg_metric,
        "geometry_gate": {
            "pass": bool(geometry_gate),
            "bilateral_tip_z_difference_mm": tip_z_delta,
            "criteria": "each line length >=50 mm, RMS <=1.5 mm, bilateral tip-z difference <=12 mm",
        },
        "axes": {
            side: {
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in fit.items()
            }
            for side, fit in fits.items()
        },
        "limitations": [
            "Distal anchor is estimated from CT metal blooming and is not manually corrected.",
            "B33005 level centers are provisional until the implant record is confirmed.",
            "Directional segment rotation (DiODe) is not estimated.",
            "Expert CT/T1 fusion and lead-axis visual QC is required before VTA, PAM, or clinical use.",
        ],
    }
    with (out_dir / "electrode_localization_provisional.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", default=list(DEFAULT_SUBJECTS))
    parser.add_argument("--threshold", type=float, default=3000.0)
    args = parser.parse_args()

    summary = []
    for subject in args.subjects:
        try:
            report = process_subject(args.dataset.resolve(), subject, args.threshold)
            summary.append(
                {
                    "subject": subject,
                    "run_status": "completed_provisional",
                    "geometry_gate_pass": report["geometry_gate"]["pass"],
                    "registration_gate": report["registration_metric"]["automated_gate"],
                }
            )
            print(json.dumps(summary[-1], ensure_ascii=False))
        except Exception as exc:
            summary.append({"subject": subject, "run_status": "failed", "error": str(exc)})
            print(json.dumps(summary[-1], ensure_ascii=False))

    summary_path = (
        args.dataset.resolve()
        / "derivatives"
        / "native_localization"
        / "batch_localization_summary.json"
    )
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    if any(item["run_status"] == "failed" for item in summary):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
