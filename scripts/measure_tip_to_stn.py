#!/usr/bin/env python3
"""Measure electrode tip -> STN distance in the patient's own preop T1 space.

The electrode axes are fitted in CT and carried into the batch T1
(``t1_sag.nii.gz``) by the Gate A registration.  The STN is a manual
segmentation that lives on the DWI 1 mm grid (``fa_1mm_ref``).  Bridging those
two through the pipeline's T1<->DWI matrix would drag EPI distortion into a
2-3 mm measurement, so instead we register ``SWI_on_FA_1mm.nii.gz`` -- the SWI
resampled onto the *exact same grid as the ROI* -- straight to the T1.  That is
a single structural-to-structural link whose quality we can test.

FSL matrices act in scaled-voxel space, so they are converted to world (RAS)
transforms with the same ``vox2fsl`` construction used for the CT init.  The
direction is asserted, not assumed: the transform must carry the moving image's
centre of mass onto the fixed image's.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage


def vox2fsl(img) -> np.ndarray:
    """Voxel indices -> FSL scaled-millimetre coordinates."""
    zooms = np.asarray(img.header.get_zooms()[:3], dtype=float)
    scale = np.diag(np.append(zooms, 1.0))
    if np.linalg.det(img.affine[:3, :3]) > 0:
        flip = np.eye(4)
        flip[0, 0] = -1.0
        flip[0, 3] = float(img.shape[0] - 1)
        return scale @ flip
    return scale


def fsl_mat_to_world(mat: np.ndarray, moving, fixed) -> np.ndarray:
    """FSL ``-omat`` (moving -> fixed) as a world-to-world transform."""
    return (fixed.affine @ np.linalg.inv(vox2fsl(fixed))
            @ mat @ vox2fsl(moving) @ np.linalg.inv(moving.affine))


def normalized_mi(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    aa, bb = a[mask], b[mask]
    if len(aa) > 500_000:
        step = int(math.ceil(len(aa) / 500_000))
        aa, bb = aa[::step], bb[::step]
    ar = np.percentile(aa, [0.5, 99.5])
    br = np.percentile(bb, [0.5, 99.5])
    keep = (aa >= ar[0]) & (aa <= ar[1]) & (bb >= br[0]) & (bb <= br[1])
    hist, _, _ = np.histogram2d(aa[keep], bb[keep], bins=64)
    prob = hist / max(hist.sum(), 1)
    px, py = prob.sum(axis=1), prob.sum(axis=0)
    hx = -np.sum(px[px > 0] * np.log(px[px > 0]))
    hy = -np.sum(py[py > 0] * np.log(py[py > 0]))
    hxy = -np.sum(prob[prob > 0] * np.log(prob[prob > 0]))
    return float((hx + hy) / hxy) if hxy > 0 else float("nan")


def centre_of_mass(img) -> np.ndarray:
    data = np.asarray(img.dataobj, dtype=np.float32)
    keep = data > 0.05 * data.max()
    idx = np.column_stack(np.where(keep))
    return nib.affines.apply_affine(img.affine, np.average(idx, axis=0, weights=data[keep]))


def surface_distance(point: np.ndarray, world: np.ndarray) -> float:
    """Distance to the nearest ROI voxel centre; negative when inside."""
    d = float(np.min(np.linalg.norm(world - point, axis=1)))
    return d


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--axes-json", type=Path, required=True)
    parser.add_argument("--t1", type=Path, required=True)
    parser.add_argument("--swi-on-roi-grid", type=Path, required=True)
    parser.add_argument("--swi-in-t1", type=Path, required=True)
    parser.add_argument("--swi2t1-mat", type=Path, required=True)
    parser.add_argument("--roi", nargs=2, action="append", metavar=("SIDE", "PATH"), required=True)
    parser.add_argument("--contact-span-mm", type=float, default=6.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    t1 = nib.load(str(args.t1))
    swi = nib.load(str(args.swi_on_roi_grid))
    mat = np.loadtxt(str(args.swi2t1_mat))
    world = fsl_mat_to_world(mat, swi, t1)

    report: dict = {"checks": {}, "sides": {}}

    # --- sign check: the transform must move SWI's COM onto its registered copy
    com_moving = centre_of_mass(swi)
    com_fixed = centre_of_mass(nib.load(str(args.swi_in_t1)))
    fwd = nib.affines.apply_affine(world, com_moving)
    inv = nib.affines.apply_affine(np.linalg.inv(world), com_moving)
    err_fwd = float(np.linalg.norm(fwd - com_fixed))
    err_inv = float(np.linalg.norm(inv - com_fixed))
    report["checks"]["sign"] = {
        "com_moving_world": com_moving.round(3).tolist(),
        "com_registered_world": com_fixed.round(3).tolist(),
        "error_forward_mm": round(err_fwd, 4),
        "error_inverse_mm": round(err_inv, 4),
        "pass": bool(err_fwd < 1.0 and err_fwd < 0.2 * err_inv),
    }

    # --- registration quality: must beat +-5 voxel shifted controls
    fixed = np.asarray(t1.dataobj, dtype=np.float32)
    moved = np.asarray(nib.load(str(args.swi_in_t1)).dataobj, dtype=np.float32)
    mask = (fixed > np.percentile(fixed, 60)) & (moved > np.percentile(moved[moved > 0], 30))
    base = normalized_mi(fixed, moved, mask)
    controls = {}
    for axis, name in enumerate(("LR", "AP", "SI")):
        for delta in (-5, 5):
            shift = np.zeros(3)
            shift[axis] = delta
            controls[f"{name}{delta:+d}vox"] = normalized_mi(
                fixed, ndimage.shift(moved, shift, order=0, mode="constant", cval=0.0), mask)
    best = max(controls.values())
    report["checks"]["registration"] = {
        "nmi": round(base, 5),
        "controls": {k: round(v, 5) for k, v in controls.items()},
        "strongest_control": max(controls, key=controls.get),
        "margin": round(base - best, 5),
        "pass": bool(base > best),
    }

    axes = json.loads(args.axes_json.read_text(encoding="utf-8"))["axes"]
    for side, path in args.roi:
        roi = nib.load(str(path))
        if not np.allclose(roi.affine, swi.affine, atol=1e-4) or roi.shape != swi.shape:
            raise SystemExit(f"{path} is not on the same grid as the registered SWI")
        voxels = np.column_stack(np.where(np.asarray(roi.dataobj) > 0))
        roi_world = nib.affines.apply_affine(world @ roi.affine, voxels)
        centroid = roi_world.mean(axis=0)

        fit = axes[side]
        tip = np.asarray(fit["tip"], dtype=float)
        axis_dir = np.asarray(fit["axis"], dtype=float)
        axis_dir = axis_dir / np.linalg.norm(axis_dir)
        # The fitted axis points from the tip up the shaft; contacts sit above it.
        contacts = tip + np.linspace(0.0, args.contact_span_mm, 61)[:, None] * axis_dir

        along = (roi_world - tip) @ axis_dir
        radial = np.linalg.norm(np.cross(roi_world - tip, axis_dir), axis=1)
        inside = radial < 0.75  # ROI voxels the shaft passes through

        report["sides"][side] = {
            "roi": str(Path(path).name),
            "roi_voxels": int(len(voxels)),
            "roi_centroid_t1_world": centroid.round(3).tolist(),
            "tip_t1_world": tip.round(3).tolist(),
            "tip_to_centroid_mm": round(float(np.linalg.norm(tip - centroid)), 3),
            "tip_to_nearest_roi_voxel_mm": round(surface_distance(tip, roi_world), 3),
            "tip_inside_roi": bool(surface_distance(tip, roi_world) <= 0.87),
            "contact_span_min_distance_mm": round(
                float(min(surface_distance(c, roi_world) for c in contacts)), 3),
            "roi_voxels_pierced_by_shaft": int(inside.sum()),
            "shaft_traversal_mm": (round(float(along[inside].max() - along[inside].min()), 2)
                                   if inside.any() else 0.0),
            "centroid_offset_from_tip_xyz_mm": (centroid - tip).round(3).tolist(),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    sign = report["checks"]["sign"]
    reg = report["checks"]["registration"]
    print(f"sign check    : forward {sign['error_forward_mm']:.3f} mm  "
          f"inverse {sign['error_inverse_mm']:.3f} mm  -> "
          f"{'PASS' if sign['pass'] else 'FAIL'}")
    print(f"registration  : NMI {reg['nmi']:.5f}  strongest control "
          f"{reg['strongest_control']} {max(reg['controls'].values()):.5f}  "
          f"margin {reg['margin']:+.5f}  -> {'PASS' if reg['pass'] else 'FAIL'}")
    for side, r in report["sides"].items():
        print(f"\n{side}  ({r['roi_voxels']} vox)")
        print(f"  tip                     {r['tip_t1_world']}")
        print(f"  STN centroid            {r['roi_centroid_t1_world']}")
        print(f"  tip -> centroid         {r['tip_to_centroid_mm']:.2f} mm   "
              f"offset {r['centroid_offset_from_tip_xyz_mm']}")
        print(f"  tip -> nearest STN vox  {r['tip_to_nearest_roi_voxel_mm']:.2f} mm  "
              f"(inside: {r['tip_inside_roi']})")
        print(f"  contact span closest    {r['contact_span_min_distance_mm']:.2f} mm")
        print(f"  shaft pierces           {r['roi_voxels_pierced_by_shaft']} STN voxels, "
              f"{r['shaft_traversal_mm']} mm of travel")


if __name__ == "__main__":
    main()
