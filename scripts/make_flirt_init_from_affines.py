#!/usr/bin/env python3
"""Build a FLIRT -init matrix from the two NIfTI world affines.

FLIRT optimises in scaled-voxel space and ignores the sform/qform orientation,
so an input and reference stored in different orientations (e.g. an LPS CT
against a PIR MPRAGE) start ~90 degrees apart -- beyond any sane -searchr*
range.  Seeding with the scanner-derived world alignment removes that gap and
leaves the optimiser only the true inter-session patient motion to solve.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np


def vox2fsl(img: nib.Nifti1Image) -> np.ndarray:
    zooms = np.asarray(img.header.get_zooms()[:3], dtype=float)
    scale = np.diag(np.append(zooms, 1.0))
    if np.linalg.det(img.affine[:3, :3]) > 0:
        # Neurological storage: FLIRT flips the first axis.
        nx = img.shape[0]
        flip = np.eye(4)
        flip[0, 0] = -1.0
        flip[0, 3] = float(nx - 1)
        return scale @ flip
    return scale


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-image", type=Path, required=True)
    parser.add_argument("--ref-image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    src = nib.load(str(args.in_image))
    ref = nib.load(str(args.ref_image))
    init = vox2fsl(ref) @ np.linalg.inv(ref.affine) @ src.affine @ np.linalg.inv(vox2fsl(src))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in init:
            handle.write("  ".join(f"{v:.10f}" for v in row) + "\n")

    rotation = init[:3, :3]
    angle = np.degrees(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1)))
    print(f"in  orientation: {''.join(nib.aff2axcodes(src.affine))}")
    print(f"ref orientation: {''.join(nib.aff2axcodes(ref.affine))}")
    print(f"seed rotation magnitude: {angle:.2f} deg")
    print(f"seed translation (mm): {np.round(init[:3, 3], 2).tolist()}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
