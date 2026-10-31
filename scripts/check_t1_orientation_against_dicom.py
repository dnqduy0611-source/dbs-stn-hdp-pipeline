#!/usr/bin/env python3
"""Confirm a converted T1 NIfTI carries the scanner's own left/right.

A left-right flip shared by both images is invisible in a CT/MRI fusion
overlay, so the "no left-right flip" QC item cannot be settled from the
montage.  It can be settled against the source DICOM: ImageOrientationPatient
defines the row/column direction cosines in LPS, and the NIfTI affine must
agree with them after the LPS->RAS sign change.
"""

from __future__ import annotations

import argparse
import glob
import os
import struct
from pathlib import Path

import nibabel as nib
import numpy as np

SERIES_DESCRIPTION = (0x0008, 0x103E)
IMAGE_ORIENTATION = (0x0020, 0x0037)
IMAGE_POSITION = (0x0020, 0x0032)
INSTANCE_NUMBER = (0x0020, 0x0013)


def read_tags(path: str) -> dict:
    with open(path, "rb") as handle:
        buf = handle.read(200000)
    out = {}
    for tag, vr in ((SERIES_DESCRIPTION, b"LO"), (IMAGE_ORIENTATION, b"DS"),
                    (IMAGE_POSITION, b"DS"), (INSTANCE_NUMBER, b"IS")):
        i = buf.find(struct.pack("<HH", *tag) + vr)
        if i >= 0:
            length = struct.unpack_from("<H", buf, i + 6)[0]
            out[tag] = buf[i + 8:i + 8 + length].decode("latin-1", "ignore").strip("\x00 ")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dicom-dir", type=Path, required=True)
    parser.add_argument("--series-match", required=True)
    parser.add_argument("--nifti", type=Path, required=True)
    args = parser.parse_args()

    matches = []
    for path in glob.glob(str(args.dicom_dir / "**" / "*"), recursive=True):
        if not os.path.isfile(path):
            continue
        tags = read_tags(path)
        if args.series_match.lower() in tags.get(SERIES_DESCRIPTION, "").lower():
            matches.append((int(tags.get(INSTANCE_NUMBER, "0")), path, tags))
    if not matches:
        raise SystemExit(f"no DICOM matching {args.series_match!r}")
    matches.sort()
    _, path, tags = matches[0]

    values = [float(v) for v in tags[IMAGE_ORIENTATION].split("\\")]
    row, col = np.array(values[:3]), np.array(values[3:])
    normal = np.cross(row, col)
    dicom = np.column_stack([row, col, normal])
    dicom[0] *= -1.0  # LPS -> RAS
    dicom[1] *= -1.0

    img = nib.load(str(args.nifti))
    direction = img.affine[:3, :3]
    direction = direction / np.linalg.norm(direction, axis=0)

    det_nifti = float(np.linalg.det(direction))
    det_dicom = float(np.linalg.det(dicom))
    flipped = np.sign(det_nifti) != np.sign(det_dicom)

    print(f"series        : {tags[SERIES_DESCRIPTION]}  ({len(matches)} files)")
    print(f"IOP (LPS)     : {values}")
    print(f"NIfTI storage : {''.join(nib.aff2axcodes(img.affine))}")
    print(f"det(NIfTI) {det_nifti:+.4f}   det(DICOM) {det_dicom:+.4f}")
    print("VERDICT       : " + ("LEFT-RIGHT FLIP" if flipped else "no flip, handedness matches scanner"))

    # The NIfTI axes are a permutation of the DICOM ones; compare as sets of
    # direction cosines rather than requiring the same column order.
    overlap = np.abs(dicom.T @ direction)
    print(f"axis correspondence (max |cos| per DICOM axis): "
          f"{np.round(overlap.max(axis=1), 4).tolist()}")


if __name__ == "__main__":
    main()
