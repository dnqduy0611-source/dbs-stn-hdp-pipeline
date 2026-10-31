#!/usr/bin/env python3
"""Convert one uncompressed DICOM series to NIfTI without pydicom.

Written because pydicom is not installed in this environment and the earlier
ad-hoc converter was never saved.  Two traps this file exists to close:

* ``PixelRepresentation`` (0028,0103) must decide signedness.  Decoding signed
  CT pixels as unsigned wraps every negative HU by +65536, which yields
  ``max = 64511`` instead of 31743 and silently destroys downstream
  registration.  See the sub-01 step 1 notes.
* Sequences must be skipped, not walked.  A naive sequential reader derails on
  the first undefined-length sequence and then reports ``None`` for every tag
  in groups 0010/0018/0020.

Only uncompressed little-endian transfer syntaxes are supported; anything else
raises rather than guessing.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import nibabel as nib
import numpy as np

EXPLICIT_LE = "1.2.840.10008.1.2.1"
IMPLICIT_LE = "1.2.840.10008.1.2"
LONG_VRS = {b"OB", b"OW", b"OF", b"SQ", b"UT", b"UN", b"UC", b"UR"}

ITEM = (0xFFFE, 0xE000)
ITEM_END = (0xFFFE, 0xE00D)
SEQ_END = (0xFFFE, 0xE0DD)


def _transfer_syntax(blob: bytes) -> tuple[str, int]:
    """Return (transfer syntax UID, offset of the main dataset)."""
    if blob[128:132] != b"DICM":
        raise ValueError("missing DICM magic")
    i = 132
    uid = EXPLICIT_LE
    while i + 8 <= len(blob):
        group, element = struct.unpack_from("<HH", blob, i)
        if group != 0x0002:
            break
        vr = blob[i + 4 : i + 6]
        if vr in LONG_VRS:
            length = struct.unpack_from("<I", blob, i + 8)[0]
            header = 12
        else:
            length = struct.unpack_from("<H", blob, i + 6)[0]
            header = 8
        if (group, element) == (0x0002, 0x0010):
            uid = blob[i + header : i + header + length].decode("latin-1").strip("\x00 ")
        i += header + length
    return uid, i


def parse(blob: bytes) -> dict[tuple[int, int], bytes]:
    """Top-level tags only; sequences are skipped whole."""
    uid, i = _transfer_syntax(blob)
    if uid not in (EXPLICIT_LE, IMPLICIT_LE):
        raise ValueError(f"unsupported transfer syntax {uid} (compressed?)")
    explicit = uid == EXPLICIT_LE
    out: dict[tuple[int, int], bytes] = {}
    depth = 0
    while i + 8 <= len(blob):
        group, element = struct.unpack_from("<HH", blob, i)
        tag = (group, element)
        if tag in (ITEM, ITEM_END, SEQ_END):
            length = struct.unpack_from("<I", blob, i + 4)[0]
            i += 8
            if tag == SEQ_END:
                depth = max(depth - 1, 0)
            elif tag == ITEM and length != 0xFFFFFFFF:
                i += length
            continue
        if explicit:
            vr = blob[i + 4 : i + 6]
            if vr in LONG_VRS:
                length = struct.unpack_from("<I", blob, i + 8)[0]
                header = 12
            else:
                length = struct.unpack_from("<H", blob, i + 6)[0]
                header = 8
        else:
            vr = b"??"
            length = struct.unpack_from("<I", blob, i + 4)[0]
            header = 8
        if length == 0xFFFFFFFF:
            depth += 1
            i += header
            continue
        if vr == b"SQ":
            i += header + length
            continue
        if depth == 0:
            out[tag] = blob[i + header : i + header + length]
        if tag == (0x7FE0, 0x0010):
            break
        i += header + length
    return out


def text(tags, tag, default=None):
    raw = tags.get(tag)
    return default if raw is None else raw.decode("latin-1").strip("\x00 ")


def numbers(tags, tag):
    value = text(tags, tag)
    if value is None:
        return []
    return [float(part) for part in value.split("\\") if part.strip()]


def ushort(tags, tag, default=0):
    raw = tags.get(tag)
    return default if raw is None or len(raw) < 2 else struct.unpack_from("<H", raw, 0)[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dicom-dir", type=Path, required=True)
    parser.add_argument("--series-number", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    slices = []
    for path in sorted(args.dicom_dir.rglob("*")):
        if not path.is_file():
            continue
        try:
            tags = parse(path.read_bytes())
        except ValueError:
            continue
        if (0x7FE0, 0x0010) not in tags:
            continue
        series = text(tags, (0x0020, 0x0011))
        if series is None or int(series) != args.series_number:
            continue
        slices.append((path, tags))
    if not slices:
        raise SystemExit(f"no images found for series {args.series_number}")

    first = slices[0][1]
    rows = ushort(first, (0x0028, 0x0010))
    cols = ushort(first, (0x0028, 0x0011))
    bits = ushort(first, (0x0028, 0x0100), 16)
    signed = ushort(first, (0x0028, 0x0103), 0) == 1
    if bits != 16:
        raise SystemExit(f"unexpected BitsAllocated {bits}")
    orientation = np.asarray(numbers(first, (0x0020, 0x0037)), dtype=float)
    spacing = numbers(first, (0x0028, 0x0030))
    if orientation.size != 6 or len(spacing) != 2:
        raise SystemExit("missing ImageOrientationPatient/PixelSpacing")
    row_dir, col_dir = orientation[:3], orientation[3:]
    normal = np.cross(row_dir, col_dir)

    # Order by projection on the slice normal; ImagePositionPatient is the
    # authority, InstanceNumber is not (GE renumbers reformats).
    keyed = []
    for path, tags in slices:
        position = np.asarray(numbers(tags, (0x0020, 0x0032)), dtype=float)
        if position.size != 3:
            raise SystemExit(f"{path.name}: missing ImagePositionPatient")
        keyed.append((float(position @ normal), position, path, tags))
    keyed.sort(key=lambda item: item[0])

    projections = np.array([item[0] for item in keyed])
    gaps = np.diff(projections)
    if gaps.size and (gaps.max() - gaps.min()) > 0.01:
        print(f"warning: irregular slice spacing {gaps.min():.4f}..{gaps.max():.4f} mm")
    slice_gap = float(np.median(gaps)) if gaps.size else 1.0

    volume = np.empty((cols, rows, len(keyed)), dtype=np.int16)
    for index, (_, _, _, tags) in enumerate(keyed):
        raw = tags[(0x7FE0, 0x0010)][: rows * cols * 2]
        plane = np.frombuffer(raw, dtype="<i2" if signed else "<u2").reshape(rows, cols)
        plane = plane.astype(np.int32)
        if not signed:
            # Only correct if the series really is signed CT stored unsigned.
            plane = np.where(plane > 32767, plane - 65536, plane)
        slope = numbers(tags, (0x0028, 0x1053)) or [1.0]
        intercept = numbers(tags, (0x0028, 0x1052)) or [0.0]
        plane = plane * slope[0] + intercept[0]
        volume[:, :, index] = np.rint(plane).astype(np.int16).T

    # DICOM patient coordinates are LPS; NIfTI world is RAS.
    affine = np.eye(4)
    affine[:3, 0] = row_dir * spacing[1]
    affine[:3, 1] = col_dir * spacing[0]
    affine[:3, 2] = normal * slice_gap
    affine[:3, 3] = keyed[0][1]
    affine[:2, :] *= -1.0

    image = nib.Nifti1Image(volume, affine)
    image.header.set_slope_inter(1, 0)
    image.header.set_xyzt_units("mm")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, str(args.output))

    report = {
        "dicom_dir": str(args.dicom_dir),
        "series_number": args.series_number,
        "series_description": text(first, (0x0008, 0x103E)),
        "patient_name": text(first, (0x0010, 0x0010)),
        "patient_id": text(first, (0x0010, 0x0020)),
        "study_date": text(first, (0x0008, 0x0020)),
        "accession": text(first, (0x0008, 0x0050)),
        "convolution_kernel": text(first, (0x0018, 0x1210)),
        "slice_thickness_mm": numbers(first, (0x0018, 0x0050)),
        "pixel_representation_signed": bool(signed),
        "slice_count": len(keyed),
        "shape": list(volume.shape),
        "zooms_mm": [abs(spacing[1]), abs(spacing[0]), abs(slice_gap)],
        "orientation": "".join(nib.aff2axcodes(affine)),
        "min_hu": int(volume.min()),
        "max_hu": int(volume.max()),
        "fraction_below_minus_1500": float(np.mean(volume < -1500)),
        "voxels_above_3000": int((volume > 3000).sum()),
        "output": str(args.output),
    }
    print(json.dumps(report, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
