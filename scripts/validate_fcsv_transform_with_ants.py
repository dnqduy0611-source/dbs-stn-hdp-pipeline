#!/usr/bin/env python3
"""
Validate the direct Python point transform against ANTs antsApplyTransformsToPoints.

ANTs/ITK expects physical points in LPS. Slicer stores fiducials in RAS.
For point MNI -> anchorNative, ANTs should be given the image transform
anchorNative -> MNI, because point transforms are applied in the opposite
direction of image transforms.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

import numpy as np

from convert_leaddbs_fcsv_to_ossdbs_points import (
    DEFAULT_FCSV,
    DEFAULT_MNI_TO_ANCHOR_FIELD,
    DEFAULT_OUTPUT,
    mni_ras_to_anchor_ras,
    read_fcsv,
)


ROOT = Path(r"D:\dbs_workspace")
DEFAULT_ANTS = ROOT / "tools" / "ants" / "ants-2.6.5" / "bin" / "antsApplyTransformsToPoints"


def ras_to_lps(points: np.ndarray) -> np.ndarray:
    out = points.copy()
    out[:, 0] *= -1
    out[:, 1] *= -1
    return out


def write_ants_csv(path: Path, labels: list[str], lps: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "z", "t", "label"])
        for i, (label, p) in enumerate(zip(labels, lps), 1):
            writer.writerow([p[0], p[1], p[2], 0, i])


def read_ants_csv(path: Path) -> np.ndarray:
    rows = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append([float(row["x"]), float(row["y"]), float(row["z"])])
    return np.array(rows, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ants", type=Path, default=DEFAULT_ANTS)
    parser.add_argument("--fcsv", type=Path, default=DEFAULT_FCSV)
    parser.add_argument("--transform", type=Path, default=DEFAULT_MNI_TO_ANCHOR_FIELD)
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    points = read_fcsv(args.fcsv)
    labels = [p["label"] for p in points]
    mni_ras = np.vstack([p["mni_ras"] for p in points])
    python_anchor_ras = mni_ras_to_anchor_ras(mni_ras, args.transform)

    ants_input = args.out_prefix.with_name(args.out_prefix.name + "_ants_input_mni_lps.csv")
    ants_output = args.out_prefix.with_name(args.out_prefix.name + "_ants_output_anchor_lps.csv")
    report_path = args.out_prefix.with_name(args.out_prefix.name + "_ants_validation.json")

    write_ants_csv(ants_input, labels, ras_to_lps(mni_ras))

    cmd = [
        "wsl",
        "bash",
        "-lc",
        (
            "export LD_LIBRARY_PATH=/mnt/d/dbs_workspace/tools/ants/ants-2.6.5/lib:$LD_LIBRARY_PATH; "
            f"/mnt/d/dbs_workspace/tools/ants/ants-2.6.5/bin/antsApplyTransformsToPoints "
            f"-d 3 -i '{str(ants_input).replace('D:', '/mnt/d').replace(chr(92), '/')}' "
            f"-o '{str(ants_output).replace('D:', '/mnt/d').replace(chr(92), '/')}' "
            f"-t '{str(args.transform).replace('D:', '/mnt/d').replace(chr(92), '/')}'"
        ),
    ]
    subprocess.run(cmd, check=True)

    ants_anchor_ras = ras_to_lps(read_ants_csv(ants_output))
    deltas = python_anchor_ras - ants_anchor_ras
    distances = np.linalg.norm(deltas, axis=1)

    report = {
        "ants_binary": str(args.ants),
        "ants_input_csv": str(ants_input),
        "ants_output_csv": str(ants_output),
        "transform": str(args.transform),
        "coordinate_convention": "Input MNI RAS -> LPS for ANTs; output LPS -> RAS for comparison.",
        "max_abs_axis_delta_mm": np.max(np.abs(deltas), axis=0).tolist(),
        "mean_distance_mm": float(np.mean(distances)),
        "max_distance_mm": float(np.max(distances)),
        "per_point": [
            {
                "label": label,
                "python_anchor_ras": py.tolist(),
                "ants_anchor_ras": ants.tolist(),
                "delta_python_minus_ants_mm": delta.tolist(),
                "distance_mm": float(dist),
            }
            for label, py, ants, delta, dist in zip(labels, python_anchor_ras, ants_anchor_ras, deltas, distances)
        ],
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote: {ants_input}")
    print(f"Wrote: {ants_output}")
    print(f"Wrote: {report_path}")
    print(f"Mean Python-vs-ANTs distance: {report['mean_distance_mm']:.6g} mm")
    print(f"Max Python-vs-ANTs distance: {report['max_distance_mm']:.6g} mm")


if __name__ == "__main__":
    main()
