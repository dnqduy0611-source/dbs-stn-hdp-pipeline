#!/usr/bin/env python3
"""Physiology-oriented post-filter for ipsilateral HDP pilot bundles.

The filter is intentionally applied after endpoint, ordered PLIC, Z-floor,
length/tortuosity, and zero-midline filtering. It rejects streamlines with
excessive global detour, sharp accumulated bending, superior backtracking, or
multiple entries into the terminal STN ROI.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("in_tck")
ap.add_argument("out_tck")
ap.add_argument("in_weights")
ap.add_argument("out_weights")
ap.add_argument("--stn", required=True)
ap.add_argument("--rejected-tck")
ap.add_argument("--rejected-weights")
ap.add_argument("--metrics-tsv")
ap.add_argument("--max-tortuosity", type=float, default=1.25)
ap.add_argument("--max-bend-deg", type=float, default=60.0)
ap.add_argument("--bend-window-mm", type=float, default=5.0)
ap.add_argument("--max-superior-backtrack-mm", type=float, default=2.0)
ap.add_argument("--resample-step-mm", type=float, default=1.0)
a = ap.parse_args()


def resample(streamline: np.ndarray, step: float) -> np.ndarray:
    distances = np.linalg.norm(np.diff(streamline, axis=0), axis=1)
    cumulative = np.r_[0.0, np.cumsum(distances)]
    locations = np.arange(0.0, cumulative[-1], step)
    if not len(locations) or locations[-1] < cumulative[-1]:
        locations = np.r_[locations, cumulative[-1]]
    return np.column_stack(
        [np.interp(locations, cumulative, streamline[:, axis]) for axis in range(3)]
    )


stn_image = nib.load(a.stn)
stn = np.asanyarray(stn_image.dataobj) > 0.5
world_to_voxel = np.linalg.inv(stn_image.affine)


def sample_stn(points: np.ndarray) -> np.ndarray:
    ijk = np.rint(nib.affines.apply_affine(world_to_voxel, points)).astype(int)
    valid = np.all((ijk >= 0) & (ijk < np.asarray(stn.shape)), axis=1)
    result = np.zeros(len(points), dtype=bool)
    q = ijk[valid]
    result[valid] = stn[q[:, 0], q[:, 1], q[:, 2]]
    return result


def count_components(values: np.ndarray) -> int:
    return int(np.sum(values & np.r_[True, ~values[:-1]]))


def metrics(streamline: np.ndarray) -> dict[str, float | int | bool]:
    # HDP runs from the high cortical endpoint to the lower STN endpoint.
    if streamline[0, 2] < streamline[-1, 2]:
        streamline = streamline[::-1]
    points = resample(streamline, a.resample_step_mm)
    arc = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    chord = float(np.linalg.norm(points[-1] - points[0]))
    tortuosity = arc / chord if chord > 1e-6 else np.inf

    dz = np.diff(points[:, 2])
    superior_backtrack = float(dz[dz > 0].sum())

    window = max(1, int(round(a.bend_window_mm / a.resample_step_mm)))
    bends = []
    for index in range(window, len(points) - window):
        before = points[index] - points[index - window]
        after = points[index + window] - points[index]
        denominator = np.linalg.norm(before) * np.linalg.norm(after)
        if denominator <= 1e-8:
            continue
        cosine = np.clip(np.dot(before, after) / denominator, -1.0, 1.0)
        bends.append(float(np.degrees(np.arccos(cosine))))
    max_bend = max(bends) if bends else 0.0

    stn_samples = sample_stn(points)
    stn_components = count_components(stn_samples)
    terminal_stn = bool(stn_samples[-1])

    keep = (
        tortuosity <= a.max_tortuosity
        and max_bend <= a.max_bend_deg
        and superior_backtrack <= a.max_superior_backtrack_mm
        and stn_components == 1
        and terminal_stn
    )
    return {
        "arc_mm": arc,
        "tortuosity": tortuosity,
        "max_bend_deg": max_bend,
        "superior_backtrack_mm": superior_backtrack,
        "stn_components": stn_components,
        "terminal_stn": terminal_stn,
        "keep": keep,
    }


tractogram = nib.streamlines.load(a.in_tck)
streamlines = list(tractogram.streamlines)
weights = np.loadtxt(a.in_weights, comments="#", ndmin=1).ravel()
if weights.size != len(streamlines):
    raise SystemExit(
        f"weight/streamline mismatch: {weights.size} != {len(streamlines)}"
    )

rows = [metrics(streamline) for streamline in streamlines]
keep = np.asarray([row["keep"] for row in rows], dtype=bool)
nib.streamlines.save(tractogram.tractogram[keep], a.out_tck, header=tractogram.header)
np.savetxt(a.out_weights, weights[keep], fmt="%.10g")

if a.rejected_tck:
    nib.streamlines.save(
        tractogram.tractogram[~keep], a.rejected_tck, header=tractogram.header
    )
if a.rejected_weights:
    np.savetxt(a.rejected_weights, weights[~keep], fmt="%.10g")
if a.metrics_tsv:
    with Path(a.metrics_tsv).open("w", encoding="utf-8") as stream:
        stream.write(
            "index\tkeep\tarc_mm\ttortuosity\tmax_bend_deg\t"
            "superior_backtrack_mm\tstn_components\tterminal_stn\tweight\n"
        )
        for index, (row, weight) in enumerate(zip(rows, weights)):
            stream.write(
                f"{index}\t{int(row['keep'])}\t{row['arc_mm']:.6f}\t"
                f"{row['tortuosity']:.6f}\t{row['max_bend_deg']:.6f}\t"
                f"{row['superior_backtrack_mm']:.6f}\t"
                f"{row['stn_components']}\t{int(row['terminal_stn'])}\t"
                f"{weight:.10g}\n"
            )

print(
    f"in={len(streamlines)} keep={int(keep.sum())} "
    f"rejected={int((~keep).sum())} "
    f"sift2={weights.sum():.6f}->{weights[keep].sum():.6f}"
)
print(
    f"thresholds: tort<={a.max_tortuosity:g} "
    f"bend{2*a.bend_window_mm:g}mm<={a.max_bend_deg:g}deg "
    f"superior_backtrack<={a.max_superior_backtrack_mm:g}mm "
    "stn_components=1 terminal_stn=1"
)
