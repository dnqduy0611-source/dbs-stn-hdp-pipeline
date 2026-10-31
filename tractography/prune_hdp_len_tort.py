#!/usr/bin/env python3
"""Keep HDP streamlines below arc-length and tortuosity limits."""
import argparse
import sys

import nibabel as nib
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("in_tck")
ap.add_argument("out_tck")
ap.add_argument("in_w", nargs="?")
ap.add_argument("out_w", nargs="?")
ap.add_argument("--maxlen", type=float, default=95.0)
ap.add_argument("--maxtort", type=float, default=1.4)
a = ap.parse_args()

tf = nib.streamlines.load(a.in_tck)
streamlines = list(tf.streamlines)
if not streamlines:
    nib.streamlines.save(tf.tractogram, a.out_tck, header=tf.header)
    if a.out_w:
        open(a.out_w, "w", encoding="utf-8").close()
    print("in=0 out=0")
    sys.exit(0)

lengths = np.array([
    np.linalg.norm(np.diff(s, axis=0), axis=1).sum() for s in streamlines
])
end_to_end = np.array([
    np.linalg.norm(s[-1] - s[0]) for s in streamlines
])
tortuosity = np.divide(
    lengths, end_to_end, out=np.full_like(lengths, np.inf), where=end_to_end > 1e-6
)
keep = (lengths <= a.maxlen) & (tortuosity <= a.maxtort)

nib.streamlines.save(tf.tractogram[keep], a.out_tck, header=tf.header)
if a.in_w and a.out_w:
    weights = np.loadtxt(a.in_w, comments="#", ndmin=1).ravel()
    if weights.size != len(streamlines):
        raise SystemExit(
            f"weight/streamline mismatch: {weights.size} != {len(streamlines)}"
        )
    np.savetxt(a.out_w, weights[keep], fmt="%.10g")

print(
    f"in={len(streamlines)} out={int(keep.sum())} "
    f"maxlen={a.maxlen:g} maxtort={a.maxtort:g}"
)
if keep.any():
    print(
        f"kept_length_mean={lengths[keep].mean():.3f} "
        f"kept_length_max={lengths[keep].max():.3f} "
        f"kept_tort_mean={tortuosity[keep].mean():.4f} "
        f"kept_tort_max={tortuosity[keep].max():.4f}"
    )
