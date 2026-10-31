#!/usr/bin/env python3
"""Fit a full midsagittal plane after sanitising non-finite image voxels.

Plane convention: x(Y,Z) = XC + AY*Y + AZ*Z.
This is a finite-safe derivative of the cohort msp_fit.py used for the
quarantined sub-09 pilot; it does not alter the cohort source script.
"""
import argparse

import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import minimize

ap = argparse.ArgumentParser()
ap.add_argument("image")
ap.add_argument("--step", type=int, default=3)
ap.add_argument("--thr", type=float, default=0.05)
ap.add_argument("--zrange", type=float, nargs=2, default=None)
ap.add_argument("--label", default="")
ap.add_argument("--maxiter", type=int, default=1800)
a = ap.parse_args()

im = nib.load(a.image)
vol = np.asarray(im.dataobj, dtype=np.float32)
if vol.ndim > 3:
    vol = vol[..., 0]
nonfinite = int((~np.isfinite(vol)).sum())
vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)
aff = im.affine
inv = np.linalg.inv(aff)

s = a.step
ii, jj, kk = np.meshgrid(
    *[np.arange(0, n, s) for n in vol.shape[:3]], indexing="ij"
)
idx = np.stack([ii.ravel(), jj.ravel(), kk.ravel()]).astype(np.float64)
val = vol[ii.ravel(), jj.ravel(), kk.ravel()]
p = (aff[:3, :3] @ idx) + aff[:3, 3:4]
mask = np.isfinite(val) & (val > a.thr)
if a.zrange:
    mask &= (p[2] >= a.zrange[0]) & (p[2] <= a.zrange[1])
p, val = p[:, mask], val[mask]
if val.size < 1000:
    raise SystemExit("FATAL: too few finite suprathreshold samples")

print("=" * 68)
print(f"FINITE-SAFE MAT PHANG DOC GIUA   {a.label}")
print("=" * 68)
print(f"anh        : {a.image}")
print(f"nonfinite  : {nonfinite} voxel replaced by 0")
print(f"diem dung  : {val.size}  (nguong {a.thr:.2f}, lay thua {s})")
print(f"dai world-Z: {p[2].min():.1f} .. {p[2].max():.1f} mm")


def objective(params):
    ny, nz, d = params
    n0 = np.array([1.0, ny, nz], dtype=float)
    norm = np.linalg.norm(n0)
    n = n0 / norm
    dist = n @ p - d / norm
    reflected = p - 2.0 * n[:, None] * dist[None, :]
    vox = (inv[:3, :3] @ reflected) + inv[:3, 3:4]
    ok = np.all(
        (vox >= 0)
        & (vox <= (np.asarray(vol.shape[:3], dtype=float)[:, None] - 1)),
        axis=0,
    )
    if ok.sum() < 1000:
        return 1.0
    mirrored = map_coordinates(
        vol, vox[:, ok], order=1, mode="constant", cval=0.0
    )
    x = val[ok]
    finite = np.isfinite(x) & np.isfinite(mirrored)
    if finite.sum() < 1000:
        return 1.0
    x = x[finite] - x[finite].mean()
    y = mirrored[finite] - mirrored[finite].mean()
    denominator = np.sqrt(np.dot(x, x) * np.dot(y, y))
    if not np.isfinite(denominator) or denominator <= 0:
        return 1.0
    score = float(np.dot(x, y) / denominator)
    return -score if np.isfinite(score) else 1.0


x0 = np.array([0.0, 0.0, float(np.median(p[0]))])
initial = -objective(x0)
print(f"khoi tao   : ny=0 nz=0 d={x0[2]:.2f}   (NCC ban dau {initial:.4f})")
res = minimize(
    objective,
    x0,
    method="Nelder-Mead",
    options=dict(xatol=1e-4, fatol=1e-7, maxiter=a.maxiter, maxfev=a.maxiter),
)
ny, nz, d = map(float, res.x)
ncc = -float(res.fun)
if not np.isfinite(ncc):
    raise SystemExit("FATAL: fitted NCC is not finite")
nrm = np.linalg.norm([1.0, ny, nz])
print(f"hoi tu     : {res.success}  sau {res.nfev} lan goi,  NCC = {ncc:.4f}")
print()
print(f"MAT PHANG  :  x(Y,Z) = {d:.3f}  {-ny:+.5f}*Y  {-nz:+.5f}*Z")
print(f"   nghieng theo Z = {np.degrees(np.arctan(-nz)):+.2f} deg")
print(f"   nghieng theo Y = {np.degrees(np.arctan(-ny)):+.2f} deg")
print(
    "   phap tuyen chuan hoa = "
    f"({1/nrm:.5f}, {ny/nrm:.5f}, {nz/nrm:.5f})"
)
print("=" * 68)
