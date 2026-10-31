#!/usr/bin/env python3
"""Re-fit the sampling axis from bare shaft, then re-read the marker.

The cross-section diagnostic showed that on sub-08's left lead the metal
centroid walks 0.5 mm away from the assumed axis across the marker window, in a
direction that barely rotates with depth.  That is a displacement, not a
rotation, and it is far larger than the lobe asymmetry DiODe is trying to read -
so it dominates the centre-of-mass and forces the upper and lower readings onto
opposite sides of the axis, which is the 180 deg that gets reported.

The fix has to avoid a circularity: fitting the axis through the marker would
regress out the lobes themselves.  So the axis is fit only on *bare shaft*,
below the marker and above it, where the cross-section is rotationally
symmetric and the centroid therefore tracks the true centreline.  The marker
window between them is interpolated, never fit.

Success criterion, decided before looking: the refit must (a) drive the
residual eccentricity in the marker window down toward the noise, and (b) leave
a lobe direction that *rotates* with depth instead of sitting still.  If the
eccentricity collapses but nothing rotates, the marker is genuinely unresolved
and no axis correction will save this lead.

Read-only.  Measures no roll and approves no gate.
"""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

BASE = Path(r"D:\ossdbs_env\dbs_stn_hdp_pipeline\lead_dbs_work"
            r"\ElectrodeLocalizationBatch20260807\derivatives\native_localization")

METAL_HU = 1500.0
DISC_R_MM = 2.2
STEP_MM = 0.12
# Bare shaft either side of the marker: above the top contact (which ends near
# 10.5 mm from the tip) and above the marker.  Nothing inside 12.5-16.0 is fit.
SHAFT_LO = (10.8, 12.4)
SHAFT_HI = (16.4, 22.0)
MARKER = (12.6, 16.2)


def frame(axis):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    seed = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(axis, seed)
    e1 /= np.linalg.norm(e1)
    return axis, e1, np.cross(axis, e1)


def disc(radius, step):
    g = np.arange(-int(radius / step), int(radius / step) + 1) * step
    uu, vv = np.meshgrid(g, g, indexing="ij")
    keep = (uu ** 2 + vv ** 2) <= radius ** 2
    return uu[keep], vv[keep]


def centroid_at(data, inv, origin, ax, e1, e2, depth, uu, vv):
    pts = origin + ax * depth + uu[:, None] * e1 + vv[:, None] * e2
    vox = (inv[:3, :3] @ pts.T).T + inv[:3, 3]
    vals = ndimage.map_coordinates(data, vox.T, order=1, mode="constant", cval=-1024.0)
    m = vals >= METAL_HU
    if m.sum() < 5:
        return None
    w = vals[m] - METAL_HU
    return (float((uu[m] * w).sum() / w.sum()),
            float((vv[m] * w).sum() / w.sum()),
            vals, m)


def run(subject, side_name):
    diode = BASE / f"sub-{subject}" / "diode"
    summ = json.loads((diode / f"sub-{subject}_B33005_diode_summary.json")
                      .read_text(encoding="utf-8"))
    s = next(x for x in summ if x["side_name"] == side_name)
    img = nib.load(str(diode / f"sub-{subject}_ses-postop_CT_original.nii"))
    data, inv = img.get_fdata(), np.linalg.inv(img.affine)

    k0 = np.array([s["k0_ras_x"], s["k0_ras_y"], s["k0_ras_z"]])
    ax, e1, e2 = frame([s["axis_ras_x"], s["axis_ras_y"], s["axis_ras_z"]])
    origin = k0 - ax * 1.65          # physical tip
    uu, vv = disc(DISC_R_MM, STEP_MM)

    depths = np.arange(10.0, 22.01, 0.2)
    got = {}
    for d in depths:
        r = centroid_at(data, inv, origin, ax, e1, e2, d, uu, vv)
        if r is not None:
            got[round(float(d), 2)] = (r[0], r[1])

    fit_d = np.array([d for d in got
                      if SHAFT_LO[0] <= d <= SHAFT_LO[1] or SHAFT_HI[0] <= d <= SHAFT_HI[1]])
    if fit_d.size < 8:
        print(f"sub-{subject} {side_name}: not enough bare shaft")
        return
    cu = np.array([got[d][0] for d in fit_d])
    cv = np.array([got[d][1] for d in fit_d])
    su, iu = np.polyfit(fit_d, cu, 1)
    sv, iv = np.polyfit(fit_d, cv, 1)
    tilt = np.degrees(np.arctan(np.hypot(su, sv)))
    rms = float(np.sqrt(np.mean((cu - (su * fit_d + iu)) ** 2
                                + (cv - (sv * fit_d + iv)) ** 2)))

    new_ax = ax + su * e1 + sv * e2
    new_ax /= np.linalg.norm(new_ax)
    new_origin = origin + iu * e1 + iv * e2
    nax, n1, n2 = frame(new_ax)

    print(f"== sub-{subject} {side_name}   polar1 {s['polar1_deg']:.1f}   "
          f"dCOG(DiODe) {s['upper_lower_cog_difference_deg']:.1f}")
    print(f"   shaft fit on {fit_d.size} depths   axis correction {tilt:.2f} deg   "
          f"shaft residual RMS {rms:.3f} mm")
    print(f"{'depth':>6} {'ecc_old':>8} {'ecc_new':>8} {'lobe_old':>9} {'lobe_new':>9}  profile (refit axis)")
    out = []
    for d in np.arange(MARKER[0], MARKER[1] + 0.01, 0.2):
        d = round(float(d), 2)
        old = got.get(d)
        r = centroid_at(data, inv, new_origin, nax, n1, n2, d, uu, vv)
        if old is None or r is None:
            continue
        ecc_old = np.hypot(*old)
        ecc_new = np.hypot(r[0], r[1])
        vals, m = r[2], r[3]
        rr = np.hypot(uu, vv)
        band = m & (rr >= 0.30) & (rr <= 1.10)
        if band.sum() < 10:
            continue
        ang = np.degrees(np.arctan2(vv[band], uu[band])) % 360.0
        hist, _ = np.histogram(ang, bins=36, range=(0, 360), weights=vals[band] - METAL_HU)
        hist = ndimage.gaussian_filter1d(hist, 1.0, mode="wrap")
        ph = np.degrees(np.arctan2((hist * np.sin(np.radians(np.arange(5, 360, 10)))).sum(),
                                   (hist * np.cos(np.radians(np.arange(5, 360, 10)))).sum())) % 360
        top = hist.max() if hist.max() > 0 else 1.0
        bars = "".join(" .:-=+*#%@"[min(9, int(9 * h / top))] for h in hist)
        lobe_old = np.degrees(np.arctan2(old[1], old[0])) % 360
        print(f"{d:6.2f} {ecc_old:8.3f} {ecc_new:8.3f} {lobe_old:9.0f} {ph:9.0f}  {bars}")
        out.append((d, ecc_old, ecc_new, ph))
    if out:
        arr = np.array(out)
        print(f"   ecc in marker window: old mean {arr[:,1].mean():.3f} mm "
              f"-> new mean {arr[:,2].mean():.3f} mm")
        unw = np.unwrap(np.radians(arr[:, 3]))
        print(f"   lobe direction (refit) swings {np.degrees(unw.max()-unw.min()):.0f} deg "
              f"over {arr[0,0]:.1f}-{arr[-1,0]:.1f} mm")
    print()


if __name__ == "__main__":
    for subj, side in (("sub-08", "left"), ("sub-08", "right"),
                       ("sub-01", "left"), ("sub-01", "right")):
        run(subj, side)
