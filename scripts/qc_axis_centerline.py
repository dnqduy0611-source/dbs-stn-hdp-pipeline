"""Sample the metal centreline along the electrode shaft, perpendicular to the axis.

Subject-generic version of qc_sub-02_axis_centerline.py.  Same method,
same output schema, so qc_marker_band_axis_refit.py can consume either.

At each depth t along the given axis, sample a disc PERPENDICULAR to the axis
(not an axial slice - an axial slice through a tilted lead is an ellipse and its
centroid is biased by thresholding), take the intensity-weighted centroid of the
voxels above THRESH_HU inside RADIUS_MM, and record its (u,v) offset from the
axis in the perpendicular frame.

The marker band is EXCLUDED from every fit.  The marker is a partial cylinder,
so its centroid is off-axis BY DESIGN - that asymmetry is the roll signal DiODe
lives on.  Including it would drag the centreline toward the marker slot.

Nothing here writes to any pending JSON or approves anything.

    python qc_axis_centerline.py sub-01
"""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates

NATIVE = Path(
    r"D:\ossdbs_env\dbs_stn_hdp_pipeline\lead_dbs_work"
    r"\ElectrodeLocalizationBatch20260807\derivatives\native_localization"
)

# subject -> acquisition -> (CT relative to the subject dir, axes json relative
# to the subject dir).  The axes json may use either the pending-file field
# names (tip_ras_mm / axis_unit_ras) or the bridge names (tip / axis).
SUBJECTS = {
    "sub-01": {
        "bone0p625": (
            "diode/sub-01_ses-postop_CT_original.nii",
            "native_ct_axis_qc/axes_for_pitch_contact_refined.json",
        ),
        "std1p25": (
            "diode/std1p25/sub-01_ses-postop_acq-std1p25_CT_original.nii",
            "native_ct_axis_qc/std1p25/axes_for_pitch_contact_refined.json",
        ),
    },
    # sub-02 keeps both acquisitions in ONE pending file, nested under
    # ["acquisitions"][acq]["axes"]; read_axes_block below detects that shape.
    "sub-02": {
        "bone0p625": (
            "diode/bone0p625/sub-02_ses-postop_acq-bone0p625_CT_original.nii",
            "native_ct_axis_qc/electrode_axes_contact_refined_pending.json",
        ),
        "std1p25": (
            "diode/std1p25/sub-02_ses-postop_acq-std1p25_CT_original.nii",
            "native_ct_axis_qc/electrode_axes_contact_refined_pending.json",
        ),
    },
}

THRESH_HU = 2000.0        # same threshold ea_diode_medtronic uses for its own refit
RADIUS_MM = 2.0           # lead is 1.36 mm across; 2 mm keeps blooming, drops streaks
DEPTH_MM = np.arange(0.0, 26.01, 0.25)   # from the physical tip upward
MARKER_MM = 14.4          # nominal centre of the directional marker, from the tip
MARKER_HALF = 2.5         # exclusion half-width around it
CONTACT_BAND = (0.5, 10.0)    # what the contact fit and DiODe's own refit both saw
EXTEND_BAND = (0.5, 24.0)     # the whole visible shaft, marker cut out


def perp_frame(axis):
    """Return two unit vectors spanning the plane perpendicular to `axis`."""
    axis = axis / np.linalg.norm(axis)
    seed = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(seed, axis)) > 0.9:
        seed = np.array([1.0, 0.0, 0.0])
    u = np.cross(axis, seed)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    return u, v


def read_axis(node):
    """Accept both the pending-file and the bridge-file field names."""
    tip = node.get("tip_ras_mm", node.get("tip"))
    axis = node.get("axis_unit_ras", node.get("axis"))
    if tip is None or axis is None:
        raise KeyError("axis record has neither tip_ras_mm/axis_unit_ras nor tip/axis")
    return np.array(tip, dtype=float), np.array(axis, dtype=float)


def read_axes_block(path, acq):
    """Return {'right': ..., 'left': ...} from either JSON layout.

    sub-01 writes one file per acquisition with a flat ["axes"]; sub-02 writes one
    file for both, nested under ["acquisitions"][acq]["axes"].
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if "acquisitions" in data:
        if acq not in data["acquisitions"]:
            raise KeyError("%s has no acquisition '%s' (has %s)"
                           % (path.name, acq, sorted(data["acquisitions"])))
        return data["acquisitions"][acq]["axes"]
    return data["axes"]


def disc_centroid(img, inv_affine, centre, u, v):
    """Intensity-weighted centroid of the >THRESH voxels on one perpendicular disc.

    Returns (du, dv, mass, n_hot) with du/dv in mm relative to `centre`.
    """
    step = 0.15
    grid = np.arange(-RADIUS_MM, RADIUS_MM + 1e-9, step)
    gu, gv = np.meshgrid(grid, grid, indexing="ij")
    keep = (gu ** 2 + gv ** 2) <= RADIUS_MM ** 2
    gu, gv = gu[keep], gv[keep]

    pts = centre[None, :] + gu[:, None] * u[None, :] + gv[:, None] * v[None, :]
    vox = (inv_affine[:3, :3] @ pts.T).T + inv_affine[:3, 3]
    val = map_coordinates(img, vox.T, order=1, mode="constant", cval=-1000.0)

    hot = val > THRESH_HU
    if hot.sum() < 4:
        return np.nan, np.nan, 0.0, int(hot.sum())
    w = val[hot] - THRESH_HU          # weight by excess over threshold
    mass = float(w.sum())
    du = float((w * gu[hot]).sum() / w.sum())
    dv = float((w * gv[hot]).sum() / w.sum())
    return du, dv, mass, int(hot.sum())


def fit_centerline(depth, du, dv, axis, tip, u, v, band):
    """Least-squares re-fit of the centreline through the measured centroids."""
    sel = (
        np.isfinite(du)
        & (depth >= band[0])
        & (depth <= band[1])
        & (np.abs(depth - MARKER_MM) > MARKER_HALF)
    )
    if sel.sum() < 8:
        return None
    t = depth[sel]
    su = np.polyfit(t, du[sel], 1)     # du = su[0]*t + su[1]
    sv = np.polyfit(t, dv[sel], 1)

    new_axis = axis + su[0] * u + sv[0] * v
    new_axis /= np.linalg.norm(new_axis)
    new_tip = tip + su[1] * u + sv[1] * v

    angle = np.degrees(np.arccos(np.clip(np.dot(new_axis, axis), -1.0, 1.0)))
    p_old = tip + MARKER_MM * axis
    p_new = new_tip + MARKER_MM * new_axis
    shift_at_marker = float(np.linalg.norm(p_new - p_old))

    resid = np.hypot(du[sel] - np.polyval(su, t), dv[sel] - np.polyval(sv, t))
    return {
        "n_points": int(sel.sum()),
        "slope_u_mm_per_mm": float(su[0]),
        "slope_v_mm_per_mm": float(sv[0]),
        "intercept_u_mm": float(su[1]),
        "intercept_v_mm": float(sv[1]),
        "angle_change_deg": float(angle),
        "shift_at_marker_mm": shift_at_marker,
        "rms_residual_mm": float(np.sqrt((resid ** 2).mean())),
        "max_residual_mm": float(resid.max()),
        "new_axis_unit_ras": new_axis.tolist(),
        "new_tip_ras_mm": new_tip.tolist(),
    }


def run(subject, only_acq=None, axes_override=None, out_name="axis_centerline_check"):
    if subject not in SUBJECTS:
        raise SystemExit("no acquisition map for %s; add one to SUBJECTS" % subject)
    root = NATIVE / ("sub-%s" % subject)
    out = root / "native_ct_axis_qc" / "axis_centerline_check"
    out.mkdir(parents=True, exist_ok=True)
    plan = {k: v for k, v in SUBJECTS[subject].items()
            if only_acq is None or k == only_acq}
    if not plan:
        raise SystemExit("subject %s has no acquisition '%s' (has %s)"
                         % (subject, only_acq, sorted(SUBJECTS[subject])))

    report = {"subject": subject, "threshold_hu": THRESH_HU, "radius_mm": RADIUS_MM,
              "marker_excluded_mm": [MARKER_MM - MARKER_HALF, MARKER_MM + MARKER_HALF],
              "acquisitions": {}}
    lines = []

    def say(text=""):
        print(text)
        lines.append(text)

    for acq, (ct_rel, axes_rel) in plan.items():
        ct_path = root / ct_rel
        if axes_override is not None:
            axes_rel = str(axes_override)
            axes = read_axes_block(Path(axes_override), acq)
        else:
            axes = read_axes_block(root / axes_rel, acq)
        nii = nib.load(str(ct_path))
        img = np.asarray(nii.dataobj, dtype=np.float32)
        inv_affine = np.linalg.inv(nii.affine)
        say("=" * 74)
        say("acquisition %s   voxel %.4f %.4f %.4f mm"
            % ((acq,) + tuple(np.sqrt((nii.affine[:3, :3] ** 2).sum(axis=0)))))
        say("  ct   %s" % ct_path.name)
        say("  axes %s" % axes_rel)
        report["acquisitions"][acq] = {}

        for side in ("right", "left"):
            tip, axis = read_axis(axes[side])
            axis /= np.linalg.norm(axis)
            u, v = perp_frame(axis)

            du = np.full(DEPTH_MM.shape, np.nan)
            dv = np.full(DEPTH_MM.shape, np.nan)
            mass = np.zeros(DEPTH_MM.shape)
            nhot = np.zeros(DEPTH_MM.shape, dtype=int)
            for i, t in enumerate(DEPTH_MM):
                centre = tip + t * axis
                du[i], dv[i], mass[i], nhot[i] = disc_centroid(
                    img, inv_affine, centre, u, v)

            radial = np.hypot(du, dv)
            ok = np.isfinite(radial)

            say("")
            say("--- %s / %s ---" % (acq, side))
            say("  metal found on %d of %d discs (%.1f-%.1f mm continuous check)"
                % (ok.sum(), ok.size, DEPTH_MM[ok].min() if ok.any() else np.nan,
                   DEPTH_MM[ok].max() if ok.any() else np.nan))

            say("  offset of the metal centroid from the given axis, per band:")
            say("    %-16s %8s %8s %8s" % ("band(mm)", "median", "mean", "max"))
            for lo, hi, label in ((0.5, 10.0, "contacts 0.5-10"),
                                  (10.0, 12.0, "shaft 10-12"),
                                  (11.9, 16.9, "MARKER 11.9-16.9"),
                                  (17.0, 24.0, "shaft 17-24")):
                m = ok & (DEPTH_MM >= lo) & (DEPTH_MM <= hi)
                if m.sum() == 0:
                    continue
                say("    %-16s %8.3f %8.3f %8.3f"
                    % (label, np.median(radial[m]), radial[m].mean(), radial[m].max()))

            fits = {}
            for name, band in (("contact_band", CONTACT_BAND),
                               ("extended_band", EXTEND_BAND)):
                fit = fit_centerline(DEPTH_MM, du, dv, axis, tip, u, v, band)
                fits[name] = fit
                if fit is None:
                    say("  re-fit %-14s : too few points" % name)
                    continue
                say("  re-fit %-14s : n=%3d  angle %+.3f deg  shift at marker %.3f mm"
                    "  rms %.3f mm"
                    % (name, fit["n_points"], fit["angle_change_deg"],
                       fit["shift_at_marker_mm"], fit["rms_residual_mm"]))

            if fits["contact_band"] and fits["extended_band"]:
                a1 = np.array(fits["contact_band"]["new_axis_unit_ras"])
                a2 = np.array(fits["extended_band"]["new_axis_unit_ras"])
                between = np.degrees(np.arccos(np.clip(np.dot(a1, a2), -1, 1)))
                say("  contact-band vs extended-band centreline: %.3f deg apart"
                    % between)
                fits["bands_disagree_deg"] = float(between)

            report["acquisitions"][acq][side] = {
                "given_axis_unit_ras": axis.tolist(),
                "given_tip_ras_mm": tip.tolist(),
                "depth_mm": DEPTH_MM.tolist(),
                "offset_u_mm": np.where(np.isfinite(du), du, None).tolist(),
                "offset_v_mm": np.where(np.isfinite(dv), dv, None).tolist(),
                "radial_offset_mm": np.where(np.isfinite(radial), radial, None).tolist(),
                "n_hot_voxels": nhot.tolist(),
                "fits": fits,
            }
        say("")

    (out / ("%s.json" % out_name)).write_text(
        json.dumps(report, indent=1), encoding="utf-8")
    (out / ("%s.log" % out_name)).write_text(
        "\n".join(lines), encoding="utf-8")
    print("wrote %s" % (out / ("%s.json" % out_name)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("subject")
    ap.add_argument("--acq", default=None,
                    help="run only this acquisition instead of all of them")
    ap.add_argument("--axes-override", default=None,
                    help="axes JSON to use instead of the one in SUBJECTS; "
                         "same two layouts are accepted")
    ap.add_argument("--out-name", default="axis_centerline_check",
                    help="basename of the .json/.log written into "
                         "native_ct_axis_qc/axis_centerline_check/")
    a = ap.parse_args()
    run(a.subject, only_acq=a.acq, axes_override=a.axes_override,
        out_name=a.out_name)
