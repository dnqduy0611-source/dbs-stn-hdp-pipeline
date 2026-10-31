#!/usr/bin/env python3
"""Loc soi theo so lan bat cheo duong giua, dung MAT PHANG DAY DU do bang msp_fit.py.

Khac hai ban truoc:
  * sub-01/prune_drtt_midline.py  - hardcode thu muc ROI cua sub-01.
  * sub-06/prune_midline.py  - bat buoc --roi-dir, nhung mat phang chi co
    dang x = a*Z + b, KHONG co so hang Y. Voi ca sub-01 dau xoay ngang 4.81 deg nen
    thanh phan Y lon hon thanh phan Z -> mo hinh chi-Z bo sot.
  * Ban nay khong doc ROI de suy duong giua nua. He so mat phang truyen thang vao
    tu msp_fit.py (do bang doi xung guong tren chinh anh giai phau). Ly do: R_RN cua
    sub-01 la ANH GUONG cua L_RN (592 vs 593 voxel), nen trung diem cap RN chi tra lai
    mat phang guong da dung luc tao ROI - vong luan quan, khong phai do dac.

Mat phang:  x_giua(Y, Z) = XC + AY*Y + AZ*Z

DRTT: cat dung MOT lan, tai decussation SCP (Z thap)  -> --ncross 1 --zmax -5
HDP : bo cung ben, KHONG duoc cat                     -> --ncross 0
"""
import argparse
import sys

import nibabel as nib
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("in_tck")
ap.add_argument("out_tck", nargs="?")
ap.add_argument("in_w", nargs="?")
ap.add_argument("out_w", nargs="?")
ap.add_argument("--plane", type=float, nargs=3, required=True, metavar=("XC", "AY", "AZ"),
                help="x_giua(Y,Z) = XC + AY*Y + AZ*Z  (lay tu msp_fit.py)")
ap.add_argument("--sgn-rois", nargs=2, metavar=("L_ROI", "R_ROI"), default=None,
                help="mot cap ROI trai/phai BAT KY de biet chieu +x la ben nao. "
                     "Dung ROI KHONG phai anh guong (vd PG). Neu bo qua thi phai --sgn.")
ap.add_argument("--sgn", type=float, default=None,
                help="+1 neu +x la ben PHAI, -1 neu nguoc lai")
ap.add_argument("--ncross", type=int, default=1)
ap.add_argument("--zmax", type=float, default=-5.0)
ap.add_argument("--hyst", type=float, default=1.0,
                help="nguong tre (mm). Dem doi dau tran khien soi chay tiep tuyen sat "
                     "mat phang bi dem thanh nhieu lan cat. Dat 0 de dem tran.")
ap.add_argument("--report", action="store_true")
a = ap.parse_args()

XC, AY, AZ = a.plane


def cx(f):
    im = nib.load(f)
    idx = np.argwhere(np.asarray(im.dataobj) > 0.5)
    if idx.size == 0:
        sys.exit("FATAL: ROI rong: %s" % f)
    m = im.affine
    return float((m[0, 0] * idx[:, 0] + m[0, 1] * idx[:, 1]
                  + m[0, 2] * idx[:, 2] + m[0, 3]).mean())


if a.sgn is not None:
    SGN = float(np.sign(a.sgn))
    src = "tham so --sgn"
elif a.sgn_rois:
    xl, xr = cx(a.sgn_rois[0]), cx(a.sgn_rois[1])
    SGN = 1.0 if xr > xl else -1.0
    src = "cap ROI (L x=%.1f, R x=%.1f)" % (xl, xr)
else:
    sys.exit("FATAL: phai cho --sgn hoac --sgn-rois")

print("mat phang : x_giua(Y,Z) = %.3f %+.5f*Y %+.5f*Z" % (XC, AY, AZ))
print("            nghieng theo Z %+.2f deg, theo Y %+.2f deg"
      % (np.degrees(np.arctan(AZ)), np.degrees(np.arctan(AY))))
print("chieu     : SGN=%+.0f  (%s)" % (SGN, src))

tf = nib.streamlines.load(a.in_tck)
sl = list(tf.streamlines)
if not sl:
    print("in=0 out=0 (rong)")
    sys.exit(0)


def crossings(s):
    """(so lan cat, list Z tai diem cat) voi nguong tre a.hyst."""
    d = (s[:, 0] - (XC + AY * s[:, 1] + AZ * s[:, 2])) * SGN
    if a.hyst <= 0:
        sign = np.sign(d)
        sign[sign == 0] = 1.0
        ci = np.where(sign[:-1] != sign[1:])[0]
        return len(ci), list(s[ci, 2])
    st = np.zeros(len(d), dtype=int)
    st[d > a.hyst] = 1
    st[d < -a.hyst] = -1
    nz = np.nonzero(st)[0]
    if nz.size < 2:
        return 0, []
    seq, idx = st[nz], nz
    zs = []
    for f in np.where(seq[:-1] != seq[1:])[0]:
        i0, i1 = idx[f], idx[f + 1]
        zs.append(float(s[i0 + int(np.argmin(np.abs(d[i0:i1 + 1]))), 2]))
    return len(zs), zs


keep, ncross, zcross = [], [], []
for s in sl:
    n, zs = crossings(s)
    zc = np.array(zs) if zs else np.array([np.nan])
    ncross.append(n)
    zcross.append(float(np.nanmax(zc)))
    if a.ncross == 0:
        keep.append(n == 0)
    else:
        keep.append(n == a.ncross and float(np.nanmax(zc)) <= a.zmax)

keep = np.array(keep)
ncross = np.array(ncross)
zcross = np.array(zcross)
n_in, n_out = len(sl), int(keep.sum())

cond = "%d lan cheo" % a.ncross + ("" if a.ncross == 0 else ", Z<=%.0f" % a.zmax)
print("in=%d out=%d (giu %.1f%%)  nguong: %s" % (n_in, n_out, 100.0 * n_out / n_in, cond))
for k in sorted(set(ncross.tolist())):
    print("  cat %d lan : %5d soi" % (k, int((ncross == k).sum())))
one = ncross == 1
if one.any():
    z = zcross[one]
    print("  Z cho bat cheo (soi cat 1 lan, n=%d): min %.1f  p5 %.1f  trung vi %.1f  "
          "p95 %.1f  max %.1f"
          % (int(one.sum()), z.min(), np.percentile(z, 5), np.median(z),
             np.percentile(z, 95), z.max()))
    hi = np.where(one & (zcross > a.zmax))[0]
    if len(hi) and a.ncross == 1:
        print("  cat 1 lan nhung Z qua cao: %d soi" % len(hi))

if a.report or not a.out_tck:
    sys.exit(0)

nib.streamlines.save(tf.tractogram[keep], a.out_tck, header=tf.header)
if a.in_w and a.out_w:
    w = np.loadtxt(a.in_w, comments="#", ndmin=1).ravel()
    if w.size != n_in:
        sys.exit("FATAL: so trong so (%d) != so soi (%d)" % (w.size, n_in))
    np.savetxt(a.out_w, w[keep][None, :], fmt="%.6f")
    print("  SIFT2 sum: %.1f -> %.1f" % (w.sum(), w[keep].sum()))
