#!/usr/bin/env python3
"""Trich mat phang doc giua (MSP) tu chinh anh giai phau, bang doi xung guong 3D.

Vi sao khong dung trung diem cap ROI: R_RN cua ca sub-01 duoc tao bang cach LAY GUONG
L_RN (592 vs 593 voxel). Trung diem cap do chi tra lai mat phang guong da dung luc
tao ROI, khong phai do dac giai phau. Vong luan quan.

Cach lam: tim mat phang  x + ny*y + nz*z = d  sao cho anh phan chieu qua no khop
voi chinh no nhat (tuong quan chuan hoa lon nhat).
  -> do nghieng theo Z (cuon dau, roll)  = -nz   , goc = atan(-nz)
  -> do nghieng theo Y (xoay dau, yaw)   = -ny
  -> mat phang: x(Y,Z) = d - ny*Y - nz*Z

Chay khong tham so gi them se in ca hai so de so voi mo hinh cu x = a*Z + b.
"""
import argparse

import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import minimize

ap = argparse.ArgumentParser()
ap.add_argument("image", help="anh giai phau trong khong gian DWI, vd fa_1mm_ref.nii.gz")
ap.add_argument("--step", type=int, default=2, help="lay mau thua moi <step> voxel")
ap.add_argument("--thr", type=float, default=0.05, help="nguong tach nao")
ap.add_argument("--zrange", type=float, nargs=2, default=None,
                help="chi khop trong dai world-Z nay (mm)")
ap.add_argument("--label", default="")
A = ap.parse_args()

im = nib.load(A.image)
vol = np.asarray(im.dataobj, dtype=np.float32)
if vol.ndim > 3:
    vol = vol[..., 0]
aff = im.affine
inv = np.linalg.inv(aff)

s = A.step
ii, jj, kk = np.meshgrid(*[np.arange(0, n, s) for n in vol.shape[:3]], indexing="ij")
idx = np.stack([ii.ravel(), jj.ravel(), kk.ravel()]).astype(np.float64)
val = vol[ii.ravel(), jj.ravel(), kk.ravel()]

P = (aff[:3, :3] @ idx) + aff[:3, 3:4]          # world coords, shape (3, N)

m = val > A.thr
if A.zrange:
    m &= (P[2] >= A.zrange[0]) & (P[2] <= A.zrange[1])
P, val = P[:, m], val[m]
print("=" * 68)
print("MAT PHANG DOC GIUA bang doi xung guong   %s" % A.label)
print("=" * 68)
print("anh        : %s" % A.image)
print("diem dung  : %d  (nguong %.2f, lay thua %d)" % (val.size, A.thr, s))
print("dai world-Z: %.1f .. %.1f mm" % (P[2].min(), P[2].max()))


def ncc(params):
    ny, nz, d = params
    n = np.array([1.0, ny, nz])
    n /= np.linalg.norm(n)
    dist = n @ P - d / np.linalg.norm([1.0, ny, nz])
    Pr = P - 2.0 * n[:, None] * dist[None, :]           # phan chieu qua mat phang
    vox = (inv[:3, :3] @ Pr) + inv[:3, 3:4]
    ok = np.all((vox >= 0) & (vox <= (np.array(vol.shape[:3])[:, None] - 1)), axis=0)
    if ok.sum() < 1000:
        return 1.0
    r = map_coordinates(vol, vox[:, ok], order=1, mode="constant", cval=0.0)
    a, b = val[ok], r
    a = a - a.mean(); b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return 1.0 if den == 0 else -float((a * b).sum() / den)


x0 = np.array([0.0, 0.0, float(np.median(P[0]))])
print("khoi tao   : ny=0 nz=0 d=%.2f   (NCC ban dau %.4f)" % (x0[2], -ncc(x0)))

res = minimize(ncc, x0, method="Nelder-Mead",
               options=dict(xatol=1e-4, fatol=1e-7, maxiter=4000, maxfev=4000))
ny, nz, d = res.x
nrm = np.linalg.norm([1.0, ny, nz])

print("hoi tu     : %s  sau %d lan goi,  NCC = %.4f" % (res.success, res.nfev, -res.fun))
print()
print("MAT PHANG  :  x(Y,Z) = %.3f  %+.5f*Y  %+.5f*Z" % (d, -ny, -nz))
print("   nghieng theo Z (cuon dau) = %.5f  ->  %+.2f deg" % (-nz, np.degrees(np.arctan(-nz))))
print("   nghieng theo Y (xoay dau) = %.5f  ->  %+.2f deg" % (-ny, np.degrees(np.arctan(-ny))))
print("   phap tuyen chuan hoa      = (%.5f, %.5f, %.5f)" % (1 / nrm, ny / nrm, nz / nrm))
print()

# so voi mo hinh cu chi phu thuoc Z: lay Y trung binh cua vung dung de khop
ybar = float(P[1].mean())
a_z = -nz
b_z = d - ny * ybar
print("QUY VE mo hinh x = a*Z + b  (lay Y = %.1f mm, trung binh vung khop):" % ybar)
print("   x = %.4f*Z + %.3f" % (a_z, b_z))
print()

zlo, zhi = float(P[2].min()), float(P[2].max())
xc = b_z + a_z * 0.5 * (zlo + zhi)
print("LECH so voi hang-x (lay hang-x = gia tri o giua dai Z = %.2f mm):" % xc)
for z in (zlo, 0.5 * (zlo + zhi), zhi):
    print("   Z = %+7.1f mm  ->  x_giua = %+7.2f mm   (lech %+.2f mm)"
          % (z, b_z + a_z * z, b_z + a_z * z - xc))
amp = abs(a_z) * (zhi - zlo)
print("   bien do lech tren toan dai Z = %.2f mm" % amp)
print("   -> %s" % ("PHAI dung mat phang nghieng." if amp >= 1.0
                    else "duoi 1 mm, hang-x dung duoc."))
print("=" * 68)
