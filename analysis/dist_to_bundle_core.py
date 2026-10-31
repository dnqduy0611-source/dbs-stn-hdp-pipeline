"""Do khoang cach tu tiep diem den TAM BO (thay vi den soi gan nhat).

Ly do: khoang cach den soi gan nhat bao hoa o 0.00mm khi dien cuc nam TRONG bo,
nen mat het suc phan biet. Khoang cach den tam thi khong bao hoa.

Hai dinh nghia "tam" duoc tinh:
  A. loi cuc bo  - moi soi lay diem gan tiep diem nhat, roi lay trong tam cua
                   cac diem do (chi tinh soi co khoang cach <= R_LOCAL).
                   Khong phu thuoc huong bo.
  B. lat cat ngang - trong tam cac diem nam trong lat +-SLAB mm quanh z tiep diem.
                   So sanh duoc voi khung "muc axial" cua pipeline.

Kem theo do TAN cua bo (RMS quanh trong tam) va ti so d/RMS = "nam sau bao nhieu
lan ban kinh bo". Do la dai luong khong thu nguyen, so duoc giua cac ca.

CONG KIEM (chay dau tien, bat buoc dat): doc .tck bang tay roi tai lap
center_to_tract_edge_mm cua pipeline. Lech > 0.01mm => sai he toa do, dung lai.
"""
import csv
import sys
from pathlib import Path

import numpy as np
from nibabel.streamlines import load as tck_load

ROOT = Path(r"D:\ossdbs_env\dbs_stn_hdp_pipeline")
CASES = [
    ("sub-06", "sub-06"),
    ("sub-02", "sub-02"),
    ("sub-09", "sub-09"),
]
METRICS = "derivatives/electrode_tract_metrics_20260824/contact_level_to_tract_distances.csv"

R_LOCAL = 10.0   # mm - soi phai lai gan tiep diem hon muc nay moi duoc tinh vao loi
SLAB = 2.0       # mm - nua be day lat cat ngang


def wsl_to_win(p):
    """/mnt/d/... -> D:/..."""
    if p.startswith("/mnt/"):
        return Path(p[5].upper() + ":/" + p[7:])
    return Path(p)


_cache = {}


def points_of(path):
    """Tra ve mo hinh bo: diem gop, va bo doan thang (A, AB, ab2, chi so bat dau
    tung soi) de tinh khoang cach den DOAN THANG chu khong phai den dinh.

    Pipeline dung khoang cach den doan thang; do den dinh lech toi 0.26mm
    (dung bang cot pointcloud_minus_exact_mm). Phai khop thi cong kiem moi that.
    """
    key = str(path)
    if key not in _cache:
        sl = [s.astype(np.float64) for s in tck_load(str(path)).tractogram.streamlines]
        A = np.vstack([s[:-1] for s in sl])
        B = np.vstack([s[1:] for s in sl])
        AB = B - A
        ab2 = np.einsum("ij,ij->i", AB, AB)
        ab2[ab2 == 0] = 1e-12
        nseg = np.array([len(s) - 1 for s in sl])
        starts = np.concatenate([[0], np.cumsum(nseg)[:-1]])
        _cache[key] = (np.vstack(sl), A, AB, ab2, starts, len(sl))
    return _cache[key]


def seg_dist(contact, A, AB, ab2):
    """Khoang cach va diem chieu tu contact den moi doan thang."""
    t = np.clip(np.einsum("ij,ij->i", contact - A, AB) / ab2, 0.0, 1.0)
    proj = A + t[:, None] * AB
    return np.linalg.norm(proj - contact, axis=1), proj


def core_metrics(contact, A, AB, ab2, starts, nstream):
    """Dinh nghia A: trong tam cac diem gan nhat cua TUNG soi."""
    d, proj = seg_dist(contact, A, AB, ab2)
    # gom theo soi: doan thang xep lien tiep theo soi nen dung reduceat
    dmin = np.minimum.reduceat(d, starts)
    idx = np.array([starts[i] + int(np.argmin(d[starts[i]:(starts[i + 1] if i + 1 < len(starts) else len(d))]))
                    for i in range(len(starts))])
    pts = proj[idx]
    keep = dmin <= R_LOCAL
    if keep.sum() < 3:
        return None
    sel = pts[keep]
    cen = sel.mean(axis=0)
    rms = float(np.sqrt(((sel - cen) ** 2).sum(axis=1).mean()))
    return {
        "d_core": float(np.linalg.norm(contact - cen)),
        "rms": rms,
        "n": int(keep.sum()),
        "n_tot": nstream,
    }


def slab_metrics(contact, allpts):
    """Dinh nghia B: trong tam lat cat ngang quanh z cua tiep diem."""
    m = np.abs(allpts[:, 2] - contact[2]) <= SLAB
    if m.sum() < 10:
        return None
    sel = allpts[m]
    cen = sel.mean(axis=0)
    # do trong mat phang xy vi day la lat cat ngang
    d_xy = float(np.linalg.norm(contact[:2] - cen[:2]))
    rms_xy = float(np.sqrt(((sel[:, :2] - cen[:2]) ** 2).sum(axis=1).mean()))
    return {"d_slab": d_xy, "rms_slab": rms_xy, "n": int(m.sum())}


def main():
    gate_worst = 0.0
    gate_rows = 0
    out = []

    for pdir, lab in CASES:
        f = ROOT / "patients" / pdir / METRICS
        for r in csv.DictReader(open(f)):
            if r["tract"] not in ("HDP", "DRTT"):
                continue
            if r["variant"].startswith("std"):      # giu bien the CT chinh
                continue
            contact = np.array([float(r["contact_dwi_ras_" + a]) for a in "xyz"])
            tf = wsl_to_win(r["tract_file"])
            if not tf.exists():
                print("THIEU FILE:", tf)
                continue
            allpts, A, AB, ab2, starts, nstream = points_of(tf)

            # --- CONG KIEM: tai lap khoang cach den soi gan nhat
            mine = float(seg_dist(contact, A, AB, ab2)[0].min())
            theirs = float(r["center_to_tract_edge_mm"])
            gate_worst = max(gate_worst, abs(mine - theirs))
            gate_rows += 1

            a = core_metrics(contact, A, AB, ab2, starts, nstream)
            b = slab_metrics(contact, allpts)
            out.append((lab, r["side"], r["level"], r["tract"], theirs,
                        float(r["electrode_surface_to_tract_edge_mm"]), a, b))

    print(f"CONG KIEM: {gate_rows} dong, lech lon nhat so voi pipeline = {gate_worst:.4f} mm")
    if gate_worst > 0.01:
        print("  => KHONG DAT. He toa do khong khop, moi so duoi day vo nghia.")
        sys.exit(1)
    print("  => DAT.\n")

    hdr = (f"{'ca':6}{'ben':4}{'muc':5}{'bo':6}"
           f"{'soi gan':>8}{'be mat':>8}"
           f"{'d_loi':>8}{'RMS':>7}{'d/RMS':>7}{'n_soi':>8}"
           f"{'d_lat':>8}{'RMS_lat':>9}{'d/RMS':>7}")
    print(hdr)
    print("-" * len(hdr))
    for lab, side, lvl, tract, near, surf, a, b in out:
        sa = (f"{a['d_core']:8.2f}{a['rms']:7.2f}{a['d_core']/a['rms']:7.2f}"
              f"{a['n']:5d}/{a['n_tot']:<4d}" if a else f"{'--':>8}{'--':>7}{'--':>7}{'--':>8}")
        sb = (f"{b['d_slab']:8.2f}{b['rms_slab']:9.2f}{b['d_slab']/b['rms_slab']:7.2f}"
              if b else f"{'--':>8}{'--':>9}{'--':>7}")
        print(f"{lab:6}{side:4}{lvl:5}{tract:6}{near:8.2f}{surf:8.2f}{sa}{sb}")


if __name__ == "__main__":
    main()
