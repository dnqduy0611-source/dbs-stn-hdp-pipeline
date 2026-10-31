"""So DINH NGHIA khoang cang tiep diem -> bo: d_core (dang dung) vs Prent 2019 (trung vi).

Cau hoi (28/08/2026): `d_hdp` trong cac phan tich cua du an co duoc do giong
Prent et al. 2019 J Neurosurg 133:433-42 khong?

Prent, muc "Distance Estimation":
    "The minimum perpendicular distance from the center of each electrode contact
     to each fiber was calculated per tract. The median was then used to robustly
     estimate the distance to the center of the tract for each electrode contact."
  => d_prent = median_over_fibers( min_perp_dist(contact, fiber) )   [TRUNG VI cua SO VO HUONG]

Cong thuc dang dung (`remeasure_all_contact_tract.measure_core`):
    1. min_perp_dist cho tung soi          <- GIONG Prent
    2. giu soi co dmin <= 10 mm            <- Prent KHONG co buoc nay
    3. lay DIEM gan nhat cua moi soi giu lai, TRUNG BINH VECTO thanh mot tam `cen`
    4. d_core = |contact - cen|            <- KHOANG CACH TOI TAM, khong phai trung vi
  => |mean(vector)| != mean(|vector|). Cac diem gan nhat vay quanh tiep diem theo
     nhieu huong nen phep trung binh vecto TRIET TIEU lan nhau; d_core luon <= d_prent,
     va bang 0 khi tiep diem nam giua bo.

Script nay chay lai TOAN COHORT bang dung chuoi toa do / cong kiem cua
`remeasure_all_contact_tract.py` (monkeypatch `measure_core`), ghi ca bon thuoc do
canh nhau roi in bang doi chieu + tuong quan.

Chay:  wsl -d Ubuntu-22.04 -- python3 /mnt/d/ossdbs_env/dbs_stn_hdp_pipeline/clinical/compare_prent_distance.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import remeasure_all_contact_tract as R  # noqa: E402

EXTRA: list[dict] = []
_orig_measure_core = R.measure_core


def measure_both(bundle, contact):
    """Goi ham goc de giu y nguyen d_core, roi tinh them thuoc do Prent."""
    res = _orig_measure_core(bundle, contact)
    if res is None:
        return None

    A, AB, ab2, starts = bundle["A"], bundle["AB"], bundle["ab2"], bundle["starts"]
    t = np.clip(np.einsum("ij,ij->i", contact - A, AB) / ab2, 0.0, 1.0)
    proj = A + t[:, None] * AB
    d = np.linalg.norm(proj - contact, axis=1)
    n = len(starts)
    idx_min = np.array([starts[i] + np.argmin(d[starts[i]: (starts[i + 1] if i + 1 < n else len(d))])
                        for i in range(n)])
    dmin = d[idx_min]                      # mot gia tri cho MOI soi cua bo

    keep = dmin <= R.R_LOCAL_MM
    extra = {
        "d_prent_mm": float(np.median(dmin)),            # dung Prent: MOI soi
        "d_prent_local_mm": float(np.median(dmin[keep])) if keep.any() else float("nan"),
        "d_prent_p25_mm": float(np.percentile(dmin, 25)),
        "d_prent_p75_mm": float(np.percentile(dmin, 75)),
        "n_fibers_total": int(n),
    }
    w = bundle["w"]
    if w is not None and w.sum() > 0:       # trung vi CO TRONG SO SIFT2
        order = np.argsort(dmin)
        cw = np.cumsum(w[order]) / w.sum()
        extra["d_prent_sift2_mm"] = float(dmin[order][np.searchsorted(cw, 0.5)])
    EXTRA.append(extra)
    return res


def main():
    R.measure_core = measure_both
    rows, manifest = [], {}
    for name, case in R.CASES.items():
        try:
            R.process(name, case, rows, manifest)
        except Exception as exc:
            print(f"  !! {name} THAT BAI: {type(exc).__name__}: {exc}")

    if len(rows) != len(EXTRA):
        raise AssertionError(f"lech so dong: rows={len(rows)} extra={len(EXTRA)} "
                             "-> thu tu goi measure_core khong khop, KHONG dung ket qua")

    for row, extra in zip(rows, EXTRA):
        for k, v in extra.items():
            row[k] = (round(v, 4) if isinstance(v, float) else v)

    out = R.OUT / "compare_prent_vs_dcore.csv"
    fields = list(rows[0].keys())
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nDa ghi {len(rows)} dong -> {out}")
    report(rows)


def spearman(a, b):
    def rank(x):
        order = np.argsort(np.argsort(x))
        return order.astype(float)
    ra, rb = rank(np.asarray(a)), rank(np.asarray(b))
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))


def report(rows):
    ok = [r for r in rows if r["variant"] == "primary" and r["gate3_contact"] == "PASS"]
    for tract in ("HDP", "DRTT"):
        sub = [r for r in ok if r["tract"] == tract]
        if not sub:
            continue
        core = np.array([r["d_core_mm"] for r in sub], float)
        prent = np.array([r["d_prent_mm"] for r in sub], float)
        local = np.array([r["d_prent_local_mm"] for r in sub], float)
        dmin = np.array([r["d_min_mm"] for r in sub], float)
        print(f"\n===== {tract}  (n = {len(sub)} tiep diem)")
        for label, v in (("d_core   (dang dung)", core), ("d_prent  (moi soi)", prent),
                         ("d_prent  (<=10mm)", local), ("d_min    (soi gan nhat)", dmin)):
            print(f"  {label:24s} trung vi {np.nanmedian(v):6.2f}   "
                  f"khoang {np.nanmin(v):5.2f} - {np.nanmax(v):6.2f}")
        ratio = prent / np.where(core == 0, np.nan, core)
        print(f"  ty so d_prent / d_core : trung vi {np.nanmedian(ratio):.2f}  "
              f"({np.nanmin(ratio):.2f} - {np.nanmax(ratio):.2f})")
        print(f"  rho(d_core, d_prent)          = {spearman(core, prent):+.3f}")
        m = ~np.isnan(local)
        print(f"  rho(d_core, d_prent <=10mm)   = {spearman(core[m], local[m]):+.3f}")
        n_clamp = int((core < 0.5).sum())
        print(f"  so tiep diem co d_core < 0,5 mm (bi 'kep' vao giua bo): {n_clamp}/{len(sub)}")


if __name__ == "__main__":
    main()
