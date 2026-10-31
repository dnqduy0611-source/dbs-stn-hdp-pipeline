"""(1) SAI SO CUA CHINH PHEP DO d_core / d_min  -- chay TUNG TIEP DIEM.

Cau hoi: bien thien 0,49mm (trong ban cau) va 3,65mm (giua ban cau) cua d_core la
GIAI PHAU hay la NHIEU CUA PHEP DO? Neu sai so do > 0,6mm (thang vat ly cua toan
bo dai nguong 3->5 mA) thi bai toan "khoang cach -> nguong" dong lai ve nguyen tac.

KHONG co ban dung CST thu hai o bat ky ca nao (da kiem: 20 file, 10 ca, 1 ban/ca)
=> khong the do sai so GIUA CAC LAN CHAY THAT. Bon nguon nhieu mo phong duoc tren
chinh bo da co:

  A. BOOTSTRAP SOI     - lay lai mau soi co hoan lai. Do nhieu "bo soi nao tinh co
                         duoc dung lai" khi FOD/seed giu nguyen. Day la thanh phan
                         LON NHAT ma tckgen ngau nhien gay ra ([[tckgen-randomseed-
                         khong-tai-lap]]: so soi bo cuoi la Poisson).
  B. LAY MAU CON 50%   - mo phong mot lan chay nang suat thap hon.
  C. QUET R_LOCAL      - R_LOCAL=10mm la THAM SO TU CHON cua chinh dinh nghia
                         d_core. Neu doi R lam d_core doi >1mm thi d_core phan anh
                         lua chon tham so nhieu ngang giai phau.
  D. RUNG TIEP DIEM    - dich tam tiep diem theo N(0, sigma^2 I), sigma = 0,5 va
                         1,0mm = sai so dinh vi CT + dang ky CT->T1->DWI. Day la
                         nguon nhieu DOC LAP voi tractography.

DIEM MAU CHOT PHAI TACH: sai so TUYET DOI cua mot tiep diem vs sai so cua HIEU
d_core(K1)-d_core(K0) trong CUNG ban cau. Hai tiep diem cung ban cau dung CHUNG
mot bo => sai so cua chung TUONG QUAN, nen HIEU chinh xac hon nhieu so voi tri
tuyet doi. Neu dung vay thi:
    - so sanh TRONG ban cau (do sau)      -> it bi nhieu do
    - so sanh GIUA ban cau (khoang cach)  -> chinh la cai bi nhieu do giet
va ket qua am tinh o §3.2 cua bao cao 28/08 co loi giai thich CO CHE.

Bootstrap dung cong thuc tat (cache diem gan nhat tung soi) de chay nhanh; moi ca
deu ASSERT lai voi `measure_core` goc, sai lech phai < 1e-9 mm.

Chay: wsl -d Ubuntu-22.04 -- python3 /mnt/d/ossdbs_env/dbs_stn_hdp_pipeline/clinical/cst_dcore_measurement_error.py
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remeasure_all_contact_tract import (           # noqa: E402
    CASES, ROOT, R_LOCAL_MM, contacts_in_dwi, load_bundle, measure_core,
)
from measure_cst_contact_distance import CST        # noqa: E402

OUT = ROOT / "reports/electrode_cst_distance_20260828"
NBOOT = 2000
NJIT = 40
R_SWEEP = (5.0, 7.5, 10.0, 12.5, 15.0)
SIG_JIT = (0.5, 1.0)
SEED = 20260828


def per_streamline_nearest(bundle, contact):
    """Diem gan nhat tren TUNG soi + khoang cach. Cache de bootstrap cho nhanh."""
    A, AB, ab2, starts = bundle["A"], bundle["AB"], bundle["ab2"], bundle["starts"]
    t = np.clip(np.einsum("ij,ij->i", contact - A, AB) / ab2, 0.0, 1.0)
    proj = A + t[:, None] * AB
    d = np.linalg.norm(proj - contact, axis=1)
    ends = np.append(starts[1:], len(d)) if len(starts) > 1 else np.array([len(d)])
    idx = np.array([starts[i] + int(np.argmin(d[starts[i]:ends[i]]))
                    for i in range(len(starts))])
    return proj[idx], d[idx]


def dcore_from(pts, dmin, contact, r=R_LOCAL_MM):
    keep = dmin <= r
    if not keep.any():
        return np.nan
    return float(np.linalg.norm(contact - pts[keep].mean(axis=0)))


def main():
    rng = np.random.default_rng(SEED)
    rows, per_hemi = [], []

    for name, case in CASES.items():
        cst_dir = CST.get(name)
        if cst_dir is None:
            continue
        dwi = contacts_in_dwi(case)
        print(f"\n===== {name}")
        for side in ("L", "R"):
            path = cst_dir / f"{side}_CST_final.tck"
            if not path.exists():
                continue
            b = load_bundle(path)
            n_str = b["n"]
            cache, base = {}, {}
            for lv in range(4):
                c = dwi[side][lv]
                if np.isnan(c).any():
                    continue
                pts, dmin = per_streamline_nearest(b, c)
                cache[lv] = (pts, dmin, c)
                # --- ASSERT lai voi cong thuc goc ---
                m = measure_core(b, c)
                dc = dcore_from(pts, dmin, c)
                assert abs(dc - m["d_core_mm"]) < 1e-9, (name, side, lv, dc, m)
                assert abs(float(dmin.min()) - m["d_min_mm"]) < 1e-9
                base[lv] = dict(d_core=dc, d_min=float(dmin.min()),
                                n_local=int((dmin <= R_LOCAL_MM).sum()))

            if not cache:
                continue
            lvs = sorted(cache)

            # ---------- A. bootstrap soi (CHUNG chi so cho ca ban cau) ----------
            bt_core = {lv: np.empty(NBOOT) for lv in lvs}
            bt_min = {lv: np.empty(NBOOT) for lv in lvs}
            for bi in range(NBOOT):
                idx = rng.integers(0, n_str, n_str)
                for lv in lvs:
                    pts, dmin, c = cache[lv]
                    bt_core[lv][bi] = dcore_from(pts[idx], dmin[idx], c)
                    bt_min[lv][bi] = float(dmin[idx].min())

            # ---------- B. lay mau con 50% ----------
            half = max(2, n_str // 2)
            sub_core = {lv: np.empty(NBOOT // 4) for lv in lvs}
            for bi in range(NBOOT // 4):
                idx = rng.choice(n_str, half, replace=False)
                for lv in lvs:
                    pts, dmin, c = cache[lv]
                    sub_core[lv][bi] = dcore_from(pts[idx], dmin[idx], c)

            for lv in lvs:
                pts, dmin, c = cache[lv]
                rec = dict(case=name, side=side, level=f"K{lv}",
                           n_streamlines=n_str, **{f"base_{k}": v for k, v in base[lv].items()})
                rec["boot_sd_dcore"] = float(np.nanstd(bt_core[lv], ddof=1))
                rec["boot_p2p5_dcore"] = float(np.nanpercentile(bt_core[lv], 2.5))
                rec["boot_p97p5_dcore"] = float(np.nanpercentile(bt_core[lv], 97.5))
                rec["boot_sd_dmin"] = float(np.nanstd(bt_min[lv], ddof=1))
                rec["boot_bias_dmin"] = float(np.nanmean(bt_min[lv]) - base[lv]["d_min"])
                rec["sub50_sd_dcore"] = float(np.nanstd(sub_core[lv], ddof=1))
                rec["sub50_bias_dcore"] = float(np.nanmean(sub_core[lv]) - base[lv]["d_core"])
                # ---------- C. quet R_LOCAL ----------
                for r in R_SWEEP:
                    rec[f"dcore_R{r:g}"] = dcore_from(pts, dmin, c, r)
                vals = [rec[f"dcore_R{r:g}"] for r in (7.5, 10.0, 12.5)]
                rec["R_range_7p5_12p5"] = float(np.nanmax(vals) - np.nanmin(vals))
                # ---------- D. rung tiep diem ----------
                for sig in SIG_JIT:
                    jj = np.empty(NJIT)
                    for k in range(NJIT):
                        cj = c + rng.normal(0.0, sig, 3)
                        pj, dj = per_streamline_nearest(b, cj)
                        jj[k] = dcore_from(pj, dj, cj)
                    rec[f"jit{sig:g}_sd_dcore"] = float(np.nanstd(jj, ddof=1))
                    rec[f"jit{sig:g}_bias_dcore"] = float(np.nanmean(jj) - base[lv]["d_core"])
                rows.append(rec)

            # ---------- sai so cua HIEU trong cung ban cau ----------
            for i in range(len(lvs) - 1):
                a, z = lvs[i], lvs[i + 1]
                diff = bt_core[z] - bt_core[a]
                per_hemi.append(dict(
                    case=name, side=side, pair=f"K{a}->K{z}",
                    base_diff=base[z]["d_core"] - base[a]["d_core"],
                    sd_diff=float(np.nanstd(diff, ddof=1)),
                    sd_abs_a=float(np.nanstd(bt_core[a], ddof=1)),
                    sd_abs_z=float(np.nanstd(bt_core[z], ddof=1)),
                    corr=float(np.corrcoef(bt_core[a], bt_core[z])[0, 1]),
                    frac_same_sign=float(np.mean(np.sign(diff) == np.sign(
                        base[z]["d_core"] - base[a]["d_core"])))))
            print(f"  {side}: {n_str} soi | boot SD d_core = "
                  + " ".join(f"K{lv}:{np.nanstd(bt_core[lv], ddof=1):.3f}" for lv in lvs))

    # ================= tong hop =================
    def col(k, src=rows):
        return np.array([r[k] for r in src if r.get(k) is not None
                         and not (isinstance(r[k], float) and np.isnan(r[k]))], float)

    k0 = [r for r in rows if r["level"] == "K0"]
    var_between_k0 = float(np.var(col("base_d_core", k0), ddof=1))
    var_noise_boot = float(np.mean(col("boot_sd_dcore", k0) ** 2))
    var_noise_jit05 = float(np.mean(col("jit0.5_sd_dcore", k0) ** 2))
    var_noise_jit10 = float(np.mean(col("jit1_sd_dcore", k0) ** 2))
    rel_boot = 1 - var_noise_boot / var_between_k0
    rel_both = 1 - (var_noise_boot + var_noise_jit05) / var_between_k0
    rel_jit10 = 1 - (var_noise_boot + var_noise_jit10) / var_between_k0

    summ = {
        "n_contacts": len(rows), "n_hemi_pairs": len(per_hemi),
        "n_boot": NBOOT, "n_jitter": NJIT, "seed": SEED,
        "d_core": {
            "boot_sd_mm": [float(np.median(col("boot_sd_dcore"))),
                           float(col("boot_sd_dcore").min()),
                           float(col("boot_sd_dcore").max())],
            "sub50_sd_mm": float(np.median(col("sub50_sd_dcore"))),
            "R_sweep_range_7p5_12p5_mm": [float(np.median(col("R_range_7p5_12p5"))),
                                          float(col("R_range_7p5_12p5").max())],
            # LUU Y: khoa la f"jit{sig:g}_sd_dcore" nen sig=1.0 -> "jit1_", KHONG
            # phai "jit1.0_". Da tung viet nham va nhan NaN im lang o day.
            "jit0p5_sd_mm": float(np.median(col("jit0.5_sd_dcore"))),
            "jit1p0_sd_mm": float(np.median(col("jit1_sd_dcore"))),
        },
        "d_min": {
            "boot_sd_mm": float(np.median(col("boot_sd_dmin"))),
            "boot_bias_mm": float(np.median(col("boot_bias_dmin"))),
        },
        "hieu_trong_ban_cau": {
            "sd_diff_mm": float(np.median([h["sd_diff"] for h in per_hemi])),
            "sd_abs_mm": float(np.median([h["sd_abs_z"] for h in per_hemi])),
            "corr_giua_2_tang": float(np.median([h["corr"] for h in per_hemi])),
            "frac_same_sign": float(np.median([h["frac_same_sign"] for h in per_hemi])),
            "n_pair_dau_on_dinh_95": int(sum(h["frac_same_sign"] >= 0.95 for h in per_hemi)),
        },
        "do_tin_cay_K0": {
            "var_between_hemi_k0": var_between_k0,
            "var_noise_bootstrap": var_noise_boot,
            "var_noise_jitter0p5": var_noise_jit05,
            "reliability_chi_bootstrap": rel_boot,
            "var_noise_jitter1p0": var_noise_jit10,
            "reliability_boot_va_jitter0p5": rel_both,
            "reliability_boot_va_jitter1p0": rel_jit10,
            "he_so_suy_giam_rho_sqrt_jit0p5": float(np.sqrt(max(rel_both, 0.0))),
            "he_so_suy_giam_rho_sqrt_jit1p0": float(np.sqrt(max(rel_jit10, 0.0))),
        },
    }

    (OUT / "cst_dcore_measurement_error.json").write_text(
        json.dumps({"summary": summ, "per_contact": rows, "per_pair": per_hemi},
                   indent=1, ensure_ascii=False), encoding="utf-8")
    with open(OUT / "cst_dcore_measurement_error.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print("\n" + "=" * 78)
    print(json.dumps(summ, indent=1, ensure_ascii=False))
    print("=" * 78)
    print("\n--- SD bootstrap d_core theo TUNG TIEP DIEM (mm) ---")
    print("%-20s %-4s %-4s %8s %8s %8s %8s %8s" % (
        "ca", "ben", "tang", "d_core", "SD_boot", "SD_jit.5", "R7.5-12.5", "n_soi"))
    for r in rows:
        print("%-20s %-4s %-4s %8.3f %8.3f %8.3f %8.3f %8d" % (
            r["case"], r["side"], r["level"], r["base_d_core"], r["boot_sd_dcore"],
            r["jit0.5_sd_dcore"], r["R_range_7p5_12p5"], r["n_streamlines"]))
    print("\n--- HIEU d_core giua hai tang lien tiep, cung ban cau ---")
    print("%-20s %-4s %-9s %9s %8s %8s %7s" % (
        "ca", "ben", "cap", "hieu", "SD_hieu", "SD_tuyetdoi", "r"))
    for h in per_hemi:
        print("%-20s %-4s %-9s %+9.3f %8.3f %8.3f %7.3f" % (
            h["case"], h["side"], h["pair"], h["base_diff"], h["sd_diff"],
            h["sd_abs_z"], h["corr"]))
    print(f"\nDa ghi -> {OUT/'cst_dcore_measurement_error.json'}")


if __name__ == "__main__":
    main()
