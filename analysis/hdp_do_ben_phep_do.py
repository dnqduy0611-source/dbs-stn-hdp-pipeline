"""Do BEN cua chinh phep do `d_prent` (HDP) - ban doi ung cua
`cst_dcore_measurement_error.py` cho nhanh HDP.

VI SAO PHAI CHAY: bao cao sai so phep do (BAO_CAO_SAI_SO_PHEP_DO.md) chi chay cho
CST, va no tim ra dieu nghiem trong: thu hang GIUA cac ban cau cua `d_core` gan nhu
khong ben (rho giua R=5 va R=10 chi +0,121). Ma toan bo tin hieu cua HDP nam dung o
thanh phan GIUA ban cau (Mundlak p ,001) - thanh phan KHONG duoc huong phep triet
tieu sai so ma thanh phan TRONG ban cau duoc huong.

BON NGUON NHIEU cua bao cao CST, ap lai cho d_prent:
  A. bootstrap soi          lay lai mau soi co hoan lai, %d lan
  B. lay mau con 50%%        bo ngau nhien nua so soi, %d lan
  C. quet R_LOCAL           KHONG AP DUOC - xem duoi
  D. rung tiep diem         dich tam ~ N(0, sigma^2 I), sigma = 0,5 va 1,0 mm, %d lan

NGUON C KHONG AP DUOC CHO d_prent - VA DAY LA PHAT BIEU DUOC DO, KHONG PHAI GIA DINH.
  d_prent = trung vi TREN MOI SOI cua khoang cach vuong goc nho nhat. Khong co qua
  cau ban kinh R trong dinh nghia => khong co tham so tu chon nao de quet.
  Co che giet d_core - "khoang cach toi trong tam cua phan bo tinh co lot vao qua cau
  10 mm", bi kep tran o ~R khi chi 8%% soi lot vao - khong co cho bam.
  De KHONG chi noi mieng, script van tinh bien the `d_prent_local` (co loc <= R) o
  nam gia tri R va in ra do dich chuyen cua NO. Cot do la doi chung: no PHAI dich,
  trong khi d_prent chinh dung yen tuyet doi.

BUOC 2: quy sai so do ra HE SO SUY GIAM cua rho -0,66 (o dau bang §6).
  do tin cay = 1 - var(sai so) / var(quan sat giua ban cau)
  he so suy giam = sqrt(do tin cay);  rho that = rho quan sat / he so suy giam
  Kem mot mo phong: cong THEM mot lieu nhieu doc lap roi tinh lai rho, 5000 lan.

GIA DINH PHAI NEU RO: sai so cua d_hdp muc BAN CAU duoc lay bang TRUNG VI sai so cua
4 tiep diem trong ban cau do - tuc GIA DINH KHONG co loi ich trung binh hoa. Do la
gia dinh BAO THU (bao cao CST do duoc tuong quan 0,866 giua cac tang, nen sai so
gan nhu khong triet tieu khi trung binh - gia dinh nay dung voi thuc te do).

CHI DOC. Khong ghi de bat ky file du lieu goc nao.
Chay:  wsl -d Ubuntu-22.04 -- python3 /mnt/d/ossdbs_env/dbs_stn_hdp_pipeline/clinical/hdp_do_ben_phep_do.py
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import remeasure_all_contact_tract as R  # noqa: E402

NB_FIB, NB_SUB, NB_JIT, NB_SIM = 2000, 500, 200, 5000
SIGMAS = (0.5, 1.0)
RS = (5.0, 7.5, 10.0, 12.5, 15.0)
SEED = 20260829

ROOT = R.ROOT
DU = ROOT / "reports/HDP_TONG_HOP_20260828/du_lieu"
PUB = ROOT / "reports/electrode_tract_remeasure_20260827/compare_prent_vs_dcore.csv"

SHORT = {"sub-06": "sub-06", "sub-04": "sub-04",
         "sub-02": "sub-02", "sub-03": "sub-03",
         "sub-08": "sub-08", "sub-09": "sub-09",
         "sub-05": "sub-05", "sub-01": "sub-01", "sub-10": "sub-10"}


def per_fiber_dmin(b, contact):
    """Khoang cach vuong goc nho nhat cho TUNG soi. Vecto hoa bang reduceat."""
    t = np.clip(np.einsum("ij,ij->i", contact - b["A"], b["AB"]) / b["ab2"], 0.0, 1.0)
    d = np.linalg.norm(b["A"] + t[:, None] * b["AB"] - contact, axis=1)
    return np.minimum.reduceat(d, b["starts"])


def check_reduceat(b, contact):
    """Cong kiem: reduceat phai trung khop vong lap goc cua measure_core."""
    st = b["starts"]
    t = np.clip(np.einsum("ij,ij->i", contact - b["A"], b["AB"]) / b["ab2"], 0.0, 1.0)
    d = np.linalg.norm(b["A"] + t[:, None] * b["AB"] - contact, axis=1)
    ref = np.array([d[st[i]:(st[i + 1] if i + 1 < len(st) else len(d))].min()
                    for i in range(len(st))])
    return float(np.abs(ref - np.minimum.reduceat(d, st)).max())


def measure_contact(b, contact, rng):
    dmin = per_fiber_dmin(b, contact)
    n = len(dmin)
    d0 = float(np.median(dmin))
    out = {"d_prent_mm": d0, "n_soi": n,
           "p25": float(np.percentile(dmin, 25)),
           "p75": float(np.percentile(dmin, 75))}

    boot = np.median(dmin[rng.integers(0, n, (NB_FIB, n))], axis=1)
    out["sd_boot_soi"] = float(boot.std(ddof=1))
    out["lech_boot"] = float(boot.mean() - d0)

    k = max(2, n // 2)
    sub = np.median(np.take_along_axis(
        np.broadcast_to(dmin, (NB_SUB, n)),
        np.argsort(rng.random((NB_SUB, n)), axis=1)[:, :k], axis=1), axis=1)
    out["sd_mau_con_50"] = float(sub.std(ddof=1))

    for sg in SIGMAS:
        v = np.array([np.median(per_fiber_dmin(b, contact + rng.normal(0, sg, 3)))
                      for _ in range(NB_JIT)])
        out["sd_rung_%.1f" % sg] = float(v.std(ddof=1))
        out["lech_rung_%.1f" % sg] = float(v.mean() - d0)

    for Rv in RS:
        keep = dmin <= Rv
        out["d_local_R%.1f" % Rv] = (float(np.median(dmin[keep])) if keep.any()
                                     else float("nan"))
    return out


def phase1(s):
    rng = np.random.default_rng(SEED)
    pub = {(r["case"], r["side"], r["level"]): float(r["d_prent_mm"])
           for r in csv.DictReader(open(PUB, encoding="utf-8"))
           if r["tract"] == "HDP" and r["variant"] == "primary"}
    rows, gate = [], []
    for name, case in R.CASES.items():
        dwi = R.contacts_in_dwi(case, None)
        for (side, tract), path in case["tracts"].items():
            if tract != "HDP" or not Path(path).exists():
                continue
            b = R.load_bundle(path)
            if b is None:
                continue
            for lv in range(4):
                c = dwi[side][lv]
                if np.isnan(c).any():
                    continue
                if not rows:
                    gate.append(check_reduceat(b, c))
                m = measure_contact(b, c, rng)
                key = (name, side, "K%d" % lv)
                m["lech_voi_ban_cong_bo"] = (abs(m["d_prent_mm"] - pub[key])
                                             if key in pub else float("nan"))
                m.update(bn=SHORT[name], case=name, ben=side, muc="K%d" % lv)
                rows.append(m)
        s.append("  %-18s xong (%d tiep diem)" % (name, len(rows)))
    return rows, (gate[0] if gate else float("nan"))


def fmt(s, rows, gate_max):
    # NGUONG 1e-4: compare_prent_vs_dcore.csv luu d_prent LAM TRON 4 CHU SO, nen
    # nua buoc lam tron la 5e-5. Nguong 1e-6 cua ban dau la SAI - no bat chinh phep
    # lam tron cua file, khong bat sai lech tinh toan. Da kiem: lech toi da dung
    # bang 5,000e-05 mm o moi tiep diem lech.
    dv = np.array([r["lech_voi_ban_cong_bo"] for r in rows], float)
    bad = int((dv > 1e-4).sum())
    s += ["", "=" * 92,
          "0. CONG KIEM", "=" * 92,
          "  reduceat vs vong lap goc            lech toi da %.2e mm" % gate_max,
          "  d_prent tinh lai vs ban da cong bo  %d/%d tiep diem lech > 1e-4 mm"
          " (lech toi da %.3e = nua buoc lam tron 4 chu so cua file goc)"
          % (bad, len(rows), np.nanmax(dv))]

    def col(k):
        return np.array([r[k] for r in rows], float)

    s += ["", "=" * 92,
          "1. NHIEU NGAU NHIEN tren d_prent - %d tiep diem HDP" % len(rows),
          "=" * 92,
          "%-30s %12s %10s %10s" % ("nguon", "SD trung vi", "nho nhat", "lon nhat")]
    for k, lab in (("sd_boot_soi", "A. bootstrap soi"),
                   ("sd_mau_con_50", "B. lay mau con 50%"),
                   ("sd_rung_0.5", "D. rung tiep diem 0,5 mm"),
                   ("sd_rung_1.0", "D. rung tiep diem 1,0 mm")):
        v = col(k)
        s.append("%-30s %12.3f %10.3f %10.3f"
                 % (lab, np.median(v), v.min(), v.max()))
    for k, lab in (("lech_boot", "thien lech bootstrap"),
                   ("lech_rung_0.5", "thien lech rung 0,5 mm"),
                   ("lech_rung_1.0", "thien lech rung 1,0 mm")):
        v = col(k)
        s.append("%-30s %12.3f %10.3f %10.3f"
                 % (lab, np.median(v), v.min(), v.max()))

    s += ["", "=" * 92,
          "2. NGUON C (quet R_LOCAL) - KHONG AP DUOC CHO d_prent",
          "   Cot trai: d_prent CHINH (khong co tham so R).",
          "   Cot phai: bien the d_prent_local (CO loc <= R) - de doi chung.",
          "=" * 92,
          "%-14s %12s %12s %12s" % ("R (mm)", "trung vi", "bien do 18bc", "SD")]
    for Rv in RS:
        v = col("d_local_R%.1f" % Rv)
        v = v[np.isfinite(v)]
        s.append("%-14.1f %12.3f %12.3f %12.3f"
                 % (Rv, np.median(v), v.max() - v.min(), v.std(ddof=1)))
    v0 = col("d_prent_mm")
    s.append("%-14s %12.3f %12.3f %12.3f  <- KHONG phu thuoc R"
             % ("d_prent", np.median(v0), v0.max() - v0.min(), v0.std(ddof=1)))
    k0 = np.array([r["d_local_R5.0"] for r in rows], float)
    k1 = np.array([r["d_local_R10.0"] for r in rows], float)
    m = np.isfinite(k0) & np.isfinite(k1)
    s.append("")
    s.append("  rho thu hang d_prent_local: R=5 vs R=10  ->  %+.3f  (n=%d)"
             % (stats.spearmanr(k0[m], k1[m]).statistic, m.sum()))
    s.append("  (o CST, cung phep nay tren d_core cho +0,121 - do la o giet d_core)")
    return s


def phase2(s, rows):
    rng = np.random.default_rng(SEED)
    err = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for sg in SIGMAS:
            err[(r["bn"], r["ben"])][sg].append(
                np.hypot(r["sd_boot_soi"], r["sd_rung_%.1f" % sg]))

    cfg = list(csv.DictReader(open(DU / "hdp_27_cau_hinh_lam_sang.csv",
                                   encoding="utf-8")))
    g = defaultdict(list)
    for r in cfg:
        g[(r["patient"], r["lead"])].append((float(r["d_hdp"]),
                                             float(r["imp_hemi3"])))
    hemi = sorted(g)
    d = np.array([np.mean([a for a, _ in g[h]]) for h in hemi])
    y = np.array([np.mean([b for _, b in g[h]]) for h in hemi])
    pat = [h[0] for h in hemi]
    rho = stats.spearmanr(d, y).statistic

    s += ["", "=" * 92,
          "3. HE SO SUY GIAM ap len rho o dau bang §6  (muc BAN CAU, n=%d)" % len(d),
          "=" * 92,
          "  rho quan sat (d_hdp x imp_hemi3) = %+.4f" % rho,
          "  phuong sai GIUA ban cau cua d_hdp = %.4f mm^2" % d.var(ddof=1),
          "",
          "%-22s %9s %9s %8s %12s %10s %19s %7s"
          % ("kich ban sai so", "SD sai so", "do tin cay", "he suy",
             "rho hieu chinh", "rho mo phong", "KTC95 mo phong", "giu dau")]

    def line(lab, se):
        rel = 1.0 - (se ** 2).mean() / d.var(ddof=1)
        if rel <= 0:
            att, hc = float("nan"), "KHONG XAC DINH"
        else:
            att = np.sqrt(rel)
            hc = ("%+.4f" % (rho / att)) if abs(rho / att) <= 1 else "NGOAI MIEN"
        sim = np.array([stats.spearmanr(d + rng.normal(0, se), y).statistic
                        for _ in range(NB_SIM)])
        lo, hi = np.percentile(sim, [2.5, 97.5])
        s.append("%-22s %9.3f %9.3f %8.3f %12s %+10.4f %19s %6.1f%%"
                 % (lab, np.median(se), rel, att, hc, np.median(sim),
                    "[%+.3f; %+.3f]" % (lo, hi),
                    100 * np.mean(np.sign(sim) == np.sign(rho))))
        return hi

    se0 = np.array([np.median([r["sd_boot_soi"] for r in rows
                               if (r["bn"], r["ben"]) == h]) for h in hemi])
    line("chi bootstrap soi", se0)
    for sg in SIGMAS:
        line("bootstrap + rung %.1f" % sg,
             np.array([np.median(err[h][sg]) for h in hemi]))
    s += ["",
          "  do tin cay = 1 - var(sai so)/var(quan sat). he suy = sqrt(do tin cay).",
          "  rho hieu chinh = rho quan sat / he suy. Sai so do LAM YEU rho, nen ban",
          "  hieu chinh luon LON HON ve tri tuyet doi: sai so KHONG THE TAO RA ket qua,",
          "  chi co the che giau mot ket qua manh hon.",
          "  'NGOAI MIEN' = cong thuc hieu chinh co dien tra ve |rho| > 1, tuc gia dinh",
          "  sai so do da vuot qua do lon ma du lieu chiu duoc => KHONG duoc trich con so",
          "  do. Cot dung de doc trong tinh huong do la cot MO PHONG ben phai.",
          "  rho mo phong = cong THEM mot lieu nhieu doc lap NUA vao 18 gia tri ban cau",
          "  roi tinh lai rho, %d lan. Day la phep kiem BI QUAN kep (du lieu goc da chua"
          % NB_SIM,
          "  san mot lieu nhieu roi).",
          "",
          "  GIA DINH PHAI NEU: (a) sai so ban cau = TRUNG VI sai so 4 tiep diem, tuc",
          "  gia dinh khong co loi ich trung binh hoa - bao thu; (b) nhieu rung doc lap",
          "  giua cac ban cau. Trong thuc te, hai dien cuc cua CUNG mot benh nhan dung",
          "  chung mot chuoi dang ky CT->T1->DWI nen sai so co tuong quan trong benh",
          "  nhan; mo phong doc lap la xap xi, khong phai mo hinh day du."]
    return s


def main():
    s = ["", "Dang do tung ca (moi tiep diem: %d bootstrap soi, %d mau con, "
         "%d x2 rung):" % (NB_FIB, NB_SUB, NB_JIT)]
    rows, gate_max = phase1(s)
    s = fmt(s, rows, gate_max)
    s = phase2(s, rows)
    s.append("")
    sys.stdout.buffer.write("\n".join(s).encode("utf-8", "replace"))

    keys = ["bn", "case", "ben", "muc", "n_soi", "d_prent_mm", "p25", "p75",
            "sd_boot_soi", "lech_boot", "sd_mau_con_50",
            "sd_rung_0.5", "lech_rung_0.5", "sd_rung_1.0", "lech_rung_1.0",
            "lech_voi_ban_cong_bo"] + ["d_local_R%.1f" % r for r in RS]
    with open(DU / "hdp_do_ben_phep_do.csv", "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 5) if isinstance(v, float) else v)
                        for k, v in r.items() if k in keys})


if __name__ == "__main__":
    main()
