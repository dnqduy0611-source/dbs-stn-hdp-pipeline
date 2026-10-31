"""Bootstrap cum benh nhan cho TOAN BO bang tuong quan §6 cua bao cao HDP.

VI SAO: nhanh CST bat buoc moi lan trich rho phai kem KTC bootstrap cum benh nhan
(§4 cua BAO_CAO_CST_TONG_HOP.md). Nhanh HDP chi ap ky luat do cho §8 (Huong A);
bang chinh §6 - o dau bang rho -0,6605 p ,00018 - van bao p tinh tren n = 27 cau
hinh, trong khi chinh §10 cua bao cao thua nhan n hieu dung la 18. Hai chuan.

BA MUC SUY LUAN, tu long den chat:
  1. p Spearman tho tren n cau hinh              <- cai dang duoc trich, NOI QUA
  2. KTC95 bootstrap cum benh nhan (9 cum)       <- ky luat cua nhanh CST
  3. p hoan vi KHOI benh nhan o muc ban cau      <- chat nhat, hop le nhat

Muc 3 chi chay duoc o MUC BAN CAU vi moi benh nhan co dung 2 ban cau => 9 khoi
kich thuoc bang nhau, hoan vi khoi la mot phep ngau nhien hoa hop le (9! = 362880).
O muc cau hinh, so dong moi benh nhan khac nhau (2..6) nen khong hoan vi khoi duoc.

LUU Y KHI DOC MUC BAN CAU: 12/18 ban cau chi co 1 cau hinh, nhung 5/18 ban cau co
outcome KHAC NHAU giua cac cau hinh (cac ca nhieu mode kich thich). Gop ve ban cau
lay TRUNG BINH => mat phan bien thien do. Do la gia phai tra de co mau so trung thuc.

CHI DOC. Khong ghi de bat ky file du lieu nao.
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(r"D:\ossdbs_env\dbs_stn_hdp_pipeline")
SRC = ROOT / "reports/HDP_TONG_HOP_20260828/du_lieu/hdp_27_cau_hinh_lam_sang.csv"
OUT = ROOT / "reports/HDP_TONG_HOP_20260828/du_lieu"
NBOOT = 10000
NPERM = 20000
SEED = 20260829

XS = (("d_hdp", "d_hdp"), ("pct_vta", "pct_vta"))
YS = (("imp_hemi3", "imp_hemi3"), ("post_hemi3", "post_hemi3"),
      ("imp_tremor", "imp_tremor"), ("imp_rigbrady", "imp_rigbrady"),
      ("chg_hemi3", "chg_hemi3"))


def fnum(v):
    v = (v or "").strip()
    if v in ("", "nan", "NaN", "NA"):
        return float("nan")
    return float(v)


def load():
    rows = list(csv.DictReader(open(SRC, encoding="utf-8")))
    for r in rows:
        r["_pat"] = r["patient"]
        r["_hemi"] = (r["patient"], r["lead"])
    return rows


def to_hemi(rows, xcol, ycol):
    """Gop ve ban cau: trung binh x va y trong tung ban cau."""
    g = defaultdict(list)
    for r in rows:
        x, y = fnum(r[xcol]), fnum(r[ycol])
        if np.isnan(x) or np.isnan(y):
            continue
        g[r["_hemi"]].append((x, y))
    out = []
    for (pat, lead), v in sorted(g.items()):
        a = np.array(v, float)
        out.append((pat, lead, a[:, 0].mean(), a[:, 1].mean()))
    return out


def boot_ci(x, y, pat, rng, nboot=NBOOT):
    """KTC95 percentile, bootstrap CUM benh nhan (lay lai mau cac benh nhan)."""
    g = defaultdict(list)
    for i, p in enumerate(pat):
        g[p].append(i)
    keys = sorted(g)
    K = len(keys)
    vals = []
    for _ in range(nboot):
        idx = np.concatenate([g[keys[j]] for j in rng.integers(0, K, K)])
        xs, ys = x[idx], y[idx]
        if len(np.unique(xs)) < 2 or len(np.unique(ys)) < 2:
            continue
        r = stats.spearmanr(xs, ys).statistic
        if np.isfinite(r):
            vals.append(r)
    v = np.array(vals)
    if len(v) < nboot * 0.5:
        return None
    lo, hi = np.percentile(v, [2.5, 97.5])
    return dict(lo=float(lo), hi=float(hi), med=float(np.median(v)),
                chua_0=bool(lo <= 0 <= hi), n_dung=int(len(v)),
                doi_dau=float(np.mean(np.sign(v) != np.sign(np.median(v)))))


def perm_block(x, y, pat, rng, nperm=NPERM):
    """p hoan vi KHOI benh nhan. Moi benh nhan = 1 khoi 2 ban cau (L,R) giu nguyen
    thu tu; hoan vi xem khoi outcome cua benh nhan nao di voi khoi khoang cach nao.
    Hai phia."""
    g = defaultdict(list)
    for i, p in enumerate(pat):
        g[p].append(i)
    keys = sorted(g)
    order = [g[k] for k in keys]
    flat = np.concatenate(order)
    obs = stats.spearmanr(x[flat], y[flat]).statistic
    tang = defaultdict(list)
    for i, blk in enumerate(order):
        tang[len(blk)].append(i)
    if max(len(v) for v in tang.values()) < 4:
        return None
    ge = 0
    for _ in range(nperm):
        pm = list(range(len(keys)))
        for _sz, mem in tang.items():
            for a, b in zip(mem, rng.permutation(len(mem))):
                pm[a] = mem[b]
        yy = np.concatenate([order[j] for j in pm])
        r = stats.spearmanr(x[flat], y[yy]).statistic
        if np.isfinite(r) and abs(r) >= abs(obs) - 1e-12:
            ge += 1
    return dict(obs=float(obs), p=(ge + 1) / (nperm + 1))


def run(rows, tag, s, L):
    rng = np.random.default_rng(SEED)
    for lvl in ("cau_hinh", "ban_cau"):
        s += ["", "-" * 104,
              "  %s  |  muc = %s" % (tag, lvl),
              "-" * 104,
              "%-9s %-14s %4s %4s %9s %10s %20s %6s %10s"
              % ("x", "y", "n", "nBN", "rho", "p tho", "KTC95 bootstrap cum",
                 "chua0", "p hoan vi")]
        for xc, xl in XS:
            for yc, yl in YS:
                if lvl == "cau_hinh":
                    dat = [(r["_pat"], r["lead"], fnum(r[xc]), fnum(r[yc]))
                           for r in rows]
                    dat = [d for d in dat if not (np.isnan(d[2]) or np.isnan(d[3]))]
                else:
                    dat = to_hemi(rows, xc, yc)
                if len(dat) < 6:
                    continue
                pat = [d[0] for d in dat]
                x = np.array([d[2] for d in dat], float)
                y = np.array([d[3] for d in dat], float)
                rho, p = stats.spearmanr(x, y)
                bc = boot_ci(x, y, pat, rng)
                pb = perm_block(x, y, pat, rng) if lvl == "ban_cau" else None
                ktc = ("[%+.3f; %+.3f]" % (bc["lo"], bc["hi"])) if bc else "(that bai)"
                c0 = ("CO" if bc["chua_0"] else "khong") if bc else "-"
                pp = ("%.4f" % pb["p"]) if pb else "-"
                s.append("%-9s %-14s %4d %4d %+9.4f %10.5f %20s %6s %10s%s"
                         % (xl, yl, len(dat), len(set(pat)), rho, p, ktc, c0, pp,
                            "  <<<" if bc and not bc["chua_0"] else ""))
                L.append([tag, lvl, xl, yl, len(dat), len(set(pat)),
                          round(rho, 4), round(p, 5),
                          round(bc["lo"], 4) if bc else "",
                          round(bc["hi"], 4) if bc else "",
                          round(bc["med"], 4) if bc else "",
                          ("khong" if bc["chua_0"] else "co") if bc else "",
                          round(bc["doi_dau"], 4) if bc else "",
                          round(pb["p"], 5) if pb else ""])
    return s


def main():
    rows = load()
    s, L = [], []
    s = run(rows, "tat_ca_ke_ca_doan", s, L)
    s = run([r for r in rows if r["directed"] == "0"], "chinh_vong_tron", s, L)
    s += ["",
          "chua0 = CO nghia la KTC95 CHUA so 0 => KHONG tach khoi 0 => o do TAN RA.",
          "p tho = Spearman tren n cau hinh, KHONG hieu chinh cum. Day la con so",
          "        dang duoc trich trong §6 hien tai.",
          "p hoan vi = hoan vi KHOI benh nhan, hai phia, %d lan, chi o muc ban cau." % NPERM,
          "        Khoi chi hoan vi trong nhom CUNG KICH THUOC (khi mot BN thieu mot",
          "        ban cau thi BN do chi doi cho voi cac BN cung so ban cau).",
          "bootstrap: %d lan, lay lai mau 9 benh nhan co hoan lai, KTC percentile." % NBOOT,
          ""]
    sys.stdout.buffer.write("\n".join(s).encode("utf-8", "replace"))
    with open(OUT / "hdp_bootstrap_cum_muc6.csv", "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["bo_du_lieu", "muc", "x", "y", "n", "n_bn", "rho", "p_tho",
                    "boot_lo", "boot_hi", "boot_med", "tach_khoi_0",
                    "ty_le_doi_dau", "p_hoan_vi_khoi"])
        w.writerows(L)


if __name__ == "__main__":
    main()
