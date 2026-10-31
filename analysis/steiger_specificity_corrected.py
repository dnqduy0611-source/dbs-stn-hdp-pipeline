"""HUONG A - kiem tinh DAC HIEU giai phau, chay tren DIEM DA SUA (sub-09 + sub-02).

CAU HOI: thuoc do cua ta co that su do "gan bo nao" khong, hay chi do DO SAU
dien cuc doi lot? Neu dac hieu that thi phai thay hinh chu X (phan ly kep):

                   cung/cham        run
   d_HDP   ->      MANH             yeu
   d_DRTT  ->      yeu              MANH

Bon phep so Williams/Steiger, moi phep CHUNG MOT BIEN (overlapping correlations):

  A. HDP  co dac hieu khong?  r(dHDP , cc)  vs  r(dHDP , run)   chung dHDP
  B. DRTT co dac hieu khong?  r(dDRTT, run) vs  r(dDRTT, cc)    chung dDRTT
  C. Bo nao du bao cung/cham? r(dHDP , cc)  vs  r(dDRTT, cc)    chung cc
  D. Bo nao du bao run?       r(dDRTT, run) vs  r(dHDP , run)   chung run

Quy uoc dau: khoang cach nho = tot, nen gia thuyet dung => rho AM. "Manh hon"
nghia la AM HON. Cot 'hieu' duoc in sao cho gia thuyet dung => hieu AM.

KHAC ban 27/08 (`steiger_specificity_test.py`):
  - doc `reports/spss_updrs_tract_20260827/*.csv` (dung tu workbook 13:47 da sua),
    khong doc `clinical_contact_level_20260827/` cua ban cu;
  - dung `d_hdp`/`d_drtt` muc TIEP DIEM, khong dung `d_core_*`;
  - them DON VI VTA: `pct_vta` (%soi HDP trong VTA) vs `pct_drtt`, de xem FEM co
    cuu duoc tinh dac hieu ma khoang cach khong co hay khong.

HAI CANH BAO giu nguyen tu ban cu:
 1. Williams gia dinh mau doc lap + chuan hai bien. Ca hai deu SAI o day (tiep diem
    trong cung nguoi phu thuoc nhau; ta dung Spearman). p Williams la CAN DUOI
    lac quan - that su con yeu hon the.
 2. Vi vay moi phep so kem CLUSTER BOOTSTRAP theo BENH NHAN cho hieu so. Do moi la
    con so nen trich dan. Nhung chi co 8-9 cum => van hep qua muc.

Chay:  python3 clinical/steiger_specificity_corrected.py
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
TRACT = ROOT / "reports/spss_updrs_tract_20260827"
VTA = ROOT / "reports/vta_fem_20260828/spss/vta_segment.csv"
OUT = ROOT / "reports/vta_fem_20260828/steiger_specificity_corrected.json"

N_BOOT = 20_000
SEED = 20260828

CC, RUN = "imp_rigbrady", "imp_tremor"


def load(path: Path, hdp: str, drtt: str) -> list[dict]:
    """Complete-case: phai co CA HAI thuoc do va CA HAI ket cuc cung luc.

    Nguong nen giong `include_hdp`/`include_drtt` cua build_contact_dataset.py
    (cung/cham nen >= 1, run nen >= 2) nhung o day phai thoa CA HAI cung luc,
    nen n nho hon bang chinh. Do la dung, khong phai loi.
    """
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            try:
                row = {
                    "patient": raw["patient"],
                    "lead": raw["lead"],
                    "hdp": float(raw[hdp]),
                    "drtt": float(raw[drtt]),
                    CC: float(raw[CC]),
                    RUN: float(raw[RUN]),
                    "base_rigbrady": float(raw["base_rigbrady"]),
                    "base_tremor": float(raw["base_tremor"]),
                }
            except (ValueError, KeyError):
                continue
            if row["base_rigbrady"] >= 1 and row["base_tremor"] >= 2:
                rows.append(row)
    return rows


def rho(rows: list[dict], a: str, b: str) -> float:
    x = [r[a] for r in rows]
    y = [r[b] for r in rows]
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return math.nan
    return float(stats.spearmanr(x, y).statistic)


def williams(r_jk: float, r_jh: float, r_kh: float, n: int) -> tuple[float, float, int]:
    """Williams (1959) / Steiger (1980), hai tuong quan CHUNG bien j. df = n - 3."""
    if n < 5 or any(math.isnan(v) for v in (r_jk, r_jh, r_kh)):
        return math.nan, math.nan, n - 3
    det = (1 - r_jk ** 2 - r_jh ** 2 - r_kh ** 2 + 2 * r_jk * r_jh * r_kh)
    mean_r = (r_jk + r_jh) / 2
    denom = (2 * (n - 1) / (n - 3) * det + mean_r ** 2 * (1 - r_kh) ** 3)
    if denom <= 0:
        return math.nan, math.nan, n - 3
    t = (r_jk - r_jh) * math.sqrt((n - 1) * (1 + r_kh) / denom)
    return t, float(2 * stats.t.sf(abs(t), n - 3)), n - 3


def cluster_bootstrap_diff(rows, pair1, pair2) -> dict:
    """KTC 95% cho HIEU rho1 - rho2, lay lai mau theo BENH NHAN (khong theo dong)."""
    by_patient: dict[str, list[dict]] = {}
    for row in rows:
        by_patient.setdefault(row["patient"], []).append(row)
    patients = sorted(by_patient)
    rng = np.random.default_rng(SEED)
    diffs = []
    for _ in range(N_BOOT):
        picked = rng.integers(0, len(patients), len(patients))
        sample = [r for i in picked for r in by_patient[patients[i]]]
        d1, d2 = rho(sample, *pair1), rho(sample, *pair2)
        if not (math.isnan(d1) or math.isnan(d2)):
            diffs.append(d1 - d2)
    if len(diffs) < 100:
        return {"lo": math.nan, "hi": math.nan, "n_ok": len(diffs), "p_two_sided": math.nan}
    array = np.array(diffs)
    share = float(min((array <= 0).mean(), (array >= 0).mean()))
    return {
        "lo": float(np.percentile(array, 2.5)),
        "hi": float(np.percentile(array, 97.5)),
        "n_ok": len(diffs),
        "p_two_sided": min(1.0, 2 * share),
    }


def patient_level(rows: list[dict]) -> list[dict]:
    by_patient: dict[str, list[dict]] = {}
    for row in rows:
        by_patient.setdefault(row["patient"], []).append(row)
    return [{"patient": p, "lead": "-",
             **{k: float(np.mean([r[k] for r in g])) for k in ("hdp", "drtt", CC, RUN)}}
            for p, g in sorted(by_patient.items())]


COMPARISONS = [
    ("A. HDP  co dac hieu?", ("hdp", CC), ("hdp", RUN), "HDP am hon o cung/cham"),
    ("B. DRTT co dac hieu?", ("drtt", RUN), ("drtt", CC), "DRTT am hon o run"),
    ("C. Ai du bao cung/cham?", ("hdp", CC), ("drtt", CC), "HDP am hon DRTT"),
    ("D. Ai du bao run?", ("drtt", RUN), ("hdp", RUN), "DRTT am hon HDP"),
]


def run_unit(label: str, rows: list[dict], flip: bool) -> dict:
    """flip=True khi thuoc do la KHOANG CACH (nho = tot) -> giu nguyen dau.

    Voi %VTA thi lon = tot, nen doi dau de moi bang doc cung mot kieu:
    'gia thuyet dung => rho AM, hieu AM'.
    """
    if not flip:
        rows = [dict(r, hdp=-r["hdp"], drtt=-r["drtt"]) for r in rows]

    n = len(rows)
    n_pat = len({r["patient"] for r in rows})
    n_hemi = len({(r["patient"], r["lead"]) for r in rows})
    print()
    print("=" * 104)
    print(f"{label}   n = {n} dong / {n_hemi} ban cau / {n_pat} benh nhan")
    print("=" * 104)
    if n < 5:
        print("  n qua nho - bo qua.")
        return {"unit": label, "n": n, "skipped": True}

    print(f"  Hai thuoc do di cung nhau:  rho(HDP, DRTT) = {rho(rows,'hdp','drtt'):+.3f}")
    print(f"  Hai ket cuc di cung nhau:   rho(%cc, %run) = {rho(rows,CC,RUN):+.3f}")
    print()
    print(f"  {'phep so':<24}{'rho 1':>8}{'rho 2':>8}{'hieu':>8}"
          f"{'t':>7}{'df':>4}{'p Wil':>8}{'  CI95 boot theo BN':>22}{'p boot':>8}  dat?")
    print("  " + "-" * 100)

    results = []
    for title, pair1, pair2, expect in COMPARISONS:
        r1, r2 = rho(rows, *pair1), rho(rows, *pair2)
        shared = set(pair1) & set(pair2)
        a, b = sorted((set(pair1) | set(pair2)) - shared)
        r_kh = rho(rows, a, b)
        t, p, df = williams(r1, r2, r_kh, n)
        boot = cluster_bootstrap_diff(rows, pair1, pair2)
        ok = "CO" if (r1 - r2) < 0 and p < 0.05 else ("dung chieu" if (r1 - r2) < 0 else "KHONG")
        print(f"  {title:<24}{r1:>+8.3f}{r2:>+8.3f}{r1 - r2:>+8.3f}"
              f"{t:>7.2f}{df:>4}{p:>8.4f}"
              + f"[{boot['lo']:+.2f}, {boot['hi']:+.2f}]".rjust(22)
              + f"{boot['p_two_sided']:>8.4f}  {ok}")
        results.append({
            "comparison": title, "expected": expect,
            "rho1": r1, "rho2": r2, "difference": r1 - r2,
            "r_between_the_other_two": r_kh,
            "williams_t": t, "williams_df": df, "williams_p": p,
            "cluster_bootstrap": boot, "verdict": ok,
        })

    interaction = ((rho(rows, "hdp", CC) - rho(rows, "hdp", RUN))
                   - (rho(rows, "drtt", CC) - rho(rows, "drtt", RUN)))
    print()
    print(f"  Tuong tac bac hai (hai bo co PHAN VAI nguoc nhau khong): {interaction:+.3f}"
          "   (gia thuyet dung => AM)")

    return {"unit": label, "n": n, "n_hemispheres": n_hemi, "n_patients": n_pat,
            "rho_hdp_drtt": rho(rows, "hdp", "drtt"), "rho_outcomes": rho(rows, CC, RUN),
            "interaction": interaction, "comparisons": results}


def main() -> None:
    contact = load(TRACT / "contact.csv", "d_hdp", "d_drtt")
    modehemi = load(TRACT / "mode_contact.csv", "d_hdp", "d_drtt")
    hemi = load(TRACT / "hemisphere.csv", "d_hdp", "d_drtt")
    vta_dist = load(VTA, "d_hdp", "d_drtt")
    vta_pct = load(VTA, "pct_vta", "pct_drtt")

    units = [
        run_unit("1. KHOANG CACH - don vi TIEP DIEM", contact, True),
        run_unit("2. KHOANG CACH - gop ve MUC BENH NHAN", patient_level(contact), True),
        run_unit("3. KHOANG CACH - don vi BAN CAU (1 mode/BN)", hemi, True),
        run_unit("4. KHOANG CACH - don vi MODE x BAN CAU", modehemi, True),
        run_unit("5. KHOANG CACH - don vi CAU HINH VTA", vta_dist, True),
        run_unit("6. %SOI TRONG VTA (FEM) - don vi CAU HINH", vta_pct, False),
    ]
    OUT.write_text(json.dumps({
        "generated": "2026-08-28",
        "source": ("workbook 28/08 (diem sub-09 + sub-02 da sua; sub-05 lead phai C+9- = K1); "
                   "thuoc do khoang cach = trung vi Prent 2019 (d_prent_mm), "
                   "xem run_spss_preliminary_analysis.METRIC"),
        "outcome_cc": CC, "outcome_run": RUN,
        "note": ("Complete-case ca hai thuoc do va ca hai ket cuc, nen n nho hon bang "
                 "chinh va rho khong trung tung con so. Don vi 6 da DOI DAU %VTA de "
                 "moi bang doc cung kieu 'gia thuyet dung => rho AM'."),
        "n_bootstrap": N_BOOT, "seed": SEED, "units": units,
    }, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print()
    print(f"Da ghi: {OUT}")


if __name__ == "__main__":
    main()
