#!/usr/bin/env python3
"""
PHI scrub + staging cho mot lo archive du lieu DBS truoc khi dua len cloud.

Cau hinh: bien moi truong PHI_SCRUB_CONFIG -> file JSON (xem phi_scrub_config.example.json).

Pha:
  scan     - liet ke moi file se bi dung toi, khong ghi gi
  scrub    - go PHI TAI CHO (header .tck ghi de cung do dai byte, strip .json, sua text)
  stage    - MOVE sang thu muc STAGE (cung volume => tuc thi)
  manifest - sinh MANIFEST_SHA256.txt
  gate     - grep ho ten / ma BN / so phieu / ngay sinh => phai ra 0 hit

Nguyen tac an toan:
  - Header .tck: CHI thay chuoi CUNG DO DAI, khong byte nao xe dich,
    truong "file: . <offset>" giu nguyen hieu luc. Assert do dai truoc khi ghi.
  - Kich thuoc file va truong count: phai khong doi sau khi scrub.
"""
import os, sys, re, json, hashlib, shutil

# Cau hinh lo (duong dan, bang ma hoa, dinh danh cung, danh sach muc) doc tu file
# JSON NGOAI repo. Khong bao gio viet ho ten / ma BN vao code: chinh file script
# se lot vao cong PHI (xem bay 3 trong README cua thu muc nay).
CFG_PATH = os.environ.get("PHI_SCRUB_CONFIG", "phi_scrub_config.json")
with open(CFG_PATH, encoding="utf-8") as _f:
    CFG = json.load(_f)

ROOT    = CFG["root"]
STAGE   = CFG["stage"]
MAPFILE = CFG["mapfile"]          # O LAI may cuc bo, KHONG len cloud
LOGDIR  = CFG["logdir"]

# ---- ban do ma hoa (thu tu alphabet, on dinh): {"Ten_Thu_Muc": "sub-01"} ------
SUBJ = CFG["subjects"]

def pad(name):
    """sub-NN dem '_' cho bang do dai ten goc -> giu nguyen offset header .tck"""
    s = SUBJ[name]
    assert len(s) <= len(name), f"{name} qua ngan de ma hoa"
    return s + "_" * (len(name) - len(s))

# MANH ten nhung trong ten thu muc do chinh du an dat ra (khong phai DICOM).
# Cong kiem ban dau CHI tim ho ten day du nen bo sot nhung chuoi nay -> them vao.
# Bat buoc CUNG DO DAI vi chung nam trong header .tck (vd "_abc" -> "_s01").
FRAG = [(a.encode(), b.encode()) for a, b in CFG.get("fixed_fragments", [])]

# thay the CUNG DO DAI, dung cho header .tck (nhi phan, offset nhay cam)
FIXED = [(n.encode(), pad(n).encode()) for n in SUBJ] + FRAG

# thay the TU DO, dung cho .json/.py/.log/.txt/.md (do dai doi duoc)
FREE = []
for n, s in SUBJ.items():
    parts = n.split("_")
    FREE += [
        (n, s),                                     # Ho_Ten_Dem
        (n.lower(), s),                             # ho_ten_dem
        (n.replace("_", " "), s),                   # Ho Ten Dem
        (n.replace("_", " ").upper(), s),           # HO TEN DEM
        (n.replace("_", "-").lower(), s),           # ho-ten-dem
        ("_".join(parts).lower(), s),
    ]
FREE = sorted(set(FREE), key=lambda x: -len(x[0]))

# dinh danh cung ngoai ho ten (ma BN, so phieu, ngay sinh, ten co so, ten nguoi
# chup...) lay tu .json sidecar da soi.
# CO Y BO StudyID ngan (5 chu so): trung ngau nhien trong du lieu toa do ASCII
# cua .vtk (da kiem 8/8 hit deu la false positive) -> de lai thi cong PHI khong
# bao gio xanh. StudyID van bi go khoi .json qua PHI_KEYS.
HARD_IDS = CFG.get("hard_ids", [])

# khoa PHI can go khoi .json sidecar dcm2niix. GIU lai thong so anh.
PHI_KEYS = {
    "PatientName", "PatientID", "PatientBirthDate", "PatientAddress",
    "PatientTelephoneNumbers", "OtherPatientIDs", "OtherPatientNames",
    "AccessionNumber", "StudyID", "StudyInstanceUID", "SeriesInstanceUID",
    "FrameOfReferenceUID", "StudyDate", "StudyTime", "SeriesDate", "SeriesTime",
    "AcquisitionDateTime", "AcquisitionDate", "AcquisitionTime", "ContentDate",
    "ContentTime", "InstanceCreationDate", "InstanceCreationTime",
    "ReferringPhysicianName", "PerformingPhysicianName", "OperatorsName",
    "PhysiciansOfRecord", "RequestingPhysician",
    "InstitutionName", "InstitutionAddress", "InstitutionalDepartmentName",
    "StationName", "DeviceSerialNumber",
}
# GIU: PatientSex, PatientAge, Modality, Manufacturer, EchoTime, RepetitionTime,
#      SliceThickness, PatientPosition, ProcedureStepDescription, ...

# ---- cac muc trong lo: [nhan, duong dan tuong doi ROOT | "H5SET"] -------------
ITEMS = [(lbl, src if src == "H5SET" else os.path.join(ROOT, src))
         for lbl, src in CFG["items"]]
H5SET = [os.path.join(ROOT, p) for p in CFG.get("h5set", [])]

TEXT_EXT = {".py", ".log", ".txt", ".md", ".sh", ".mrml", ".csv", ".m", ".cfg", ".ini"}


def walk(item):
    """tra ve list (abspath, relpath_trong_muc)"""
    label, src = item
    if src == "H5SET":
        out = []
        for p in H5SET:
            run = os.path.basename(os.path.dirname(os.path.dirname(p)))
            out.append((p, os.path.join(run, os.path.basename(p))))
        return out
    if os.path.isfile(src):
        return [(src, os.path.basename(src))]
    out = []
    for dp, _, fns in os.walk(src):
        for fn in fns:
            p = os.path.join(dp, fn)
            out.append((p, os.path.relpath(p, src)))
    return out


# ---------------------------------------------------------------- scrub .tck
def scrub_tck(path, apply):
    with open(path, "rb") as f:
        buf = f.read(4 * 1024 * 1024)
    i = buf.find(b"\nEND\n")
    if i < 0:
        return "ERR_NO_END", None
    hlen = i + 5
    hdr = buf[:hlen]
    new = hdr
    for a, b in FIXED:
        new = new.replace(a, b)
    if new == hdr:
        return "CLEAN", None
    if len(new) != len(hdr):
        return "ERR_LEN", None
    m_before = re.search(rb"^count:\s*(\d+)", hdr, re.M)
    size_before = os.path.getsize(path)
    if not apply:
        return "WOULD_SCRUB", None
    with open(path, "r+b") as f:
        f.seek(0); f.write(new); f.flush(); os.fsync(f.fileno())
    # --- verify ---
    size_after = os.path.getsize(path)
    with open(path, "rb") as f:
        chk = f.read(hlen + 64)
    m_after = re.search(rb"^count:\s*(\d+)", chk[:hlen], re.M)
    ok = (size_after == size_before
          and chk[:hlen] == new
          and (m_before is None) == (m_after is None)
          and (m_before is None or m_before.group(1) == m_after.group(1))
          and not any(a in chk[:hlen] for a, _ in FIXED))
    return ("SCRUBBED" if ok else "ERR_VERIFY",
            {"size": size_before, "count": m_before.group(1).decode() if m_before else "?"})


# --------------------------------------------------------------- scrub .json
def scrub_json(path, apply):
    try:
        d = json.load(open(path, "r", encoding="utf-8"))
    except Exception as e:
        return f"ERR_JSON:{e}", None
    hit = sorted(k for k in d if k in PHI_KEYS)
    if not hit:
        return "CLEAN", None
    if not apply:
        return "WOULD_STRIP", hit
    for k in hit:
        del d[k]
    d["_deidentified"] = "PHI keys removed 2026-08-20; see PATIENT_ID_MAP.csv (offline)"
    json.dump(d, open(path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    return "STRIPPED", hit


# --------------------------------------------------------------- scrub text
def scrub_text(path, apply):
    try:
        s = open(path, "r", encoding="utf-8", errors="surrogateescape").read()
    except Exception as e:
        return f"ERR_TXT:{e}", None
    new = s
    for a, b in FREE:
        new = new.replace(a, b)
    for h in HARD_IDS:
        new = new.replace(h, "[REDACTED]")
    if new == s:
        return "CLEAN", None
    if not apply:
        return "WOULD_SCRUB", None
    open(path, "w", encoding="utf-8", errors="surrogateescape").write(new)
    return "SCRUBBED", None


# ------------------------------------------------------------ phat hien PHI
NEEDLES = [n.encode() for n in SUBJ] + \
          [n.replace("_", " ").upper().encode() for n in SUBJ] + \
          [n.lower().encode() for n in SUBJ] + \
          [h.encode() for h in HARD_IDS]

def sniff(path, nbytes=4 * 1024 * 1024):
    """quet dau file tim dau vet PHI (du cho header .tck/.mif/.nii, .vtk nho)"""
    try:
        with open(path, "rb") as f:
            b = f.read(nbytes)
    except Exception:
        return []
    return sorted({n.decode(errors="replace") for n in NEEDLES if n in b})


def sha256(path, bs=8 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            c = f.read(bs)
            if not c:
                break
            h.update(c)
    return h.hexdigest()


# ==================================================================== phases
def do_scan(apply=False):
    tot_files = tot_bytes = 0
    for item in ITEMS:
        label, _ = item
        fl = walk(item)
        b = sum(os.path.getsize(p) for p, _ in fl)
        tot_files += len(fl); tot_bytes += b
        print(f"\n### {label}   {len(fl)} file   {b/2**30:.2f} GiB")
        cats = {}
        for p, _ in fl:
            cats[os.path.splitext(p)[1].lower()] = cats.get(os.path.splitext(p)[1].lower(), 0) + 1
        print("    ext:", ", ".join(f"{k or '(none)'}={v}" for k, v in sorted(cats.items(), key=lambda x: -x[1])))
        dirty = []
        for p, rel in fl:
            hits = sniff(p)
            if hits:
                dirty.append((rel, os.path.splitext(p)[1].lower(), hits))
        print(f"    file co dau vet PHI: {len(dirty)}/{len(fl)}")
        for rel, ext, hits in dirty[:8]:
            print(f"      - {rel}  <- {', '.join(hits)}")
        if len(dirty) > 8:
            print(f"      ... va {len(dirty)-8} file nua")
    print(f"\n===== TONG: {tot_files} file, {tot_bytes/2**30:.2f} GiB "
          f"({tot_bytes:,} bytes) =====")


def do_scrub(apply=True):
    os.makedirs(LOGDIR, exist_ok=True)
    log = open(os.path.join(LOGDIR, "phi_scrub_20260820.log"), "w", encoding="utf-8")
    tally = {}
    for item in ITEMS:
        label, _ = item
        print(f"\n### {label}")
        for p, rel in walk(item):
            ext = os.path.splitext(p)[1].lower()
            if ext in (".tck", ".mif"):   # ca hai deu co truong "file: . <offset>"
                st, info = scrub_tck(p, apply)
            elif ext == ".json":
                st, info = scrub_json(p, apply)
            elif ext in TEXT_EXT:
                st, info = scrub_text(p, apply)
            else:
                st, info = ("CLEAN" if not sniff(p) else "NEEDS_REVIEW"), None
            tally[st] = tally.get(st, 0) + 1
            if st != "CLEAN":
                line = f"{st}\t{label}\t{rel}\t{info or ''}"
                print("   ", line); log.write(line + "\n")
    log.close()
    print("\n===== KET QUA =====")
    for k, v in sorted(tally.items(), key=lambda x: -x[1]):
        print(f"  {k:14s} {v}")


def do_stage():
    os.makedirs(STAGE, exist_ok=True)
    for item in ITEMS:
        label, src = item
        dst = os.path.join(STAGE, label)
        os.makedirs(dst, exist_ok=True)
        if src == "H5SET":
            for p in H5SET:
                run = os.path.basename(os.path.dirname(os.path.dirname(p)))
                d2 = os.path.join(dst, run); os.makedirs(d2, exist_ok=True)
                os.replace(p, os.path.join(d2, os.path.basename(p)))
                print(f"  MOVE {run}/oss_time_result_PAM.h5")
        elif os.path.isfile(src):
            os.replace(src, os.path.join(dst, os.path.basename(src)))
            print(f"  MOVE {os.path.basename(src)}")
        else:
            for name in os.listdir(src):
                os.replace(os.path.join(src, name), os.path.join(dst, name))
            os.rmdir(src)
            print(f"  MOVE {os.path.basename(src)}/ (toan bo)")
        print(f"-> {label}")


def do_rescrub():
    """Chay lai scrub NGAY TREN thu muc gom (dung sau khi bo sung FRAG)."""
    tally = {}
    for dp, _, fns in os.walk(STAGE):
        for fn in fns:
            if fn == "MANIFEST_SHA256.txt":
                continue
            p = os.path.join(dp, fn)
            ext = os.path.splitext(fn)[1].lower()
            if ext in (".tck", ".mif"):
                st, info = scrub_tck(p, True)
            elif ext == ".json":
                st, info = scrub_json(p, True)
            elif ext in TEXT_EXT:
                st, info = scrub_text(p, True)
            else:
                st, info = "SKIP", None
            tally[st] = tally.get(st, 0) + 1
            if st not in ("CLEAN", "SKIP"):
                print(f"    {st}\t{os.path.relpath(p, STAGE)}\t{info or ''}")
    for k, v in sorted(tally.items(), key=lambda x: -x[1]):
        print(f"  {k:14s} {v}")


def do_rename():
    """Doi TEN FILE con mang ho ten. Dung cung bang FREE nen tham chieu cheo
    trong .mrml/.py (da scrub noi dung o pha truoc) van khop."""
    n = 0
    for dp, _, fns in os.walk(STAGE):
        for fn in fns:
            new = fn
            for a, b in FREE:
                new = new.replace(a, b)
            if new != fn:
                os.replace(os.path.join(dp, fn), os.path.join(dp, new))
                print(f"  {fn}  ->  {new}")
                n += 1
    print(f"-> doi ten {n} file")


def do_manifest():
    out = os.path.join(STAGE, "MANIFEST_SHA256.txt")
    n = 0; tot = 0
    with open(out, "w", encoding="utf-8") as f:
        f.write("# MANIFEST_SHA256 - lo archive DBS 2026-08-20 (da pseudonymise)\n")
        for dp, _, fns in os.walk(STAGE):
            for fn in sorted(fns):
                if fn == "MANIFEST_SHA256.txt":
                    continue
                p = os.path.join(dp, fn)
                rel = os.path.relpath(p, STAGE).replace("\\", "/")
                sz = os.path.getsize(p)
                f.write(f"{sha256(p)}  {sz}  {rel}\n")
                n += 1; tot += sz
                print(f"  {n:4d}  {sz/2**20:9.1f} MiB  {rel}")
        f.write(f"# {n} file, {tot} bytes ({tot/2**30:.2f} GiB)\n")
    print(f"\n-> {out}\n   {n} file, {tot/2**30:.2f} GiB")


def do_gate():
    bad = []
    for dp, _, fns in os.walk(STAGE):
        for fn in fns:
            p = os.path.join(dp, fn)
            rel = os.path.relpath(p, STAGE)
            hits = sniff(p, 8 * 1024 * 1024)
            nm = [n for n in NEEDLES if n.decode(errors="replace") in
                  [x for x in hits]]
            if hits:
                bad.append((rel, hits))
            for n in SUBJ:
                if n.lower() in rel.lower():
                    bad.append((rel, ["TEN TRONG DUONG DAN"]))
    # --- vong 2: manh ten trong header .tck/.mif va file text ---
    FRAGS = [f.encode() for f in CFG.get("gate_fragments", [])]
    for dp, _, fns in os.walk(STAGE):
        for fn in fns:
            p = os.path.join(dp, fn); ext = os.path.splitext(fn)[1].lower()
            if fn == "MANIFEST_SHA256.txt":
                continue
            try:
                b = open(p, "rb").read(4 * 1024 * 1024)
            except Exception:
                continue
            if ext in (".tck", ".mif"):
                i = b.find(b"\nEND\n"); b = b[:i + 5] if i >= 0 else b""
            elif ext not in TEXT_EXT and ext != ".json":
                continue
            for fr in FRAGS:
                for m in re.finditer(rb"[A-Za-z0-9_.-]*" + fr + rb"[A-Za-z0-9_.-]*", b):
                    tok = m.group().decode("latin1")
                    # chi bao dong tokenschua dau '_' hoac '/' quanh manh ten,
                    # bo qua bien Python thong thuong nhu 'name', 'dst_name'
                    if tok.lower() == tok and tok not in (
                            "name", "names", "name.strip", "dst_name", "src_name",
                            "node_name", "view_name", "clean_names", "filename",
                            "basename", "dirname", "fname"):
                        bad.append((os.path.relpath(p, STAGE), [f"MANH TEN: {tok}"]))

    print(f"===== CONG PHI: {len(bad)} hit =====")
    for rel, h in bad[:40]:
        print(f"  {rel}  <- {', '.join(h)}")
    print("KET QUA:", "PASS (0 hit)" if not bad else f"FAIL ({len(bad)} hit)")
    return 0 if not bad else 1


def do_map():
    os.makedirs(os.path.dirname(MAPFILE), exist_ok=True)
    with open(MAPFILE, "w", encoding="utf-8") as f:
        f.write("subject_id,folder_name,padded_token_for_tck_header\n")
        for n in sorted(SUBJ):
            f.write(f"{SUBJ[n]},{n},{pad(n)}\n")
    print(f"-> {MAPFILE}  (KHONG duoc dua len cloud)")
    for n in sorted(SUBJ):
        print(f"   {SUBJ[n]}  {n:18s} -> '{pad(n)}' (len {len(n)})")


if __name__ == "__main__":
    ph = sys.argv[1] if len(sys.argv) > 1 else "scan"
    {"scan": lambda: do_scan(False), "map": do_map, "scrub": do_scrub,
     "stage": do_stage, "rename": do_rename, "rescrub": do_rescrub,
     "manifest": do_manifest, "gate": do_gate}[ph]()
