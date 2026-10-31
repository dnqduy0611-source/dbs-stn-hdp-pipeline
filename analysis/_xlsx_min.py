"""Doc .xlsx bang THU VIEN CHUAN (openpyxl khong con trong may 28/08).

Chi doc, khong ghi. Tra ve list[list] gia tri o theo dung thu tu cot A,B,C...
So duoc ep ve float; chuoi giu nguyen; o rong = None.
"""
import re
import zipfile
from xml.etree import ElementTree as ET

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
RNS = "{http://schemas.openxmlformats.org/package/2006/relationships}"


def _col_idx(ref):
    m = re.match(r"([A-Z]+)", ref)
    n = 0
    for ch in m.group(1):
        n = n * 26 + (ord(ch) - 64)
    return n - 1


def read_sheet(path, sheet_name):
    with zipfile.ZipFile(path) as z:
        wb = ET.fromstring(z.read("xl/workbook.xml"))
        rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        tgt = {r.get("Id"): r.get("Target") for r in rels}
        target = None
        for s in wb.iter(NS + "sheet"):
            if s.get("name") == sheet_name:
                target = tgt[s.get(
                    "{http://schemas.openxmlformats.org/officeDocument/"
                    "2006/relationships}id")]
        if target is None:
            raise KeyError(sheet_name)
        if not target.startswith("xl/"):
            target = "xl/" + target.lstrip("/")

        sst = []
        if "xl/sharedStrings.xml" in z.namelist():
            ss = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in ss.iter(NS + "si"):
                sst.append("".join(t.text or "" for t in si.iter(NS + "t")))

        sh = ET.fromstring(z.read(target))

    rows = []
    for r in sh.iter(NS + "row"):
        cells = {}
        for c in r.iter(NS + "c"):
            ref, typ = c.get("r"), c.get("t")
            v = c.find(NS + "v")
            isel = c.find(NS + "is")
            if typ == "s" and v is not None:
                val = sst[int(v.text)]
            elif typ == "inlineStr" and isel is not None:
                val = "".join(t.text or "" for t in isel.iter(NS + "t"))
            elif v is None:
                val = None
            else:
                try:
                    val = float(v.text)
                    if val == int(val):
                        val = int(val)
                except (TypeError, ValueError):
                    val = v.text
            if val == "":
                val = None
            cells[_col_idx(ref)] = val
        width = (max(cells) + 1) if cells else 0
        rows.append([cells.get(i) for i in range(width)])
    return rows


def sheet_names(path):
    with zipfile.ZipFile(path) as z:
        wb = ET.fromstring(z.read("xl/workbook.xml"))
        return [s.get("name") for s in wb.iter(NS + "sheet")]
