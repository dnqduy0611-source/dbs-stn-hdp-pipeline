"""Nap khai bao du lieu UPDRS ma khong khoi tao/ghi workbook.

``build_updrs_workbook.py`` vua la nguon du lieu vua la script tao Excel. Cac
phan tich chi can phan khai bao truoc khoi ``style``; loader nay cat dung ranh
gioi do va bo cac import openpyxl de phan tich chi-doc chay doc lap.
"""
from pathlib import Path


def load_updrs_declarations(source: Path):
    text = Path(source).read_text(encoding="utf-8")
    marker = "# ---------------------------------------------------------------- style"
    if marker not in text:
        raise RuntimeError(f"Khong tim thay moc du lieu/style trong {source}")

    declarations = text[: text.index(marker)]
    declarations = "\n".join(
        line for line in declarations.splitlines()
        if not line.startswith("from openpyxl")
    )
    def get_column_letter(n):
        out = ""
        while n:
            n, rem = divmod(n - 1, 26)
            out = chr(65 + rem) + out
        return out

    ns = {"get_column_letter": get_column_letter}
    exec(compile(declarations, str(source), "exec"), ns)
    return ns
