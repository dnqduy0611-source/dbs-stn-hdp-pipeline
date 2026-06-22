from __future__ import annotations

from pathlib import Path


FILES = [
    Path(r"D:\dbs_workspace\dbs_stn_hdp_pipeline\patients\patient_example\derivatives\lead_dbs\sub-PatientExample_B33005_contacts_from_diode.dwi.vtk"),
    Path(r"D:\dbs_workspace\dbs_stn_hdp_pipeline\patients\patient_example\derivatives\tracts\official\L_CST_current_official.vtk"),
    Path(r"D:\dbs_workspace\dbs_stn_hdp_pipeline\patients\patient_example\derivatives\tracts\official\L_DRTT_iFOD2_strictEndpoint_nodil_to_LPG_official_max160.vtk"),
    Path(r"D:\dbs_workspace\dbs_stn_hdp_pipeline\patients\patient_example\derivatives\tracts\official\L_HDP_refined_official_max80.vtk"),
]


def add_header(path: Path) -> Path:
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(text) < 4 or not text[0].startswith("# vtk DataFile"):
        raise ValueError(f"Not an ASCII legacy VTK file: {path}")
    if any(line.strip() == "SPACE=RAS" for line in text[:8]):
        out_path = path
    else:
        text.insert(2, "SPACE=RAS")
        out_path = path.with_name(path.stem + "_slicerRAS.vtk")
        out_path.write_text("\n".join(text) + "\n", encoding="utf-8")
    return out_path


def main() -> None:
    for path in FILES:
        out = add_header(path)
        print(out)


if __name__ == "__main__":
    main()
