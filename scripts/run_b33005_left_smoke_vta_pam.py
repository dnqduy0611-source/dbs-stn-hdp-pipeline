from __future__ import annotations

import csv
import copy
import json
import os
import platform
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np


ROOT = Path(r"D:\dbs_workspace")
PATIENT_DIR = ROOT / "dbs_stn_hdp_pipeline" / "patients" / "patient_example"
LEAD_DIR = PATIENT_DIR / "derivatives" / "lead_dbs"
TRACT_DIR = PATIENT_DIR / "derivatives" / "tracts" / "official"
OUT_DIR = PATIENT_DIR / "derivatives" / "ossdbs" / "smoke_left_LK1A_current_angle"

CONTACTS_CSV = LEAD_DIR / "sub-PatientExample_B33005_contacts_from_diode.csv"
MRI_PATH = ROOT / "exam_nifti" / "brain_segmented.nii.gz"
DTI_PATH = ROOT / "exam_nifti" / "dti_tensor.nii.gz"

TRACTS = {
    "L_CST": TRACT_DIR / "L_CST_current_official.tck",
    "L_DRTT": TRACT_DIR / "L_DRTT_iFOD2_strictEndpoint_nodil_to_LPG_official_max160.tck",
    "L_HDP": TRACT_DIR / "L_HDP_refined_official_max80.vtk",
}

ACTIVE_CONTACT = "L_K1A"
ACTIVE_CONTACT_ID = 2
RETURN_CONTACT_ID = 1
CURRENT_A = 0.001
PULSE_WIDTH_US = 60.0
FREQUENCY_HZ = 130.0
AXON_DIAM_UM = 5.7
N_RANVIER = 21
N_POINTS = 11 * (N_RANVIER - 1) + 1
MAX_FIBERS_PER_TRACT = 10
TRUNC_MM = 14.0


def configure_windows_neuron() -> None:
    if platform.system() != "Windows":
        return
    nrn_bin = Path(r"C:\nrn\bin")
    os.environ["NEURONHOME"] = r"C:\nrn"
    os.environ["PATH"] = str(nrn_bin) + ";" + os.environ.get("PATH", "")
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    if nrn_bin.exists():
        os.add_dll_directory(str(nrn_bin))


def patch_neuron_compile_for_windows() -> None:
    if platform.system() != "Windows":
        return
    import neuron as nrn
    from ossdbs.axon_processing.neuron_model import NeuronSimulator

    def patched_compile(self):
        try:
            subprocess.run(
                self.neuron_executable,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=os.path.abspath(self._neuron_workdir),
                shell=True,
                check=False,
            )
        except Exception:
            pass
        try:
            nrn.load_mechanisms(self._neuron_workdir)
        except RuntimeError:
            pass

    NeuronSimulator.compile_neuron_files = patched_compile


def load_contacts() -> dict[str, dict[str, float | str]]:
    with CONTACTS_CSV.open(newline="", encoding="utf-8") as fp:
        return {row["contact_name"]: row for row in csv.DictReader(fp)}


def dwi_point(row: dict[str, float | str]) -> np.ndarray:
    return np.array(
        [float(row["dwi_ras_x"]), float(row["dwi_ras_y"]), float(row["dwi_ras_z"])],
        dtype=float,
    )


def read_tck(path: Path) -> list[np.ndarray]:
    fibers: list[np.ndarray] = []
    current: list[list[float]] = []
    with path.open("rb") as fp:
        header = {}
        while True:
            line = fp.readline().decode("latin-1").strip()
            if line == "END":
                break
            if ":" in line:
                key, value = line.split(":", 1)
                header[key.strip()] = value.strip()
        fp.seek(int(header.get("file", ". 0").split()[-1]))
        while True:
            raw = fp.read(12)
            if len(raw) < 12:
                break
            x, y, z = struct.unpack("<fff", raw)
            if np.isinf(x):
                if current:
                    fibers.append(np.asarray(current, dtype=float))
                break
            if np.isnan(x):
                if current:
                    fibers.append(np.asarray(current, dtype=float))
                    current = []
            else:
                current.append([x, y, z])
    return fibers


def read_ascii_vtk_polyline(path: Path) -> list[np.ndarray]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    points: list[list[float]] = []
    line_cells: list[list[int]] = []
    i = 0
    while i < len(lines):
        parts = lines[i].split()
        if parts[:1] == ["POINTS"]:
            n_points = int(parts[1])
            i += 1
            values: list[float] = []
            while len(values) < n_points * 3:
                values.extend(float(v) for v in lines[i].split())
                i += 1
            points = np.asarray(values, dtype=float).reshape(-1, 3).tolist()
            continue
        if parts[:1] == ["LINES"]:
            n_lines = int(parts[1])
            i += 1
            for _ in range(n_lines):
                cell = [int(v) for v in lines[i].split()]
                line_cells.append(cell[1:])
                i += 1
            continue
        i += 1
    point_array = np.asarray(points, dtype=float)
    return [point_array[idx, :] for idx in line_cells if len(idx) >= 2]


def load_streamlines(path: Path) -> list[np.ndarray]:
    if path.suffix.lower() == ".tck":
        return read_tck(path)
    if path.suffix.lower() == ".vtk":
        return read_ascii_vtk_polyline(path)
    raise ValueError(f"Unsupported tract file: {path}")


def trim_and_resample(streamline: np.ndarray, center: np.ndarray) -> np.ndarray | None:
    if len(streamline) < 5:
        return None
    seg = np.linalg.norm(np.diff(streamline, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    if arc[-1] <= 0:
        return None
    closest = int(np.argmin(np.linalg.norm(streamline - center, axis=1)))
    mid = arc[closest]
    keep = (arc >= mid - TRUNC_MM / 2.0) & (arc <= mid + TRUNC_MM / 2.0)
    trimmed = streamline[keep]
    if len(trimmed) < 5:
        return None
    seg = np.linalg.norm(np.diff(trimmed, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    if arc[-1] <= 0:
        return None
    target = np.linspace(0.0, arc[-1], N_POINTS)
    out = np.zeros((N_POINTS, 3), dtype=float)
    for axis in range(3):
        out[:, axis] = np.interp(target, arc, trimmed[:, axis])
    return out


def build_pathway_h5(active_point: np.ndarray) -> tuple[Path, Path, dict[str, int]]:
    h5_path = OUT_DIR / "pathways_left_smoke.h5"
    json_path = OUT_DIR / "pathways_left_smoke.json"
    counts: dict[str, int] = {}
    with h5py.File(h5_path, "w") as h5:
        for name, path in TRACTS.items():
            raw = load_streamlines(path)
            raw.sort(key=lambda fiber: float(np.min(np.linalg.norm(fiber - active_point, axis=1))))
            selected: list[np.ndarray] = []
            for fiber in raw:
                resampled = trim_and_resample(fiber, active_point)
                if resampled is not None:
                    selected.append(resampled)
                if len(selected) >= MAX_FIBERS_PER_TRACT:
                    break
            if not selected:
                raise RuntimeError(f"No usable fibers near {ACTIVE_CONTACT} for {name}")
            group = h5.create_group(name)
            for idx, fiber in enumerate(selected):
                group.create_dataset(f"axon{idx}", data=fiber.astype(np.float64))
            counts[name] = len(selected)
    payload = {
        "Axon_Model_Type": "MRG2002",
        "connectome_name": "PatientExample_Left_Official_Tracts",
        "axon_diams": [AXON_DIAM_UM for _ in counts],
        "n_Ranvier": [N_RANVIER for _ in counts],
        "N_seeded_neurons": [counts[name] for name in counts],
        "N_orig_neurons": [counts[name] for name in counts],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return h5_path, json_path, counts


def make_contacts() -> list[dict]:
    contacts = []
    for contact_id in range(1, 9):
        contact = {
            "Contact_ID": contact_id,
            "Active": False,
            "Current[A]": 0.0,
            "Voltage[V]": 0.0,
            "Floating": True,
        }
        if contact_id == RETURN_CONTACT_ID:
            contact.update({"Active": True, "Floating": False, "Current[A]": CURRENT_A, "Voltage[V]": 0.0})
        elif contact_id == ACTIVE_CONTACT_ID:
            contact.update({"Active": True, "Floating": False, "Current[A]": -CURRENT_A, "Voltage[V]": 1.0})
        contacts.append(contact)
    return contacts


def build_settings(h5_path: Path, json_path: Path, contacts: dict[str, dict[str, float | str]]) -> dict:
    k0 = dwi_point(contacts["L_K0"])
    k3 = dwi_point(contacts["L_K3"])
    active = dwi_point(contacts[ACTIVE_CONTACT])
    axis = k3 - k0
    axis = axis / np.linalg.norm(axis)
    tip = k0 - axis * 1.65
    rotation = float(contacts[ACTIVE_CONTACT]["diode_roll_y_deg"])
    center = active
    settings = {
        "Electrodes": [
            {
                "Name": "MedtronicSenSightB33005",
                "Rotation[Degrees]": rotation,
                "Direction": {"x[mm]": float(axis[0]), "y[mm]": float(axis[1]), "z[mm]": float(axis[2])},
                "TipPosition": {"x[mm]": float(tip[0]), "y[mm]": float(tip[1]), "z[mm]": float(tip[2])},
                "Contacts": make_contacts(),
            }
        ],
        "BrainRegion": {
            "Center": {"x[mm]": float(center[0]), "y[mm]": float(center[1]), "z[mm]": float(center[2])},
            "Dimension": {"x[mm]": 34.0, "y[mm]": 34.0, "z[mm]": 34.0},
            "Shape": "Ellipsoid",
        },
        "MaterialDistribution": {
            "MRIPath": str(MRI_PATH),
            "DiffusionTensorActive": True,
            "DTIPath": str(DTI_PATH),
            "WMMasking": True,
        },
        "StimulationSignal": {
            "Type": "Rectangle",
            "Frequency[Hz]": FREQUENCY_HZ,
            "PulseWidth[us]": PULSE_WIDTH_US,
            "SpectrumMode": "FullSpectrum",
            "CutoffFrequency": 20000.0,
            "CurrentControlled": True,
        },
        "DielectricModel": {"Type": "ColeCole4"},
        "EQSMode": False,
        "FEMOrder": 2,
        "Solver": {"Type": "CG", "Preconditioner": "local", "MaximumSteps": 10000, "Precision": 1e-10},
        "Mesh": {
            "MeshingHypothesis": {"Type": "Default", "MaxMeshSize": 2.0},
            "MaterialRefinementSteps": 0,
            "HPRefinement": {"Active": False},
            "AdaptiveMeshRefinement": {"Active": False},
        },
        "PointModel": {
            "Pathway": {"Active": True, "FileName": str(h5_path), "ExportField": True},
            "Lattice": {
                "Active": False,
                "Center": {"x[mm]": float(center[0]), "y[mm]": float(center[1]), "z[mm]": float(center[2])},
                "Shape": {"x": 18, "y": 18, "z": 18},
                "Direction": {"x[mm]": float(axis[0]), "y[mm]": float(axis[1]), "z[mm]": float(axis[2])},
                "PointDistance[mm]": 0.5,
                "CollapseVTA": False,
                "ExportField": True,
            },
        },
        "OutputPath": str(OUT_DIR / "ossdbs_run"),
        "StimulationFolder": str(OUT_DIR),
        "ExportVTK": False,
        "ExportElectrode": True,
        "ComputeImpedance": True,
        "ActivationThresholdVTA[V-per-m]": None,
        "PathwayFile": str(json_path),
        "CalcAxonActivation": True,
        "StimSets": {"Active": False},
        "FailFlag": "b33005_left_smoke",
    }
    return settings


def build_vta_settings(settings: dict) -> dict:
    vta = copy.deepcopy(settings)
    vta["OutputPath"] = str(OUT_DIR / "ossdbs_vta_1V")
    vta["FailFlag"] = "b33005_left_vta"
    vta["PathwayFile"] = None
    vta["CalcAxonActivation"] = False
    vta["PointModel"]["Pathway"]["Active"] = False
    vta["PointModel"]["Lattice"]["Active"] = True
    vta["PointModel"]["Lattice"]["ExportField"] = True
    vta["ActivationThresholdVTA[V-per-m]"] = 200.0
    vta["StimulationSignal"]["CurrentControlled"] = False
    for contact in vta["Electrodes"][0]["Contacts"]:
        contact["Current[A]"] = 0.0
        if contact["Contact_ID"] == RETURN_CONTACT_ID:
            contact.update({"Active": True, "Floating": False, "Voltage[V]": 0.0})
        elif contact["Contact_ID"] == ACTIVE_CONTACT_ID:
            contact.update({"Active": True, "Floating": False, "Voltage[V]": -1.0})
        else:
            contact.update({"Active": False, "Floating": True, "Voltage[V]": 0.0})
    return vta


def summarize_outputs(settings: dict, counts: dict[str, int]) -> None:
    output = Path(settings["OutputPath"])
    status_files = sorted(output.glob("Pathway_status*.json"))
    csv_files = sorted(output.glob("Axon_state*.csv"))
    summary = {
        "active_contact": ACTIVE_CONTACT,
        "active_contact_id_proxy": ACTIVE_CONTACT_ID,
        "return_contact_id": RETURN_CONTACT_ID,
        "current_A": CURRENT_A,
        "pulse_width_us": PULSE_WIDTH_US,
        "frequency_Hz": FREQUENCY_HZ,
        "tract_counts": counts,
        "output_path": str(output),
        "pathway_status_files": [str(p) for p in status_files],
        "axon_state_csv_files": [str(p) for p in csv_files],
        "settings_json": str(OUT_DIR / "ossdbs_settings.json"),
    }
    (OUT_DIR / "smoke_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    configure_windows_neuron()
    import ossdbs
    from ossdbs.api import run_PAM
    from ossdbs.main import main_run

    patch_neuron_compile_for_windows()
    ossdbs.set_logger()
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)

    contacts = load_contacts()
    active_point = dwi_point(contacts[ACTIVE_CONTACT])
    h5_path, json_path, counts = build_pathway_h5(active_point)
    settings = build_settings(h5_path, json_path, contacts)
    vta_settings = build_vta_settings(settings)
    settings_path = OUT_DIR / "ossdbs_settings.json"
    vta_settings_path = OUT_DIR / "ossdbs_vta_1V_settings.json"
    settings_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    vta_settings_path.write_text(json.dumps(vta_settings, indent=2), encoding="utf-8")

    print(f"Settings: {settings_path}")
    print(f"Pathway H5: {h5_path}")
    print(f"Pathway JSON: {json_path}")
    print(f"Fiber counts: {counts}")

    start = time.time()
    main_run(settings)
    settings_for_pam = dict(settings)
    settings_for_pam["Scaling"] = 1.0
    settings_for_pam["ScalingIndex"] = None
    settings_for_pam["CurrentVector"] = None
    run_PAM(settings_for_pam)
    main_run(vta_settings)
    print(f"Elapsed seconds: {time.time() - start:.1f}")
    summarize_outputs(settings, counts)


if __name__ == "__main__":
    sys.exit(main())
