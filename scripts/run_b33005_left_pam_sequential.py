from __future__ import annotations

import argparse
import csv
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
import numpy as np


ROOT = Path(r"D:\dbs_workspace")
PATIENT_DIR = ROOT / "dbs_stn_hdp_pipeline" / "patients" / "patient_example"
LEAD_DIR = PATIENT_DIR / "derivatives" / "lead_dbs"
TRACT_DIR = PATIENT_DIR / "derivatives" / "tracts" / "official"
OUT_DIR = PATIENT_DIR / "derivatives" / "ossdbs" / "pam_left_LK1A_current_1mA_sequential"

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


def patch_neuron_for_windows_and_sequential() -> None:
    import neuron as nrn
    from ossdbs.axon_processing.neuron_model import NeuronSimulator
    from ossdbs.axon_processing.utilities import (
        create_leaddbs_outputs,
        create_paraview_outputs,
        store_axon_statuses,
    )

    if platform.system() == "Windows":

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
            except Exception as exc:
                print(f"WARNING: mknrndll failed: {exc}")
            try:
                nrn.load_mechanisms(self._neuron_workdir)
            except RuntimeError:
                pass

        NeuronSimulator.compile_neuron_files = patched_compile

    def patched_run_neuron(self, v_ext: np.ndarray, extra_initialization: bool = False) -> bool:
        neuron_home = os.environ.get("NEURONHOME", "")
        if neuron_home:
            hoc_lib = Path(neuron_home) / "share" / "nrn" / "lib" / "hoc"
            if hoc_lib.exists():
                try:
                    nrn.h.load_file(str((hoc_lib / "stdlib.hoc")).replace("\\", "/"))
                    nrn.h.load_file(str((hoc_lib / "stdrun.hoc")).replace("\\", "/"))
                except Exception:
                    pass

        hoc_path = str(Path(self._neuron_workdir) / self.hoc_file).replace("\\", "/")
        ok = nrn.h.load_file(hoc_path)
        if not ok:
            raise RuntimeError(f"NEURON failed to load hoc file: {hoc_path}")

        if extra_initialization:
            for proc_name in ("deletenodes", "createnodes", "dependent_var", "initialize"):
                try:
                    getattr(nrn.h, proc_name)()
                except (AttributeError, RuntimeError) as exc:
                    print(f"WARNING: NEURON {proc_name} failed: {exc}")

        if not hasattr(nrn.h, "setupAPWatcher_0") or not hasattr(nrn.h, "setupAPWatcher_1"):
            raise RuntimeError(f"NEURON hoc file loaded but AP watcher procedures are missing: {hoc_path}")

        nrn.h.setupAPWatcher_0()
        nrn.h.setupAPWatcher_1()

        time_step = self.signal_dict["time_step"]
        run_time = self.signal_dict["N_time_steps"] * self.signal_dict["time_step"]
        nrn.h.dt = time_step
        nrn.h.tstop = run_time
        nrn.h.n_pulse = 1
        nrn.h.v_init = self._v_init

        for i in range(v_ext.shape[0]):
            nrn.h.wf[i] = nrn.h.Vector(v_ext[i, :])

        nrn.h.stimul()
        nrn.h.run()
        return bool(nrn.h.stoprun)

    NeuronSimulator.run_NEURON = patched_run_neuron

    def sequential_check_pathway_activation(
        self, pathway_dataset, pathway_idx, pathway_name=None, scaling_index=None
    ):
        n_neurons = self.get_N_seeded_neurons(pathway_idx)
        axon_diam = self.get_axon_diam(pathway_idx)
        n_ranvier = self.get_n_Ranvier(pathway_idx)
        orig_n_neurons = self.get_N_orig_neurons(pathway_idx)

        axon_morphology = self._ax_morph_class()
        axon_morphology.update_axon_morphology(axon_diam, n_Ranvier=n_ranvier)
        n_segments_actual = axon_morphology.get_n_segments(self.downsampled)
        steps_per_ms = int(1.0 / self.signal_dict["time_step"])
        hoc_signature = (n_ranvier, steps_per_ms, float(axon_diam), bool(self.downsampled))
        if getattr(self, "_pam_hoc_signature", None) != hoc_signature:
            self.modify_hoc_file(n_ranvier, steps_per_ms, axon_morphology)
            self._pam_hoc_signature = hoc_signature

        pre_status = pathway_dataset["Status"]
        axon_leaddbs = np.zeros((n_neurons * n_segments_actual, 5), float)
        activated = []
        not_activated = []

        print(f"PAM sequential pathway={pathway_name}, axons={n_neurons}")
        for neuron_index in range(n_neurons):
            neuron = pathway_dataset["axon" + str(neuron_index)]
            start = neuron_index * n_segments_actual
            stop = (neuron_index + 1) * n_segments_actual
            axon_leaddbs[start:stop, :3] = np.array(neuron["Points[mm]"])
            axon_leaddbs[start:stop, 3] = neuron_index + 1

            if pre_status[neuron_index] != 0:
                axon_leaddbs[start:stop, 4] = pre_status[neuron_index]
                continue

            neuron_time_sol = np.array(neuron["Potential[V]"])
            if self.downsampled:
                neuron_time_sol = self.upsample_voltage(
                    neuron_time_sol, axon_diam, axon_morphology
                )
            spike = self.run_NEURON(self.get_v_ext(neuron_time_sol), self._extra_initialization)
            if spike:
                activated.append(neuron_index)
                axon_leaddbs[start:stop, 4] = 1
            else:
                not_activated.append(neuron_index)
                axon_leaddbs[start:stop, 4] = 0
            if (neuron_index + 1) % 10 == 0 or neuron_index + 1 == n_neurons:
                print(
                    f"  {pathway_name}: {neuron_index + 1}/{n_neurons} "
                    f"done, activated={len(activated)}"
                )

        create_leaddbs_outputs(
            self.output_path,
            axon_leaddbs,
            self.connectome_name,
            scaling_index=scaling_index,
            pathway_name=pathway_name,
        )
        if scaling_index is None:
            create_paraview_outputs(
                self.output_path,
                axon_leaddbs,
                scaling_index=scaling_index,
                pathway_name=pathway_name,
            )

        percent_activated = np.round(100.0 * len(activated) / float(orig_n_neurons), 2)
        percent_damaged = np.round(
            100.0 * np.sum(np.isclose(pre_status, -1.0)) / float(orig_n_neurons), 2
        )
        percent_csf = np.round(
            100.0 * np.sum(np.isclose(pre_status, -2.0)) / float(orig_n_neurons), 2
        )
        store_axon_statuses(
            self.output_path,
            percent_activated,
            percent_damaged,
            percent_csf,
            scaling_index=scaling_index,
            pathway_name=pathway_name,
        )

    NeuronSimulator.check_pathway_activation = sequential_check_pathway_activation


def load_contacts() -> dict[str, dict[str, str]]:
    with CONTACTS_CSV.open(newline="", encoding="utf-8") as fp:
        return {row["contact_name"]: row for row in csv.DictReader(fp)}


def dwi_point(row: dict[str, str]) -> np.ndarray:
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


def build_pathway_h5(active_point: np.ndarray, max_fibers_per_tract: int | None) -> tuple[Path, Path, dict[str, int]]:
    h5_path = OUT_DIR / "pathways_left_LK1A_official.h5"
    json_path = OUT_DIR / "pathways_left_LK1A_official.json"
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
                if max_fibers_per_tract is not None and len(selected) >= max_fibers_per_tract:
                    break
            if not selected:
                raise RuntimeError(f"No usable fibers near {ACTIVE_CONTACT} for {name}")
            group = h5.create_group(name)
            for idx, fiber in enumerate(selected):
                group.create_dataset(f"axon{idx}", data=fiber.astype(np.float64))
            counts[name] = len(selected)
    payload = {
        "Axon_Model_Type": "MRG2002",
        "connectome_name": "PatientExample_Left_Official_Tracts_LK1A",
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


def build_settings(h5_path: Path, json_path: Path, contacts: dict[str, dict[str, str]]) -> dict:
    k0 = dwi_point(contacts["L_K0"])
    k3 = dwi_point(contacts["L_K3"])
    active = dwi_point(contacts[ACTIVE_CONTACT])
    axis = k3 - k0
    axis = axis / np.linalg.norm(axis)
    tip = k0 - axis * 1.65
    rotation = float(contacts[ACTIVE_CONTACT]["diode_roll_y_deg"])
    return {
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
            "Center": {"x[mm]": float(active[0]), "y[mm]": float(active[1]), "z[mm]": float(active[2])},
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
            "Lattice": {"Active": False},
            "VoxelLattice": {"Active": False},
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
        "FailFlag": "b33005_left_LK1A_pam_seq",
    }


def collect_status(settings: dict, counts: dict[str, int], started: float) -> dict:
    output = Path(settings["OutputPath"])
    status = {}
    for path in sorted(output.glob("Pathway_status*.json")):
        status[path.stem.replace("Pathway_status_", "")] = json.loads(path.read_text(encoding="utf-8"))
    summary = {
        "active_contact": ACTIVE_CONTACT,
        "active_contact_id_proxy": ACTIVE_CONTACT_ID,
        "return_contact_id": RETURN_CONTACT_ID,
        "current_A": CURRENT_A,
        "pulse_width_us": PULSE_WIDTH_US,
        "frequency_Hz": FREQUENCY_HZ,
        "axon_diam_um": AXON_DIAM_UM,
        "n_ranvier": N_RANVIER,
        "trunc_mm": TRUNC_MM,
        "tract_counts": counts,
        "status": status,
        "output_path": str(output),
        "elapsed_s": time.time() - started,
    }
    (OUT_DIR / "pam_sequential_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-fibers-per-tract", type=int, default=50)
    parser.add_argument("--keep-existing", action="store_true")
    parser.add_argument("--skip-fem", action="store_true")
    args = parser.parse_args()

    configure_windows_neuron()
    import ossdbs
    from ossdbs.api import run_PAM
    from ossdbs.main import main_run

    patch_neuron_for_windows_and_sequential()
    ossdbs.set_logger()

    if OUT_DIR.exists() and not args.keep_existing:
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    contacts = load_contacts()
    active_point = dwi_point(contacts[ACTIVE_CONTACT])
    max_fibers = None if args.max_fibers_per_tract <= 0 else args.max_fibers_per_tract
    h5_path, json_path, counts = build_pathway_h5(active_point, max_fibers)
    settings = build_settings(h5_path, json_path, contacts)
    settings_path = OUT_DIR / "ossdbs_pam_settings.json"
    settings_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    print(f"Settings: {settings_path}")
    print(f"Pathway H5: {h5_path}")
    print(f"Pathway JSON: {json_path}")
    print(f"Fiber counts: {counts}")

    started = time.time()
    if not args.skip_fem:
        main_run(settings)

    settings_for_pam = dict(settings)
    settings_for_pam["Scaling"] = 1.0
    settings_for_pam["ScalingIndex"] = None
    settings_for_pam["CurrentVector"] = None
    run_PAM(settings_for_pam)
    summary = collect_status(settings, counts, started)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
