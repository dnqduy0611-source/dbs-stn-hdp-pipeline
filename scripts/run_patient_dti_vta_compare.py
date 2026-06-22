from __future__ import annotations

import csv
import json
import os
import platform
import shutil
import time
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import pandas as pd
from dipy.core.gradients import gradient_table
from dipy.io import read_bvals_bvecs
from dipy.reconst.dti import TensorModel


ROOT = Path(r"D:\dbs_workspace")
PATIENT_DIR = ROOT / "dbs_stn_hdp_pipeline" / "patients" / "patient_example"
OUT_DIR = PATIENT_DIR / "derivatives" / "ossdbs" / "vta_postprocess_left_LK1A_current_1mA_patient_dti"
OLD_SUMMARY = (
    PATIENT_DIR
    / "derivatives"
    / "ossdbs"
    / "vta_postprocess_left_LK1A_current_1mA"
    / "L_K1A_current_1mA_vta_postprocess_summary.json"
)

RAW_DWI = PATIENT_DIR / "raw" / "nifti" / "DTI_ep2d_diff_mddw_30_csd_7.nii.gz"
BVAL = PATIENT_DIR / "raw" / "nifti" / "DTI_ep2d_diff_mddw_30_csd_7.bval"
BVEC = PATIENT_DIR / "raw" / "nifti" / "DTI_ep2d_diff_mddw_30_csd_7.bvec"
DWI_MASK = PATIENT_DIR / "derivatives" / "dwi" / "mask.nii.gz"
FIVE_TT_DWI = PATIENT_DIR / "derivatives" / "registration" / "5tt_dwi.nii.gz"
CONTACTS_CSV = PATIENT_DIR / "derivatives" / "lead_dbs" / "sub-PatientExample_B33005_contacts_from_diode.csv"

DTI_TENSOR = OUT_DIR / "dti_tensor_patient_raw_ossdbs.nii.gz"
FA_MAP = OUT_DIR / "dti_fa_patient_raw.nii.gz"
MATERIAL_MAP = OUT_DIR / "brain_segmented_from_5tt_dwi_ossdbs.nii.gz"

ACTIVE_CONTACT = "L_K1A"
ACTIVE_CONTACT_ID = 2
RETURN_CONTACT_ID = 1
CURRENT_A = 0.001
PULSE_WIDTH_US = 60.0
FREQUENCY_HZ = 130.0
GRID_SHAPE = (25, 25, 25)
GRID_SPACING_MM = 0.6
THRESHOLD_VM = 200.0


def configure_windows_neuron() -> None:
    if platform.system() != "Windows":
        return
    nrn_bin = Path(r"C:\nrn\bin")
    os.environ["NEURONHOME"] = r"C:\nrn"
    os.environ["PATH"] = str(nrn_bin) + ";" + os.environ.get("PATH", "")
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    if nrn_bin.exists():
        os.add_dll_directory(str(nrn_bin))


def load_contacts() -> dict[str, dict[str, str]]:
    with CONTACTS_CSV.open(newline="", encoding="utf-8") as fp:
        return {row["contact_name"]: row for row in csv.DictReader(fp)}


def dwi_point(row: dict[str, str]) -> np.ndarray:
    return np.array(
        [float(row["dwi_ras_x"]), float(row["dwi_ras_y"]), float(row["dwi_ras_z"])],
        dtype=float,
    )


def compute_patient_tensor() -> dict:
    img = nib.load(str(RAW_DWI))
    data = img.get_fdata(dtype=np.float32)
    bvals, bvecs = read_bvals_bvecs(str(BVAL), str(BVEC))
    gtab = gradient_table(bvals, bvecs)
    mask = nib.load(str(DWI_MASK)).get_fdata() > 0

    tenfit = TensorModel(gtab).fit(data, mask=mask)
    lower_tri = tenfit.lower_triangular()
    tensor_ossdbs = np.zeros_like(lower_tri, dtype=np.float32)
    tensor_ossdbs[..., 0] = lower_tri[..., 0]
    tensor_ossdbs[..., 1] = lower_tri[..., 1]
    tensor_ossdbs[..., 2] = lower_tri[..., 3]
    tensor_ossdbs[..., 3] = lower_tri[..., 2]
    tensor_ossdbs[..., 4] = lower_tri[..., 4]
    tensor_ossdbs[..., 5] = lower_tri[..., 5]
    tensor_ossdbs[~mask, :] = 0.0

    nib.save(nib.Nifti1Image(tensor_ossdbs, img.affine, img.header), str(DTI_TENSOR))
    fa = np.asarray(tenfit.fa, dtype=np.float32)
    fa[~np.isfinite(fa)] = 0.0
    fa[~mask] = 0.0
    nib.save(nib.Nifti1Image(fa, img.affine, img.header), str(FA_MAP))

    return {
        "raw_dwi": str(RAW_DWI),
        "bval": str(BVAL),
        "bvec": str(BVEC),
        "mask": str(DWI_MASK),
        "tensor": str(DTI_TENSOR),
        "fa": str(FA_MAP),
        "shape": list(tensor_ossdbs.shape),
        "voxel_size": [float(v) for v in img.header.get_zooms()[:3]],
        "b_values": [float(v) for v in np.unique(bvals)],
        "tensor_min": float(np.nanmin(tensor_ossdbs)),
        "tensor_max": float(np.nanmax(tensor_ossdbs)),
        "fa_mean_masked": float(fa[mask].mean()),
        "fa_max": float(fa.max()),
    }


def make_material_map_from_5tt() -> dict:
    img = nib.load(str(FIVE_TT_DWI))
    data = img.get_fdata(dtype=np.float32)
    if data.ndim != 4 or data.shape[3] < 4:
        raise ValueError(f"Expected 5TT image with shape (x,y,z,5), got {data.shape}")

    label = np.zeros(data.shape[:3], dtype=np.int16)
    argmax = np.argmax(data[..., :4], axis=3)
    maxval = np.max(data[..., :4], axis=3)
    in_brain = maxval > 0.05
    label[in_brain & ((argmax == 0) | (argmax == 1))] = 1  # gray matter
    label[in_brain & (argmax == 2)] = 2  # white matter
    label[in_brain & (argmax == 3)] = 3  # CSF

    out_img = nib.Nifti1Image(label, img.affine, img.header)
    out_img.header.set_data_dtype(np.int16)
    nib.save(out_img, str(MATERIAL_MAP))

    values, counts = np.unique(label, return_counts=True)
    return {
        "five_tt": str(FIVE_TT_DWI),
        "material_map": str(MATERIAL_MAP),
        "shape": list(label.shape),
        "voxel_size": [float(v) for v in img.header.get_zooms()[:3]],
        "labels": {int(v): int(c) for v, c in zip(values, counts)},
        "label_meaning": {"0": "Unknown/background", "1": "Gray matter", "2": "White matter", "3": "CSF"},
    }


def make_grid(center: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axes = [
        (np.arange(n, dtype=float) - (n - 1) / 2.0) * GRID_SPACING_MM
        for n in GRID_SHAPE
    ]
    xx, yy, zz = np.meshgrid(*axes, indexing="ij")
    offsets = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    points = center.reshape(1, 3) + offsets
    return points, offsets


def write_grid_pathway_h5(points: np.ndarray) -> Path:
    h5_path = OUT_DIR / "vta_grid_points_as_pathway.h5"
    with h5py.File(h5_path, "w") as h5:
        group = h5.create_group("VTA_GRID")
        for idx, point in enumerate(points):
            group.create_dataset(f"axon{idx}", data=point.reshape(1, 3).astype(np.float64))
    return h5_path


def make_contacts() -> list[dict]:
    contacts = []
    for contact_id in range(1, 9):
        item = {
            "Contact_ID": contact_id,
            "Active": False,
            "Current[A]": 0.0,
            "Voltage[V]": 0.0,
            "Floating": True,
        }
        if contact_id == RETURN_CONTACT_ID:
            item.update({"Active": True, "Floating": False, "Current[A]": CURRENT_A, "Voltage[V]": 0.0})
        elif contact_id == ACTIVE_CONTACT_ID:
            item.update({"Active": True, "Floating": False, "Current[A]": -CURRENT_A, "Voltage[V]": 1.0})
        contacts.append(item)
    return contacts


def build_settings(h5_path: Path, contacts: dict[str, dict[str, str]], center: np.ndarray) -> dict:
    k0 = dwi_point(contacts["L_K0"])
    k3 = dwi_point(contacts["L_K3"])
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
            "Center": {"x[mm]": float(center[0]), "y[mm]": float(center[1]), "z[mm]": float(center[2])},
            "Dimension": {"x[mm]": 34.0, "y[mm]": 34.0, "z[mm]": 34.0},
            "Shape": "Ellipsoid",
        },
        "MaterialDistribution": {
            "MRIPath": str(MATERIAL_MAP),
            "DiffusionTensorActive": True,
            "DTIPath": str(DTI_TENSOR),
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
        "OutputPath": str(OUT_DIR / "ossdbs_grid_field"),
        "StimulationFolder": str(OUT_DIR),
        "ExportVTK": False,
        "ExportElectrode": True,
        "ComputeImpedance": True,
        "ActivationThresholdVTA[V-per-m]": None,
        "PathwayFile": None,
        "CalcAxonActivation": False,
        "StimSets": {"Active": False},
        "FailFlag": "b33005_left_vta_patient_dti",
    }


def save_grid_outputs(field_csv: Path, center: np.ndarray, offsets: np.ndarray) -> dict:
    df = pd.read_csv(field_csv)
    valid = (df["inside_csf"].to_numpy() == 0) & (df["inside_encap"].to_numpy() == 0)
    efield_vm = np.zeros(np.prod(GRID_SHAPE), dtype=np.float32)
    coords = np.round(df[["x-pt", "y-pt", "z-pt"]].to_numpy(dtype=float), 6)
    all_points = np.round(center.reshape(1, 3) + offsets, 6)
    index = {tuple(point): i for i, point in enumerate(all_points)}
    for row_idx, point in enumerate(coords):
        grid_idx = index.get(tuple(point))
        if grid_idx is not None and valid[row_idx]:
            efield_vm[grid_idx] = float(df.iloc[row_idx]["magnitude"]) * 1000.0
    vta = (efield_vm >= THRESHOLD_VM).astype(np.uint8)
    efield_grid = efield_vm.reshape(GRID_SHAPE)
    vta_grid = vta.reshape(GRID_SHAPE)

    origin = center - ((np.array(GRID_SHAPE, dtype=float) - 1.0) / 2.0) * GRID_SPACING_MM
    affine = np.eye(4)
    affine[0, 0] = GRID_SPACING_MM
    affine[1, 1] = GRID_SPACING_MM
    affine[2, 2] = GRID_SPACING_MM
    affine[:3, 3] = origin

    efield_nii = OUT_DIR / "L_K1A_current_1mA_patient_dti_efield_magnitude_Vm.nii.gz"
    vta_nii = OUT_DIR / "L_K1A_current_1mA_patient_dti_VTA_threshold_200Vm.nii.gz"
    nib.save(nib.Nifti1Image(efield_grid, affine), str(efield_nii))
    nib.save(nib.Nifti1Image(vta_grid, affine), str(vta_nii))

    thresholded_csv = OUT_DIR / "L_K1A_current_1mA_patient_dti_vta_grid_thresholded.csv"
    out_df = pd.DataFrame(all_points, columns=["dwi_ras_x", "dwi_ras_y", "dwi_ras_z"])
    out_df["efield_Vm"] = efield_vm
    out_df["vta_200Vm"] = vta
    out_df.to_csv(thresholded_csv, index=False)

    voxel_volume = GRID_SPACING_MM**3
    return {
        "field_csv": str(field_csv),
        "thresholded_csv": str(thresholded_csv),
        "efield_nifti": str(efield_nii),
        "vta_nifti": str(vta_nii),
        "threshold_Vm": THRESHOLD_VM,
        "grid_shape": list(GRID_SHAPE),
        "grid_spacing_mm": GRID_SPACING_MM,
        "vta_voxels": int(vta.sum()),
        "vta_volume_mm3": float(vta.sum() * voxel_volume),
        "max_efield_Vm": float(efield_vm.max()),
        "mean_nonzero_efield_Vm": float(efield_vm[efield_vm > 0].mean()) if np.any(efield_vm > 0) else 0.0,
        "valid_exported_points": int(valid.sum()),
        "total_grid_points": int(np.prod(GRID_SHAPE)),
    }


def compare_to_old(new_summary: dict) -> dict:
    if not OLD_SUMMARY.exists():
        return {"old_summary": str(OLD_SUMMARY), "available": False}
    old = json.loads(OLD_SUMMARY.read_text(encoding="utf-8"))
    keys = ["vta_voxels", "vta_volume_mm3", "max_efield_Vm", "valid_exported_points"]
    return {
        "old_summary": str(OLD_SUMMARY),
        "available": True,
        "old": {key: old.get(key) for key in keys},
        "new": {key: new_summary.get(key) for key in keys},
        "delta": {
            key: (
                None
                if old.get(key) is None or new_summary.get(key) is None
                else float(new_summary[key]) - float(old[key])
            )
            for key in keys
        },
    }


def main() -> int:
    configure_windows_neuron()
    import ossdbs
    from ossdbs.main import main_run

    ossdbs.set_logger()
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)

    tensor_summary = compute_patient_tensor()
    material_summary = make_material_map_from_5tt()
    contacts = load_contacts()
    center = dwi_point(contacts[ACTIVE_CONTACT])
    points, offsets = make_grid(center)
    h5_path = write_grid_pathway_h5(points)
    settings = build_settings(h5_path, contacts, center)
    settings_path = OUT_DIR / "ossdbs_patient_dti_grid_settings.json"
    settings_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")

    start = time.time()
    main_run(settings)
    field_csv = Path(settings["OutputPath"]) / "E_field_PAM.csv"
    fem_summary = save_grid_outputs(field_csv, center, offsets)
    fem_summary["elapsed_s"] = time.time() - start
    fem_summary["tensor"] = tensor_summary
    fem_summary["material_map"] = material_summary
    fem_summary["settings_json"] = str(settings_path)
    fem_summary["comparison_to_previous_exam_nifti"] = compare_to_old(fem_summary)

    summary_path = OUT_DIR / "L_K1A_current_1mA_patient_dti_vta_postprocess_summary.json"
    summary_path.write_text(json.dumps(fem_summary, indent=2), encoding="utf-8")
    print(json.dumps(fem_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
