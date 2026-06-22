from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path


ROOT = Path(r"D:\dbs_workspace")
BASE_SCRIPT = ROOT / "dbs_stn_hdp_pipeline" / "scripts" / "run_b33005_left_pam_sequential.py"
PATIENT_DIR = ROOT / "dbs_stn_hdp_pipeline" / "patients" / "patient_example"
SOURCE_DIR = (
    PATIENT_DIR
    / "derivatives"
    / "ossdbs"
    / "vta_postprocess_left_LK1A_current_1mA_patient_dti_normtrace"
)
OUT_DIR = PATIENT_DIR / "derivatives" / "ossdbs" / "pam_left_LK1A_current_1mA_patient_dti_normtrace_sequential"
MRI_PATH = (
    PATIENT_DIR
    / "derivatives"
    / "ossdbs"
    / "vta_postprocess_left_LK1A_current_1mA_patient_dti"
    / "brain_segmented_from_5tt_dwi_ossdbs.nii.gz"
)
DTI_PATH = SOURCE_DIR / "dti_tensor_patient_normtrace_ossdbs.nii.gz"


def load_base_module():
    spec = importlib.util.spec_from_file_location("pam_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load base PAM script: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    base = load_base_module()
    base.OUT_DIR = OUT_DIR
    base.MRI_PATH = MRI_PATH
    base.DTI_PATH = DTI_PATH

    base.configure_windows_neuron()
    import ossdbs
    from ossdbs.api import run_PAM
    from ossdbs.main import main_run

    base.patch_neuron_for_windows_and_sequential()
    ossdbs.set_logger()

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    contacts = base.load_contacts()
    active_point = base.dwi_point(contacts[base.ACTIVE_CONTACT])
    h5_path, json_path, counts = base.build_pathway_h5(active_point, max_fibers_per_tract=None)
    settings = base.build_settings(h5_path, json_path, contacts)
    settings["MaterialDistribution"]["MRIPath"] = str(MRI_PATH)
    settings["MaterialDistribution"]["DTIPath"] = str(DTI_PATH)
    settings["OutputPath"] = str(OUT_DIR / "ossdbs_run")
    settings["StimulationFolder"] = str(OUT_DIR)
    settings["FailFlag"] = "b33005_left_LK1A_pam_patient_dti_normtrace_seq"

    settings_path = OUT_DIR / "ossdbs_pam_patient_dti_normtrace_settings.json"
    settings_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")

    print(f"Settings: {settings_path}")
    print(f"Material MRI: {MRI_PATH}")
    print(f"DTI tensor: {DTI_PATH}")
    print(f"Pathway H5: {h5_path}")
    print(f"Pathway JSON: {json_path}")
    print(f"Fiber counts: {counts}")

    started = time.time()
    main_run(settings)

    settings_for_pam = dict(settings)
    settings_for_pam["Scaling"] = 1.0
    settings_for_pam["ScalingIndex"] = None
    settings_for_pam["CurrentVector"] = None
    run_PAM(settings_for_pam)

    summary = base.collect_status(settings, counts, started)
    summary["settings_json"] = str(settings_path)
    summary["material_mri"] = str(MRI_PATH)
    summary["dti_tensor"] = str(DTI_PATH)
    summary["workflow"] = "sequential PAM, patient DTI trace-normalized conductivity tensor"
    (OUT_DIR / "pam_patient_dti_normtrace_sequential_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
