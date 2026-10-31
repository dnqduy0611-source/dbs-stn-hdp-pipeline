from __future__ import annotations

import argparse
import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from scipy import sparse


class FEMAuditComplete(RuntimeError):
    """Stop OSS-DBS immediately after the first FEM matrix is assembled."""


def reconstruct_symmetric_tensor(components: np.ndarray) -> np.ndarray:
    """Reconstruct OSS-DBS [xx, xy, xz, yy, yz, zz] data as 3x3 matrices."""
    if components.ndim != 4 or components.shape[-1] != 6:
        raise ValueError(
            f"Expected tensor image shape (x, y, z, 6), got {components.shape}"
        )

    tensor = np.empty(components.shape[:3] + (3, 3), dtype=np.float64)
    tensor[..., 0, 0] = components[..., 0]
    tensor[..., 0, 1] = tensor[..., 1, 0] = components[..., 1]
    tensor[..., 0, 2] = tensor[..., 2, 0] = components[..., 2]
    tensor[..., 1, 1] = components[..., 3]
    tensor[..., 1, 2] = tensor[..., 2, 1] = components[..., 4]
    tensor[..., 2, 2] = components[..., 5]
    return tensor


def _ellipsoid_mask(
    image: nib.spatialimages.SpatialImage, settings: dict[str, Any]
) -> np.ndarray:
    region = settings["BrainRegion"]
    center = np.array(
        [region["Center"][f"{axis}[mm]"] for axis in "xyz"], dtype=float
    )
    dimensions = np.array(
        [region["Dimension"][f"{axis}[mm]"] for axis in "xyz"], dtype=float
    )
    indices = np.indices(image.shape[:3], dtype=float).reshape(3, -1).T
    coordinates = nib.affines.apply_affine(image.affine, indices)
    normalized_radius_squared = np.sum(
        ((coordinates - center) / (0.5 * dimensions)) ** 2, axis=1
    )
    return (normalized_radius_squared <= 1.0).reshape(image.shape[:3])


def audit_tensor(
    tensor_path: Path,
    material_path: Path,
    dwi_mask_path: Path,
    settings: dict[str, Any],
) -> dict[str, Any]:
    tensor_image = nib.load(str(tensor_path))
    components = tensor_image.get_fdata(dtype=np.float32)
    material = nib.load(str(material_path)).get_fdata(dtype=np.float32)
    dwi_mask = nib.load(str(dwi_mask_path)).get_fdata(dtype=np.float32) > 0

    if material.shape != components.shape[:3] or dwi_mask.shape != components.shape[:3]:
        raise ValueError("Tensor, material map, and DWI mask geometries do not match")

    tensor = reconstruct_symmetric_tensor(components)
    transpose = np.swapaxes(tensor, -1, -2)
    asymmetry = np.abs(tensor - transpose)
    eigenvalues = np.linalg.eigvalsh(tensor)
    minimum_eigenvalue = eigenvalues[..., 0]
    trace = np.trace(tensor, axis1=-2, axis2=-1)

    white_matter = material == 2
    fem_domain = _ellipsoid_mask(tensor_image, settings)
    active_dti = white_matter & fem_domain
    nonpositive = minimum_eigenvalue <= 0

    active_eigenvalues = eigenvalues[active_dti]
    active_trace = trace[active_dti]
    if active_eigenvalues.size == 0:
        raise ValueError("No white-matter tensor voxels intersect the FEM domain")

    return {
        "tensor_path": str(tensor_path),
        "component_order": ["Dxx", "Dxy", "Dxz", "Dyy", "Dyz", "Dzz"],
        "shape": list(components.shape),
        "all_values_finite": bool(np.isfinite(components).all()),
        "max_absolute_asymmetry": float(asymmetry.max()),
        "exactly_symmetric_after_reconstruction": bool(np.all(asymmetry == 0)),
        "white_matter_voxels": int(white_matter.sum()),
        "white_matter_outside_dwi_mask": int((white_matter & ~dwi_mask).sum()),
        "nonpositive_tensor_voxels_in_white_matter": int(
            (white_matter & nonpositive).sum()
        ),
        "nonpositive_tensor_voxels_in_white_matter_outside_dwi_mask": int(
            (white_matter & ~dwi_mask & nonpositive).sum()
        ),
        "active_white_matter_voxels_in_fem_domain": int(active_dti.sum()),
        "nonpositive_tensor_voxels_in_active_fem_white_matter": int(
            (active_dti & nonpositive).sum()
        ),
        "active_fem_white_matter_minimum_eigenvalue": float(
            active_eigenvalues.min()
        ),
        "active_fem_white_matter_trace": {
            "min": float(active_trace.min()),
            "median": float(np.median(active_trace)),
            "max": float(active_trace.max()),
        },
    }


def _sparse_frobenius_norm(matrix: sparse.spmatrix) -> float:
    return float(np.sqrt(np.sum(np.abs(matrix.data) ** 2)))


def matrix_metrics(matrix: Any, free_dofs: Any) -> dict[str, Any]:
    rows, columns, values = matrix.COO()
    assembled = sparse.coo_matrix(
        (np.asarray(values), (np.asarray(rows), np.asarray(columns))),
        shape=(matrix.height, matrix.width),
    ).tocsr()
    assembled.sum_duplicates()

    free = np.asarray(free_dofs, dtype=bool)
    free_matrix = assembled[free][:, free].tocsr()
    transpose_difference = free_matrix - free_matrix.transpose()
    hermitian_difference = free_matrix - free_matrix.getH()

    matrix_norm = _sparse_frobenius_norm(free_matrix)
    transpose_norm = _sparse_frobenius_norm(transpose_difference)
    hermitian_norm = _sparse_frobenius_norm(hermitian_difference)
    max_transpose_difference = (
        float(np.max(np.abs(transpose_difference.data)))
        if transpose_difference.nnz
        else 0.0
    )

    rng = np.random.default_rng(20260629)
    rayleigh_quotients = []
    for _ in range(5):
        vector = rng.standard_normal(free_matrix.shape[0])
        value = np.vdot(vector, free_matrix @ vector) / np.vdot(vector, vector)
        rayleigh_quotients.append(
            {"real": float(np.real(value)), "imag": float(np.imag(value))}
        )

    diagonal = free_matrix.diagonal()
    return {
        "matrix_shape": list(assembled.shape),
        "matrix_nonzeros": int(assembled.nnz),
        "free_dofs": int(free.sum()),
        "free_matrix_nonzeros": int(free_matrix.nnz),
        "dtype": str(free_matrix.dtype),
        "frobenius_norm": matrix_norm,
        "transpose_symmetry": {
            "absolute_frobenius_residual": transpose_norm,
            "relative_frobenius_residual": (
                transpose_norm / matrix_norm if matrix_norm else 0.0
            ),
            "maximum_absolute_entry_difference": max_transpose_difference,
        },
        "hermitian_symmetry": {
            "absolute_frobenius_residual": hermitian_norm,
            "relative_frobenius_residual": (
                hermitian_norm / matrix_norm if matrix_norm else 0.0
            ),
        },
        "diagonal": {
            "minimum_real": float(np.min(np.real(diagonal))),
            "maximum_real": float(np.max(np.real(diagonal))),
            "nonpositive_real_count": int(np.sum(np.real(diagonal) <= 0)),
        },
        "random_real_vector_rayleigh_quotients": rayleigh_quotients,
    }


def audit_first_fem_matrix(settings: dict[str, Any]) -> dict[str, Any]:
    import ossdbs
    from ossdbs.fem.solver import CGSolver
    from ossdbs.main import main_run

    result: dict[str, Any] = {}
    original_bvp = CGSolver.bvp

    def audited_bvp(self, bilinear_form, linear_form, grid_function) -> None:
        bilinear_form.Assemble()
        linear_form.Assemble()
        result.update(
            matrix_metrics(bilinear_form.mat, grid_function.space.FreeDofs())
        )
        raise FEMAuditComplete

    with tempfile.TemporaryDirectory(prefix="ossdbs_fem_symmetry_audit_") as temp:
        audit_settings = json.loads(json.dumps(settings))
        audit_settings["OutputPath"] = str(Path(temp) / "output")
        audit_settings["StimulationFolder"] = temp
        audit_settings["FailFlag"] = "fem_symmetry_audit"
        audit_settings["ExportElectrode"] = False
        audit_settings["ExportVTK"] = False

        CGSolver.bvp = audited_bvp
        try:
            ossdbs.set_logger()
            main_run(audit_settings)
        except FEMAuditComplete:
            pass
        finally:
            CGSolver.bvp = original_bvp
            # OSS-DBS attaches a FileHandler under OutputPath. Close it before
            # TemporaryDirectory removes the audit-only output on Windows.
            logging.shutdown()

    if not result:
        raise RuntimeError("OSS-DBS completed without exposing a CG FEM matrix")
    result["audited_frequency_hz"] = 0.0
    result["scope"] = "assembled free-DOF matrix before the first CG solve"
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit OSS-DBS DTI tensor and assembled FEM matrix symmetry"
    )
    parser.add_argument("--tensor", type=Path, required=True)
    parser.add_argument("--material-map", type=Path, required=True)
    parser.add_argument("--dwi-mask", type=Path, required=True)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--skip-fem", action="store_true", help="Only audit the tensor image"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = json.loads(args.settings.read_text(encoding="utf-8"))
    report = {
        "tensor": audit_tensor(
            args.tensor, args.material_map, args.dwi_mask, settings
        )
    }
    if not args.skip_fem:
        report["fem"] = audit_first_fem_matrix(settings)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
