#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to


NRRD_TO_NUMPY_DTYPE = {
    "uchar": np.uint8,
    "unsigned char": np.uint8,
    "uint8": np.uint8,
    "short": np.int16,
    "signed short": np.int16,
    "ushort": np.uint16,
    "unsigned short": np.uint16,
    "int": np.int32,
    "signed int": np.int32,
    "uint": np.uint32,
    "unsigned int": np.uint32,
}


@dataclass
class SegmentResult:
    name: str
    label_value: int
    source_mask: str
    dwi_mask: str
    dilated_mask: str | None
    source_voxels: int
    dwi_voxels: int
    dilated_voxels: int | None
    source_center_ras: list[float] | None
    dwi_center_ras: list[float] | None
    warning: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export Slicer .seg.nrrd labelmap segments to NIfTI masks and resample "
            "them to MRtrix DWI geometry for tractography ROI filtering."
        )
    )
    parser.add_argument(
        "patient_dir",
        nargs="?",
        help="Patient directory. Kept for backward compatibility with old scripts.",
    )
    parser.add_argument(
        "--seg",
        help="Input Slicer .seg.nrrd. Relative paths are resolved from patient_dir.",
    )
    parser.add_argument(
        "--template",
        help="DWI-space template image, preferably derivatives/dwi/dwi_bias.mif.",
    )
    parser.add_argument(
        "--out-dir",
        help="Output directory. Relative paths are resolved from patient_dir.",
    )
    parser.add_argument(
        "--segment",
        action="append",
        default=[],
        help="Only export selected segment name. Can be supplied multiple times.",
    )
    parser.add_argument(
        "--dilate",
        type=int,
        default=1,
        help="Number of DWI-space maskfilter dilation passes to create. Use 0 to disable.",
    )
    parser.add_argument(
        "--stride",
        default="-1,2,3",
        help="NIfTI strides for DWI-space outputs for Slicer compatibility.",
    )
    parser.add_argument(
        "--reference-nifti",
        help=(
            "NIfTI image that defines the target DWI grid for direct ROI projection. "
            "Defaults to derivatives/dwi/fa.nii.gz when patient_dir is supplied; "
            "otherwise a temporary NIfTI is generated from --template using mrconvert."
        ),
    )
    parser.add_argument(
        "--flirt-mat",
        help=(
            "Optional FSL FLIRT affine from the segmentation source space to DWI. "
            "When supplied, masks are resampled with nearest-neighbour interpolation "
            "instead of direct header-coordinate projection."
        ),
    )
    parser.add_argument(
        "--report",
        help="Markdown QC report path. Relative paths are resolved from patient_dir.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs.")
    return parser.parse_args()


def resolve_path(path: str | None, patient_dir: Path | None, default: Path | None = None) -> Path:
    if path is None:
        if default is None:
            raise ValueError("Missing path and no default was supplied")
        return default
    p = Path(path)
    if p.is_absolute():
        return p
    if patient_dir is not None:
        return patient_dir / p
    return p.resolve()


def safe_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")
    if not cleaned:
        raise ValueError(f"Invalid segment name: {name!r}")
    return cleaned


def split_nrrd_header(raw: bytes) -> tuple[str, bytes]:
    for sep in (b"\n\n", b"\r\n\r\n"):
        idx = raw.find(sep)
        if idx >= 0:
            return raw[:idx].decode("latin1"), raw[idx + len(sep) :]
    raise ValueError("Could not find NRRD header/data separator")


def parse_nrrd_vector(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",")]


def parse_space_directions(header: str) -> np.ndarray:
    match = re.search(r"^space directions:\s*(.*)$", header, re.M)
    if not match:
        raise ValueError("Missing NRRD 'space directions' field")
    vectors = re.findall(r"\(([^)]*)\)", match.group(1))
    if len(vectors) != 3:
        raise ValueError("Only 3D Slicer binary labelmap segmentations are supported")
    matrix = np.eye(4)
    matrix[:3, :3] = np.array([parse_nrrd_vector(v) for v in vectors], dtype=float).T
    return matrix


def parse_space_origin(header: str) -> np.ndarray:
    match = re.search(r"^space origin:\s*\((.*)\)$", header, re.M)
    if not match:
        raise ValueError("Missing NRRD 'space origin' field")
    return np.array(parse_nrrd_vector(match.group(1)), dtype=float)


def parse_segments(header: str) -> list[tuple[int, str, int]]:
    segments: dict[int, dict[str, str]] = {}
    pattern = r"^Segment(\d+)_(Name|LabelValue):=([^\n\r]+)$"
    for match in re.finditer(pattern, header, re.M):
        idx, key, value = match.groups()
        segments.setdefault(int(idx), {})[key] = value.strip()

    parsed: list[tuple[int, str, int]] = []
    for idx, meta in sorted(segments.items()):
        if "Name" not in meta or "LabelValue" not in meta:
            continue
        parsed.append((idx, meta["Name"], int(meta["LabelValue"])))
    if not parsed:
        raise ValueError("No Slicer segments found in NRRD header")
    return parsed


def read_slicer_seg(seg_path: Path) -> tuple[np.ndarray, np.ndarray, list[tuple[int, str, int]]]:
    raw = seg_path.read_bytes()
    header, payload = split_nrrd_header(raw)

    sizes_match = re.search(r"^sizes:\s*(.*)$", header, re.M)
    if not sizes_match:
        raise ValueError("Missing NRRD 'sizes' field")
    sizes = tuple(int(v) for v in sizes_match.group(1).split())
    if len(sizes) != 3:
        raise ValueError(f"Expected a 3D labelmap, got sizes={sizes}")

    dtype_text = re.search(r"^type:\s*(.*)$", header, re.M)
    if not dtype_text:
        raise ValueError("Missing NRRD 'type' field")
    dtype_key = dtype_text.group(1).strip().lower()
    if dtype_key not in NRRD_TO_NUMPY_DTYPE:
        raise ValueError(f"Unsupported NRRD type: {dtype_key}")
    dtype = NRRD_TO_NUMPY_DTYPE[dtype_key]

    encoding_match = re.search(r"^encoding:\s*(.*)$", header, re.M)
    encoding = encoding_match.group(1).strip().lower() if encoding_match else "raw"
    if encoding in {"gzip", "gz"}:
        payload = gzip.decompress(payload)
    elif encoding not in {"raw"}:
        raise ValueError(f"Unsupported NRRD encoding: {encoding}")

    arr = np.frombuffer(payload, dtype=dtype)
    expected = int(np.prod(sizes))
    if arr.size != expected:
        raise ValueError(f"NRRD payload voxel count mismatch: got {arr.size}, expected {expected}")
    arr = arr.reshape(sizes, order="F")

    space_match = re.search(r"^space:\s*(.*)$", header, re.M)
    space = space_match.group(1).strip().lower() if space_match else ""
    affine = parse_space_directions(header)
    affine[:3, 3] = parse_space_origin(header)
    if space in {"left-posterior-superior", "lps"}:
        affine = np.diag([-1.0, -1.0, 1.0, 1.0]) @ affine
    elif space not in {"right-anterior-superior", "ras"}:
        raise ValueError(f"Unsupported or missing NRRD space: {space!r}")

    return arr, affine, parse_segments(header)


def write_mask(mask: np.ndarray, affine: np.ndarray, out_path: Path, force: bool) -> None:
    if out_path.exists() and not force:
        raise FileExistsError(f"Output exists, use --force: {out_path}")
    img = nib.Nifti1Image(mask.astype(np.uint8), affine)
    img.header.set_xyzt_units("mm")
    img.set_qform(affine, code=1)
    img.set_sform(affine, code=1)
    nib.save(img, str(out_path))


def mask_count_and_center(path: Path) -> tuple[int, list[float] | None]:
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)
    nz = np.argwhere(data > 0)
    if len(nz) == 0:
        return 0, None
    center = nib.affines.apply_affine(img.affine, nz.mean(axis=0))
    return int(len(nz)), [round(float(v), 3) for v in center]


def run_command(cmd: list[str]) -> None:
    print("+ " + " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)


def require_tool(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Required command not found in PATH: {name}")


def ensure_reference_nifti(template_path: Path, reference_nifti: Path | None) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    if reference_nifti is not None:
        if not reference_nifti.exists():
            raise FileNotFoundError(f"Missing reference NIfTI: {reference_nifti}")
        return reference_nifti, None

    if template_path.suffix in {".nii", ".gz"} or str(template_path).endswith(".nii.gz"):
        return template_path, None

    require_tool("mrconvert")
    tmpdir = tempfile.TemporaryDirectory(prefix="seg_to_dwi_")
    tmpref = Path(tmpdir.name) / "template_ref.nii.gz"
    cmd = ["mrconvert", str(template_path), str(tmpref), "-coord", "3", "0", "-strides", "-1,2,3", "-force", "-quiet"]
    run_command(cmd)
    return tmpref, tmpdir


def project_mask_to_reference(source_path: Path, reference_path: Path, out_path: Path, force: bool) -> None:
    if out_path.exists() and not force:
        raise FileExistsError(f"Output exists, use --force: {out_path}")

    source = nib.load(str(source_path))
    reference = nib.load(str(reference_path))
    source_mask = nib.Nifti1Image(
        (np.asanyarray(source.dataobj) > 0).astype(np.uint8), source.affine
    )
    resampled = resample_from_to(
        source_mask,
        (reference.shape[:3], reference.affine),
        order=0,
        mode="constant",
        cval=0,
    )
    out = (np.asanyarray(resampled.dataobj) > 0).astype(np.uint8)

    img = nib.Nifti1Image(out, reference.affine)
    img.header.set_xyzt_units("mm")
    img.set_qform(reference.affine, code=1)
    img.set_sform(reference.affine, code=1)
    nib.save(img, str(out_path))


def resample_mask_with_flirt(
    source_path: Path,
    reference_path: Path,
    transform_path: Path,
    out_path: Path,
    force: bool,
) -> None:
    if out_path.exists() and not force:
        raise FileExistsError(f"Output exists, use --force: {out_path}")

    require_tool("flirt")
    cmd = [
        "flirt",
        "-in",
        str(source_path),
        "-ref",
        str(reference_path),
        "-applyxfm",
        "-init",
        str(transform_path),
        "-interp",
        "nearestneighbour",
        "-out",
        str(out_path),
    ]
    run_command(cmd)


def write_report(report_path: Path, args: argparse.Namespace, results: list[SegmentResult]) -> None:
    lines = [
        "# Slicer Segmentation to DWI ROI QC",
        "",
        f"- Source segmentation: `{args.seg}`",
        f"- DWI template: `{args.template}`",
        f"- Output directory: `{args.out_dir}`",
        f"- Source-to-DWI FLIRT matrix: `{args.flirt_mat or ''}`",
        f"- Dilation passes: `{args.dilate}`",
        "",
        "## Segment Results",
        "",
        "| Segment | Label | Source voxels | DWI voxels | Dilated voxels | Warning |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for r in results:
        warning = r.warning or ""
        dilated = "" if r.dilated_voxels is None else str(r.dilated_voxels)
        lines.append(
            f"| {r.name} | {r.label_value} | {r.source_voxels} | {r.dwi_voxels} | "
            f"{dilated} | {warning} |"
        )
    lines.extend(["", "## Outputs", ""])
    for r in results:
        lines.extend(
            [
                f"### {r.name}",
                "",
                f"- Source mask: `{r.source_mask}`",
                f"- DWI mask: `{r.dwi_mask}`",
                f"- Dilated DWI mask: `{r.dilated_mask or ''}`",
                f"- Source center RAS: `{r.source_center_ras}`",
                f"- DWI center RAS: `{r.dwi_center_ras}`",
                "",
            ]
        )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    patient_dir = Path(args.patient_dir).resolve() if args.patient_dir else None

    default_seg = patient_dir / "derivatives/rois/L_DRTT.seg.nrrd" if patient_dir else None
    default_template = patient_dir / "derivatives/dwi/dwi_bias.mif" if patient_dir else None
    default_reference = patient_dir / "derivatives/dwi/fa.nii.gz" if patient_dir else None
    default_out_dir = patient_dir / "derivatives/rois" if patient_dir else Path.cwd()
    default_report = patient_dir / "qc/reports/seg_to_dwi_roi_qc.md" if patient_dir else Path.cwd() / "seg_to_dwi_roi_qc.md"

    seg_path = resolve_path(args.seg, patient_dir, default_seg)
    template_path = resolve_path(args.template, patient_dir, default_template)
    reference_nifti = resolve_path(args.reference_nifti, patient_dir, default_reference) if (args.reference_nifti or default_reference) else None
    flirt_mat = resolve_path(args.flirt_mat, patient_dir) if args.flirt_mat else None
    out_dir = resolve_path(args.out_dir, patient_dir, default_out_dir)
    report_path = resolve_path(args.report, patient_dir, default_report)

    args.seg = str(seg_path)
    args.template = str(template_path)
    args.out_dir = str(out_dir)
    args.flirt_mat = str(flirt_mat) if flirt_mat else None

    required_paths = [seg_path, template_path]
    if flirt_mat is not None:
        required_paths.append(flirt_mat)
    for path in required_paths:
        if not path.exists():
            print(f"Missing input: {path}", file=sys.stderr)
            return 1
    out_dir.mkdir(parents=True, exist_ok=True)

    reference_path, tmp_reference_dir = ensure_reference_nifti(template_path, reference_nifti)
    if args.dilate > 0:
        require_tool("maskfilter")

    data, affine, segments = read_slicer_seg(seg_path)
    selected = set(args.segment)

    results: list[SegmentResult] = []
    for _idx, name, label_value in segments:
        if selected and name not in selected:
            continue
        stem = safe_name(name)
        source_path = out_dir / f"{stem}_source.nii.gz"
        dwi_path = out_dir / f"{stem}_dwi.nii.gz"
        dilated_path = out_dir / f"{stem}_dwi_dil{args.dilate}.nii.gz" if args.dilate > 0 else None

        source_mask = (data == label_value).astype(np.uint8)
        write_mask(source_mask, affine, source_path, args.force)
        source_voxels, source_center = mask_count_and_center(source_path)

        if flirt_mat is not None:
            resample_mask_with_flirt(
                source_path,
                reference_path,
                flirt_mat,
                dwi_path,
                args.force,
            )
        else:
            project_mask_to_reference(source_path, reference_path, dwi_path, args.force)
        dwi_voxels, dwi_center = mask_count_and_center(dwi_path)

        dilated_voxels = None
        if dilated_path is not None:
            cmd = [
                "maskfilter",
                str(dwi_path),
                "dilate",
                str(dilated_path),
                "-npass",
                str(args.dilate),
                "-quiet",
            ]
            if args.force:
                cmd.append("-force")
            run_command(cmd)
            dilated_voxels, _ = mask_count_and_center(dilated_path)

        warning = None
        if source_voxels == 0:
            warning = "empty source ROI"
        elif dwi_voxels == 0:
            warning = "ROI lost after DWI resampling; check coverage/registration"
        elif dwi_voxels <= 3:
            warning = "very small DWI ROI; consider QC or dilation"

        result = SegmentResult(
            name=name,
            label_value=label_value,
            source_mask=str(source_path),
            dwi_mask=str(dwi_path),
            dilated_mask=str(dilated_path) if dilated_path else None,
            source_voxels=source_voxels,
            dwi_voxels=dwi_voxels,
            dilated_voxels=dilated_voxels,
            source_center_ras=source_center,
            dwi_center_ras=dwi_center,
            warning=warning,
        )
        results.append(result)

    if selected and not results:
        print(f"No selected segments found. Requested: {sorted(selected)}", file=sys.stderr)
        return 1

    write_report(report_path, args, results)
    print()
    print(json.dumps([asdict(r) for r in results], indent=2))
    print(f"\nQC report: {report_path}")

    if tmp_reference_dir is not None:
        tmp_reference_dir.cleanup()

    if any(r.warning for r in results):
        print("\nWarnings were generated. Review the QC report before tract filtering.", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
