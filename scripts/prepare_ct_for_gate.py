#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import SimpleITK as sitk


def find_series_dirs(root: Path) -> list[Path]:
    """Return directories that contain at least one .dcm file."""
    out: list[Path] = []
    for d in sorted(p for p in root.rglob("*") if p.is_dir()):
        if any(f.suffix.lower() == ".dcm" for f in d.iterdir() if f.is_file()):
            out.append(d)
    return out


def load_dicom_series(series_dir: Path) -> sitk.Image:
    reader = sitk.ImageSeriesReader()
    try:
        series_ids = list(reader.GetGDCMSeriesIDs(str(series_dir)) or [])
    except Exception:
        series_ids = []

    if series_ids:
        files = reader.GetGDCMSeriesFileNames(str(series_dir), series_ids[0])
    else:
        files = sorted(str(p) for p in series_dir.glob("*.dcm"))

    if not files:
        raise FileNotFoundError(f"No DICOM files found in {series_dir}")

    reader.SetFileNames(files)
    return reader.Execute()


def convert_one(series_dir: Path, out_mhd: Path) -> dict[str, object]:
    out_mhd.parent.mkdir(parents=True, exist_ok=True)
    img = load_dicom_series(series_dir)
    sitk.WriteImage(img, str(out_mhd), useCompression=False)

    size = tuple(int(v) for v in img.GetSize())
    spacing = tuple(float(v) for v in img.GetSpacing())
    return {
        "source_series_dir": str(series_dir),
        "ct_mhd": str(out_mhd),
        "size_xyz": size,
        "spacing_xyz_mm": spacing,
    }


def pick_first_n(series_dirs: list[Path], n: int) -> list[Path]:
    return series_dirs[: max(0, int(n))]


def process_group(group_name: str, root: Path, out_root: Path, n_each: int) -> list[dict[str, object]]:
    series_dirs = find_series_dirs(root)
    selected = pick_first_n(series_dirs, n_each)
    rows: list[dict[str, object]] = []

    for i, series_dir in enumerate(selected, start=1):
        case_id = f"{group_name}_{i:03d}"
        case_out = out_root / group_name / case_id
        out_mhd = case_out / "ct.mhd"
        info = convert_one(series_dir, out_mhd)
        info["group"] = group_name
        info["case_id"] = case_id
        rows.append(info)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert selected DICOM CTs to MHD for OpenGATE")
    ap.add_argument("--lung-root", type=Path, required=True)
    ap.add_argument("--colorectal-root", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, default=Path("mc/ct_for_gate"))
    ap.add_argument("--n-each", type=int, default=4)
    args = ap.parse_args()

    all_rows: list[dict[str, object]] = []
    all_rows.extend(process_group("lung", args.lung_root, args.out_root, args.n_each))
    all_rows.extend(process_group("colorectal", args.colorectal_root, args.out_root, args.n_each))

    manifest_json = args.out_root / "manifest.json"
    manifest_csv = args.out_root / "manifest.csv"
    args.out_root.mkdir(parents=True, exist_ok=True)

    with open(manifest_json, "w", encoding="utf-8") as f:
        json.dump(all_rows, f, indent=2)

    with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["group", "case_id", "source_series_dir", "ct_mhd", "size_xyz", "spacing_xyz_mm"],
        )
        writer.writeheader()
        for r in all_rows:
            writer.writerow(r)

    print(f"Converted CTs: {len(all_rows)}")
    print(f"Manifest JSON: {manifest_json}")
    print(f"Manifest CSV:  {manifest_csv}")


if __name__ == "__main__":
    main()
