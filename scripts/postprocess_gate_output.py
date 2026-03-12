#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def hu_to_spr(hu: np.ndarray) -> np.ndarray:
    # Keep HU bins consistent with gate_voxelized_ct_beam.py voxel_materials.
    # Representative SPR values are approximate class centroids for analysis.
    spr = np.full_like(hu, 1.0, dtype=np.float32)
    spr[hu <= -950] = 0.01  # air
    spr[(hu > -950) & (hu <= -500)] = 0.26  # lung
    spr[(hu > -500) & (hu <= -50)] = 0.92  # adipose
    spr[(hu > -50) & (hu <= 20)] = 1.00  # water/fluid
    spr[(hu > 20) & (hu <= 100)] = 1.05  # muscle
    spr[(hu > 100) & (hu <= 300)] = 1.16  # cartilage/trabecular-like
    spr[hu > 300] = 1.65  # cortical bone
    return spr


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert OpenGATE MHD dose output to dose.npy/spr.npy")
    parser.add_argument("--dose-mhd", type=Path, required=True)
    parser.add_argument("--ct-mhd", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    dose_img = sitk.ReadImage(str(args.dose_mhd))
    dose = sitk.GetArrayFromImage(dose_img).astype(np.float32)
    dose = np.clip(dose, 0.0, None)

    ct_img = sitk.ReadImage(str(args.ct_mhd))
    hu = sitk.GetArrayFromImage(ct_img).astype(np.float32)
    spr = hu_to_spr(hu)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "dose.npy", dose)
    np.save(args.out_dir / "spr.npy", spr)
    print(f"Saved {(args.out_dir / 'dose.npy')} and {(args.out_dir / 'spr.npy')}")


if __name__ == "__main__":
    main()
