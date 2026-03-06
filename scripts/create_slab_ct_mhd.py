#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk


HU_MAP = {
    "water": 0,
    "bone": 1200,
    "lung": -750,
    "air": -1000,
}


def _parse_csv_list(text: str) -> list[str]:
    return [x.strip().lower() for x in text.split(",") if x.strip()]


def _parse_csv_float(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create slab CT phantom in MHD format")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--nx", type=int, default=75)
    parser.add_argument("--ny", type=int, default=75)
    parser.add_argument("--nz", type=int, default=100)
    parser.add_argument("--sx", type=float, default=2.0, help="spacing mm")
    parser.add_argument("--sy", type=float, default=2.0, help="spacing mm")
    parser.add_argument("--sz", type=float, default=2.0, help="spacing mm")
    parser.add_argument(
        "--layer-order",
        type=str,
        default="water,bone,lung,water",
        help="Comma-separated layer names along +z",
    )
    parser.add_argument(
        "--layer-thickness-mm",
        type=str,
        default="40,30,50,80",
        help="Comma-separated layer thicknesses in mm",
    )
    args = parser.parse_args()

    layer_order = _parse_csv_list(args.layer_order)
    layer_thickness = _parse_csv_float(args.layer_thickness_mm)
    if len(layer_order) != len(layer_thickness):
        raise ValueError("layer-order and layer-thickness-mm must have same number of elements")
    for mat in layer_order:
        if mat not in HU_MAP:
            raise ValueError(f"Unknown material '{mat}', valid: {sorted(HU_MAP.keys())}")

    hu = np.zeros((args.nz, args.ny, args.nx), dtype=np.int16)  # water ~ 0 HU

    z0 = 0
    for mat, thick_mm in zip(layer_order, layer_thickness):
        nvox = int(round(thick_mm / args.sz))
        z1 = min(args.nz, z0 + max(nvox, 1))
        hu[z0:z1, :, :] = HU_MAP[mat]
        z0 = z1
        if z0 >= args.nz:
            break

    img = sitk.GetImageFromArray(hu)
    img.SetSpacing((args.sx, args.sy, args.sz))
    img.SetOrigin((0.0, 0.0, 0.0))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(args.out))
    print(f"Wrote CT phantom: {args.out}")


if __name__ == "__main__":
    main()
