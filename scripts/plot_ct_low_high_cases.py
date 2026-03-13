#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Plot CT (SPR), low dose and high dose for NPZ cases")
    ap.add_argument("--input-dir", type=Path, required=True, help="Directory containing .npz files")
    ap.add_argument("--pattern", type=str, default="*.npz", help="Glob pattern for case selection")
    ap.add_argument("--limit", type=int, default=0, help="Max number of cases to process (0 = all)")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/ct_low_high_cases"),
        help="Output directory for figures",
    )
    ap.add_argument(
        "--input-dose-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to low-dose input channel for plotting",
    )
    ap.add_argument(
        "--ct-channel",
        type=int,
        choices=[1],
        default=1,
        help="Channel index used as CT/SPR map in input tensor",
    )
    ap.add_argument(
        "--percentile",
        type=float,
        default=99.5,
        help="Robust percentile used for color scale clipping",
    )
    return ap.parse_args()


def robust_vmax(arr: np.ndarray, pct: float, eps: float = 1e-8) -> float:
    vmax = float(np.percentile(arr, pct))
    return max(vmax, eps)


def xz_center(vol: np.ndarray) -> np.ndarray:
    _, ny, _ = vol.shape
    return vol[:, ny // 2, :]


def save_case_plot(npz_path: Path, out_path: Path, input_dose_scale: float, percentile: float) -> None:
    with np.load(npz_path) as z:
        required = {"input", "target", "spr", "energy_mev"}
        missing = required - set(z.files)
        if missing:
            raise KeyError(f"{npz_path.name}: missing keys {sorted(missing)}")

        inp = z["input"].astype(np.float32)  # [2, D, H, W]
        target = z["target"].astype(np.float32)  # [D, H, W]
        energy = float(z["energy_mev"].item())

    low = inp[0] * float(input_dose_scale)
    ct = inp[1]

    peak_z = int(np.argmax(target.max(axis=(1, 2))))
    cy = target.shape[1] // 2

    low_xy = low[peak_z]
    high_xy = target[peak_z]
    ct_xy = ct[peak_z]

    low_xz = xz_center(low)
    high_xz = xz_center(target)
    ct_xz = xz_center(ct)

    dose_vmax = robust_vmax(np.concatenate([low_xy.ravel(), high_xy.ravel()]), percentile)
    dose_xz_vmax = robust_vmax(np.concatenate([low_xz.ravel(), high_xz.ravel()]), percentile)
    ct_vmax = robust_vmax(np.concatenate([ct_xy.ravel(), ct_xz.ravel()]), percentile)

    err_xy = np.abs(low_xy - high_xy)
    err_xz = np.abs(low_xz - high_xz)
    err_vmax = robust_vmax(np.concatenate([err_xy.ravel(), err_xz.ravel()]), percentile)

    fig, axs = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)

    im00 = axs[0, 0].imshow(ct_xy, cmap="gray", origin="lower", vmin=0, vmax=ct_vmax)
    axs[0, 0].set_title(f"CT/SPR XY (z={peak_z})")
    im01 = axs[0, 1].imshow(low_xy, cmap="inferno", origin="lower", vmin=0, vmax=dose_vmax)
    axs[0, 1].set_title("Low dose XY")
    im02 = axs[0, 2].imshow(high_xy, cmap="inferno", origin="lower", vmin=0, vmax=dose_vmax)
    axs[0, 2].set_title("High dose XY")
    im03 = axs[0, 3].imshow(err_xy, cmap="magma", origin="lower", vmin=0, vmax=err_vmax)
    axs[0, 3].set_title("|Low-High| XY")

    im10 = axs[1, 0].imshow(ct_xz, cmap="gray", origin="lower", aspect="auto", vmin=0, vmax=ct_vmax)
    axs[1, 0].set_title(f"CT/SPR XZ (y={cy})")
    im11 = axs[1, 1].imshow(low_xz, cmap="inferno", origin="lower", aspect="auto", vmin=0, vmax=dose_xz_vmax)
    axs[1, 1].set_title("Low dose XZ")
    im12 = axs[1, 2].imshow(high_xz, cmap="inferno", origin="lower", aspect="auto", vmin=0, vmax=dose_xz_vmax)
    axs[1, 2].set_title("High dose XZ")
    im13 = axs[1, 3].imshow(err_xz, cmap="magma", origin="lower", aspect="auto", vmin=0, vmax=err_vmax)
    axs[1, 3].set_title("|Low-High| XZ")

    for ax in axs.ravel():
        ax.set_xlabel("x voxel")
        ax.set_ylabel("y/z voxel")

    c0 = fig.colorbar(im00, ax=[axs[0, 0], axs[1, 0]], fraction=0.02, pad=0.01)
    c0.set_label("CT/SPR")
    c1 = fig.colorbar(im01, ax=[axs[0, 1], axs[0, 2], axs[1, 1], axs[1, 2]], fraction=0.02, pad=0.01)
    c1.set_label("Dose")
    c2 = fig.colorbar(im03, ax=[axs[0, 3], axs[1, 3]], fraction=0.02, pad=0.01)
    c2.set_label("Absolute error")

    fig.suptitle(
        f"{npz_path.name} | E={energy:.1f} MeV | input-dose-scale={input_dose_scale:g}",
        fontsize=11,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    files = sorted(args.input_dir.glob(args.pattern))
    if not files:
        raise FileNotFoundError(f"No files found in {args.input_dir} with pattern {args.pattern!r}")

    if args.limit > 0:
        files = files[: args.limit]

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for i, f in enumerate(files, start=1):
        out_png = args.out_dir / f"{f.stem}_ct_low_high.png"
        save_case_plot(f, out_png, args.input_dose_scale, args.percentile)
        print(f"[{i}/{len(files)}] {out_png}")


if __name__ == "__main__":
    main()
