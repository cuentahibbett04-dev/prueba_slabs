#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Create a multi-case beam-parallel (XZ) montage with dose overlay on CT/SPR"
    )
    ap.add_argument("--input-dir", type=Path, required=True, help="Directory with .npz cases")
    ap.add_argument("--pattern", type=str, default="*.npz", help="Glob pattern to select cases")
    ap.add_argument("--out", type=Path, default=Path("artifacts/beam_parallel_montage.png"))
    ap.add_argument("--limit", type=int, default=10, help="Maximum number of cases (0 = all)")
    ap.add_argument("--cols", type=int, default=5, help="Number of columns in montage")
    ap.add_argument(
        "--dose-source",
        type=str,
        choices=["target", "low"],
        default="target",
        help="Dose channel used for overlay",
    )
    ap.add_argument(
        "--input-dose-scale",
        type=float,
        default=1.0,
        help="Scale factor for low-dose channel when --dose-source low",
    )
    ap.add_argument("--y-band-half", type=int, default=2, help="Half-width around center y for XZ averaging")
    ap.add_argument(
        "--focus-mode",
        type=str,
        choices=["center", "peak"],
        default="peak",
        help="Use geometric center or global peak-dose y for XZ band extraction",
    )
    ap.add_argument(
        "--peak-source",
        type=str,
        choices=["target", "selected"],
        default="target",
        help="Dose volume used to find peak when --focus-mode peak",
    )
    ap.add_argument(
        "--align-peak",
        action="store_true",
        help="Shift XZ maps so the peak is centered (helps compare dispersion across cases)",
    )
    ap.add_argument("--dose-percentile", type=float, default=99.7, help="Robust vmax percentile for dose")
    ap.add_argument("--ct-percentile", type=float, default=99.5, help="Robust vmax percentile for CT")
    ap.add_argument("--alpha-gamma", type=float, default=0.6, help="Alpha shaping exponent for dose overlay")
    ap.add_argument("--dpi", type=int, default=180)
    return ap.parse_args()


def robust_normalize(a: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.5, eps: float = 1e-8) -> np.ndarray:
    lo = float(np.percentile(a, low_pct))
    hi = float(np.percentile(a, high_pct))
    if hi <= lo + eps:
        hi = lo + eps
    x = (a - lo) / (hi - lo)
    return np.clip(x, 0.0, 1.0)


def xz_center_band(vol: np.ndarray, half_width: int) -> np.ndarray:
    _, ny, _ = vol.shape
    cy = ny // 2
    y0 = max(0, cy - half_width)
    y1 = min(ny, cy + half_width + 1)
    return vol[:, y0:y1, :].mean(axis=1)


def xz_band_at_y(vol: np.ndarray, y_center: int, half_width: int) -> np.ndarray:
    _, ny, _ = vol.shape
    yc = int(max(0, min(y_center, ny - 1)))
    y0 = max(0, yc - half_width)
    y1 = min(ny, yc + half_width + 1)
    return vol[:, y0:y1, :].mean(axis=1)


def peak_zyx(vol: np.ndarray) -> tuple[int, int, int]:
    return tuple(int(v) for v in np.unravel_index(np.argmax(vol), vol.shape))


def shift_no_wrap_2d(xz: np.ndarray, dz: int, dx: int) -> np.ndarray:
    """Shift a 2D map without wrap-around; uncovered area is filled with zeros."""
    out = np.zeros_like(xz)
    h, w = xz.shape

    src_z0 = max(0, -dz)
    src_z1 = min(h, h - dz) if dz >= 0 else h
    dst_z0 = max(0, dz)
    dst_z1 = dst_z0 + (src_z1 - src_z0)

    src_x0 = max(0, -dx)
    src_x1 = min(w, w - dx) if dx >= 0 else w
    dst_x0 = max(0, dx)
    dst_x1 = dst_x0 + (src_x1 - src_x0)

    if src_z1 > src_z0 and src_x1 > src_x0:
        out[dst_z0:dst_z1, dst_x0:dst_x1] = xz[src_z0:src_z1, src_x0:src_x1]
    return out


def align_peak_to_center_2d(xz: np.ndarray, z_peak: int, x_peak: int) -> np.ndarray:
    zc = xz.shape[0] // 2
    xc = xz.shape[1] // 2
    dz = zc - int(z_peak)
    dx = xc - int(x_peak)
    return shift_no_wrap_2d(xz, dz=dz, dx=dx)


def load_case(
    npz_path: Path,
    dose_source: str,
    input_dose_scale: float,
    y_band_half: int,
    focus_mode: str,
    peak_source: str,
    align_peak: bool,
) -> tuple[np.ndarray, np.ndarray, str]:
    with np.load(npz_path) as z:
        required = {"input", "target", "spr", "energy_mev"}
        missing = required - set(z.files)
        if missing:
            raise KeyError(f"{npz_path.name}: missing keys {sorted(missing)}")

        inp = z["input"].astype(np.float32)  # [2, D, H, W]
        target = z["target"].astype(np.float32)  # [D, H, W]

    ct = inp[1]
    low = inp[0] * float(input_dose_scale)
    dose = target if dose_source == "target" else low

    peak_vol = target if peak_source == "target" else dose
    z_peak, y_peak, x_peak = peak_zyx(peak_vol)

    if focus_mode == "peak":
        ct_xz = xz_band_at_y(ct, y_peak, y_band_half)
        dose_xz = xz_band_at_y(dose, y_peak, y_band_half)
    else:
        ct_xz = xz_center_band(ct, y_band_half)
        dose_xz = xz_center_band(dose, y_band_half)

    if align_peak:
        ct_xz = align_peak_to_center_2d(ct_xz, z_peak=z_peak, x_peak=x_peak)
        dose_xz = align_peak_to_center_2d(dose_xz, z_peak=z_peak, x_peak=x_peak)

    return ct_xz, dose_xz, npz_path.stem


def main() -> None:
    args = parse_args()

    files = sorted(args.input_dir.glob(args.pattern))
    if not files:
        raise FileNotFoundError(f"No NPZ files found in {args.input_dir} with pattern {args.pattern!r}")

    if args.limit > 0:
        files = files[: args.limit]

    cols = max(1, int(args.cols))
    rows = math.ceil(len(files) / cols)

    fig, axs = plt.subplots(rows, cols, figsize=(4.0 * cols, 1.9 * rows), constrained_layout=True)
    axs = np.array(axs).reshape(rows, cols)

    for i, npz_path in enumerate(files):
        r = i // cols
        c = i % cols
        ax = axs[r, c]

        ct_xz, dose_xz, name = load_case(
            npz_path,
            dose_source=args.dose_source,
            input_dose_scale=args.input_dose_scale,
            y_band_half=args.y_band_half,
            focus_mode=args.focus_mode,
            peak_source=args.peak_source,
            align_peak=bool(args.align_peak),
        )

        ct_norm = robust_normalize(ct_xz, low_pct=1.0, high_pct=args.ct_percentile)
        dose_norm = robust_normalize(dose_xz, low_pct=0.0, high_pct=args.dose_percentile)
        alpha = np.clip(dose_norm, 0.0, 1.0) ** float(args.alpha_gamma)

        ax.imshow(ct_norm, cmap="bone", origin="lower", aspect="auto")
        ax.imshow(dose_norm, cmap="turbo", origin="lower", aspect="auto", alpha=alpha)
        ax.set_title(name, fontsize=11)
        ax.axis("off")

    # Hide empty cells in last row if any
    for i in range(len(files), rows * cols):
        r = i // cols
        c = i % cols
        axs[r, c].axis("off")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi)
    plt.close(fig)
    print(args.out)


if __name__ == "__main__":
    main()
