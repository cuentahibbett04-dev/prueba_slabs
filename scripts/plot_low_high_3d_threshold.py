#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="3D scatter for low/high dose using relative threshold")
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("artifacts/low_high_3d.png"))
    ap.add_argument(
        "--rel-threshold",
        type=float,
        default=0.001,
        help="Keep voxels with value >= rel-threshold * max(volume). 0.001 = 0.1%%",
    )
    ap.add_argument(
        "--input-dose-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to low-dose input channel before thresholding",
    )
    ap.add_argument("--max-points", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def sample_masked_points(vol: np.ndarray, rel_threshold: float, max_points: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vmax = float(np.max(vol))
    thr = float(rel_threshold) * vmax
    mask = vol >= thr
    z, y, x = np.where(mask)
    if z.size == 0:
        return (
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
            np.array([], dtype=np.float32),
        )

    v = vol[z, y, x].astype(np.float32)
    if z.size > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(z.size, size=max_points, replace=False)
        z, y, x, v = z[idx], y[idx], x[idx], v[idx]
    return z, y, x, v


def main() -> None:
    args = parse_args()

    with np.load(args.npz) as z:
        required = {"input", "target", "spr", "energy_mev"}
        missing = required - set(z.files)
        if missing:
            raise KeyError(f"{args.npz.name}: missing keys {sorted(missing)}")
        inp = z["input"].astype(np.float32)
        target = z["target"].astype(np.float32)

    low = inp[0] * float(args.input_dose_scale)
    high = target

    lz, ly, lx, lv = sample_masked_points(low, args.rel_threshold, args.max_points, args.seed)
    hz, hy, hx, hv = sample_masked_points(high, args.rel_threshold, args.max_points, args.seed)

    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    if lz.size > 0:
        sc1 = ax1.scatter(lx, ly, lz, c=lv, s=1, cmap="turbo", alpha=0.65)
        fig.colorbar(sc1, ax=ax1, fraction=0.046, pad=0.04, label="Low dose")
    ax1.set_title(f"Low 3D (>= {args.rel_threshold*100:.3f}% of max)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")

    if hz.size > 0:
        sc2 = ax2.scatter(hx, hy, hz, c=hv, s=1, cmap="turbo", alpha=0.65)
        fig.colorbar(sc2, ax=ax2, fraction=0.046, pad=0.04, label="High dose")
    ax2.set_title(f"High 3D (>= {args.rel_threshold*100:.3f}% of max)")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("z")

    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180)
    plt.close(fig)
    print(args.out)


if __name__ == "__main__":
    main()
