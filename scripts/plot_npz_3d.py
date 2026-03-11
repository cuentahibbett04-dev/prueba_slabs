#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def sample_top_voxels(
    vol: np.ndarray,
    percentile: float,
    max_points: int,
    seed: int,
    nonzero_only: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    thr = float(np.percentile(vol, percentile))
    mask = vol >= thr
    if nonzero_only:
        mask = mask & (vol > 0)
    z, y, x = np.where(mask)
    if z.size == 0:
        return (
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
            np.array([], dtype=np.float32),
        )

    values = vol[z, y, x].astype(np.float32)
    if z.size > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(z.size, size=max_points, replace=False)
        z, y, x, values = z[idx], y[idx], x[idx], values[idx]
    return z, y, x, values


def to_3d(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 3:
        return arr.astype(np.float32)
    if arr.ndim == 4 and arr.shape[0] == 1:
        return arr[0].astype(np.float32)
    raise ValueError(f"Expected 3D array or [1,D,H,W], got shape={arr.shape}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Render 3D scatter of pred/target from NPZ")
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("artifacts/pred_target_3d.png"))
    ap.add_argument("--percentile", type=float, default=99.5, help="Keep voxels >= this percentile")
    ap.add_argument("--max-points", type=int, default=25000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--pred-nonzero-only",
        action="store_true",
        help="For prediction, keep only voxels with value > 0",
    )
    args = ap.parse_args()

    with np.load(args.npz, allow_pickle=True) as z:
        if "pred" not in z.files or "target" not in z.files:
            raise KeyError("NPZ must contain keys: pred and target")
        pred = to_3d(z["pred"])
        target = to_3d(z["target"])

    pz, py, px, pv = sample_top_voxels(
        pred,
        args.percentile,
        args.max_points,
        args.seed,
        nonzero_only=bool(args.pred_nonzero_only),
    )
    tz, ty, tx, tv = sample_top_voxels(target, args.percentile, args.max_points, args.seed)

    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    if pz.size > 0:
        sc1 = ax1.scatter(px, py, pz, c=pv, s=1, cmap="inferno", alpha=0.65)
        fig.colorbar(sc1, ax=ax1, fraction=0.046, pad=0.04, label="Pred intensity")
    ax1.set_title(f"Prediction 3D (p>={args.percentile:g})")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")

    if tz.size > 0:
        sc2 = ax2.scatter(tx, ty, tz, c=tv, s=1, cmap="inferno", alpha=0.65)
        fig.colorbar(sc2, ax=ax2, fraction=0.046, pad=0.04, label="Target intensity")
    ax2.set_title(f"Target 3D (p>={args.percentile:g})")
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
