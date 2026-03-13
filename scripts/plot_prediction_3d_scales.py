#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from proton_denoise.data import ProtonDoseDataset
from proton_denoise.model import load_model_from_checkpoint


def sample_masked_points(
    vol: np.ndarray,
    rel_threshold: float,
    max_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    vmax = float(np.max(vol))
    p99 = float(np.percentile(vol, 99.5))
    thr_abs = float(rel_threshold) * vmax

    mask = vol >= thr_abs
    z, y, x = np.where(mask)
    if z.size == 0:
        stats = {
            "max": vmax,
            "p99_5": p99,
            "threshold_abs": thr_abs,
            "selected_points": 0.0,
        }
        return (
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
            np.array([], dtype=np.float32),
            stats,
        )

    vals = vol[z, y, x].astype(np.float32)
    if z.size > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(z.size, size=max_points, replace=False)
        z, y, x, vals = z[idx], y[idx], x[idx], vals[idx]

    stats = {
        "max": vmax,
        "p99_5": p99,
        "threshold_abs": thr_abs,
        "selected_points": float(z.size),
    }
    return z, y, x, vals, stats


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description="3D plot of Reference/Prediction/Low with independent scales")
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--out", type=Path, default=Path("artifacts/pred_ref_low_3d.png"))
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--input-norm-mode", choices=["none", "per_channel_max", "global_max"], default="none")
    ap.add_argument("--input-dose-scale", type=float, default=1.0)
    ap.add_argument("--crop-shape", type=int, nargs=3, default=None, metavar=("D", "H", "W"))
    ap.add_argument("--crop-focus", choices=["center", "maxdose"], default="center")
    ap.add_argument("--no-normalize-target", action="store_true")
    ap.add_argument(
        "--plot-target-from-npz-raw",
        action="store_true",
        help=(
            "Use raw target from NPZ for plotting. If dataset target normalization is enabled, "
            "prediction is multiplied by raw target max to compare in the same physical scale."
        ),
    )
    ap.add_argument("--low-plot-mode", choices=["unscaled", "model_input"], default="unscaled")
    ap.add_argument(
        "--plot-low-from-npz-raw",
        action="store_true",
        help="Use raw input[0] from NPZ for plotting (no additional loader scaling)",
    )
    ap.add_argument("--rel-threshold", type=float, default=0.001, help="Relative threshold (0.001 = 0.1%)")
    ap.add_argument("--max-points", type=int, default=70000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    ds = ProtonDoseDataset(
        Path(args.data_root) / args.split,
        normalize_target=not args.no_normalize_target,
        input_norm_mode=args.input_norm_mode,
        input_dose_scale=args.input_dose_scale,
        crop_shape=args.crop_shape,
        crop_focus=args.crop_focus,
    )
    if args.index < 0 or args.index >= len(ds):
        raise IndexError(f"Index {args.index} out of bounds for split '{args.split}' with {len(ds)} samples")

    item = ds[args.index]
    x = item["input"].unsqueeze(0).to(device)
    ref_norm = item["target"][0].cpu().numpy()
    low_model = item["input"][0].cpu().numpy()

    npz_path = Path(item["path"])
    raw_low = None
    raw_target = None
    with np.load(npz_path) as z:
        raw_low = z["input"][0].astype(np.float32)
        raw_target = z["target"].astype(np.float32)

    if args.crop_shape is not None:
        crop_shape = tuple(int(v) for v in args.crop_shape)
        if args.crop_focus == "maxdose":
            center = tuple(int(v) for v in np.unravel_index(np.argmax(raw_target), raw_target.shape))
        else:
            d, h, w = raw_target.shape
            center = (d // 2, h // 2, w // 2)
        raw_low = ProtonDoseDataset._crop_or_pad_3d(raw_low, crop_shape, center)
        raw_target = ProtonDoseDataset._crop_or_pad_3d(raw_target, crop_shape, center)

    if args.plot_low_from_npz_raw:
        low = raw_low
    elif args.low_plot_mode == "unscaled" and float(args.input_dose_scale) != 0.0:
        low = low_model / float(args.input_dose_scale)
    else:
        low = low_model

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = load_model_from_checkpoint(ckpt, in_channels=2, out_channels=1).to(device)
    model.eval()
    pred_norm = model(x).cpu().numpy()[0, 0]

    if args.plot_target_from_npz_raw:
        ref = raw_target
        if args.no_normalize_target:
            pred = pred_norm
        else:
            tmax = float(np.max(raw_target))
            pred = pred_norm * tmax if tmax > 0.0 else pred_norm
    else:
        ref = ref_norm
        pred = pred_norm

    rz, ry, rx, rv, rstats = sample_masked_points(ref, args.rel_threshold, args.max_points, args.seed)
    pz, py, px, pv, pstats = sample_masked_points(pred, args.rel_threshold, args.max_points, args.seed)
    lz, ly, lx, lv, lstats = sample_masked_points(low, args.rel_threshold, args.max_points, args.seed)

    fig = plt.figure(figsize=(18, 6))
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")

    if rz.size > 0:
        sc1 = ax1.scatter(rx, ry, rz, c=rv, s=1, cmap="inferno", alpha=0.65)
        fig.colorbar(sc1, ax=ax1, fraction=0.046, pad=0.04, label="Reference scale")
    ax1.set_title(
        "Reference\\n"
        f"max={rstats['max']:.3e}, p99.5={rstats['p99_5']:.3e}, thr={rstats['threshold_abs']:.3e}"
    )
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")

    if pz.size > 0:
        sc2 = ax2.scatter(px, py, pz, c=pv, s=1, cmap="inferno", alpha=0.65)
        fig.colorbar(sc2, ax=ax2, fraction=0.046, pad=0.04, label="Prediction scale")
    ax2.set_title(
        "Prediction\\n"
        f"max={pstats['max']:.3e}, p99.5={pstats['p99_5']:.3e}, thr={pstats['threshold_abs']:.3e}"
    )
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("z")

    if lz.size > 0:
        sc3 = ax3.scatter(lx, ly, lz, c=lv, s=1, cmap="inferno", alpha=0.65)
        fig.colorbar(sc3, ax=ax3, fraction=0.046, pad=0.04, label="Input low scale")
    ax3.set_title(
        f"Input low [{args.low_plot_mode}]\\n"
        f"max={lstats['max']:.3e}, p99.5={lstats['p99_5']:.3e}, thr={lstats['threshold_abs']:.3e}"
    )
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_zlabel("z")

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180)
    plt.close(fig)

    meta = {
        "index": int(args.index),
        "energy_mev": float(item["energy_mev"].item()),
        "rel_threshold": float(args.rel_threshold),
        "low_plot_mode": str(args.low_plot_mode),
        "plot_low_from_npz_raw": bool(args.plot_low_from_npz_raw),
        "plot_target_from_npz_raw": bool(args.plot_target_from_npz_raw),
        "no_normalize_target": bool(args.no_normalize_target),
        "input_norm_mode": str(args.input_norm_mode),
        "input_dose_scale": float(args.input_dose_scale),
        "reference": rstats,
        "prediction": pstats,
        "input_low": lstats,
    }
    meta_path = args.out.with_suffix(".json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(args.out)
    print(meta_path)


if __name__ == "__main__":
    main()
