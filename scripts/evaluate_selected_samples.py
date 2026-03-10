#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from proton_denoise.data import ProtonDoseDataset
from proton_denoise.metrics import bragg_peak_index, central_axis_profile
from proton_denoise.model import load_model_from_checkpoint


def core_mean_depth_profile(dose: np.ndarray, half_width: int = 2) -> np.ndarray:
    nz, ny, nx = dose.shape
    cy, cx = ny // 2, nx // 2
    y0, y1 = max(0, cy - half_width), min(ny, cy + half_width + 1)
    x0, x1 = max(0, cx - half_width), min(nx, cx + half_width + 1)
    return dose[:, y0:y1, x0:x1].mean(axis=(1, 2))


def longitudinal_xz_slice(dose: np.ndarray) -> np.ndarray:
    """Return beam-aligned x-z slice through central y."""
    _, ny, _ = dose.shape
    cy = ny // 2
    return dose[:, cy, :]


def evaluate_one(
    model: torch.nn.Module,
    ds: ProtonDoseDataset,
    idx: int,
    out_dir: Path,
    device: torch.device,
    voxel_mm: tuple[float, float, float],
    low_plot_scale: float,
    ref_pred_only: bool,
) -> dict[str, float]:
    item = ds[idx]
    x = item["input"].unsqueeze(0).to(device)
    y = item["target"][0].cpu().numpy()

    with torch.no_grad():
        pred = model(x).cpu().numpy()[0, 0]

    noisy = item["input"][0].cpu().numpy()
    noisy_plot = noisy * low_plot_scale

    p_y = central_axis_profile(y)
    p_pred = central_axis_profile(pred)
    p_noisy = central_axis_profile(noisy)
    p_noisy_plot = central_axis_profile(noisy_plot)

    pc_y = core_mean_depth_profile(y)
    pc_pred = core_mean_depth_profile(pred)
    pc_noisy_plot = core_mean_depth_profile(noisy_plot)

    z_peak = bragg_peak_index(p_y)
    z_mm = np.arange(len(p_y)) * voxel_mm[2]

    mae = float(np.mean(np.abs(pred - y)))
    peak_err_mm = float(abs(bragg_peak_index(p_pred) - bragg_peak_index(p_y)) * voxel_mm[2])

    sample_out = out_dir / f"sample_{idx:03d}"
    sample_out.mkdir(parents=True, exist_ok=True)

    pred_xz = longitudinal_xz_slice(pred)
    ref_xz = longitudinal_xz_slice(y)
    noisy_xz = longitudinal_xz_slice(noisy)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(ref_xz, cmap="inferno", origin="lower", aspect="auto")
    axs[0].set_title("Reference (x-z)")
    axs[1].imshow(pred_xz, cmap="inferno", origin="lower", aspect="auto")
    axs[1].set_title("Prediction (x-z)")
    axs[2].imshow(noisy_xz, cmap="inferno", origin="lower", aspect="auto")
    axs[2].set_title("Input low (x-z)")
    for ax in axs:
        ax.set_xlabel("x voxel")
        ax.set_ylabel("z voxel")
    fig.tight_layout()
    fig.savefig(sample_out / "input_pred_ref_slice.png", dpi=170)
    plt.close(fig)

    plt.figure(figsize=(6, 4))
    plt.plot(z_mm, p_y, label="Reference (axis)", linewidth=2)
    plt.plot(z_mm, p_pred, label="Prediction (axis)", linewidth=1.8)
    plt.plot(z_mm, pc_y, label="Reference (core mean)", linestyle="--")
    plt.plot(z_mm, pc_pred, label="Prediction (core mean)", linestyle="--")
    if not ref_pred_only:
        plt.plot(z_mm, p_noisy_plot, label=f"Input low (axis) x{low_plot_scale:g}", alpha=0.85)
        plt.plot(z_mm, pc_noisy_plot, label=f"Input low (core mean) x{low_plot_scale:g}", linestyle="--")
    plt.xlabel("Depth (mm)")
    plt.ylabel("Normalized dose")
    plt.title(f"Depth-dose analysis (idx={idx}, E={float(item['energy_mev'].item()):.1f} MeV)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(sample_out / "depth_profile.png", dpi=170)
    plt.close()

    return {
        "index": float(idx),
        "energy_mev": float(item["energy_mev"].item()),
        "mae_pred_vs_ref": mae,
        "peak_error_mm": peak_err_mm,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate selected sample indices and export input/pred/ref plots")
    parser.add_argument("--data-root", type=str, default="data_real_photon_1000")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="test")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--indices", type=int, nargs="+", required=True)
    parser.add_argument("--out-dir", type=str, default="artifacts/selected_eval")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--low-plot-scale", type=float, default=50.0)
    parser.add_argument("--ref-pred-only", action="store_true")
    parser.add_argument("--input-norm-mode", choices=["none", "per_channel_max", "global_max"], default="none")
    parser.add_argument("--input-dose-scale", type=float, default=1.0)
    parser.add_argument("--no-normalize-target", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    ds = ProtonDoseDataset(
        Path(args.data_root) / args.split,
        normalize_target=not args.no_normalize_target,
        input_norm_mode=args.input_norm_mode,
        input_dose_scale=args.input_dose_scale,
    )

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = load_model_from_checkpoint(ckpt, in_channels=2, out_channels=1).to(device)
    model.eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for idx in args.indices:
        if idx < 0 or idx >= len(ds):
            raise IndexError(f"Index {idx} out of bounds for split '{args.split}' with {len(ds)} samples")
        results.append(
            evaluate_one(
                model,
                ds,
                idx,
                out_dir,
                device,
                voxel_mm=(2.0, 2.0, 2.0),
                low_plot_scale=float(args.low_plot_scale),
                ref_pred_only=bool(args.ref_pred_only),
            )
        )

    for r in results:
        print(r)


if __name__ == "__main__":
    main()
