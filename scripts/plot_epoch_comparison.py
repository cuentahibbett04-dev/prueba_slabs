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
from proton_denoise.model import load_model_from_checkpoint


def longitudinal_xz_slice(vol: np.ndarray, half_width_y: int = 2) -> np.ndarray:
    """Beam-centered x-z slice averaged over a y-band to reduce voxel noise."""
    _, ny, _ = vol.shape
    cy = ny // 2
    y0, y1 = max(0, cy - half_width_y), min(ny, cy + half_width_y + 1)
    return vol[:, y0:y1, :].mean(axis=1)


def robust_vmax(arrays: list[np.ndarray], pct: float = 99.5, eps: float = 1e-8) -> float:
    vals = np.concatenate([a.ravel() for a in arrays])
    vmax = float(np.percentile(vals, pct))
    return max(vmax, eps)


def load_pred(ckpt_path: Path, x: torch.Tensor, device: torch.device) -> tuple[np.ndarray, int]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    epoch = int(ckpt.get("epoch", -1))
    model = load_model_from_checkpoint(ckpt, in_channels=2, out_channels=1).to(device)
    model.eval()
    with torch.no_grad():
        yhat = model(x.to(device)).cpu().numpy()[0, 0]
    return yhat, epoch


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare epoch predictions in a single image")
    ap.add_argument("--data-root", default="data_real_photon_1000")
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--sample-index", type=int, default=0)
    ap.add_argument("--ckpt10", type=Path, required=True)
    ap.add_argument("--ckpt15", type=Path, required=True)
    ap.add_argument("--ckpt20", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("artifacts/epoch_compare_10_15_20.png"))
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--band-half-width", type=int, default=2, help="Half-width (in y voxels) for x-z band averaging")
    ap.add_argument("--input-plot-scale", type=float, default=50.0, help="Scale factor for low-input in plots")
    ap.add_argument("--plot-clamp-min", type=float, default=0.0, help="Minimum value used for plotting")
    ap.add_argument("--input-norm-mode", choices=["none", "per_channel_max", "global_max"], default="none")
    ap.add_argument("--input-dose-scale", type=float, default=1.0)
    ap.add_argument("--no-normalize-target", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    ds = ProtonDoseDataset(
        Path(args.data_root) / args.split,
        normalize_target=not args.no_normalize_target,
        input_norm_mode=args.input_norm_mode,
        input_dose_scale=args.input_dose_scale,
    )
    item = ds[args.sample_index]
    x = item["input"].unsqueeze(0)
    y = item["target"][0].numpy()
    input_low = item["input"][0].numpy()
    input_low_plot = input_low * float(args.input_plot_scale)

    pred10, ep10 = load_pred(args.ckpt10, x, device)
    pred15, ep15 = load_pred(args.ckpt15, x, device)
    pred20, ep20 = load_pred(args.ckpt20, x, device)

    cmin = float(args.plot_clamp_min)
    y = np.clip(y, cmin, None)
    input_low_plot = np.clip(input_low_plot, cmin, None)
    pred10 = np.clip(pred10, cmin, None)
    pred15 = np.clip(pred15, cmin, None)
    pred20 = np.clip(pred20, cmin, None)

    ref_xz = longitudinal_xz_slice(y, half_width_y=args.band_half_width)
    p10_xz = longitudinal_xz_slice(pred10, half_width_y=args.band_half_width)
    p15_xz = longitudinal_xz_slice(pred15, half_width_y=args.band_half_width)
    p20_xz = longitudinal_xz_slice(pred20, half_width_y=args.band_half_width)
    in_xz = longitudinal_xz_slice(input_low_plot, half_width_y=args.band_half_width)

    e10 = np.abs(p10_xz - ref_xz)
    e15 = np.abs(p15_xz - ref_xz)
    e20 = np.abs(p20_xz - ref_xz)
    ein = np.abs(in_xz - ref_xz)

    mae10 = float(np.mean(np.abs(pred10 - y)))
    mae15 = float(np.mean(np.abs(pred15 - y)))
    mae20 = float(np.mean(np.abs(pred20 - y)))
    maein = float(np.mean(np.abs(input_low_plot - y)))

    # Keep target/preds visible: use robust shared scale excluding noisy input.
    vmax_refpred = robust_vmax([ref_xz, p10_xz, p15_xz, p20_xz], pct=99.5)
    # Input gets its own robust scale to avoid flattening the others.
    vmax_input = robust_vmax([in_xz], pct=99.5)

    err_vmax = float(max(e10.max(), e15.max(), e20.max(), ein.max(), 1e-8))

    fig, axs = plt.subplots(2, 5, figsize=(18, 7), constrained_layout=True)

    im0 = axs[0, 0].imshow(ref_xz, cmap="inferno", origin="lower", aspect="auto", vmin=0, vmax=vmax_refpred)
    axs[0, 0].set_title("Target (Reference)")
    axs[0, 1].imshow(p10_xz, cmap="inferno", origin="lower", aspect="auto", vmin=0, vmax=vmax_refpred)
    axs[0, 1].set_title(f"Prediction E{ep10}\nMAE={mae10:.3e}")
    axs[0, 2].imshow(p15_xz, cmap="inferno", origin="lower", aspect="auto", vmin=0, vmax=vmax_refpred)
    axs[0, 2].set_title(f"Prediction E{ep15}\nMAE={mae15:.3e}")
    axs[0, 3].imshow(p20_xz, cmap="inferno", origin="lower", aspect="auto", vmin=0, vmax=vmax_refpred)
    axs[0, 3].set_title(f"Prediction E{ep20}\nMAE={mae20:.3e}")
    im_in = axs[0, 4].imshow(in_xz, cmap="inferno", origin="lower", aspect="auto", vmin=0, vmax=vmax_input)
    axs[0, 4].set_title(f"Input low x{args.input_plot_scale:g}\nMAE={maein:.3e}")

    em0 = axs[1, 0].imshow(np.zeros_like(ref_xz), cmap="magma", origin="lower", aspect="auto", vmin=0, vmax=err_vmax)
    axs[1, 0].set_title("|Target-Target| = 0")
    axs[1, 1].imshow(e10, cmap="magma", origin="lower", aspect="auto", vmin=0, vmax=err_vmax)
    axs[1, 1].set_title(f"|E{ep10} - Target|")
    axs[1, 2].imshow(e15, cmap="magma", origin="lower", aspect="auto", vmin=0, vmax=err_vmax)
    axs[1, 2].set_title(f"|E{ep15} - Target|")
    axs[1, 3].imshow(e20, cmap="magma", origin="lower", aspect="auto", vmin=0, vmax=err_vmax)
    axs[1, 3].set_title(f"|E{ep20} - Target|")
    axs[1, 4].imshow(ein, cmap="magma", origin="lower", aspect="auto", vmin=0, vmax=err_vmax)
    axs[1, 4].set_title(f"|Input x{args.input_plot_scale:g} - Target|")

    for r in range(2):
        for c in range(5):
            axs[r, c].set_xlabel("x voxel")
            axs[r, c].set_ylabel("z voxel")

    cbar0 = fig.colorbar(im0, ax=axs[0, :4], fraction=0.02, pad=0.01)
    cbar0.set_label("Dose scale (target + predictions)")
    cbar_in = fig.colorbar(im_in, ax=axs[0, 4], fraction=0.04, pad=0.01)
    cbar_in.set_label("Dose scale (input)")
    cbar1 = fig.colorbar(em0, ax=axs[1, :], fraction=0.02, pad=0.01)
    cbar1.set_label("Absolute error vs target")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180)
    plt.close(fig)

    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
