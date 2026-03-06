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
from proton_denoise.model import ResUNet3D


def load_prediction(ckpt_path: Path, x: torch.Tensor, device: torch.device) -> tuple[np.ndarray, int]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    base_channels = int(ckpt.get("base_channels", 16))
    epoch = int(ckpt.get("epoch", -1))
    output_activation = str(ckpt.get("output_activation", "identity"))
    model = ResUNet3D(
        in_channels=2,
        out_channels=1,
        base_channels=base_channels,
        output_activation=output_activation,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    with torch.no_grad():
        yhat = model(x.to(device)).cpu().numpy()[0, 0]
    return yhat, epoch


def estimate_beam_center_xy(ref: np.ndarray, entrance_slices: int = 12) -> tuple[int, int]:
    """Estimate beam center from dose-weighted centroid near beam entrance."""
    nz, ny, nx = ref.shape
    z1 = min(max(entrance_slices, 1), nz)
    slab = np.clip(ref[:z1], 0.0, None)
    w = slab.sum(axis=0)  # [y, x]
    s = float(w.sum())
    if s <= 0:
        return ny // 2, nx // 2
    yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    cy = int(round(float((w * yy).sum() / s)))
    cx = int(round(float((w * xx).sum() / s)))
    cy = int(np.clip(cy, 0, ny - 1))
    cx = int(np.clip(cx, 0, nx - 1))
    return cy, cx


def depth_profile_center(vol: np.ndarray, cy: int, cx: int, half_width_xy: int = 2) -> np.ndarray:
    nz, ny, nx = vol.shape
    y0, y1 = max(0, cy - half_width_xy), min(ny, cy + half_width_xy + 1)
    x0, x1 = max(0, cx - half_width_xy), min(nx, cx + half_width_xy + 1)
    return vol[:, y0:y1, x0:x1].mean(axis=(1, 2))


def lateral_x_profile(vol: np.ndarray, z_idx: int, cy: int, half_width_y: int = 2, half_width_z: int = 1) -> np.ndarray:
    nz, ny, _ = vol.shape
    y0, y1 = max(0, cy - half_width_y), min(ny, cy + half_width_y + 1)
    z0, z1 = max(0, z_idx - half_width_z), min(nz, z_idx + half_width_z + 1)
    return vol[z0:z1, y0:y1, :].mean(axis=(0, 1))


def lateral_y_profile(vol: np.ndarray, z_idx: int, cx: int, half_width_x: int = 2, half_width_z: int = 1) -> np.ndarray:
    nz, _, nx = vol.shape
    x0, x1 = max(0, cx - half_width_x), min(nx, cx + half_width_x + 1)
    z0, z1 = max(0, z_idx - half_width_z), min(nz, z_idx + half_width_z + 1)
    return vol[z0:z1, :, x0:x1].mean(axis=(0, 2))


def main() -> None:
    ap = argparse.ArgumentParser(description="Beam-centered profile comparison across epochs")
    ap.add_argument("--data-root", default="data_real_photon_1000")
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--sample-index", type=int, default=0)
    ap.add_argument("--ckpt10", type=Path, required=True)
    ap.add_argument("--ckpt15", type=Path, required=True)
    ap.add_argument("--ckpt20", type=Path, required=True)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--out", type=Path, default=Path("artifacts/beam_profiles_epoch_compare.png"))
    ap.add_argument("--input-scale", type=float, default=50.0, help="Scale factor for low-input curve")
    ap.add_argument(
        "--match-input-to-ref-scale",
        action="store_true",
        help="Auto-scale noisy input so its depth-profile peak matches the reference peak",
    )
    ap.add_argument("--voxel-mm", type=float, nargs=3, default=[2.0, 2.0, 2.0])
    ap.add_argument("--depth-half-width-xy", type=int, default=2, help="Half-width for x/y averaging in depth profile")
    ap.add_argument("--lateral-half-width", type=int, default=2, help="Half-width around beam center for lateral averaging")
    ap.add_argument("--lateral-z-half-width", type=int, default=1, help="Half-width in z around z_peak for lateral averaging")
    ap.add_argument("--plot-clamp-min", type=float, default=0.0, help="Minimum value used for plotting")
    ap.add_argument("--input-norm-mode", choices=["none", "per_channel_max", "global_max"], default="none")
    ap.add_argument("--input-dose-scale", type=float, default=1.0)
    ap.add_argument("--no-normalize-target", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    vx, vy, vz = map(float, args.voxel_mm)

    ds = ProtonDoseDataset(
        Path(args.data_root) / args.split,
        normalize_target=not args.no_normalize_target,
        input_norm_mode=args.input_norm_mode,
        input_dose_scale=args.input_dose_scale,
    )
    item = ds[args.sample_index]

    x = item["input"].unsqueeze(0)
    ref = item["target"][0].numpy()
    inp_raw = item["input"][0].numpy()

    p10, ep10 = load_prediction(args.ckpt10, x, device)
    p15, ep15 = load_prediction(args.ckpt15, x, device)
    p20, ep20 = load_prediction(args.ckpt20, x, device)

    cmin = float(args.plot_clamp_min)
    ref = np.clip(ref, cmin, None)
    inp_raw = np.clip(inp_raw, cmin, None)
    p10 = np.clip(p10, cmin, None)
    p15 = np.clip(p15, cmin, None)
    p20 = np.clip(p20, cmin, None)

    cy, cx = estimate_beam_center_xy(ref)

    # Optional: align noisy input amplitude to reference for easier visual comparison.
    input_scale_eff = float(args.input_scale)
    if args.match_input_to_ref_scale:
        ref_tmp = depth_profile_center(ref, cy=cy, cx=cx, half_width_xy=args.depth_half_width_xy)
        inp_tmp = depth_profile_center(inp_raw, cy=cy, cx=cx, half_width_xy=args.depth_half_width_xy)
        ref_peak = float(np.max(ref_tmp))
        inp_peak = float(np.max(inp_tmp))
        if inp_peak > 1e-12:
            input_scale_eff = ref_peak / inp_peak

    inp = inp_raw * input_scale_eff

    # Use reference peak depth as common comparison plane for lateral profiles.
    z_peak = int(np.argmax(depth_profile_center(ref, cy=cy, cx=cx, half_width_xy=args.depth_half_width_xy)))

    prof_depth_ref = depth_profile_center(ref, cy=cy, cx=cx, half_width_xy=args.depth_half_width_xy)
    prof_depth_inp = depth_profile_center(inp, cy=cy, cx=cx, half_width_xy=args.depth_half_width_xy)
    prof_depth_10 = depth_profile_center(p10, cy=cy, cx=cx, half_width_xy=args.depth_half_width_xy)
    prof_depth_15 = depth_profile_center(p15, cy=cy, cx=cx, half_width_xy=args.depth_half_width_xy)
    prof_depth_20 = depth_profile_center(p20, cy=cy, cx=cx, half_width_xy=args.depth_half_width_xy)

    prof_x_ref = lateral_x_profile(ref, z_peak, cy=cy, half_width_y=args.lateral_half_width, half_width_z=args.lateral_z_half_width)
    prof_x_inp = lateral_x_profile(inp, z_peak, cy=cy, half_width_y=args.lateral_half_width, half_width_z=args.lateral_z_half_width)
    prof_x_10 = lateral_x_profile(p10, z_peak, cy=cy, half_width_y=args.lateral_half_width, half_width_z=args.lateral_z_half_width)
    prof_x_15 = lateral_x_profile(p15, z_peak, cy=cy, half_width_y=args.lateral_half_width, half_width_z=args.lateral_z_half_width)
    prof_x_20 = lateral_x_profile(p20, z_peak, cy=cy, half_width_y=args.lateral_half_width, half_width_z=args.lateral_z_half_width)

    prof_y_ref = lateral_y_profile(ref, z_peak, cx=cx, half_width_x=args.lateral_half_width, half_width_z=args.lateral_z_half_width)
    prof_y_inp = lateral_y_profile(inp, z_peak, cx=cx, half_width_x=args.lateral_half_width, half_width_z=args.lateral_z_half_width)
    prof_y_10 = lateral_y_profile(p10, z_peak, cx=cx, half_width_x=args.lateral_half_width, half_width_z=args.lateral_z_half_width)
    prof_y_15 = lateral_y_profile(p15, z_peak, cx=cx, half_width_x=args.lateral_half_width, half_width_z=args.lateral_z_half_width)
    prof_y_20 = lateral_y_profile(p20, z_peak, cx=cx, half_width_x=args.lateral_half_width, half_width_z=args.lateral_z_half_width)

    z_mm = np.arange(len(prof_depth_ref)) * vz
    x_mm = (np.arange(len(prof_x_ref)) - cx) * vx
    y_mm = (np.arange(len(prof_y_ref)) - cy) * vy

    fig, axs = plt.subplots(1, 3, figsize=(17, 4.5), constrained_layout=True)

    axs[0].plot(z_mm, prof_depth_ref, label="Reference", linewidth=2)
    axs[0].plot(z_mm, prof_depth_10, label=f"Pred E{ep10}")
    axs[0].plot(z_mm, prof_depth_15, label=f"Pred E{ep15}")
    axs[0].plot(z_mm, prof_depth_20, label=f"Pred E{ep20}")
    axs[0].plot(z_mm, prof_depth_inp, label=f"Input x{input_scale_eff:g}", alpha=0.8)
    axs[0].set_title(f"Depth Profile (beam axis, xy band={2*args.depth_half_width_xy+1})")
    axs[0].set_xlabel("z (mm)")
    axs[0].set_ylabel("Normalized dose")
    axs[0].set_ylim(bottom=0.0)
    axs[0].grid(alpha=0.25)

    axs[1].plot(x_mm, prof_x_ref, label="Reference", linewidth=2)
    axs[1].plot(x_mm, prof_x_10, label=f"Pred E{ep10}")
    axs[1].plot(x_mm, prof_x_15, label=f"Pred E{ep15}")
    axs[1].plot(x_mm, prof_x_20, label=f"Pred E{ep20}")
    axs[1].plot(x_mm, prof_x_inp, label=f"Input x{input_scale_eff:g}", alpha=0.8)
    axs[1].set_title(
        f"Centered Lateral X (z_peak={z_peak}, y band={2*args.lateral_half_width+1}, "
        f"z band={2*args.lateral_z_half_width+1})"
    )
    axs[1].set_xlabel("x offset from beam center (mm)")
    axs[1].set_ylabel("Normalized dose")
    axs[1].set_ylim(bottom=0.0)
    axs[1].grid(alpha=0.25)

    axs[2].plot(y_mm, prof_y_ref, label="Reference", linewidth=2)
    axs[2].plot(y_mm, prof_y_10, label=f"Pred E{ep10}")
    axs[2].plot(y_mm, prof_y_15, label=f"Pred E{ep15}")
    axs[2].plot(y_mm, prof_y_20, label=f"Pred E{ep20}")
    axs[2].plot(y_mm, prof_y_inp, label=f"Input x{input_scale_eff:g}", alpha=0.8)
    axs[2].set_title(
        f"Centered Lateral Y (z_peak={z_peak}, x band={2*args.lateral_half_width+1}, "
        f"z band={2*args.lateral_z_half_width+1})"
    )
    axs[2].set_xlabel("y offset from beam center (mm)")
    axs[2].set_ylabel("Normalized dose")
    axs[2].set_ylim(bottom=0.0)
    axs[2].grid(alpha=0.25)

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180)
    plt.close(fig)

    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
