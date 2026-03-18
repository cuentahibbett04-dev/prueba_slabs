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


def _get_ckpt_data_prep(ckpt: dict) -> dict:
    dp = ckpt.get("data_prep", {})
    if not isinstance(dp, dict):
        return {}
    out: dict[str, object] = {}
    if "normalize_target" in dp:
        out["normalize_target"] = bool(dp["normalize_target"])
    if "input_norm_mode" in dp and isinstance(dp["input_norm_mode"], str):
        out["input_norm_mode"] = dp["input_norm_mode"]
    if "input_dose_scale" in dp:
        out["input_dose_scale"] = float(dp["input_dose_scale"])
    if "crop_shape" in dp and dp["crop_shape"] is not None:
        out["crop_shape"] = tuple(int(v) for v in dp["crop_shape"])
    if "crop_focus" in dp and isinstance(dp["crop_focus"], str):
        out["crop_focus"] = dp["crop_focus"]
    return out


def _estimate_beam_center_xy(vol: np.ndarray) -> tuple[int, int]:
    w = np.clip(vol, 0.0, None).sum(axis=0)
    s = float(w.sum())
    ny, nx = w.shape
    if s <= 0:
        return ny // 2, nx // 2
    yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    cy = int(round(float((w * yy).sum() / s)))
    cx = int(round(float((w * xx).sum() / s)))
    return int(np.clip(cy, 0, ny - 1)), int(np.clip(cx, 0, nx - 1))


def _crop_raw_like_dataset(raw_arr: np.ndarray, raw_target: np.ndarray, crop_shape: tuple[int, int, int] | None, crop_focus: str) -> np.ndarray:
    if crop_shape is None:
        return raw_arr
    if crop_focus == "maxdose":
        center = tuple(int(v) for v in np.unravel_index(np.argmax(raw_target), raw_target.shape))
    else:
        d, h, w = raw_target.shape
        center = (d // 2, h // 2, w // 2)
    return ProtonDoseDataset._crop_or_pad_3d(raw_arr, crop_shape, center)


def _add_limit_text(ax: plt.Axes, text: str) -> None:
    ax.text(
        0.02,
        0.04,
        text,
        transform=ax.transAxes,
        fontsize=8,
        color="white",
        ha="left",
        va="bottom",
        bbox={"facecolor": "black", "alpha": 0.45, "pad": 3, "edgecolor": "none"},
    )


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description="Paper-like beamlet slice figure with density and dose comparisons")
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--input-norm-mode", choices=["none", "per_channel_max", "global_max", "coupled_target_max"], default="none")
    ap.add_argument("--input-dose-scale", type=float, default=1.0)
    ap.add_argument("--crop-shape", type=int, nargs=3, default=None, metavar=("D", "H", "W"))
    ap.add_argument("--crop-focus", choices=["center", "maxdose"], default="center")
    ap.add_argument("--no-normalize-target", action="store_true")
    ap.add_argument("--no-use-checkpoint-data-prep", action="store_true")
    ap.add_argument("--voxel-mm", type=float, nargs=3, default=[2.0, 2.0, 2.0], metavar=("VX", "VY", "VZ"))
    ap.add_argument(
        "--slice-offsets-mm",
        type=float,
        nargs=3,
        default=[-2.0, 0.0, 2.0],
        metavar=("OFF1", "OFF2", "OFF3"),
        help="Three transverse offsets in mm relative to beam central axis (along y)",
    )
    ap.add_argument("--density-key", choices=["spr", "input1"], default="spr")
    ap.add_argument(
        "--high-noise-scale-mode",
        choices=["auto", "none", "times_history_ratio", "divide_history_ratio"],
        default="auto",
        help=(
            "Scaling mode for high-noise MC dose read from NPZ input[0]. "
            "'auto' chooses the variant (none, *ratio, /ratio) whose p99 best matches GT p99."
        ),
    )
    ap.add_argument("--dose-vmax-percentile", type=float, default=99.8)
    ap.add_argument("--density-vmin", type=float, default=None)
    ap.add_argument("--density-vmax", type=float, default=None)
    ap.add_argument(
        "--arrow-mm",
        type=float,
        nargs=4,
        action="append",
        default=None,
        metavar=("X0", "Z0", "X1", "Z1"),
        help="Optional arrow in mm on dose panels; can be passed multiple times",
    )
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    ckpt_data_prep = _get_ckpt_data_prep(ckpt)

    normalize_target = not args.no_normalize_target
    input_norm_mode = args.input_norm_mode
    input_dose_scale = float(args.input_dose_scale)
    crop_shape = tuple(args.crop_shape) if args.crop_shape is not None else None
    crop_focus = args.crop_focus

    if not args.no_use_checkpoint_data_prep and ckpt_data_prep:
        normalize_target = bool(ckpt_data_prep.get("normalize_target", normalize_target))
        input_norm_mode = str(ckpt_data_prep.get("input_norm_mode", input_norm_mode))
        input_dose_scale = float(ckpt_data_prep.get("input_dose_scale", input_dose_scale))
        crop_shape = ckpt_data_prep.get("crop_shape", crop_shape)
        crop_focus = str(ckpt_data_prep.get("crop_focus", crop_focus))

    ds = ProtonDoseDataset(
        Path(args.data_root) / args.split,
        normalize_target=normalize_target,
        input_norm_mode=input_norm_mode,
        input_dose_scale=input_dose_scale,
        crop_shape=crop_shape,
        crop_focus=crop_focus,
    )
    if args.index < 0 or args.index >= len(ds):
        raise IndexError(f"Index {args.index} out of bounds for split '{args.split}' with {len(ds)} samples")

    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    model = load_model_from_checkpoint(ckpt, in_channels=2, out_channels=1).to(device)
    model.eval()

    item = ds[args.index]
    x = item["input"].unsqueeze(0).to(device)
    pred_norm = model(x).detach().cpu().numpy()[0, 0]

    npz_path = Path(str(item["path"]))
    with np.load(npz_path) as z:
        raw_target = z["target"].astype(np.float32)
        raw_low = z["input"][0].astype(np.float32)
        low_events_val = int(z["low_events"].item()) if "low_events" in z else -1
        high_events_val = int(z["high_events"].item()) if "high_events" in z else -1
        if args.density_key == "spr":
            raw_density = z["spr"].astype(np.float32)
        else:
            raw_density = z["input"][1].astype(np.float32)

    raw_target = _crop_raw_like_dataset(raw_target, raw_target, crop_shape, crop_focus)
    raw_low = _crop_raw_like_dataset(raw_low, raw_target, crop_shape, crop_focus)
    raw_density = _crop_raw_like_dataset(raw_density, raw_target, crop_shape, crop_focus)

    hist_ratio = None
    if low_events_val > 0 and high_events_val > 0:
        rr = float(high_events_val) / float(low_events_val)
        if rr > 0.0:
            hist_ratio = rr

    if args.high_noise_scale_mode == "none" or hist_ratio is None:
        raw_low_scaled = raw_low
        chosen_scale_mode = "none"
    elif args.high_noise_scale_mode == "times_history_ratio":
        raw_low_scaled = raw_low * float(hist_ratio)
        chosen_scale_mode = "times_history_ratio"
    elif args.high_noise_scale_mode == "divide_history_ratio":
        raw_low_scaled = raw_low / float(hist_ratio)
        chosen_scale_mode = "divide_history_ratio"
    else:
        tgt_p99 = float(np.percentile(raw_target, 99.0))
        tgt_p99 = max(tgt_p99, 1e-12)
        cand = {
            "none": raw_low,
            "times_history_ratio": raw_low * float(hist_ratio),
            "divide_history_ratio": raw_low / float(hist_ratio),
        }
        best_key = "none"
        best_score = float("inf")
        for key, arr in cand.items():
            p99 = float(np.percentile(arr, 99.0))
            ratio = p99 / tgt_p99
            score = abs(np.log(max(ratio, 1e-12)))
            if score < best_score:
                best_score = score
                best_key = key
        raw_low_scaled = cand[best_key]
        chosen_scale_mode = best_key

    # Convert prediction to the same physical scale as raw low-noise target.
    if normalize_target:
        tmax = float(np.max(raw_target))
        pred = pred_norm * tmax if tmax > 0.0 else pred_norm
    else:
        pred = pred_norm

    norm_ref = float(np.max(raw_target))
    if norm_ref <= 0.0:
        norm_ref = 1.0

    gt_n = raw_target / norm_ref
    noisy_n = raw_low_scaled / norm_ref
    pred_n = pred / norm_ref

    cy, cx = _estimate_beam_center_xy(gt_n)
    vx, vy, vz = map(float, args.voxel_mm)
    nz, ny, nx = gt_n.shape

    offsets_mm = [float(v) for v in args.slice_offsets_mm]
    y_idxs = [int(np.clip(round(cy + off / max(vy, 1e-6)), 0, ny - 1)) for off in offsets_mm]

    dose_stack = np.concatenate([gt_n.ravel(), noisy_n.ravel(), pred_n.ravel()])
    dose_vmin = 0.0
    dose_vmax = float(np.percentile(dose_stack, float(args.dose_vmax_percentile)))
    if dose_vmax <= dose_vmin:
        dose_vmax = 1.0

    if args.density_vmin is not None:
        dens_vmin = float(args.density_vmin)
    else:
        dens_vmin = float(np.percentile(raw_density, 1.0))
    if args.density_vmax is not None:
        dens_vmax = float(args.density_vmax)
    else:
        dens_vmax = float(np.percentile(raw_density, 99.0))
    if dens_vmax <= dens_vmin:
        dens_vmax = dens_vmin + 1e-6

    fig, axs = plt.subplots(4, 3, figsize=(14, 14), constrained_layout=True)

    x_mm = np.arange(nx) * vx
    z_mm = np.arange(nz) * vz
    extent = [float(x_mm[0]), float(x_mm[-1]), float(z_mm[0]), float(z_mm[-1])]

    row_titles = [
        "Density",
        "Low-noise MC dose (GT)",
        "High-noise MC dose",
        "DeepMC prediction",
    ]

    cm_density = "gray"
    cm_dose = "inferno"

    for col, (off_mm, y_idx) in enumerate(zip(offsets_mm, y_idxs)):
        dens_xz = raw_density[:, y_idx, :]
        gt_xz = gt_n[:, y_idx, :]
        noisy_xz = noisy_n[:, y_idx, :]
        pred_xz = pred_n[:, y_idx, :]

        panels = [dens_xz, gt_xz, noisy_xz, pred_xz]
        for row, panel in enumerate(panels):
            ax = axs[row, col]
            if row == 0:
                im = ax.imshow(panel, cmap=cm_density, origin="lower", aspect="auto", extent=extent, vmin=dens_vmin, vmax=dens_vmax)
                _add_limit_text(ax, f"[{dens_vmin:.3g}, {dens_vmax:.3g}] g/cm^3")
            else:
                im = ax.imshow(panel, cmap=cm_dose, origin="lower", aspect="auto", extent=extent, vmin=dose_vmin, vmax=dose_vmax)
                _add_limit_text(ax, f"[{dose_vmin:.3g}, {dose_vmax:.3g}] Gy (norm)")
                if args.arrow_mm:
                    for arrow in args.arrow_mm:
                        x0, z0, x1, z1 = [float(v) for v in arrow]
                        ax.annotate(
                            "",
                            xy=(x1, z1),
                            xytext=(x0, z0),
                            arrowprops={"arrowstyle": "->", "color": "cyan", "lw": 1.8},
                        )

            if row == 0:
                ax.set_title(f"d_perp={off_mm:+.1f} mm")
            if col == 0:
                ax.set_ylabel(f"{row_titles[row]}\nz (mm)")
            else:
                ax.set_ylabel("z (mm)")
            ax.set_xlabel("x (mm)")

    fig.suptitle(
        f"Beamlet Slice Comparison (index={args.index}, case={npz_path.stem})\n"
        f"Dose normalized to max low-noise MC beamlet dose",
        fontsize=13,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180)
    plt.close(fig)

    meta = {
        "index": int(args.index),
        "case": npz_path.stem,
        "device": str(device),
        "data_root": str(args.data_root),
        "split": str(args.split),
        "checkpoint": str(args.checkpoint),
        "input_norm_mode": str(input_norm_mode),
        "input_dose_scale": float(input_dose_scale),
        "normalize_target": bool(normalize_target),
        "crop_shape": list(crop_shape) if crop_shape is not None else None,
        "crop_focus": str(crop_focus),
        "density_key": str(args.density_key),
        "low_events": int(low_events_val),
        "high_events": int(high_events_val),
        "history_ratio": float(hist_ratio) if hist_ratio is not None else None,
        "high_noise_scale_mode_requested": str(args.high_noise_scale_mode),
        "high_noise_scale_mode_applied": str(chosen_scale_mode),
        "slice_offsets_mm": offsets_mm,
        "slice_y_indices": y_idxs,
        "beam_center_xy": [int(cy), int(cx)],
        "dose_norm_ref_max": float(norm_ref),
        "dose_display_range_norm": [dose_vmin, dose_vmax],
        "density_display_range": [dens_vmin, dens_vmax],
        "arrows_mm": args.arrow_mm if args.arrow_mm else [],
    }
    meta_path = args.out.with_suffix(".json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(args.out)
    print(meta_path)


if __name__ == "__main__":
    main()
