#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from proton_denoise.data import ProtonDoseDataset
from proton_denoise.metrics import (
    bragg_peak_index,
    central_axis_profile,
    gamma_pass_rate,
    lateral_penumbra_width_mm,
)
from proton_denoise.model import load_model_from_checkpoint


def _weighted_center_from_xy_map(w: np.ndarray) -> tuple[int, int] | None:
    s = float(w.sum())
    if s <= 0:
        return None
    ny, nx = w.shape
    yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    cy = int(round(float((w * yy).sum() / s)))
    cx = int(round(float((w * xx).sum() / s)))
    cy = int(np.clip(cy, 0, ny - 1))
    cx = int(np.clip(cx, 0, nx - 1))
    return cy, cx


def estimate_beam_center_xy(
    ref: np.ndarray,
    low: np.ndarray,
    *,
    mode: str = "global",
    entrance_slices: int = 12,
) -> tuple[int, int]:
    """Estimate beam center in xy using robust modes and fallback to low-dose if needed."""
    nz, ny, nx = ref.shape
    refp = np.clip(ref, 0.0, None)
    lowp = np.clip(low, 0.0, None)

    if mode == "maxdose":
        if float(refp.max()) > 0:
            _, cy, cx = np.unravel_index(np.argmax(refp), refp.shape)
            return int(cy), int(cx)
        if float(lowp.max()) > 0:
            _, cy, cx = np.unravel_index(np.argmax(lowp), lowp.shape)
            return int(cy), int(cx)
        return ny // 2, nx // 2

    if mode == "entrance":
        z1 = min(max(int(entrance_slices), 1), nz)
        w_ref = refp[:z1].sum(axis=0)
        c = _weighted_center_from_xy_map(w_ref)
        if c is not None:
            return c
        w_low = lowp[:z1].sum(axis=0)
        c = _weighted_center_from_xy_map(w_low)
        if c is not None:
            return c
        return ny // 2, nx // 2

    # global: robust centroid over full volume to avoid orientation assumptions.
    w_ref = refp.sum(axis=0)
    if float(w_ref.sum()) > 0:
        nzv = w_ref[w_ref > 0]
        thr = float(np.percentile(nzv, 90.0)) if nzv.size > 0 else 0.0
        c = _weighted_center_from_xy_map(w_ref * (w_ref >= thr))
        if c is not None:
            return c

    w_low = lowp.sum(axis=0)
    c = _weighted_center_from_xy_map(w_low)
    if c is not None:
        return c
    return ny // 2, nx // 2


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


def longitudinal_xz_band(vol: np.ndarray, cy: int, half_width_y: int = 2) -> np.ndarray:
    _, ny, _ = vol.shape
    y0, y1 = max(0, cy - half_width_y), min(ny, cy + half_width_y + 1)
    return vol[:, y0:y1, :].mean(axis=1)


def robust_norm_2d(arr: np.ndarray, pct: float = 99.5, eps: float = 1e-8) -> np.ndarray:
    vmax = float(np.percentile(arr, pct))
    vmax = max(vmax, eps)
    out = arr / vmax
    return np.clip(out, 0.0, 1.0)


def _get_ckpt_data_prep(ckpt: dict) -> dict:
    dp = ckpt.get("data_prep", {})
    if not isinstance(dp, dict):
        return {}
    out = {}
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


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description="Predict and plot beam-centered profiles for selected samples")
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument("--split", type=str, choices=["train", "val", "test"], default="test")
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--indices", type=int, nargs="+", required=True)
    ap.add_argument("--out-dir", type=str, default="artifacts/predict_beam_profiles")
    ap.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    ap.add_argument(
        "--input-norm-mode",
        choices=["none", "per_channel_max", "global_max", "coupled_target_max"],
        default="none",
    )
    ap.add_argument("--input-dose-scale", type=float, default=1.0)
    ap.add_argument("--no-normalize-target", action="store_true")
    ap.add_argument("--crop-shape", type=int, nargs=3, default=None, metavar=("D", "H", "W"))
    ap.add_argument("--crop-focus", type=str, choices=["center", "maxdose"], default="center")
    ap.add_argument(
        "--no-use-checkpoint-data-prep",
        action="store_true",
        help="Disable auto-loading normalize/scale/crop settings from checkpoint data_prep",
    )
    ap.add_argument("--input-plot-scale", type=float, default=1.0)
    ap.add_argument(
        "--low-plot-mode",
        type=str,
        choices=["unscaled", "model_input"],
        default="unscaled",
        help="How to visualize low-dose channel: raw physical scale or model-input scale",
    )
    ap.add_argument("--voxel-mm", type=float, nargs=3, default=[2.0, 2.0, 2.0])
    ap.add_argument("--entrance-slices", type=int, default=12)
    ap.add_argument(
        "--beam-center-mode",
        type=str,
        choices=["global", "maxdose", "entrance"],
        default="global",
        help="How to estimate beam center in xy for profile extraction",
    )
    ap.add_argument("--depth-half-width", type=int, default=2)
    ap.add_argument("--lateral-half-width", type=int, default=2)
    ap.add_argument("--lateral-z-half-width", type=int, default=1)
    ap.add_argument("--dd", type=float, default=2.0, help="Dose-difference criterion (%) for gamma")
    ap.add_argument("--dta", type=float, default=2.0, help="Distance-to-agreement (mm) for gamma")
    ap.add_argument("--threshold", type=float, default=10.0, help="Dose threshold (%) for gamma")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    vx, vy, vz = map(float, args.voxel_mm)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    ckpt_data_prep = _get_ckpt_data_prep(ckpt)

    normalize_target = not args.no_normalize_target
    input_norm_mode = args.input_norm_mode
    input_dose_scale = float(args.input_dose_scale)
    crop_shape = tuple(args.crop_shape) if args.crop_shape is not None else None
    crop_focus = args.crop_focus

    use_ckpt_data_prep = not args.no_use_checkpoint_data_prep
    if use_ckpt_data_prep and ckpt_data_prep:
        normalize_target = bool(ckpt_data_prep.get("normalize_target", normalize_target))
        input_norm_mode = str(ckpt_data_prep.get("input_norm_mode", input_norm_mode))
        input_dose_scale = float(ckpt_data_prep.get("input_dose_scale", input_dose_scale))
        crop_shape = ckpt_data_prep.get("crop_shape", crop_shape)
        crop_focus = str(ckpt_data_prep.get("crop_focus", crop_focus))

    print(
        json.dumps(
            {
                "effective_data_prep": {
                    "normalize_target": normalize_target,
                    "input_norm_mode": input_norm_mode,
                    "input_dose_scale": input_dose_scale,
                    "crop_shape": crop_shape,
                    "crop_focus": crop_focus,
                },
                "ckpt_data_prep_found": bool(ckpt_data_prep),
                "using_ckpt_data_prep": bool(use_ckpt_data_prep and ckpt_data_prep),
                "argv": sys.argv,
            },
            indent=2,
        )
    )

    ds = ProtonDoseDataset(
        Path(args.data_root) / args.split,
        normalize_target=normalize_target,
        input_norm_mode=input_norm_mode,
        input_dose_scale=input_dose_scale,
        crop_shape=crop_shape,
        crop_focus=crop_focus,
    )

    model = load_model_from_checkpoint(ckpt, in_channels=2, out_channels=1).to(device)
    model.eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx in args.indices:
        if idx < 0 or idx >= len(ds):
            raise IndexError(f"Index {idx} out of bounds for split '{args.split}' with {len(ds)} samples")

        item = ds[idx]
        x = item["input"].unsqueeze(0).to(device)
        ref = item["target"][0].cpu().numpy()
        ct = item["input"][1].cpu().numpy()
        low_model = item["input"][0].cpu().numpy()
        if args.low_plot_mode == "unscaled" and float(args.input_dose_scale) != 0.0:
            low = low_model / float(args.input_dose_scale)
        else:
            low = low_model
        low = low * float(args.input_plot_scale)
        pred = model(x).cpu().numpy()[0, 0]

        cy, cx = estimate_beam_center_xy(
            ref,
            low,
            mode=args.beam_center_mode,
            entrance_slices=args.entrance_slices,
        )
        prof_ref = depth_profile_center(ref, cy=cy, cx=cx, half_width_xy=args.depth_half_width)
        prof_pred = depth_profile_center(pred, cy=cy, cx=cx, half_width_xy=args.depth_half_width)
        prof_low = depth_profile_center(low, cy=cy, cx=cx, half_width_xy=args.depth_half_width)

        z_peak_ref = int(np.argmax(prof_ref))
        z_peak_pred = int(np.argmax(prof_pred))
        peak_err_vox = int(abs(z_peak_pred - z_peak_ref))
        mae = float(np.mean(np.abs(pred - ref)))
        axis_ref = central_axis_profile(ref)
        axis_pred = central_axis_profile(pred)
        peak_axis_ref = int(bragg_peak_index(axis_ref))
        peak_axis_pred = int(bragg_peak_index(axis_pred))
        peak_axis_err_vox = int(abs(peak_axis_pred - peak_axis_ref))
        gamma = float(
            gamma_pass_rate(
                pred,
                ref,
                voxel_mm=(vx, vy, vz),
                dose_diff_percent=float(args.dd),
                distance_mm=float(args.dta),
                dose_threshold_percent=float(args.threshold),
            )
        )
        pen_ref = float(lateral_penumbra_width_mm(ref, voxel_mm=(vx, vy, vz), z_index=peak_axis_ref))
        pen_pred = float(lateral_penumbra_width_mm(pred, voxel_mm=(vx, vy, vz), z_index=peak_axis_pred))

        lx_ref = lateral_x_profile(
            ref,
            z_idx=z_peak_ref,
            cy=cy,
            half_width_y=args.lateral_half_width,
            half_width_z=args.lateral_z_half_width,
        )
        lx_pred = lateral_x_profile(
            pred,
            z_idx=z_peak_ref,
            cy=cy,
            half_width_y=args.lateral_half_width,
            half_width_z=args.lateral_z_half_width,
        )
        ly_ref = lateral_y_profile(
            ref,
            z_idx=z_peak_ref,
            cx=cx,
            half_width_x=args.lateral_half_width,
            half_width_z=args.lateral_z_half_width,
        )
        ly_pred = lateral_y_profile(
            pred,
            z_idx=z_peak_ref,
            cx=cx,
            half_width_x=args.lateral_half_width,
            half_width_z=args.lateral_z_half_width,
        )

        z_mm = np.arange(len(prof_ref)) * vz
        x_mm = (np.arange(len(lx_ref)) - cx) * vx
        y_mm = (np.arange(len(ly_ref)) - cy) * vy

        case_name = Path(str(item.get("path", ""))).stem if str(item.get("path", "")).strip() else f"sample_{idx:04d}"
        sample_out = out_dir / case_name
        sample_out.mkdir(parents=True, exist_ok=True)

        ref_xz = longitudinal_xz_band(ref, cy=cy, half_width_y=args.lateral_half_width)
        pred_xz = longitudinal_xz_band(pred, cy=cy, half_width_y=args.lateral_half_width)
        low_xz = longitudinal_xz_band(low, cy=cy, half_width_y=args.lateral_half_width)
        ct_xz = longitudinal_xz_band(ct, cy=cy, half_width_y=args.lateral_half_width)
        vmax_ref = float(np.percentile(ref_xz, 99.5))
        vmax_ref = max(vmax_ref, 1e-8)
        vmax_pred = float(np.percentile(pred_xz, 99.5))
        vmax_pred = max(vmax_pred, 1e-8)
        vmax_low = float(np.percentile(low_xz, 99.5))
        vmax_low = max(vmax_low, 1e-8)
        vmax_ref_pred_shared = float(np.percentile(np.concatenate([ref_xz.ravel(), pred_xz.ravel()]), 99.5))
        vmax_ref_pred_shared = max(vmax_ref_pred_shared, 1e-8)

        # Use separate robust scales so channels with very different amplitude remain visible.
        fig, axs = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
        axs[0].imshow(ref_xz, cmap="inferno", origin="lower", aspect="auto", vmin=0, vmax=vmax_ref)
        axs[0].set_title(f"Reference xz (vmax={vmax_ref:.2e})")
        axs[1].imshow(pred_xz, cmap="inferno", origin="lower", aspect="auto", vmin=0, vmax=vmax_pred)
        axs[1].set_title(f"Prediction xz (vmax={vmax_pred:.2e})")
        axs[2].imshow(low_xz, cmap="inferno", origin="lower", aspect="auto", vmin=0, vmax=vmax_low)
        label_mode = "unscaled" if args.low_plot_mode == "unscaled" else "model_input"
        axs[2].set_title(f"Input low xz [{label_mode}] (vmax={vmax_low:.2e})")
        for ax in axs:
            ax.set_xlabel("x voxel")
            ax.set_ylabel("z voxel")
        fig.savefig(sample_out / "xz_compare.png", dpi=170)
        plt.close(fig)

        # Shared scale diagnostic for reference vs prediction.
        fig, axs = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
        axs[0].imshow(ref_xz, cmap="inferno", origin="lower", aspect="auto", vmin=0, vmax=vmax_ref_pred_shared)
        axs[0].set_title("Reference xz (shared)")
        axs[1].imshow(pred_xz, cmap="inferno", origin="lower", aspect="auto", vmin=0, vmax=vmax_ref_pred_shared)
        axs[1].set_title("Prediction xz (shared)")
        axs[2].imshow(np.abs(pred_xz - ref_xz), cmap="magma", origin="lower", aspect="auto")
        axs[2].set_title("|Prediction-Reference| xz")
        for ax in axs:
            ax.set_xlabel("x voxel")
            ax.set_ylabel("z voxel")
        fig.savefig(sample_out / "xz_compare_shared.png", dpi=170)
        plt.close(fig)

        # CT overlay view for reference/prediction/input on corresponding anatomy.
        ct_bg = robust_norm_2d(ct_xz, pct=99.5)
        ref_n = robust_norm_2d(ref_xz, pct=99.5)
        pred_n = robust_norm_2d(pred_xz, pct=99.5)
        low_n = robust_norm_2d(low_xz, pct=99.5)

        fig, axs = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
        for ax, dose_n, title in [
            (axs[0], ref_n, "Reference on CT"),
            (axs[1], pred_n, "Prediction on CT"),
            (axs[2], low_n, "Input low on CT"),
        ]:
            ax.imshow(ct_bg, cmap="gray", origin="lower", aspect="auto")
            ax.imshow(dose_n, cmap="inferno", origin="lower", aspect="auto", alpha=np.clip(dose_n, 0.0, 1.0) ** 0.65)
            ax.set_title(title)
            ax.set_xlabel("x voxel")
            ax.set_ylabel("z voxel")
        fig.savefig(sample_out / "xz_overlay_ct.png", dpi=170)
        plt.close(fig)

        fig, axs = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
        axs[0].plot(z_mm, prof_ref, label="Reference", linewidth=2)
        axs[0].plot(z_mm, prof_pred, label="Prediction")
        axs[0].plot(z_mm, prof_low, label=f"Input x{args.input_plot_scale:g}", alpha=0.8)
        axs[0].set_title("Depth profile (beam-centered)")
        axs[0].set_xlabel("z (mm)")
        axs[0].set_ylabel("Dose")
        axs[0].set_ylim(bottom=0)
        axs[0].grid(alpha=0.25)
        axs[0].legend()

        axs[1].plot(x_mm, lx_ref, label="Reference", linewidth=2)
        axs[1].plot(x_mm, lx_pred, label="Prediction")
        axs[1].set_title(f"Lateral X at z_peak={z_peak_ref}")
        axs[1].set_xlabel("x offset (mm)")
        axs[1].set_ylabel("Dose")
        axs[1].set_ylim(bottom=0)
        axs[1].grid(alpha=0.25)

        axs[2].plot(y_mm, ly_ref, label="Reference", linewidth=2)
        axs[2].plot(y_mm, ly_pred, label="Prediction")
        axs[2].set_title(f"Lateral Y at z_peak={z_peak_ref}")
        axs[2].set_xlabel("y offset (mm)")
        axs[2].set_ylabel("Dose")
        axs[2].set_ylim(bottom=0)
        axs[2].grid(alpha=0.25)

        fig.savefig(sample_out / "beam_profiles.png", dpi=170)
        plt.close(fig)

        row = {
            "index": int(idx),
            "case_name": str(case_name),
            "energy_mev": float(item["energy_mev"].item()),
            "beam_center_y": int(cy),
            "beam_center_x": int(cx),
            "peak_ref_z": int(z_peak_ref),
            "peak_pred_z": int(z_peak_pred),
            "peak_abs_error_vox": int(peak_err_vox),
            "peak_abs_error_mm": float(peak_err_vox * vz),
            "peak_axis_abs_error_vox": int(peak_axis_err_vox),
            "peak_axis_abs_error_mm": float(peak_axis_err_vox * vz),
            "mae_pred_vs_ref": float(mae),
            "gamma_pass_rate": float(gamma),
            "penumbra_ref_mm": float(pen_ref),
            "penumbra_pred_mm": float(pen_pred),
            "penumbra_abs_error_mm": float(abs(pen_pred - pen_ref)),
            "ref_xz_p99_5": float(vmax_ref),
            "pred_xz_p99_5": float(vmax_pred),
            "low_xz_p99_5": float(vmax_low),
            "ref_pred_shared_xz_p99_5": float(vmax_ref_pred_shared),
            "ref_max": float(np.max(ref)),
            "low_model_max": float(np.max(low_model)),
            "low_plotted_max": float(np.max(low)),
        }
        rows.append(row)
        print(row)

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"Saved: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
