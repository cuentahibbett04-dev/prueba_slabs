#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from proton_denoise.data import ProtonDoseDataset
from proton_denoise.metrics import (
    bragg_peak_index,
    central_axis_profile,
    gamma_pass_rate,
    lateral_penumbra_width_mm,
)
from proton_denoise.model import load_model_from_checkpoint


def _parse_gamma_criteria(spec: str) -> list[tuple[float, float]]:
    """Parse gamma criteria string like: '2,2;1,1;0.5,0.5'."""
    out: list[tuple[float, float]] = []
    for token in spec.split(";"):
        tok = token.strip()
        if not tok:
            continue
        parts = [p.strip() for p in tok.split(",")]
        if len(parts) != 2:
            raise ValueError(f"Invalid gamma criterion token: {tok!r}")
        dta = float(parts[0])
        dd = float(parts[1])
        if dta <= 0 or dd <= 0:
            raise ValueError(f"Gamma criterion values must be > 0, got: {tok!r}")
        out.append((dta, dd))

    if not out:
        raise ValueError("No valid gamma criteria parsed")
    return out


def _crit_key(dta_mm: float, dd_pct: float) -> str:
    return f"{dta_mm:g}mm_{dd_pct:g}pct"


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _psnr(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0, eps: float = 1e-12) -> float:
    mse = float(np.mean((pred - target) ** 2))
    if mse <= eps:
        return float("inf")
    return float(20.0 * np.log10(max(data_range, eps)) - 10.0 * np.log10(mse))


def _ssim_global(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0, eps: float = 1e-12) -> float:
    """Global 3D SSIM approximation (single-window over the full volume)."""
    x = pred.astype(np.float64)
    y = target.astype(np.float64)
    ux = float(np.mean(x))
    uy = float(np.mean(y))
    vx = float(np.var(x))
    vy = float(np.var(y))
    vxy = float(np.mean((x - ux) * (y - uy)))

    c1 = (0.01 * max(data_range, eps)) ** 2
    c2 = (0.03 * max(data_range, eps)) ** 2
    num = (2.0 * ux * uy + c1) * (2.0 * vxy + c2)
    den = (ux * ux + uy * uy + c1) * (vx + vy + c2)
    return float(num / max(den, eps))


@torch.no_grad()
def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    test_ds = ProtonDoseDataset(
        Path(args.data_root) / "test",
        normalize_target=not args.no_normalize_target,
        input_norm_mode=args.input_norm_mode,
        input_dose_scale=args.input_dose_scale,
    )
    loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = load_model_from_checkpoint(ckpt, in_channels=2, out_channels=1).to(device)
    model.eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    voxel_mm = (2.0, 2.0, 2.0)
    gamma_criteria = _parse_gamma_criteria(args.gamma_criteria)
    gamma_mask_thr = float(args.gamma_mask_threshold_percent)
    gamma_unmasked_stride = max(int(args.gamma_unmasked_eval_stride), 1)
    gamma_masked_stride = max(int(args.gamma_masked_eval_stride), 1)
    gamma_max_points = int(args.gamma_max_eval_points)
    gamma_max_points = None if gamma_max_points <= 0 else gamma_max_points
    gamma_seed = int(args.gamma_random_seed)

    # Aggregate buckets for paper-style table.
    # Key: (comparison, masked_mode, criterion_key) -> list of gamma pass rates.
    gamma_buckets: dict[tuple[str, str, str], list[float]] = {}

    for i, batch in enumerate(loader):
        x = batch["input"].to(device)
        y = batch["target"].cpu().numpy()[0, 0]
        low = batch["input"].cpu().numpy()[0, 0]

        pred = model(x).cpu().numpy()[0, 0]

        prof_t = central_axis_profile(y)
        prof_p = central_axis_profile(pred)
        peak_t = bragg_peak_index(prof_t)
        peak_p = bragg_peak_index(prof_p)

        pen_t = lateral_penumbra_width_mm(y, voxel_mm, z_index=peak_t)
        pen_p = lateral_penumbra_width_mm(pred, voxel_mm, z_index=peak_p)

        gamma = gamma_pass_rate(
            pred,
            y,
            voxel_mm=voxel_mm,
            dose_diff_percent=args.dd,
            distance_mm=args.dta,
            dose_threshold_percent=args.threshold,
            eval_stride=gamma_masked_stride,
            max_eval_points=gamma_max_points,
            random_seed=gamma_seed,
        )

        gamma_pred_map: dict[str, float] = {}
        gamma_noisy_map: dict[str, float] = {}
        for dta_mm, dd_pct in gamma_criteria:
            k = _crit_key(dta_mm, dd_pct)

            g_pred_unmasked = gamma_pass_rate(
                pred,
                y,
                voxel_mm=voxel_mm,
                dose_diff_percent=dd_pct,
                distance_mm=dta_mm,
                dose_threshold_percent=0.0,
                eval_stride=gamma_unmasked_stride,
                max_eval_points=gamma_max_points,
                random_seed=gamma_seed,
            )
            g_noisy_unmasked = gamma_pass_rate(
                low,
                y,
                voxel_mm=voxel_mm,
                dose_diff_percent=dd_pct,
                distance_mm=dta_mm,
                dose_threshold_percent=0.0,
                eval_stride=gamma_unmasked_stride,
                max_eval_points=gamma_max_points,
                random_seed=gamma_seed,
            )
            g_pred_masked = gamma_pass_rate(
                pred,
                y,
                voxel_mm=voxel_mm,
                dose_diff_percent=dd_pct,
                distance_mm=dta_mm,
                dose_threshold_percent=gamma_mask_thr,
                eval_stride=gamma_masked_stride,
                max_eval_points=gamma_max_points,
                random_seed=gamma_seed,
            )
            g_noisy_masked = gamma_pass_rate(
                low,
                y,
                voxel_mm=voxel_mm,
                dose_diff_percent=dd_pct,
                distance_mm=dta_mm,
                dose_threshold_percent=gamma_mask_thr,
                eval_stride=gamma_masked_stride,
                max_eval_points=gamma_max_points,
                random_seed=gamma_seed,
            )

            gamma_pred_map[f"gamma_unmasked_{k}"] = g_pred_unmasked
            gamma_noisy_map[f"gamma_unmasked_{k}"] = g_noisy_unmasked
            gamma_pred_map[f"gamma_masked{gamma_mask_thr:g}pct_{k}"] = g_pred_masked
            gamma_noisy_map[f"gamma_masked{gamma_mask_thr:g}pct_{k}"] = g_noisy_masked

            gamma_buckets.setdefault(("GT/DeepMC", "unmasked", k), []).append(g_pred_unmasked)
            gamma_buckets.setdefault(("GT/Noisy", "unmasked", k), []).append(g_noisy_unmasked)
            gamma_buckets.setdefault(("GT/DeepMC", f"masked_{gamma_mask_thr:g}pct", k), []).append(g_pred_masked)
            gamma_buckets.setdefault(("GT/Noisy", f"masked_{gamma_mask_thr:g}pct", k), []).append(g_noisy_masked)

        mae_pred = _mae(pred, y)
        rmse_pred = _rmse(pred, y)
        psnr_pred = _psnr(pred, y, data_range=1.0)
        ssim_pred = _ssim_global(pred, y, data_range=1.0)

        mae_noisy = _mae(low, y)
        rmse_noisy = _rmse(low, y)
        psnr_noisy = _psnr(low, y, data_range=1.0)
        ssim_noisy = _ssim_global(low, y, data_range=1.0)

        rows.append(
            {
                "sample": i,
                "peak_target_z": peak_t,
                "peak_pred_z": peak_p,
                "peak_abs_error_vox": abs(peak_t - peak_p),
                "penumbra_target_mm": pen_t,
                "penumbra_pred_mm": pen_p,
                "penumbra_abs_error_mm": abs(pen_t - pen_p),
                "gamma_pass_rate": gamma,
                "mae_pred": mae_pred,
                "rmse_pred": rmse_pred,
                "psnr_pred": psnr_pred,
                "ssim_pred": ssim_pred,
                "mae_noisy": mae_noisy,
                "rmse_noisy": rmse_noisy,
                "psnr_noisy": psnr_noisy,
                "ssim_noisy": ssim_noisy,
                **gamma_pred_map,
                **{f"{k}_noisy": v for k, v in gamma_noisy_map.items()},
            }
        )

        if i < args.plot_samples:
            z = peak_t
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].imshow(x.cpu().numpy()[0, 0, z], cmap="inferno")
            axs[0].set_title("Input 2k")
            axs[1].imshow(pred[z], cmap="inferno")
            axs[1].set_title("Prediction")
            axs[2].imshow(y[z], cmap="inferno")
            axs[2].set_title("Target 100k")
            for ax in axs:
                ax.axis("off")
            fig.tight_layout()
            fig.savefig(out_dir / f"sample_{i:03d}_slice.png", dpi=150)
            plt.close(fig)

            z_axis = np.arange(len(prof_t)) * voxel_mm[2]
            plt.figure(figsize=(6, 4))
            plt.plot(z_axis, prof_t, label="target")
            plt.plot(z_axis, prof_p, label="pred")
            plt.xlabel("Depth (mm)")
            plt.ylabel("Normalized dose")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / f"sample_{i:03d}_profile.png", dpi=150)
            plt.close()

    csv_path = out_dir / "metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Paper-style aggregate gamma table.
    gamma_summary_rows = []
    for comparison in ["GT/DeepMC", "GT/Noisy"]:
        for masked_mode in ["unmasked", f"masked_{gamma_mask_thr:g}pct"]:
            row = {
                "comparison": comparison,
                "mode": masked_mode,
            }
            for dta_mm, dd_pct in gamma_criteria:
                k = _crit_key(dta_mm, dd_pct)
                vals = gamma_buckets.get((comparison, masked_mode, k), [])
                row[k] = float(np.mean(vals)) if vals else float("nan")
            gamma_summary_rows.append(row)

    gamma_summary_csv = out_dir / "gamma_summary.csv"
    gamma_summary_json = out_dir / "gamma_summary.json"
    with open(gamma_summary_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["comparison", "mode"] + [_crit_key(dta_mm, dd_pct) for dta_mm, dd_pct in gamma_criteria]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(gamma_summary_rows)
    with open(gamma_summary_json, "w", encoding="utf-8") as f:
        json.dump(gamma_summary_rows, f, indent=2)

    gamma_mean = float(np.mean([r["gamma_pass_rate"] for r in rows]))
    peak_err = float(np.mean([r["peak_abs_error_vox"] for r in rows]))
    pen_err = float(np.mean([r["penumbra_abs_error_mm"] for r in rows]))
    mae_pred_mean = float(np.mean([r["mae_pred"] for r in rows]))
    rmse_pred_mean = float(np.mean([r["rmse_pred"] for r in rows]))
    psnr_pred_mean = float(np.mean([r["psnr_pred"] for r in rows]))
    ssim_pred_mean = float(np.mean([r["ssim_pred"] for r in rows]))
    mae_noisy_mean = float(np.mean([r["mae_noisy"] for r in rows]))
    rmse_noisy_mean = float(np.mean([r["rmse_noisy"] for r in rows]))
    psnr_noisy_mean = float(np.mean([r["psnr_noisy"] for r in rows]))
    ssim_noisy_mean = float(np.mean([r["ssim_noisy"] for r in rows]))

    overall_summary = {
        "n_samples": len(rows),
        "gamma_pass_rate_mean": gamma_mean,
        "peak_abs_error_vox_mean": peak_err,
        "penumbra_abs_error_mm_mean": pen_err,
        "mae_pred_mean": mae_pred_mean,
        "rmse_pred_mean": rmse_pred_mean,
        "psnr_pred_mean": psnr_pred_mean,
        "ssim_pred_mean": ssim_pred_mean,
        "mae_noisy_mean": mae_noisy_mean,
        "rmse_noisy_mean": rmse_noisy_mean,
        "psnr_noisy_mean": psnr_noisy_mean,
        "ssim_noisy_mean": ssim_noisy_mean,
    }
    overall_summary_path = out_dir / "overall_summary.json"
    with open(overall_summary_path, "w", encoding="utf-8") as f:
        json.dump(overall_summary, f, indent=2)

    print(f"Mean gamma pass rate: {gamma_mean:.2f}%")
    print(f"Mean Bragg peak abs error: {peak_err:.2f} vox")
    print(f"Mean penumbra abs error: {pen_err:.2f} mm")
    print(f"Mean MAE pred/noisy: {mae_pred_mean:.6g} / {mae_noisy_mean:.6g}")
    print(f"Mean RMSE pred/noisy: {rmse_pred_mean:.6g} / {rmse_noisy_mean:.6g}")
    print(f"Mean PSNR pred/noisy: {psnr_pred_mean:.3f} / {psnr_noisy_mean:.3f}")
    print(f"Mean SSIM pred/noisy: {ssim_pred_mean:.5f} / {ssim_noisy_mean:.5f}")
    print(f"Detailed CSV: {csv_path}")
    print(f"Gamma summary CSV: {gamma_summary_csv}")
    print(f"Gamma summary JSON: {gamma_summary_json}")
    print(f"Overall summary JSON: {overall_summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate physical metrics of denoising model")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--checkpoint", type=str, default="artifacts/best_model.pt")
    parser.add_argument("--out-dir", type=str, default="artifacts/validation")
    parser.add_argument("--dd", type=float, default=2.0, help="Dose difference criterion (%)")
    parser.add_argument("--dta", type=float, default=2.0, help="Distance to agreement criterion (mm)")
    parser.add_argument("--threshold", type=float, default=10.0, help="Dose threshold (%)")
    parser.add_argument(
        "--gamma-criteria",
        type=str,
        default="2,2;1,1;0.5,0.5",
        help="Semicolon-separated DTA,DD gamma criteria. Example: '2,2;1,1;0.5,0.5'",
    )
    parser.add_argument(
        "--gamma-mask-threshold-percent",
        type=float,
        default=1.0,
        help="Mask threshold for GT-based masked gamma summary (percent of GT Dmax)",
    )
    parser.add_argument(
        "--gamma-unmasked-eval-stride",
        type=int,
        default=24,
        help="Evaluate every N-th voxel for unmasked gamma (speed/accuracy tradeoff)",
    )
    parser.add_argument(
        "--gamma-masked-eval-stride",
        type=int,
        default=8,
        help="Evaluate every N-th voxel for masked gamma",
    )
    parser.add_argument(
        "--gamma-max-eval-points",
        type=int,
        default=3000,
        help="Maximum number of evaluated voxels per gamma call (<=0 disables cap)",
    )
    parser.add_argument(
        "--gamma-random-seed",
        type=int,
        default=42,
        help="Random seed for gamma voxel subsampling when cap is active",
    )
    parser.add_argument("--plot-samples", type=int, default=3)
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument(
        "--input-norm-mode",
        choices=["none", "per_channel_max", "global_max", "coupled_target_max"],
        default="none",
    )
    parser.add_argument("--input-dose-scale", type=float, default=1.0)
    parser.add_argument("--no-normalize-target", action="store_true")
    main(parser.parse_args())
