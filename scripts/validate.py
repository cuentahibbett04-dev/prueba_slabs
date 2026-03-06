#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
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
from proton_denoise.model import ResUNet3D


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
    base_channels = int(ckpt.get("base_channels", 16))
    output_activation = str(ckpt.get("output_activation", "identity"))

    model = ResUNet3D(
        in_channels=2,
        out_channels=1,
        base_channels=base_channels,
        output_activation=output_activation,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    voxel_mm = (2.0, 2.0, 2.0)

    for i, batch in enumerate(loader):
        x = batch["input"].to(device)
        y = batch["target"].cpu().numpy()[0, 0]

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
        )

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

    gamma_mean = float(np.mean([r["gamma_pass_rate"] for r in rows]))
    peak_err = float(np.mean([r["peak_abs_error_vox"] for r in rows]))
    pen_err = float(np.mean([r["penumbra_abs_error_mm"] for r in rows]))

    print(f"Mean gamma pass rate: {gamma_mean:.2f}%")
    print(f"Mean Bragg peak abs error: {peak_err:.2f} vox")
    print(f"Mean penumbra abs error: {pen_err:.2f} mm")
    print(f"Detailed CSV: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate physical metrics of denoising model")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--checkpoint", type=str, default="artifacts/best_model.pt")
    parser.add_argument("--out-dir", type=str, default="artifacts/validation")
    parser.add_argument("--dd", type=float, default=2.0, help="Dose difference criterion (%)")
    parser.add_argument("--dta", type=float, default=2.0, help="Distance to agreement criterion (mm)")
    parser.add_argument("--threshold", type=float, default=10.0, help="Dose threshold (%)")
    parser.add_argument("--plot-samples", type=int, default=3)
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--input-norm-mode", choices=["none", "per_channel_max", "global_max"], default="none")
    parser.add_argument("--input-dose-scale", type=float, default=1.0)
    parser.add_argument("--no-normalize-target", action="store_true")
    main(parser.parse_args())
