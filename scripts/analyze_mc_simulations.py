#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def central_axis_profile(dose: np.ndarray) -> np.ndarray:
    nz, ny, nx = dose.shape
    return dose[:, ny // 2, nx // 2]


def core_mean_depth_profile(dose: np.ndarray, half_width: int = 2) -> np.ndarray:
    nz, ny, nx = dose.shape
    cy, cx = ny // 2, nx // 2
    y0, y1 = max(0, cy - half_width), min(ny, cy + half_width + 1)
    x0, x1 = max(0, cx - half_width), min(nx, cx + half_width + 1)
    return dose[:, y0:y1, x0:x1].mean(axis=(1, 2))


def cumulative_dvh(dose: np.ndarray, mask: np.ndarray, bins: int = 200) -> tuple[np.ndarray, np.ndarray]:
    d = dose[mask]
    if d.size == 0:
        x = np.linspace(0, 1, bins)
        return x, np.zeros_like(x)

    max_d = float(np.max(d))
    if max_d <= 0:
        x = np.linspace(0, 1, bins)
        return x, np.zeros_like(x)

    edges = np.linspace(0.0, max_d, bins + 1)
    hist, _ = np.histogram(d, bins=edges)
    c = np.cumsum(hist[::-1])[::-1].astype(np.float64)
    c /= c[0]
    x = 0.5 * (edges[:-1] + edges[1:])
    return x, c


def fwhm_mm(profile: np.ndarray, spacing_mm: float) -> float:
    pmax = float(np.max(profile))
    if pmax <= 0:
        return 0.0
    idx = np.where(profile >= 0.5 * pmax)[0]
    if idx.size < 2:
        return 0.0
    return float((idx[-1] - idx[0]) * spacing_mm)


def penumbra_20_80_mm(profile: np.ndarray, spacing_mm: float) -> float:
    pmax = float(np.max(profile))
    if pmax <= 0:
        return 0.0
    norm = profile / pmax
    idx20 = np.where(norm >= 0.2)[0]
    idx80 = np.where(norm >= 0.8)[0]
    if idx20.size < 2 or idx80.size < 2:
        return 0.0
    w20 = (idx20[-1] - idx20[0]) * spacing_mm
    w80 = (idx80[-1] - idx80[0]) * spacing_mm
    return float(max(0.0, w20 - w80))


def masks_from_spr(spr: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "lung": spr <= 0.5,
        "water": (spr > 0.5) & (spr < 1.3),
        "bone": spr >= 1.3,
        "all": np.ones_like(spr, dtype=bool),
    }


def analyze_sample(
    sample_dir: Path,
    out_dir: Path,
    voxel_mm: tuple[float, float, float],
    low_plot_scale: float,
) -> dict[str, float | str]:
    low = np.load(sample_dir / "low" / "dose.npy").astype(np.float32)
    high = np.load(sample_dir / "high" / "dose.npy").astype(np.float32)
    spr = np.load(sample_dir / "high" / "spr.npy").astype(np.float32)

    with open(sample_dir / "meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    high_max = float(np.max(high))
    if high_max <= 0:
        raise ValueError(f"High dose max is zero in {sample_dir}")

    low_n = low / high_max
    high_n = high / high_max
    low_n_plot = low_n * low_plot_scale

    p_low = central_axis_profile(low_n)
    p_high = central_axis_profile(high_n)
    pm_low = core_mean_depth_profile(low_n)
    pm_high = core_mean_depth_profile(high_n)
    p_low_plot = central_axis_profile(low_n_plot)
    pm_low_plot = core_mean_depth_profile(low_n_plot)

    z_mm = np.arange(low.shape[0], dtype=np.float32) * voxel_mm[2]
    peak_idx_low = int(np.argmax(pm_low))
    peak_idx_high = int(np.argmax(pm_high))

    cy, cx = low.shape[1] // 2, low.shape[2] // 2
    plane_low = low_n[peak_idx_high]
    plane_high = high_n[peak_idx_high]
    plane_low_plot = low_n_plot[peak_idx_high]
    lat_low = plane_low[cy, :]
    lat_high = plane_high[cy, :]
    lat_low_plot = plane_low_plot[cy, :]
    x_mm = np.arange(low.shape[2], dtype=np.float32) * voxel_mm[0]

    sample_out = out_dir / sample_dir.name
    sample_out.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 4))
    plt.plot(z_mm, p_high, label="High 100k (axis)", linewidth=2)
    plt.plot(z_mm, p_low_plot, label=f"Low 2k (axis) x{low_plot_scale:g}", alpha=0.8)
    plt.plot(z_mm, pm_high, label="High 100k (core mean)", linestyle="--")
    plt.plot(z_mm, pm_low_plot, label=f"Low 2k (core mean) x{low_plot_scale:g}", linestyle="--")
    plt.xlabel("Depth z (mm)")
    plt.ylabel("Normalized dose")
    plt.title(f"Depth-dose: {sample_dir.name} (E={meta.get('energy_mev', 'NA')} MeV)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(sample_out / "depth_dose_profile.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(x_mm, lat_high, label="High 100k")
    plt.plot(x_mm, lat_low_plot, label=f"Low 2k x{low_plot_scale:g}", alpha=0.85)
    plt.xlabel("Lateral x (mm)")
    plt.ylabel("Normalized dose")
    plt.title(f"Lateral profile at high peak depth (z={peak_idx_high})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(sample_out / "lateral_profile.png", dpi=160)
    plt.close()

    mats = masks_from_spr(spr)
    plt.figure(figsize=(7, 5))
    for mat_name, mask in mats.items():
        xh, yh = cumulative_dvh(high_n, mask)
        xl, yl = cumulative_dvh(low_n_plot, mask)
        plt.plot(xh, yh, label=f"{mat_name} high", linewidth=2)
        plt.plot(xl, yl, label=f"{mat_name} low x{low_plot_scale:g}", linestyle="--", alpha=0.8)
    plt.xlabel("Normalized dose")
    plt.ylabel("Volume fraction receiving >= dose")
    plt.title(f"DVH by material: {sample_dir.name}")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(sample_out / "dvh_materials.png", dpi=160)
    plt.close()

    return {
        "sample": sample_dir.name,
        "energy_mev": float(meta.get("energy_mev", -1.0)),
        "peak_depth_low_mm": peak_idx_low * voxel_mm[2],
        "peak_depth_high_mm": peak_idx_high * voxel_mm[2],
        "peak_depth_abs_error_mm": abs(peak_idx_low - peak_idx_high) * voxel_mm[2],
        "axis_mae": float(np.mean(np.abs(p_low - p_high))),
        "core_depth_mae": float(np.mean(np.abs(pm_low - pm_high))),
        "lateral_fwhm_low_mm": fwhm_mm(lat_low, voxel_mm[0]),
        "lateral_fwhm_high_mm": fwhm_mm(lat_high, voxel_mm[0]),
        "lateral_penumbra20_80_low_mm": penumbra_20_80_mm(lat_low, voxel_mm[0]),
        "lateral_penumbra20_80_high_mm": penumbra_20_80_mm(lat_high, voxel_mm[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Monte Carlo low/high simulations")
    parser.add_argument("--mc-root", type=Path, default=Path("mc_runs_opengate_small"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/sim_analysis"))
    parser.add_argument("--voxel-mm", type=float, nargs=3, default=[2.0, 2.0, 2.0])
    parser.add_argument("--low-plot-scale", type=float, default=50.0)
    args = parser.parse_args()

    sample_dirs = sorted([p for p in args.mc_root.iterdir() if p.is_dir()])
    if not sample_dirs:
        raise RuntimeError(f"No sample folders found in {args.mc_root}")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    vx, vy, vz = map(float, args.voxel_mm)
    rows: list[dict[str, float | str]] = []
    for s in sample_dirs:
        rows.append(analyze_sample(s, out_dir, (vx, vy, vz), float(args.low_plot_scale)))

    csv_path = out_dir / "summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    agg = {
        "n_samples": len(rows),
        "mean_peak_depth_abs_error_mm": float(np.mean([r["peak_depth_abs_error_mm"] for r in rows])),
        "mean_axis_mae": float(np.mean([r["axis_mae"] for r in rows])),
        "mean_core_depth_mae": float(np.mean([r["core_depth_mae"] for r in rows])),
        "mean_lateral_fwhm_abs_error_mm": float(
            np.mean([abs(r["lateral_fwhm_low_mm"] - r["lateral_fwhm_high_mm"]) for r in rows])
        ),
        "mean_lateral_penumbra_abs_error_mm": float(
            np.mean(
                [
                    abs(r["lateral_penumbra20_80_low_mm"] - r["lateral_penumbra20_80_high_mm"])
                    for r in rows
                ]
            )
        ),
    }

    with open(out_dir / "aggregate.json", "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)

    print(f"Analysis done for {len(rows)} samples")
    print(f"Summary CSV: {csv_path}")
    print(f"Aggregate JSON: {out_dir / 'aggregate.json'}")


if __name__ == "__main__":
    main()
