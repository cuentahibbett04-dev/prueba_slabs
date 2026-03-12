#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


MATERIAL_BINS: list[tuple[str, float, float]] = [
    ("air", -np.inf, 0.08),
    ("lung", 0.08, 0.50),
    ("fat", 0.50, 0.85),
    ("soft", 0.85, 1.15),
    ("trab_bone", 1.15, 1.50),
    ("cort_bone", 1.50, np.inf),
]
MATERIAL_NAMES = [b[0] for b in MATERIAL_BINS]


def central_axis_profile(dose: np.ndarray) -> np.ndarray:
    nz, ny, nx = dose.shape
    return dose[:, ny // 2, nx // 2]


def depth_profile_at(dose: np.ndarray, y: int, x: int) -> np.ndarray:
    nz, ny, nx = dose.shape
    yy = int(np.clip(y, 0, ny - 1))
    xx = int(np.clip(x, 0, nx - 1))
    return dose[:, yy, xx]


def estimate_beam_direction_xz(high_n: np.ndarray, threshold_ratio: float = 0.2) -> tuple[float, float]:
    """Estimate principal beam direction on the x-z plane from high-dose voxels.

    Returns unit vector components (dx, dz) in voxel-index space.
    """
    vmax = float(np.max(high_n))
    if (not np.isfinite(vmax)) or vmax <= 0:
        return 0.0, 1.0

    mask = high_n >= (threshold_ratio * vmax)
    coords = np.argwhere(mask)  # [z, y, x]
    if coords.shape[0] < 10:
        return 0.0, 1.0

    weights = high_n[mask].astype(np.float64)
    xz = np.stack([coords[:, 2].astype(np.float64), coords[:, 0].astype(np.float64)], axis=1)  # [x, z]

    wsum = float(np.sum(weights))
    if wsum <= 0:
        return 0.0, 1.0

    mean = np.sum(xz * weights[:, None], axis=0) / wsum
    centered = xz - mean
    cov = (centered * weights[:, None]).T @ centered / wsum

    eigvals, eigvecs = np.linalg.eigh(cov)
    v = eigvecs[:, int(np.argmax(eigvals))]  # [vx, vz]
    dx, dz = float(v[0]), float(v[1])
    n = float(np.hypot(dx, dz))
    if n <= 0:
        return 0.0, 1.0

    dx, dz = dx / n, dz / n
    # Keep forward direction mostly toward +z for consistency.
    if dz < 0:
        dx, dz = -dx, -dz
    return dx, dz


def _line_t_range(center: float, direction: float, lo: float, hi: float) -> tuple[float, float]:
    if abs(direction) < 1e-12:
        if lo <= center <= hi:
            return -np.inf, np.inf
        return np.inf, -np.inf
    t0 = (lo - center) / direction
    t1 = (hi - center) / direction
    return (min(t0, t1), max(t0, t1))


def sample_line_profile_zx(
    plane_zx: np.ndarray,
    z0: float,
    x0: float,
    dz: float,
    dx: float,
    voxel_mm: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a profile along a line on a z-x plane using bilinear interpolation.

    Returns:
      dist_mm: signed distance along beam direction (mm)
      values: interpolated dose values
    """
    nz, nx = plane_zx.shape

    tx0, tx1 = _line_t_range(x0, dx, 0.0, float(nx - 1))
    tz0, tz1 = _line_t_range(z0, dz, 0.0, float(nz - 1))
    tmin = max(tx0, tz0)
    tmax = min(tx1, tz1)
    if not np.isfinite(tmin) or not np.isfinite(tmax) or tmax < tmin:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    t = np.arange(np.floor(tmin), np.ceil(tmax) + 1, dtype=np.float64)
    zf = z0 + dz * t
    xf = x0 + dx * t

    z0i = np.floor(zf).astype(np.int64)
    x0i = np.floor(xf).astype(np.int64)
    z1i = np.clip(z0i + 1, 0, nz - 1)
    x1i = np.clip(x0i + 1, 0, nx - 1)

    az = zf - z0i
    ax = xf - x0i

    p00 = plane_zx[z0i, x0i]
    p01 = plane_zx[z0i, x1i]
    p10 = plane_zx[z1i, x0i]
    p11 = plane_zx[z1i, x1i]
    vals = (1 - az) * (1 - ax) * p00 + (1 - az) * ax * p01 + az * (1 - ax) * p10 + az * ax * p11

    vx, _, vz = voxel_mm
    step_mm = float(np.hypot(dx * vx, dz * vz))
    dist_mm = t.astype(np.float32) * step_mm
    return dist_mm, vals.astype(np.float32)


def core_mean_depth_profile(dose: np.ndarray, half_width: int = 2) -> np.ndarray:
    nz, ny, nx = dose.shape
    cy, cx = ny // 2, nx // 2
    y0, y1 = max(0, cy - half_width), min(ny, cy + half_width + 1)
    x0, x1 = max(0, cx - half_width), min(nx, cx + half_width + 1)
    return dose[:, y0:y1, x0:x1].mean(axis=(1, 2))


def core_mean_depth_profile_at(dose: np.ndarray, y: int, x: int, half_width: int = 2) -> np.ndarray:
    nz, ny, nx = dose.shape
    yy = int(np.clip(y, 0, ny - 1))
    xx = int(np.clip(x, 0, nx - 1))
    y0, y1 = max(0, yy - half_width), min(ny, yy + half_width + 1)
    x0, x1 = max(0, xx - half_width), min(nx, xx + half_width + 1)
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
    out: dict[str, np.ndarray] = {}
    for name, lo, hi in MATERIAL_BINS:
        out[name] = (spr > lo) & (spr <= hi)
    out["all"] = np.ones_like(spr, dtype=bool)
    return out


def material_code_from_spr(spr: np.ndarray) -> np.ndarray:
    """Map SPR to compact material codes following MATERIAL_BINS order."""
    code = np.zeros(spr.shape, dtype=np.int32)
    for i, (_name, lo, hi) in enumerate(MATERIAL_BINS):
        code[(spr > lo) & (spr <= hi)] = i
    return code


def sample_line_material_zx(
    spr_plane_zx: np.ndarray,
    z0: float,
    x0: float,
    dz: float,
    dx: float,
    voxel_mm: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    nz, nx = spr_plane_zx.shape
    tx0, tx1 = _line_t_range(x0, dx, 0.0, float(nx - 1))
    tz0, tz1 = _line_t_range(z0, dz, 0.0, float(nz - 1))
    tmin = max(tx0, tz0)
    tmax = min(tx1, tz1)
    if not np.isfinite(tmin) or not np.isfinite(tmax) or tmax < tmin:
        return np.array([], dtype=np.float32), np.array([], dtype=np.int32)

    t = np.arange(np.floor(tmin), np.ceil(tmax) + 1, dtype=np.float64)
    zf = z0 + dz * t
    xf = x0 + dx * t
    zi = np.clip(np.rint(zf).astype(np.int64), 0, nz - 1)
    xi = np.clip(np.rint(xf).astype(np.int64), 0, nx - 1)
    mat = material_code_from_spr(spr_plane_zx[zi, xi])

    vx, _, vz = voxel_mm
    step_mm = float(np.hypot(dx * vx, dz * vz))
    dist_mm = t.astype(np.float32) * step_mm
    return dist_mm, mat


def material_summary_text(spr: np.ndarray) -> str:
    m = masks_from_spr(spr)
    total = float(spr.size)
    parts = []
    for name in MATERIAL_NAMES:
        frac = 100.0 * float(np.count_nonzero(m[name])) / total if total > 0 else 0.0
        if frac >= 1.0:
            parts.append(f"{name}:{frac:.0f}%")
    if not parts:
        parts = ["mixed:<1% each"]
    return ", ".join(parts)


def _apply_input_norm(inp: np.ndarray, mode: str, eps: float = 1e-8) -> np.ndarray:
    out = inp.astype(np.float32, copy=True)
    if mode == "per_channel_max":
        for c in range(out.shape[0]):
            cmax = float(np.max(out[c]))
            if cmax > eps:
                out[c] = out[c] / cmax
    elif mode == "global_max":
        gmax = float(np.max(out))
        if gmax > eps:
            out = out / gmax
    return out


def predict_normalized_dose(
    model,
    low_n: np.ndarray,
    spr_raw: np.ndarray,
    *,
    device,
    input_norm_mode: str,
    input_dose_scale: float,
) -> np.ndarray:
    import torch
    from proton_denoise.physics import normalize_spr_to_01

    spr01 = normalize_spr_to_01(spr_raw)
    inp = np.stack([low_n.astype(np.float32), spr01.astype(np.float32)], axis=0)
    inp = _apply_input_norm(inp, mode=input_norm_mode)
    inp[0] = inp[0] * float(input_dose_scale)
    x = torch.from_numpy(inp[None, ...]).to(device)
    with torch.no_grad():
        pred = model(x).detach().cpu().numpy()[0, 0]
    return pred.astype(np.float32)


def analyze_sample(
    sample_dir: Path,
    out_dir: Path,
    voxel_mm: tuple[float, float, float],
    low_plot_scale: float,
    infer_ctx: dict[str, Any] | None,
) -> dict[str, float | str]:
    low_path = _resolve_low_dose_path(sample_dir)
    if low_path is None:
        raise FileNotFoundError(
            f"No low dose found in {sample_dir}. Expected low/dose.npy or low_e*/dose.npy"
        )
    low = np.load(low_path).astype(np.float32)
    high = np.load(sample_dir / "high" / "dose.npy").astype(np.float32)
    spr = np.load(sample_dir / "high" / "spr.npy").astype(np.float32)

    if not np.isfinite(low).all():
        raise ValueError(f"Low dose contains NaN/Inf in {sample_dir}")
    if not np.isfinite(high).all():
        raise ValueError(f"High dose contains NaN/Inf in {sample_dir}")
    if not np.isfinite(spr).all():
        raise ValueError(f"SPR map contains NaN/Inf in {sample_dir}")

    with open(sample_dir / "meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    events_low = int(meta.get("events_low", 2000))
    events_high = int(meta.get("events_high", 100000))

    high_max = float(np.max(high))
    if (not np.isfinite(high_max)) or high_max <= 0:
        raise ValueError(f"High dose max is zero in {sample_dir}")

    low_n = low / high_max
    high_n = high / high_max
    low_n_plot = low_n * low_plot_scale

    pred_n = None
    pred_epoch = -1
    if infer_ctx is not None:
        pred_n = predict_normalized_dose(
            infer_ctx["model"],
            low_n,
            spr,
            device=infer_ctx["device"],
            input_norm_mode=str(infer_ctx["input_norm_mode"]),
            input_dose_scale=float(infer_ctx["input_dose_scale"]),
        )
        pred_epoch = int(infer_ctx.get("epoch", -1))

    peak_idx_3d = np.unravel_index(int(np.argmax(high_n)), high_n.shape)
    peak_z, peak_y, peak_x = map(int, peak_idx_3d)

    p_low = depth_profile_at(low_n, peak_y, peak_x)
    p_high = depth_profile_at(high_n, peak_y, peak_x)
    pm_low = core_mean_depth_profile_at(low_n, peak_y, peak_x)
    pm_high = core_mean_depth_profile_at(high_n, peak_y, peak_x)
    p_low_plot = depth_profile_at(low_n_plot, peak_y, peak_x)
    pm_low_plot = core_mean_depth_profile_at(low_n_plot, peak_y, peak_x)

    peak_idx_low = int(np.argmax(pm_low))
    peak_idx_high = int(np.argmax(pm_high))

    dx, dz = estimate_beam_direction_xz(high_n, threshold_ratio=0.2)

    cy = peak_y
    plane_low = low_n[peak_idx_high]
    plane_high = high_n[peak_idx_high]
    plane_low_plot = low_n_plot[peak_idx_high]
    plane_pred = pred_n[peak_idx_high] if pred_n is not None else None
    lat_low = plane_low[cy, :]
    lat_high = plane_high[cy, :]
    lat_low_plot = plane_low_plot[cy, :]
    lat_pred = plane_pred[cy, :] if plane_pred is not None else None
    x_mm = np.arange(low.shape[2], dtype=np.float32) * voxel_mm[0]

    # Beam-axis profile on x-z plane at hotspot y.
    beam_high_plane = high_n[:, cy, :]
    beam_low_plane = low_n[:, cy, :]
    beam_low_plot_plane = low_n_plot[:, cy, :]
    beam_pred_plane = pred_n[:, cy, :] if pred_n is not None else None
    s_mm, beam_high = sample_line_profile_zx(
        beam_high_plane,
        z0=float(peak_z),
        x0=float(peak_x),
        dz=dz,
        dx=dx,
        voxel_mm=voxel_mm,
    )
    _, beam_low = sample_line_profile_zx(
        beam_low_plane,
        z0=float(peak_z),
        x0=float(peak_x),
        dz=dz,
        dx=dx,
        voxel_mm=voxel_mm,
    )
    _, beam_low_plot = sample_line_profile_zx(
        beam_low_plot_plane,
        z0=float(peak_z),
        x0=float(peak_x),
        dz=dz,
        dx=dx,
        voxel_mm=voxel_mm,
    )
    beam_pred = np.array([], dtype=np.float32)
    if beam_pred_plane is not None:
        _, beam_pred = sample_line_profile_zx(
            beam_pred_plane,
            z0=float(peak_z),
            x0=float(peak_x),
            dz=dz,
            dx=dx,
            voxel_mm=voxel_mm,
        )
    s_mat_mm, beam_mat = sample_line_material_zx(
        spr[:, cy, :],
        z0=float(peak_z),
        x0=float(peak_x),
        dz=dz,
        dx=dx,
        voxel_mm=voxel_mm,
    )
    mat_txt = material_summary_text(spr)

    sample_out = out_dir / sample_dir.name
    sample_out.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 4))
    if s_mm.size > 0:
        plt.plot(s_mm, beam_high, label=f"High {events_high//1000:g}k (beam)", linewidth=2)
        if beam_pred.size > 0:
            pred_label = f"Pred E{pred_epoch} (beam)" if pred_epoch >= 0 else "Pred (beam)"
            plt.plot(s_mm, beam_pred, label=pred_label, linewidth=1.8)
        plt.plot(
            s_mm,
            beam_low_plot,
            label=f"Low {events_low//1000:g}k (beam) x{low_plot_scale:g}",
            alpha=0.8,
        )
    plt.plot(
        np.arange(low.shape[0], dtype=np.float32) * voxel_mm[2],
        pm_high,
        label=f"High {events_high//1000:g}k (core-z)",
        linestyle="--",
    )
    plt.plot(
        np.arange(low.shape[0], dtype=np.float32) * voxel_mm[2],
        pm_low_plot,
        label=f"Low {events_low//1000:g}k (core-z) x{low_plot_scale:g}",
        linestyle="--",
    )
    if pred_n is not None:
        pm_pred = core_mean_depth_profile_at(pred_n, peak_y, peak_x)
        pred_label = f"Pred E{pred_epoch} (core-z)" if pred_epoch >= 0 else "Pred (core-z)"
        plt.plot(
            np.arange(low.shape[0], dtype=np.float32) * voxel_mm[2],
            pm_pred,
            label=pred_label,
            linestyle="-.",
        )
    plt.xlabel("Distance (mm)")
    plt.ylabel("Normalized dose")
    plt.title(
        f"Depth-dose: {sample_dir.name} (E={meta.get('energy_mev', 'NA')} MeV) "
        f"@ hotspot(y={peak_y}, x={peak_x}), dir(dx={dx:.2f}, dz={dz:.2f})\n"
        f"Materials [{mat_txt}]"
    )
    plt.legend()
    if s_mat_mm.size > 0:
        ax2 = plt.gca().twinx()
        ax2.step(s_mat_mm, beam_mat, where="mid", color="black", alpha=0.35, linewidth=1.2)
        ax2.set_yticks(list(range(len(MATERIAL_NAMES))))
        ax2.set_yticklabels(MATERIAL_NAMES)
        ax2.set_ylabel("Material")
        ax2.set_ylim(-0.5, float(len(MATERIAL_NAMES) - 0.5))
    plt.tight_layout()
    plt.savefig(sample_out / "depth_dose_profile.png", dpi=160)
    plt.close()

    # 2D visualization on beam slice (x-z plane at hotspot y).
    x_axis_mm = np.arange(low.shape[2], dtype=np.float32) * voxel_mm[0]
    z_axis_mm = np.arange(low.shape[0], dtype=np.float32) * voxel_mm[2]
    extent = [float(x_axis_mm[0]), float(x_axis_mm[-1]), float(z_axis_mm[0]), float(z_axis_mm[-1])]
    vmax = float(np.percentile(beam_high_plane, 99.5))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(np.max(beam_high_plane)) if beam_high_plane.size > 0 else 1.0
    vmax = max(vmax, 1e-8)

    fig, axs = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    im0 = axs[0].imshow(beam_high_plane, origin="lower", aspect="auto", cmap="inferno", extent=extent, vmin=0.0, vmax=vmax)
    axs[0].set_title("High (x-z beam slice)")
    axs[0].set_xlabel("x (mm)")
    axs[0].set_ylabel("z (mm)")

    axs[1].imshow(beam_low_plane, origin="lower", aspect="auto", cmap="inferno", extent=extent, vmin=0.0, vmax=vmax)
    axs[1].set_title("Low (x-z beam slice)")
    axs[1].set_xlabel("x (mm)")
    axs[1].set_ylabel("z (mm)")

    diff = np.abs(beam_high_plane - beam_low_plane)
    dvmax = float(np.percentile(diff, 99.5))
    if not np.isfinite(dvmax) or dvmax <= 0:
        dvmax = float(np.max(diff)) if diff.size > 0 else 1.0
    dvmax = max(dvmax, 1e-8)
    axs[2].imshow(diff, origin="lower", aspect="auto", cmap="magma", extent=extent, vmin=0.0, vmax=dvmax)
    axs[2].set_title("|High - Low| (x-z)")
    axs[2].set_xlabel("x (mm)")
    axs[2].set_ylabel("z (mm)")

    cbar = fig.colorbar(im0, ax=axs[:2], fraction=0.03, pad=0.02)
    cbar.set_label("Normalized dose")
    fig.suptitle(f"Beam Slice 2D: {sample_dir.name} (y={cy})")
    fig.savefig(sample_out / "beam_slice_xz_2d.png", dpi=170)
    plt.close(fig)

    plt.figure(figsize=(7, 4))
    plt.plot(x_mm, lat_high, label=f"High {events_high//1000:g}k")
    if lat_pred is not None:
        pred_label = f"Pred E{pred_epoch}" if pred_epoch >= 0 else "Pred"
        plt.plot(x_mm, lat_pred, label=pred_label, linewidth=1.8)
    plt.plot(x_mm, lat_low_plot, label=f"Low {events_low//1000:g}k x{low_plot_scale:g}", alpha=0.85)
    plt.xlabel("Lateral x (mm)")
    plt.ylabel("Normalized dose")
    plt.title(f"Lateral profile at high peak depth (z={peak_idx_high}, y={cy})\nMaterials [{mat_txt}]")
    plt.legend()
    lat_mat = material_code_from_spr(spr[peak_idx_high, cy, :])
    ax2 = plt.gca().twinx()
    ax2.step(x_mm, lat_mat, where="mid", color="black", alpha=0.35, linewidth=1.2)
    ax2.set_yticks(list(range(len(MATERIAL_NAMES))))
    ax2.set_yticklabels(MATERIAL_NAMES)
    ax2.set_ylabel("Material")
    ax2.set_ylim(-0.5, float(len(MATERIAL_NAMES) - 0.5))
    plt.tight_layout()
    plt.savefig(sample_out / "lateral_profile.png", dpi=160)
    plt.close()

    mats = masks_from_spr(spr)
    plt.figure(figsize=(7, 5))
    for mat_name, mask in mats.items():
        xh, yh = cumulative_dvh(high_n, mask)
        # DVH must use physical normalized doses, not display-scaled low_n_plot.
        xl, yl = cumulative_dvh(low_n, mask)
        plt.plot(xh, yh, label=f"{mat_name} high", linewidth=2)
        plt.plot(xl, yl, label=f"{mat_name} low", linestyle="--", alpha=0.8)
        if pred_n is not None:
            xp, yp = cumulative_dvh(pred_n, mask)
            pred_label = f"{mat_name} pred E{pred_epoch}" if pred_epoch >= 0 else f"{mat_name} pred"
            plt.plot(xp, yp, label=pred_label, linestyle="-.", alpha=0.9)
    plt.xlabel("Normalized dose")
    plt.ylabel("Volume fraction receiving >= dose")
    plt.title(f"DVH by material: {sample_dir.name}\nMaterials [{mat_txt}]")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(sample_out / "dvh_materials.png", dpi=160)
    plt.close()

    return {
        "sample": sample_dir.name,
        "energy_mev": float(meta.get("energy_mev", -1.0)),
        "beam_dir_x": dx,
        "beam_dir_z": dz,
        "hotspot_z": peak_z,
        "hotspot_y": peak_y,
        "hotspot_x": peak_x,
        "peak_depth_low_mm": peak_idx_low * voxel_mm[2],
        "peak_depth_high_mm": peak_idx_high * voxel_mm[2],
        "peak_depth_abs_error_mm": abs(peak_idx_low - peak_idx_high) * voxel_mm[2],
        "beam_profile_mae": float(np.mean(np.abs(beam_low - beam_high))) if beam_high.size > 0 else float("nan"),
        "beam_profile_pred_mae": float(np.mean(np.abs(beam_pred - beam_high))) if beam_pred.size > 0 else float("nan"),
        "axis_mae": float(np.mean(np.abs(p_low - p_high))),
        "axis_pred_mae": float(np.mean(np.abs(depth_profile_at(pred_n, peak_y, peak_x) - p_high)))
        if pred_n is not None
        else float("nan"),
        "core_depth_mae": float(np.mean(np.abs(pm_low - pm_high))),
        "core_depth_pred_mae": float(np.mean(np.abs(core_mean_depth_profile_at(pred_n, peak_y, peak_x) - pm_high)))
        if pred_n is not None
        else float("nan"),
        "lateral_fwhm_low_mm": fwhm_mm(lat_low, voxel_mm[0]),
        "lateral_fwhm_high_mm": fwhm_mm(lat_high, voxel_mm[0]),
        "lateral_fwhm_pred_mm": fwhm_mm(lat_pred, voxel_mm[0]) if lat_pred is not None else float("nan"),
        "lateral_penumbra20_80_low_mm": penumbra_20_80_mm(lat_low, voxel_mm[0]),
        "lateral_penumbra20_80_high_mm": penumbra_20_80_mm(lat_high, voxel_mm[0]),
        "lateral_penumbra20_80_pred_mm": penumbra_20_80_mm(lat_pred, voxel_mm[0])
        if lat_pred is not None
        else float("nan"),
    }


def _discover_sample_dirs(mc_root: Path) -> list[Path]:
    """Find sample folders recursively by required dose files.

    A valid sample dir must contain high/dose.npy and at least one low dose file
    (low/dose.npy or low_e*/dose.npy).
    """
    sample_set: set[Path] = set()
    for high_dose in mc_root.glob("**/high/dose.npy"):
        sample_dir = high_dose.parent.parent
        if _resolve_low_dose_path(sample_dir) is not None and sample_dir.is_dir():
            sample_set.add(sample_dir)
    return sorted(sample_set)


def _resolve_low_dose_path(sample_dir: Path) -> Path | None:
    """Resolve low-dose file from supported layouts.

    Priority:
      1) <sample>/low/dose.npy
      2) <sample>/low_eXXXXX/dose.npy (first sorted match)
    """
    p = sample_dir / "low" / "dose.npy"
    if p.exists():
        return p

    candidates = sorted(sample_dir.glob("low_e*/dose.npy"))
    if candidates:
        return candidates[0]

    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Monte Carlo low/high simulations")
    parser.add_argument("--mc-root", type=Path, default=Path("mc_runs_opengate_small"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/sim_analysis"))
    parser.add_argument("--voxel-mm", type=float, nargs=3, default=[2.0, 2.0, 2.0])
    parser.add_argument("--low-plot-scale", type=float, default=50.0)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional model checkpoint to include prediction")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument(
        "--input-norm-mode",
        choices=["none", "per_channel_max", "global_max"],
        default="per_channel_max",
        help="Must match training preprocessing for the checkpoint (most runs use per_channel_max)",
    )
    parser.add_argument("--input-dose-scale", type=float, default=1.0)
    parser.add_argument("--max-samples", type=int, default=0, help="Analyze at most N samples (0 means all)")
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Random seed used when max-samples > 0 to choose a reproducible subset",
    )
    parser.add_argument(
        "--skip-invalid",
        action="store_true",
        help="Skip invalid/corrupt samples and continue analysis",
    )
    args = parser.parse_args()

    if not args.mc_root.exists():
        raise FileNotFoundError(f"MC root does not exist: {args.mc_root}")

    sample_dirs = _discover_sample_dirs(args.mc_root)
    if not sample_dirs:
        raise RuntimeError(
            f"No sample folders found under {args.mc_root}. "
            "Expected files like <sample>/low/dose.npy and <sample>/high/dose.npy"
        )

    if args.max_samples > 0 and len(sample_dirs) > args.max_samples:
        rng = random.Random(int(args.sample_seed))
        sample_dirs = sorted(rng.sample(sample_dirs, k=int(args.max_samples)))

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    vx, vy, vz = map(float, args.voxel_mm)
    infer_ctx: dict[str, Any] | None = None
    if args.checkpoint is not None:
        import torch
        from proton_denoise.model import load_model_from_checkpoint

        device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        model = load_model_from_checkpoint(ckpt, in_channels=2, out_channels=1).to(device)
        model.eval()
        infer_ctx = {
            "model": model,
            "device": device,
            "epoch": int(ckpt.get("epoch", -1)),
            "input_norm_mode": str(args.input_norm_mode),
            "input_dose_scale": float(args.input_dose_scale),
        }

    rows: list[dict[str, float | str]] = []
    skipped = 0
    for s in sample_dirs:
        try:
            rows.append(analyze_sample(s, out_dir, (vx, vy, vz), float(args.low_plot_scale), infer_ctx=infer_ctx))
        except Exception as exc:  # pylint: disable=broad-except
            if args.skip_invalid:
                skipped += 1
                print(f"Warning: skipping {s.name}: {exc}")
                continue
            raise

    if not rows:
        raise RuntimeError("No valid samples analyzed (all failed or none selected)")

    csv_path = out_dir / "summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    agg = {
        "n_samples": len(rows),
        "mean_peak_depth_abs_error_mm": float(np.mean([r["peak_depth_abs_error_mm"] for r in rows])),
        "mean_axis_mae": float(np.mean([r["axis_mae"] for r in rows])),
        "mean_axis_pred_mae": float(np.mean([r["axis_pred_mae"] for r in rows])),
        "mean_core_depth_mae": float(np.mean([r["core_depth_mae"] for r in rows])),
        "mean_core_depth_pred_mae": float(np.mean([r["core_depth_pred_mae"] for r in rows])),
        "mean_beam_profile_mae": float(np.mean([r["beam_profile_mae"] for r in rows])),
        "mean_beam_profile_pred_mae": float(np.mean([r["beam_profile_pred_mae"] for r in rows])),
        "mean_lateral_fwhm_abs_error_mm": float(
            np.mean([abs(r["lateral_fwhm_low_mm"] - r["lateral_fwhm_high_mm"]) for r in rows])
        ),
        "mean_lateral_fwhm_pred_abs_error_mm": float(
            np.mean([abs(r["lateral_fwhm_pred_mm"] - r["lateral_fwhm_high_mm"]) for r in rows])
        ),
        "mean_lateral_penumbra_abs_error_mm": float(
            np.mean(
                [
                    abs(r["lateral_penumbra20_80_low_mm"] - r["lateral_penumbra20_80_high_mm"])
                    for r in rows
                ]
            )
        ),
        "mean_lateral_penumbra_pred_abs_error_mm": float(
            np.mean(
                [
                    abs(r["lateral_penumbra20_80_pred_mm"] - r["lateral_penumbra20_80_high_mm"])
                    for r in rows
                ]
            )
        ),
    }

    with open(out_dir / "aggregate.json", "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)

    print(f"Analysis done for {len(rows)} samples")
    print(f"Skipped samples: {skipped}")
    print(f"Summary CSV: {csv_path}")
    print(f"Aggregate JSON: {out_dir / 'aggregate.json'}")


if __name__ == "__main__":
    main()
