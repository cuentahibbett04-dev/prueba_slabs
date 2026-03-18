#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from proton_denoise.data import ProtonDoseDataset
from proton_denoise.metrics import gamma_pass_rate
from proton_denoise.model import load_model_from_checkpoint


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


def _depth_profile_band(vol: np.ndarray, cy: int, cx: int, half_width_xy: int) -> np.ndarray:
    ny, nx = vol.shape[1], vol.shape[2]
    y0, y1 = max(0, cy - half_width_xy), min(ny, cy + half_width_xy + 1)
    x0, x1 = max(0, cx - half_width_xy), min(nx, cx + half_width_xy + 1)
    return vol[:, y0:y1, x0:x1].mean(axis=(1, 2))


def _sphere_mask(shape: tuple[int, int, int], center_zyx: tuple[int, int, int], radius_mm: float, voxel_mm: tuple[float, float, float]) -> np.ndarray:
    nz, ny, nx = shape
    cz, cy, cx = center_zyx
    vx, vy, vz = voxel_mm
    zz, yy, xx = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
    dz = (zz - cz) * vz
    dy = (yy - cy) * vy
    dx = (xx - cx) * vx
    rr = np.sqrt(dx * dx + dy * dy + dz * dz)
    return rr <= float(radius_mm)


def _dvh_curve(dose: np.ndarray, mask: np.ndarray, n_bins: int = 200) -> tuple[np.ndarray, np.ndarray]:
    vals = dose[mask]
    if vals.size == 0:
        return np.array([0.0]), np.array([0.0])
    dmax = float(np.max(vals))
    grid = np.linspace(0.0, dmax, int(n_bins), dtype=np.float64)
    frac = np.array([100.0 * np.mean(vals >= d) for d in grid], dtype=np.float64)
    return grid, frac


def _dose_at_volume_percent(dose: np.ndarray, mask: np.ndarray, vol_percent: float) -> float:
    vals = dose[mask]
    if vals.size == 0:
        return float("nan")
    # Dv: dose that at least v% of volume receives.
    q = max(0.0, min(100.0, 100.0 - float(vol_percent)))
    return float(np.percentile(vals, q))


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


def _patient_id_from_path(npz_path: str, group_separator: str) -> str:
    stem = Path(npz_path).stem
    if group_separator and group_separator in stem:
        return stem.split(group_separator, 1)[0]
    return stem


def _build_groups_from_json(groups_json_path: str) -> list[dict[str, Any]]:
    with open(groups_json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict) and "groups" in payload:
        groups = payload["groups"]
    elif isinstance(payload, list):
        groups = payload
    else:
        raise ValueError("groups json must be a list or an object with key 'groups'")

    out: list[dict[str, Any]] = []
    for g in groups:
        if not isinstance(g, dict):
            raise ValueError("each group must be an object")
        pid = str(g.get("patient_id", "")).strip()
        if not pid:
            raise ValueError("group missing patient_id")
        idxs = g.get("indices", None)
        if not isinstance(idxs, list) or not idxs:
            raise ValueError(f"group '{pid}' has invalid or empty indices")
        out.append({"patient_id": pid, "indices": [int(v) for v in idxs]})
    return out


def _project_nonnegative(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _compute_lipschitz(
    beams: np.ndarray,
    ptv_mask: np.ndarray,
    oar_mask: np.ndarray,
    lambda_oar: float,
    lambda_l2: float,
) -> float:
    n = beams.shape[0]
    ptv_w = ptv_mask.astype(np.float64).reshape(1, -1)
    oar_w = oar_mask.astype(np.float64).reshape(1, -1)
    weighted = beams * (ptv_w + float(lambda_oar) * oar_w)
    h = weighted @ beams.T
    vals = np.linalg.eigvalsh(h)
    lmax = float(vals[-1]) if vals.size else 1.0
    return max(lmax + float(lambda_l2), 1e-8)


def _fista_optimize_fluence(
    beams: np.ndarray,
    ptv_mask: np.ndarray,
    oar_mask: np.ndarray,
    rx_scalar: float,
    lambda_oar: float,
    lambda_l2: float,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, dict[str, float]]:
    n_beams, n_vox = beams.shape
    if n_beams <= 0:
        raise ValueError("beams matrix has zero rows")

    ptv_w = ptv_mask.astype(np.float64).reshape(-1)
    oar_w = oar_mask.astype(np.float64).reshape(-1)
    y_ref = rx_scalar * ptv_w
    beta = ptv_w + float(lambda_oar) * oar_w

    # Precompute Hessian-like and linear terms for fast iterations.
    weighted_beams = beams * beta.reshape(1, n_vox)
    h = weighted_beams @ beams.T
    b = beams @ y_ref
    if lambda_l2 > 0.0:
        h = h + float(lambda_l2) * np.eye(n_beams, dtype=np.float64)

    lips = _compute_lipschitz(beams, ptv_mask, oar_mask, lambda_oar=lambda_oar, lambda_l2=lambda_l2)
    step = 1.0 / lips

    f = np.zeros(n_beams, dtype=np.float64)
    y = f.copy()
    t = 1.0
    prev_obj = np.inf
    converged = False
    last_iter = 0

    for it in range(1, int(max_iter) + 1):
        grad = (h @ y) - b
        f_next = _project_nonnegative(y - step * grad)

        d = beams.T @ f_next
        res_ptv = (d - rx_scalar) * ptv_w
        res_oar = d * oar_w
        obj = 0.5 * float(np.sum(res_ptv * res_ptv))
        if lambda_oar > 0.0:
            obj += 0.5 * float(lambda_oar) * float(np.sum(res_oar * res_oar))
        if lambda_l2 > 0.0:
            obj += 0.5 * float(lambda_l2) * float(np.sum(f_next * f_next))

        rel = abs(prev_obj - obj) / max(abs(prev_obj), 1e-12)
        if rel < float(tol):
            converged = True
            f = f_next
            last_iter = it
            prev_obj = obj
            break

        t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        y = f_next + ((t - 1.0) / t_next) * (f_next - f)
        f = f_next
        t = t_next
        prev_obj = obj
        last_iter = it

    return f, {
        "objective": float(prev_obj),
        "iterations": float(last_iter),
        "lipschitz": float(lips),
        "converged": float(1.0 if converged else 0.0),
    }


def _make_plan_masks(target_ref_sum: np.ndarray, voxel_mm: tuple[float, float, float], ptv_radius_mm: float, oar_radius_mm: float, oar_offset_x_mm: float) -> tuple[np.ndarray, np.ndarray]:
    zc, yc, xc = np.unravel_index(np.argmax(target_ref_sum), target_ref_sum.shape)
    ptv_mask = _sphere_mask(
        target_ref_sum.shape,
        (int(zc), int(yc), int(xc)),
        float(ptv_radius_mm),
        voxel_mm,
    )
    ox = int(round(float(oar_offset_x_mm) / max(float(voxel_mm[0]), 1e-6)))
    oar_center = (int(zc), int(yc), int(np.clip(xc + ox, 0, target_ref_sum.shape[2] - 1)))
    oar_mask = _sphere_mask(target_ref_sum.shape, oar_center, float(oar_radius_mm), voxel_mm)
    return ptv_mask, oar_mask


def _compute_hi_cn(dose: np.ndarray, ptv_mask: np.ndarray, rx_scalar: float) -> tuple[float, float]:
    d2 = _dose_at_volume_percent(dose, ptv_mask, 2.0)
    d98 = _dose_at_volume_percent(dose, ptv_mask, 98.0)
    d50 = _dose_at_volume_percent(dose, ptv_mask, 50.0)
    hi = float((d2 - d98) / max(d50, 1e-12))

    ri = dose >= float(rx_scalar)
    tv = int(np.count_nonzero(ptv_mask))
    vri = int(np.count_nonzero(ri))
    tvri = int(np.count_nonzero(ptv_mask & ri))
    cn = float((tvri * tvri) / max(tv * vri, 1))
    return hi, cn


def _gamma_pair(eval_dose: np.ndarray, ref_dose: np.ndarray, voxel_mm: tuple[float, float, float], dd: float, dta: float, threshold: float, stride: int, max_points: int | None, seed: int) -> float:
    return float(
        gamma_pass_rate(
            eval_dose,
            ref_dose,
            voxel_mm=voxel_mm,
            dose_diff_percent=float(dd),
            distance_mm=float(dta),
            dose_threshold_percent=float(threshold),
            eval_stride=max(int(stride), 1),
            max_eval_points=max_points,
            random_seed=int(seed),
        )
    )


def _build_dvh_rows(dose_grid: np.ndarray, curves: dict[str, tuple[np.ndarray, np.ndarray]]) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    n = len(dose_grid)
    for i in range(n):
        row: dict[str, float] = {"dose": float(dose_grid[i])}
        for k, (_, frac) in curves.items():
            row[k] = float(frac[i]) if i < len(frac) else float("nan")
        rows.append(row)
    return rows


def _save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _safe_get_ds_path(ds: ProtonDoseDataset, idx: int) -> str:
    # Avoid materializing arrays just to get a path string.
    try:
        return str(ds.files[idx])
    except Exception:
        # Conservative fallback in case dataset internals change.
        return str(ds[idx].get("path", ""))


def main() -> None:
    ap = argparse.ArgumentParser(description="End-to-end clinical-planning style evaluation with optimization (GT/Noisy/DeepMC)")
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument("--split", type=str, choices=["train", "val", "test"], default="test")
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--indices", type=int, nargs="+", default=None)
    ap.add_argument("--groups-json", type=str, default=None)
    ap.add_argument("--group-separator", type=str, default="_g")
    ap.add_argument("--max-patients", type=int, default=3)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--input-norm-mode", choices=["none", "per_channel_max", "global_max", "coupled_target_max"], default="none")
    ap.add_argument("--input-dose-scale", type=float, default=1.0)
    ap.add_argument("--no-normalize-target", action="store_true")
    ap.add_argument("--crop-shape", type=int, nargs=3, default=None, metavar=("D", "H", "W"))
    ap.add_argument("--crop-focus", type=str, choices=["center", "maxdose"], default="center")
    ap.add_argument("--no-use-checkpoint-data-prep", action="store_true")
    ap.add_argument("--voxel-mm", type=float, nargs=3, default=[2.0, 2.0, 2.0])
    ap.add_argument("--ptv-radius-mm", type=float, default=20.0)
    ap.add_argument("--oar-radius-mm", type=float, default=12.0)
    ap.add_argument("--oar-offset-x-mm", type=float, default=30.0)
    ap.add_argument("--depth-half-width", type=int, default=2)
    ap.add_argument("--gamma-dd", type=float, default=2.0)
    ap.add_argument("--gamma-dta", type=float, default=2.0)
    ap.add_argument("--gamma-threshold", type=float, default=1.0)
    ap.add_argument("--gamma-eval-stride", type=int, default=8)
    ap.add_argument("--gamma-max-eval-points", type=int, default=3000)
    ap.add_argument("--gamma-random-seed", type=int, default=42)
    ap.add_argument("--fista-iters", type=int, default=300)
    ap.add_argument("--fista-tol", type=float, default=1e-6)
    ap.add_argument("--lambda-oar", type=float, default=0.2)
    ap.add_argument("--lambda-l2", type=float, default=1e-4)
    ap.add_argument("--rx-scale", type=float, default=1.0)
    ap.add_argument("--low-mc-total-time-sec", type=float, default=None)
    ap.add_argument("--high-mc-total-time-sec", type=float, default=None)
    ap.add_argument(
        "--pred-cache-dir",
        type=str,
        default=None,
        help="Optional directory for per-sample prediction cache (.npz)",
    )
    ap.add_argument(
        "--read-pred-cache",
        action="store_true",
        help="Read prediction from --pred-cache-dir when available",
    )
    ap.add_argument(
        "--write-pred-cache",
        action="store_true",
        help="Write per-sample predictions to --pred-cache-dir",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _log(f"Output dir ready: {out_dir}")

    pred_cache_dir = Path(args.pred_cache_dir) if args.pred_cache_dir else None
    if pred_cache_dir is not None and (args.read_pred_cache or args.write_pred_cache):
        pred_cache_dir.mkdir(parents=True, exist_ok=True)
        _log(
            f"Prediction cache enabled: dir={pred_cache_dir}, "
            f"read={bool(args.read_pred_cache)}, write={bool(args.write_pred_cache)}"
        )

    _log(f"Loading checkpoint: {args.checkpoint}")
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
    _log(
        "Dataset loaded: "
        f"split={args.split}, samples={len(ds)}, input_norm_mode={input_norm_mode}, "
        f"crop_shape={crop_shape}, crop_focus={crop_focus}"
    )

    cuda_available = torch.cuda.is_available()
    if args.device == "cuda" and not cuda_available:
        _log("WARNING: --device cuda requested but CUDA is not available; falling back to CPU")
    device = torch.device("cuda" if cuda_available and args.device == "cuda" else "cpu")
    model = load_model_from_checkpoint(ckpt, in_channels=2, out_channels=1).to(device)
    model.eval()
    _log(f"Model loaded on device: {device}")

    vx, vy, vz = map(float, args.voxel_mm)
    voxel_mm = (vx, vy, vz)
    gmax = int(args.gamma_max_eval_points)
    gmax = None if gmax <= 0 else gmax

    all_indices = list(range(len(ds))) if args.indices is None else [int(v) for v in args.indices]
    _log(f"Planning index set prepared: n_indices={len(all_indices)}")
    for idx in all_indices:
        if idx < 0 or idx >= len(ds):
            raise IndexError(f"Index {idx} out of bounds for split '{args.split}' with {len(ds)} samples")

    if args.groups_json:
        groups = _build_groups_from_json(args.groups_json)
    else:
        by_patient: dict[str, list[int]] = {}
        for i, idx in enumerate(all_indices, start=1):
            npz_path = _safe_get_ds_path(ds, idx)
            pid = _patient_id_from_path(npz_path, str(args.group_separator))
            by_patient.setdefault(pid, []).append(int(idx))
            if i % 100 == 0 or i == len(all_indices):
                _log(f"Grouping progress: {i}/{len(all_indices)} indices")
        ordered_pids = sorted(by_patient.keys())
        groups = [{"patient_id": pid, "indices": by_patient[pid]} for pid in ordered_pids]
        if int(args.max_patients) > 0:
            groups = groups[: int(args.max_patients)]

    if not groups:
        raise ValueError("No patient groups available for planning")
    _log(f"Patient groups ready: n_groups={len(groups)}")

    patient_rows: list[dict[str, Any]] = []
    beam_rows: list[dict[str, Any]] = []
    dvh_rows: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []
    summary_patients: list[dict[str, Any]] = []

    total_infer_sec = 0.0
    total_opt_sec = 0.0

    for p_i, group in enumerate(groups, start=1):
        patient_id = str(group["patient_id"])
        indices = [int(v) for v in group["indices"]]
        if not indices:
            continue
        _log(f"[{p_i}/{len(groups)}] Patient {patient_id}: start (n_beams={len(indices)})")

        gt_list: list[np.ndarray] = []
        noisy_list: list[np.ndarray] = []
        pred_list: list[np.ndarray] = []
        ct_list: list[np.ndarray] = []

        for b_i, idx in enumerate(indices, start=1):
            item = ds[idx]
            x = item["input"].unsqueeze(0).to(device)
            target = item["target"][0].cpu().numpy().astype(np.float64)
            noisy_model = item["input"][0].cpu().numpy().astype(np.float64)
            ct = item["input"][1].cpu().numpy().astype(np.float64)

            # Bring low-stat input back to target scale.
            noisy = noisy_model.copy()
            if float(input_dose_scale) != 0.0:
                noisy = noisy / float(input_dose_scale)
            if input_norm_mode == "coupled_target_max":
                low_events_val = int(item.get("low_events", torch.tensor([-1])).item())
                high_events_val = int(item.get("high_events", torch.tensor([-1])).item())
                if low_events_val > 0 and high_events_val > 0:
                    hist_ratio = float(high_events_val) / float(low_events_val)
                    if hist_ratio > 0.0:
                        noisy = noisy / hist_ratio

            cache_hit = False
            cache_path = None
            if pred_cache_dir is not None:
                cache_path = pred_cache_dir / f"{args.split}_{idx:05d}.npz"

            if bool(args.read_pred_cache) and cache_path is not None and cache_path.exists():
                with np.load(cache_path) as zc:
                    pred = zc["pred"].astype(np.float64)
                infer_sec = 0.0
                cache_hit = True
                _log(f"[{p_i}/{len(groups)}] Patient {patient_id}: cache hit for idx={idx}")
            else:
                t0 = time.perf_counter()
                pred = model(x).detach().cpu().numpy()[0, 0].astype(np.float64)
                t1 = time.perf_counter()
                infer_sec = float(t1 - t0)

                if bool(args.write_pred_cache) and cache_path is not None:
                    np.savez_compressed(
                        cache_path,
                        pred=pred.astype(np.float32),
                        index=np.int32(idx),
                        split=str(args.split),
                        case_name=Path(str(item.get("path", ""))).stem,
                    )
                    _log(f"[{p_i}/{len(groups)}] Patient {patient_id}: cache write for idx={idx}")

            total_infer_sec += infer_sec
            if cache_hit:
                _log(
                    f"[{p_i}/{len(groups)}] Patient {patient_id}: beam {b_i}/{len(indices)} loaded from cache (idx={idx})"
                )
            else:
                _log(
                    f"[{p_i}/{len(groups)}] Patient {patient_id}: beam {b_i}/{len(indices)} "
                    f"inferred in {infer_sec:.3f}s (idx={idx})"
                )

            gt_list.append(target)
            noisy_list.append(noisy)
            pred_list.append(pred)
            ct_list.append(ct)

            beam_rows.append(
                {
                    "patient_id": patient_id,
                    "index": int(idx),
                    "case_name": Path(str(item.get("path", ""))).stem,
                    "energy_mev": float(item["energy_mev"].item()),
                    "inference_sec": infer_sec,
                    "mae_pred_vs_ref": _mae(pred, target),
                    "rmse_pred_vs_ref": _rmse(pred, target),
                    "psnr_pred_vs_ref": _psnr(pred, target, data_range=1.0),
                    "ssim_pred_vs_ref": _ssim_global(pred, target, data_range=1.0),
                    "mae_noisy_vs_ref": _mae(noisy, target),
                    "rmse_noisy_vs_ref": _rmse(noisy, target),
                    "psnr_noisy_vs_ref": _psnr(noisy, target, data_range=1.0),
                    "ssim_noisy_vs_ref": _ssim_global(noisy, target, data_range=1.0),
                }
            )

        gt = np.stack(gt_list, axis=0)
        noisy = np.stack(noisy_list, axis=0)
        pred = np.stack(pred_list, axis=0)
        ct = np.stack(ct_list, axis=0)

        n_beams = gt.shape[0]
        shape3d = tuple(int(v) for v in gt.shape[1:])
        n_vox = int(np.prod(shape3d))

        a_gt = gt.reshape(n_beams, n_vox)
        a_noisy = noisy.reshape(n_beams, n_vox)
        a_pred = pred.reshape(n_beams, n_vox)

        target_ref_sum = np.sum(gt, axis=0)
        ptv_mask, oar_mask = _make_plan_masks(
            target_ref_sum,
            voxel_mm=voxel_mm,
            ptv_radius_mm=float(args.ptv_radius_mm),
            oar_radius_mm=float(args.oar_radius_mm),
            oar_offset_x_mm=float(args.oar_offset_x_mm),
        )
        ptv_mask_flat = ptv_mask.reshape(-1)
        oar_mask_flat = oar_mask.reshape(-1)

        rx_scalar = float(args.rx_scale) * float(np.mean(target_ref_sum[ptv_mask]))

        scenario_a = {
            "GT": a_gt,
            "Noisy": a_noisy,
            "DeepMC": a_pred,
        }
        _log(f"[{p_i}/{len(groups)}] Patient {patient_id}: matrices A ready (shape={a_gt.shape})")

        # Optimization in planning space for each scenario.
        plan_dose: dict[str, np.ndarray] = {}
        fluence: dict[str, np.ndarray] = {}
        opt_meta: dict[str, dict[str, float]] = {}

        for scenario_name, a_use in scenario_a.items():
            to0 = time.perf_counter()
            _log(f"[{p_i}/{len(groups)}] Patient {patient_id}: optimize scenario={scenario_name} (FISTA)")
            f, meta = _fista_optimize_fluence(
                beams=a_use,
                ptv_mask=ptv_mask_flat,
                oar_mask=oar_mask_flat,
                rx_scalar=rx_scalar,
                lambda_oar=float(args.lambda_oar),
                lambda_l2=float(args.lambda_l2),
                max_iter=int(args.fista_iters),
                tol=float(args.fista_tol),
            )
            to1 = time.perf_counter()

            fluence[scenario_name] = f
            opt_meta[scenario_name] = meta
            opt_meta[scenario_name]["optimize_sec"] = float(to1 - to0)
            total_opt_sec += float(to1 - to0)
            _log(
                f"[{p_i}/{len(groups)}] Patient {patient_id}: scenario={scenario_name} done "
                f"in {float(to1 - to0):.3f}s, iters={int(meta.get('iterations', 0.0))}, "
                f"converged={int(meta.get('converged', 0.0) > 0.5)}"
            )

            plan_dose[scenario_name] = (a_use.T @ f).reshape(shape3d)

        # Deliverable dose always computed with GT physics matrix.
        deliverable = {
            "GT": (a_gt.T @ fluence["GT"]).reshape(shape3d),
            "Noisy": (a_gt.T @ fluence["Noisy"]).reshape(shape3d),
            "DeepMC": (a_gt.T @ fluence["DeepMC"]).reshape(shape3d),
        }
        ref_deliverable = deliverable["GT"]

        # Scenario metrics.
        for scenario_name in ["GT", "Noisy", "DeepMC"]:
            d_plan = plan_dose[scenario_name]
            d_deliv = deliverable[scenario_name]
            _log(f"[{p_i}/{len(groups)}] Patient {patient_id}: metrics scenario={scenario_name} (gamma 2/2 + 1/1)")

            gamma_2_2 = _gamma_pair(
                d_deliv,
                ref_deliverable,
                voxel_mm=voxel_mm,
                dd=2.0,
                dta=2.0,
                threshold=float(args.gamma_threshold),
                stride=int(args.gamma_eval_stride),
                max_points=gmax,
                seed=int(args.gamma_random_seed),
            )
            gamma_1_1 = _gamma_pair(
                d_deliv,
                ref_deliverable,
                voxel_mm=voxel_mm,
                dd=1.0,
                dta=1.0,
                threshold=float(args.gamma_threshold),
                stride=int(args.gamma_eval_stride),
                max_points=gmax,
                seed=int(args.gamma_random_seed),
            )

            hi, cn = _compute_hi_cn(d_deliv, ptv_mask, rx_scalar=rx_scalar)
            d95_ptv = _dose_at_volume_percent(d_deliv, ptv_mask, 95.0)
            d2_oar = _dose_at_volume_percent(d_deliv, oar_mask, 2.0)

            scenario_rows.append(
                {
                    "patient_id": patient_id,
                    "scenario": scenario_name,
                    "n_beams": int(n_beams),
                    "rx_scalar": float(rx_scalar),
                    "optimize_sec": float(opt_meta[scenario_name].get("optimize_sec", float("nan"))),
                    "fista_iters": int(opt_meta[scenario_name].get("iterations", 0.0)),
                    "fista_converged": int(opt_meta[scenario_name].get("converged", 0.0) > 0.5),
                    "fista_objective": float(opt_meta[scenario_name].get("objective", float("nan"))),
                    "mae_planning_vs_deliverable": _mae(d_plan, d_deliv),
                    "rmse_planning_vs_deliverable": _rmse(d_plan, d_deliv),
                    "mae_deliverable_vs_ref": _mae(d_deliv, ref_deliverable),
                    "rmse_deliverable_vs_ref": _rmse(d_deliv, ref_deliverable),
                    "gamma_2mm_2pct_deliverable_vs_ref": gamma_2_2,
                    "gamma_1mm_1pct_deliverable_vs_ref": gamma_1_1,
                    "hi_ptv": hi,
                    "cn_ptv": cn,
                    "d95_ptv": d95_ptv,
                    "d2_oar": d2_oar,
                }
            )

        # DVH export rows.
        dvh_dose, dvh_ptv_gt = _dvh_curve(deliverable["GT"], ptv_mask)
        _, dvh_ptv_noisy = _dvh_curve(deliverable["Noisy"], ptv_mask)
        _, dvh_ptv_deepmc = _dvh_curve(deliverable["DeepMC"], ptv_mask)
        _, dvh_oar_gt = _dvh_curve(deliverable["GT"], oar_mask)
        _, dvh_oar_noisy = _dvh_curve(deliverable["Noisy"], oar_mask)
        _, dvh_oar_deepmc = _dvh_curve(deliverable["DeepMC"], oar_mask)

        patient_dvh_rows = _build_dvh_rows(
            dvh_dose,
            {
                "ptv_gt": (dvh_dose, dvh_ptv_gt),
                "ptv_noisy": (dvh_dose, dvh_ptv_noisy),
                "ptv_deepmc": (dvh_dose, dvh_ptv_deepmc),
                "oar_gt": (dvh_dose, dvh_oar_gt),
                "oar_noisy": (dvh_dose, dvh_oar_noisy),
                "oar_deepmc": (dvh_dose, dvh_oar_deepmc),
            },
        )
        for row in patient_dvh_rows:
            row["patient_id"] = patient_id
        dvh_rows.extend(patient_dvh_rows)
        _log(f"[{p_i}/{len(groups)}] Patient {patient_id}: DVH rows added ({len(patient_dvh_rows)})")

        # Patient summary + report figure.
        cy, cx = _estimate_beam_center_xy(ref_deliverable)
        prof_gt = _depth_profile_band(ref_deliverable, cy, cx, half_width_xy=int(args.depth_half_width))
        prof_noisy = _depth_profile_band(deliverable["Noisy"], cy, cx, half_width_xy=int(args.depth_half_width))
        prof_deepmc = _depth_profile_band(deliverable["DeepMC"], cy, cx, half_width_xy=int(args.depth_half_width))
        z_axis = np.arange(len(prof_gt)) * vz

        patient_summary = {
            "patient_id": patient_id,
            "indices": indices,
            "n_beams": int(n_beams),
            "rx_scalar": float(rx_scalar),
            "ptv_voxels": int(np.count_nonzero(ptv_mask)),
            "oar_voxels": int(np.count_nonzero(oar_mask)),
        }
        summary_patients.append(patient_summary)

        fig, axs = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)

        axs[0, 0].plot(z_axis, prof_gt, label="GT deliverable", linewidth=2)
        axs[0, 0].plot(z_axis, prof_noisy, label="Noisy deliverable", alpha=0.9)
        axs[0, 0].plot(z_axis, prof_deepmc, label="DeepMC deliverable", alpha=0.9)
        axs[0, 0].set_title(f"SOBP depth profile ({patient_id})")
        axs[0, 0].set_xlabel("z (mm)")
        axs[0, 0].set_ylabel("Dose")
        axs[0, 0].legend()
        axs[0, 0].grid(alpha=0.25)

        axs[0, 1].plot(dvh_dose, dvh_ptv_gt, label="PTV GT", linewidth=2)
        axs[0, 1].plot(dvh_dose, dvh_ptv_noisy, label="PTV Noisy", alpha=0.9)
        axs[0, 1].plot(dvh_dose, dvh_ptv_deepmc, label="PTV DeepMC", alpha=0.9)
        axs[0, 1].set_title("DVH PTV (deliverable)")
        axs[0, 1].set_xlabel("Dose")
        axs[0, 1].set_ylabel("Volume >= D (%)")
        axs[0, 1].legend()
        axs[0, 1].grid(alpha=0.25)

        axs[0, 2].plot(dvh_dose, dvh_oar_gt, label="OAR GT", linewidth=2)
        axs[0, 2].plot(dvh_dose, dvh_oar_noisy, label="OAR Noisy", alpha=0.9)
        axs[0, 2].plot(dvh_dose, dvh_oar_deepmc, label="OAR DeepMC", alpha=0.9)
        axs[0, 2].set_title("DVH OAR (deliverable)")
        axs[0, 2].set_xlabel("Dose")
        axs[0, 2].set_ylabel("Volume >= D (%)")
        axs[0, 2].legend()
        axs[0, 2].grid(alpha=0.25)

        _, cy_map, _ = np.unravel_index(np.argmax(ref_deliverable), ref_deliverable.shape)
        gt_xz = ref_deliverable[:, cy_map, :]
        deepmc_xz = deliverable["DeepMC"][:, cy_map, :]
        noisy_xz = deliverable["Noisy"][:, cy_map, :]
        ct_mean = np.mean(ct, axis=0)
        ct_xz = ct_mean[:, cy_map, :]

        # Robust CT normalization for stable background contrast.
        ct_p1, ct_p99 = np.percentile(ct_xz, [1.0, 99.0])
        ct_den = max(float(ct_p99 - ct_p1), 1e-8)
        ct_bg = np.clip((ct_xz - float(ct_p1)) / ct_den, 0.0, 1.0)

        # Use shared robust scale for dose overlays.
        dose_vmax = max(
            float(np.percentile(gt_xz, 99.5)),
            float(np.percentile(deepmc_xz, 99.5)),
            float(np.percentile(noisy_xz, 99.5)),
            1e-8,
        )
        gt_n = np.clip(gt_xz / dose_vmax, 0.0, 1.0)
        deepmc_n = np.clip(deepmc_xz / dose_vmax, 0.0, 1.0)

        err_xz = np.abs(noisy_xz - gt_xz)
        err_vmax = max(float(np.percentile(err_xz, 99.5)), 1e-8)
        err_n = np.clip(err_xz / err_vmax, 0.0, 1.0)

        axs[1, 0].imshow(ct_bg, cmap="gray", origin="lower", aspect="auto")
        axs[1, 0].imshow(gt_n, cmap="inferno", origin="lower", aspect="auto", alpha=np.clip(gt_n, 0.0, 1.0) ** 0.65)
        axs[1, 0].set_title("GT deliverable on CT (xz)")

        axs[1, 1].imshow(ct_bg, cmap="gray", origin="lower", aspect="auto")
        axs[1, 1].imshow(deepmc_n, cmap="inferno", origin="lower", aspect="auto", alpha=np.clip(deepmc_n, 0.0, 1.0) ** 0.65)
        axs[1, 1].set_title("DeepMC deliverable on CT (xz)")

        axs[1, 2].imshow(ct_bg, cmap="gray", origin="lower", aspect="auto")
        im = axs[1, 2].imshow(err_n, cmap="magma", origin="lower", aspect="auto", alpha=np.clip(err_n, 0.0, 1.0) ** 0.65)
        axs[1, 2].set_title("|Noisy-GT| on CT (xz)")
        fig.colorbar(im, ax=axs[1, 2], fraction=0.046, pad=0.04)

        fig.suptitle(f"Clinical Planning Evaluation - {patient_id}", fontsize=14)
        fig.savefig(out_dir / f"clinical_plan_report_{patient_id}.png", dpi=170)
        plt.close(fig)
        _log(f"[{p_i}/{len(groups)}] Patient {patient_id}: report saved")

        patient_rows.append(
            {
                "patient_id": patient_id,
                "n_beams": int(n_beams),
                "rx_scalar": float(rx_scalar),
            }
        )
        _log(f"[{p_i}/{len(groups)}] Patient {patient_id}: completed")

    # Aggregate summary.
    summary: dict[str, Any] = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "n_patients": len(summary_patients),
        "patients": summary_patients,
        "inference_total_sec": float(total_infer_sec),
        "optimization_total_sec": float(total_opt_sec),
        "inference_mean_per_beam_sec": float(np.mean([r["inference_sec"] for r in beam_rows])) if beam_rows else float("nan"),
    }
    if args.low_mc_total_time_sec is not None:
        summary["low_mc_total_time_sec"] = float(args.low_mc_total_time_sec)
    if args.high_mc_total_time_sec is not None:
        summary["high_mc_total_time_sec"] = float(args.high_mc_total_time_sec)
    if args.low_mc_total_time_sec is not None and args.high_mc_total_time_sec is not None:
        den = float(args.low_mc_total_time_sec) + float(total_infer_sec) + float(total_opt_sec)
        if den > 0:
            summary["speedup_highmc_over_lowmc_plus_ai_plus_opt"] = float(args.high_mc_total_time_sec) / den

    # Aggregate scenario means.
    by_scenario: dict[str, list[dict[str, Any]]] = {}
    for row in scenario_rows:
        by_scenario.setdefault(str(row["scenario"]), []).append(row)
    aggregate: dict[str, dict[str, float]] = {}
    for scen, rows in by_scenario.items():
        aggregate[scen] = {
            "mean_gamma_2mm_2pct_deliverable_vs_ref": float(np.mean([r["gamma_2mm_2pct_deliverable_vs_ref"] for r in rows])),
            "mean_gamma_1mm_1pct_deliverable_vs_ref": float(np.mean([r["gamma_1mm_1pct_deliverable_vs_ref"] for r in rows])),
            "mean_hi_ptv": float(np.mean([r["hi_ptv"] for r in rows])),
            "mean_cn_ptv": float(np.mean([r["cn_ptv"] for r in rows])),
            "mean_mae_planning_vs_deliverable": float(np.mean([r["mae_planning_vs_deliverable"] for r in rows])),
            "mean_rmse_planning_vs_deliverable": float(np.mean([r["rmse_planning_vs_deliverable"] for r in rows])),
            "mean_mae_deliverable_vs_ref": float(np.mean([r["mae_deliverable_vs_ref"] for r in rows])),
            "mean_rmse_deliverable_vs_ref": float(np.mean([r["rmse_deliverable_vs_ref"] for r in rows])),
        }
    summary["aggregate_by_scenario"] = aggregate

    with open(out_dir / "clinical_plan_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    _log("Summary JSON saved")

    _save_csv(out_dir / "patient_plan_index.csv", patient_rows)
    _log("Saved patient_plan_index.csv")
    _save_csv(out_dir / "per_beam_metrics.csv", beam_rows)
    _log("Saved per_beam_metrics.csv")
    _save_csv(out_dir / "scenario_metrics.csv", scenario_rows)
    _log("Saved scenario_metrics.csv")
    _save_csv(out_dir / "dvh_curves.csv", dvh_rows)
    _log("Saved dvh_curves.csv")

    _log(
        "Run finished: "
        f"patients={len(summary_patients)}, beams={len(beam_rows)}, "
        f"inference_total_sec={total_infer_sec:.3f}, optimization_total_sec={total_opt_sec:.3f}"
    )

    print(json.dumps(summary, indent=2))
    print(f"Saved: {out_dir / 'clinical_plan_summary.json'}")
    print(f"Saved: {out_dir / 'patient_plan_index.csv'}")
    print(f"Saved: {out_dir / 'per_beam_metrics.csv'}")
    print(f"Saved: {out_dir / 'scenario_metrics.csv'}")
    print(f"Saved: {out_dir / 'dvh_curves.csv'}")


if __name__ == "__main__":
    main()
