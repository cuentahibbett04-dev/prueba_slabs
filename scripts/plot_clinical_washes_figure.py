
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from evaluate_clinical_plan import (
    _fista_optimize_fluence,
    _get_ckpt_data_prep,
    _make_plan_masks,
    _patient_id_from_path,
)
from proton_denoise.data import ProtonDoseDataset
from proton_denoise.model import load_model_from_checkpoint


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _body_bbox_from_ct(ct_slice: np.ndarray, pad: int = 8) -> tuple[int, int, int, int]:
    """Return cropped [y0:y1, x0:x1] bbox around body-like region in CT slice."""
    if ct_slice.ndim != 2:
        raise ValueError("ct_slice must be 2D")
    ny, nx = ct_slice.shape

    # Robust threshold to separate body from air/background in normalized CT channel.
    p5 = float(np.percentile(ct_slice, 5.0))
    p50 = float(np.percentile(ct_slice, 50.0))
    thr = p5 + 0.2 * (p50 - p5)
    mask = ct_slice > thr

    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        return 0, ny, 0, nx

    y0 = max(0, int(ys.min()) - int(pad))
    y1 = min(ny, int(ys.max()) + int(pad) + 1)
    x0 = max(0, int(xs.min()) - int(pad))
    x1 = min(nx, int(xs.max()) + int(pad) + 1)
    return y0, y1, x0, x1


def _gamma_map_2d(
    eval_dose: np.ndarray,
    ref_dose: np.ndarray,
    voxel_mm: tuple[float, float],
    dose_diff_percent: float,
    distance_mm: float,
    dose_threshold_percent: float,
) -> np.ndarray:
    if eval_dose.shape != ref_dose.shape:
        raise ValueError("eval_dose and ref_dose must have same shape")

    max_ref = float(np.max(ref_dose))
    out = np.full_like(ref_dose, fill_value=np.nan, dtype=np.float64)
    if max_ref <= 0.0:
        return out

    dd_crit = (float(dose_diff_percent) / 100.0) * max_ref
    thr = (float(dose_threshold_percent) / 100.0) * max_ref

    vy, vx = float(voxel_mm[0]), float(voxel_mm[1])
    ry = int(np.ceil(float(distance_mm) / max(vy, 1e-6)))
    rx = int(np.ceil(float(distance_mm) / max(vx, 1e-6)))

    ny, nx = ref_dose.shape
    valid = ref_dose >= thr

    for y in range(ny):
        for x in range(nx):
            if not valid[y, x]:
                continue
            p = eval_dose[y, x]
            gmin = np.inf
            y0, y1 = max(0, y - ry), min(ny, y + ry + 1)
            x0, x1 = max(0, x - rx), min(nx, x + rx + 1)
            for yy in range(y0, y1):
                dy_mm = abs(yy - y) * vy
                for xx in range(x0, x1):
                    dx_mm = abs(xx - x) * vx
                    dist_term = (np.sqrt(dx_mm * dx_mm + dy_mm * dy_mm) / max(float(distance_mm), 1e-8)) ** 2
                    dose_term = ((p - ref_dose[yy, xx]) / max(dd_crit, 1e-8)) ** 2
                    g = np.sqrt(dist_term + dose_term)
                    if g < gmin:
                        gmin = g
            out[y, x] = gmin

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Figure-10-like clinical wash plot for GT/Noisy/DeepMC deliverable dose")
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--groups-json", type=str, default=None)
    ap.add_argument("--group-separator", type=str, default="_g")
    ap.add_argument("--max-patients", type=int, default=3)
    ap.add_argument("--input-norm-mode", choices=["none", "per_channel_max", "global_max", "coupled_target_max"], default="none")
    ap.add_argument("--input-dose-scale", type=float, default=1.0)
    ap.add_argument("--no-normalize-target", action="store_true")
    ap.add_argument("--crop-shape", type=int, nargs=3, default=None, metavar=("D", "H", "W"))
    ap.add_argument("--crop-focus", choices=["center", "maxdose"], default="center")
    ap.add_argument("--no-use-checkpoint-data-prep", action="store_true")
    ap.add_argument("--voxel-mm", type=float, nargs=3, default=[2.0, 2.0, 2.0])
    ap.add_argument("--ptv-radius-mm", type=float, default=20.0)
    ap.add_argument("--oar-radius-mm", type=float, default=12.0)
    ap.add_argument("--oar-offset-x-mm", type=float, default=30.0)
    ap.add_argument("--fista-iters", type=int, default=300)
    ap.add_argument("--fista-tol", type=float, default=1e-6)
    ap.add_argument("--lambda-oar", type=float, default=0.2)
    ap.add_argument("--lambda-l2", type=float, default=1e-4)
    ap.add_argument("--rx-scale", type=float, default=1.0)
    ap.add_argument("--gamma-dd", type=float, default=1.0)
    ap.add_argument("--gamma-dta", type=float, default=1.0)
    ap.add_argument("--gamma-threshold", type=float, default=1.0)
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
    ap.add_argument(
        "--save-ct-overlay",
        action="store_true",
        help="Also save per-patient full-plan overlays (GT/Noisy/DeepMC) on CT background",
    )
    ap.add_argument(
        "--slice-mode",
        choices=["transverse", "longitudinal_xz", "longitudinal_yz"],
        default="transverse",
        help=(
            "Slice orientation for figure rows: "
            "transverse = axial yx @ z_ptv (paper-like), "
            "longitudinal_xz = xz @ y_center, longitudinal_yz = yz @ x_center"
        ),
    )
    ap.add_argument(
        "--crop-to-ct-body",
        action="store_true",
        help="Crop all row panels to CT body bounding box for tighter presentation framing",
    )
    ap.add_argument(
        "--crop-pad",
        type=int,
        default=8,
        help="Padding (pixels) around CT body crop box",
    )
    args = ap.parse_args()

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

    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        _log("WARNING: CUDA requested but unavailable, using CPU")
    model = load_model_from_checkpoint(ckpt, in_channels=2, out_channels=1).to(device)
    model.eval()
    _log(f"Model loaded on device: {device}")

    all_indices = list(range(len(ds)))
    if args.groups_json:
        with open(args.groups_json, "r", encoding="utf-8") as f:
            payload = json.load(f)
        groups = payload["groups"] if isinstance(payload, dict) and "groups" in payload else payload
    else:
        by_patient: dict[str, list[int]] = {}
        for idx in all_indices:
            pid = _patient_id_from_path(str(ds.files[idx]), str(args.group_separator))
            by_patient.setdefault(pid, []).append(idx)
        groups = [{"patient_id": k, "indices": v} for k, v in sorted(by_patient.items())]
        if int(args.max_patients) > 0:
            groups = groups[: int(args.max_patients)]

    if not groups:
        raise ValueError("No groups found")
    _log(f"Patient groups selected: {len(groups)}")
    _log(f"Slice mode: {args.slice_mode}")

    pred_cache_dir = Path(args.pred_cache_dir) if args.pred_cache_dir else None
    if pred_cache_dir is not None and (args.read_pred_cache or args.write_pred_cache):
        pred_cache_dir.mkdir(parents=True, exist_ok=True)
        _log(
            f"Prediction cache enabled: dir={pred_cache_dir}, "
            f"read={bool(args.read_pred_cache)}, write={bool(args.write_pred_cache)}"
        )

    vx, vy, vz = map(float, args.voxel_mm)
    rows_data: list[dict[str, object]] = []

    for i, g in enumerate(groups, start=1):
        pid = str(g["patient_id"])
        indices = [int(v) for v in g["indices"]]
        if not indices:
            continue
        _log(f"[{i}/{len(groups)}] Patient {pid}: start (beams={len(indices)})")

        gt_list: list[np.ndarray] = []
        noisy_list: list[np.ndarray] = []
        pred_list: list[np.ndarray] = []
        ct_ref = None

        for b, idx in enumerate(indices, start=1):
            item = ds[idx]
            x = item["input"].unsqueeze(0).to(device)
            target = item["target"][0].cpu().numpy().astype(np.float64)
            noisy_model = item["input"][0].cpu().numpy().astype(np.float64)

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
                cache_hit = True
            else:
                pred = model(x).detach().cpu().numpy()[0, 0].astype(np.float64)
                if bool(args.write_pred_cache) and cache_path is not None:
                    np.savez_compressed(
                        cache_path,
                        pred=pred.astype(np.float32),
                        index=np.int32(idx),
                        split=str(args.split),
                        case_name=Path(str(item.get("path", ""))).stem,
                    )
            if ct_ref is None:
                ct_ref = item["input"][1].cpu().numpy().astype(np.float64)
            if b % 10 == 0 or b == len(indices):
                if cache_hit:
                    _log(f"[{i}/{len(groups)}] Patient {pid}: beam {b}/{len(indices)} loaded from cache")
                else:
                    _log(f"[{i}/{len(groups)}] Patient {pid}: inferred beam {b}/{len(indices)}")
            gt_list.append(target)
            noisy_list.append(noisy)
            pred_list.append(pred)

        gt = np.stack(gt_list, axis=0)
        noisy = np.stack(noisy_list, axis=0)
        pred = np.stack(pred_list, axis=0)

        n_beams = gt.shape[0]
        shape3d = tuple(int(v) for v in gt.shape[1:])
        n_vox = int(np.prod(shape3d))

        a_gt = gt.reshape(n_beams, n_vox)
        a_noisy = noisy.reshape(n_beams, n_vox)
        a_pred = pred.reshape(n_beams, n_vox)

        target_ref_sum = np.sum(gt, axis=0)
        ptv_mask, oar_mask = _make_plan_masks(
            target_ref_sum,
            voxel_mm=(vx, vy, vz),
            ptv_radius_mm=float(args.ptv_radius_mm),
            oar_radius_mm=float(args.oar_radius_mm),
            oar_offset_x_mm=float(args.oar_offset_x_mm),
        )
        ptv_flat = ptv_mask.reshape(-1)
        oar_flat = oar_mask.reshape(-1)

        rx_scalar = float(args.rx_scale) * float(np.mean(target_ref_sum[ptv_mask]))

        fluence = {}
        for name, a_use in {"GT": a_gt, "Noisy": a_noisy, "DeepMC": a_pred}.items():
            _log(f"[{i}/{len(groups)}] Patient {pid}: optimize {name}")
            f, _ = _fista_optimize_fluence(
                beams=a_use,
                ptv_mask=ptv_flat,
                oar_mask=oar_flat,
                rx_scalar=rx_scalar,
                lambda_oar=float(args.lambda_oar),
                lambda_l2=float(args.lambda_l2),
                max_iter=int(args.fista_iters),
                tol=float(args.fista_tol),
            )
            fluence[name] = f

        d_gt = (a_gt.T @ fluence["GT"]).reshape(shape3d)
        d_noisy = (a_gt.T @ fluence["Noisy"]).reshape(shape3d)
        d_deepmc = (a_gt.T @ fluence["DeepMC"]).reshape(shape3d)

        z_ptv = int(np.argmax(np.sum(ptv_mask.astype(np.float64), axis=(1, 2))))
        y_ptv = int(np.argmax(np.sum(ptv_mask.astype(np.float64), axis=(0, 2))))
        x_ptv = int(np.argmax(np.sum(ptv_mask.astype(np.float64), axis=(0, 1))))

        if args.slice_mode == "transverse":
            # Axial plane (y, x), perpendicular to beam depth axis z.
            gt_slice = d_gt[z_ptv]
            noisy_slice = d_noisy[z_ptv]
            deepmc_slice = d_deepmc[z_ptv]
            ct_slice_row = ct_ref[z_ptv] if ct_ref is not None else np.zeros_like(gt_slice)
            slice_label = f"z={z_ptv}"
        elif args.slice_mode == "longitudinal_xz":
            # Longitudinal plane (z, x), parallel to beam axis z.
            gt_slice = d_gt[:, y_ptv, :]
            noisy_slice = d_noisy[:, y_ptv, :]
            deepmc_slice = d_deepmc[:, y_ptv, :]
            ct_slice_row = ct_ref[:, y_ptv, :] if ct_ref is not None else np.zeros_like(gt_slice)
            slice_label = f"y={y_ptv}"
        else:
            # Longitudinal plane (z, y), parallel to beam axis z.
            gt_slice = d_gt[:, :, x_ptv]
            noisy_slice = d_noisy[:, :, x_ptv]
            deepmc_slice = d_deepmc[:, :, x_ptv]
            ct_slice_row = ct_ref[:, :, x_ptv] if ct_ref is not None else np.zeros_like(gt_slice)
            slice_label = f"x={x_ptv}"

        diff_noisy = noisy_slice - gt_slice
        diff_deepmc = deepmc_slice - gt_slice

        g_noisy = _gamma_map_2d(
            noisy_slice,
            gt_slice,
            voxel_mm=(vy, vx),
            dose_diff_percent=float(args.gamma_dd),
            distance_mm=float(args.gamma_dta),
            dose_threshold_percent=float(args.gamma_threshold),
        )
        g_deepmc = _gamma_map_2d(
            deepmc_slice,
            gt_slice,
            voxel_mm=(vy, vx),
            dose_diff_percent=float(args.gamma_dd),
            distance_mm=float(args.gamma_dta),
            dose_threshold_percent=float(args.gamma_threshold),
        )

        rows_data.append(
            {
                "patient_id": pid,
                "z_slice": z_ptv,
                "y_slice": y_ptv,
                "x_slice": x_ptv,
                "slice_mode": str(args.slice_mode),
                "slice_label": slice_label,
                "gt": gt_slice,
                "noisy": noisy_slice,
                "deepmc": deepmc_slice,
                "diff_noisy": diff_noisy,
                "diff_deepmc": diff_deepmc,
                "gamma_noisy": g_noisy,
                "gamma_deepmc": g_deepmc,
                "ct_slice": ct_slice_row,
            }
        )

        if args.save_ct_overlay and ct_ref is not None:
            if args.slice_mode == "transverse":
                ct_slice = ct_ref[z_ptv]
                d_gt_ov = d_gt[z_ptv]
                d_noisy_ov = d_noisy[z_ptv]
                d_deepmc_ov = d_deepmc[z_ptv]
            elif args.slice_mode == "longitudinal_xz":
                ct_slice = ct_ref[:, y_ptv, :]
                d_gt_ov = d_gt[:, y_ptv, :]
                d_noisy_ov = d_noisy[:, y_ptv, :]
                d_deepmc_ov = d_deepmc[:, y_ptv, :]
            else:
                ct_slice = ct_ref[:, :, x_ptv]
                d_gt_ov = d_gt[:, :, x_ptv]
                d_noisy_ov = d_noisy[:, :, x_ptv]
                d_deepmc_ov = d_deepmc[:, :, x_ptv]

            # Robust normalization for visually stable overlays.
            ct_p1, ct_p99 = np.percentile(ct_slice, [1.0, 99.0])
            ct_den = max(float(ct_p99 - ct_p1), 1e-8)
            ct_bg = np.clip((ct_slice - float(ct_p1)) / ct_den, 0.0, 1.0)

            dmax = float(max(np.percentile(d_gt_ov, 99.5), np.percentile(d_noisy_ov, 99.5), np.percentile(d_deepmc_ov, 99.5), 1e-8))
            d_gt_n = np.clip(d_gt_ov / dmax, 0.0, 1.0)
            d_noisy_n = np.clip(d_noisy_ov / dmax, 0.0, 1.0)
            d_deepmc_n = np.clip(d_deepmc_ov / dmax, 0.0, 1.0)

            fov, aov = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
            for ax, dn, title in [
                (aov[0], d_gt_n, "GT deliverable on CT"),
                (aov[1], d_noisy_n, "Noisy deliverable on CT"),
                (aov[2], d_deepmc_n, "DeepMC deliverable on CT"),
            ]:
                ax.imshow(ct_bg, cmap="gray", origin="lower", aspect="auto")
                ax.imshow(dn, cmap="inferno", origin="lower", aspect="auto", alpha=np.clip(dn, 0.0, 1.0) ** 0.65)
                ax.set_title(title)
                ax.set_xticks([])
                ax.set_yticks([])

            overlay_path = args.out.parent / f"{args.out.stem}_{pid}_overlay_ct.png"
            fov.savefig(overlay_path, dpi=200)
            plt.close(fov)
            _log(f"[{i}/{len(groups)}] Patient {pid}: saved CT overlay {overlay_path}")
        _log(f"[{i}/{len(groups)}] Patient {pid}: row ready")

    if not rows_data:
        raise ValueError("No rows to plot")

    n_rows = len(rows_data)
    n_cols = 7
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.0 * n_rows), constrained_layout=True)
    if n_rows == 1:
        axs = np.expand_dims(axs, axis=0)

    dose_titles = ["Noisy", "DeepMC", "Ground Truth"]

    for r, row in enumerate(rows_data):
        pid = str(row["patient_id"])
        gt_slice = row["gt"]
        noisy_slice = row["noisy"]
        deepmc_slice = row["deepmc"]
        diff_noisy = row["diff_noisy"]
        diff_deepmc = row["diff_deepmc"]
        g_noisy = row["gamma_noisy"]
        g_deepmc = row["gamma_deepmc"]
        ct_slice = row["ct_slice"]

        if bool(args.crop_to_ct_body):
            y0, y1, x0, x1 = _body_bbox_from_ct(ct_slice, pad=int(args.crop_pad))
            gt_slice = gt_slice[y0:y1, x0:x1]
            noisy_slice = noisy_slice[y0:y1, x0:x1]
            deepmc_slice = deepmc_slice[y0:y1, x0:x1]
            diff_noisy = diff_noisy[y0:y1, x0:x1]
            diff_deepmc = diff_deepmc[y0:y1, x0:x1]
            g_noisy = g_noisy[y0:y1, x0:x1]
            g_deepmc = g_deepmc[y0:y1, x0:x1]

        dose_vmin = float(np.min([np.min(noisy_slice), np.min(deepmc_slice), np.min(gt_slice)]))
        dose_vmax = float(np.max([np.max(noisy_slice), np.max(deepmc_slice), np.max(gt_slice)]))
        ad = max(float(np.max(np.abs(diff_noisy))), float(np.max(np.abs(diff_deepmc))), 1e-8)

        dose_panels = [noisy_slice, deepmc_slice, gt_slice]
        for c in range(3):
            im = axs[r, c].imshow(dose_panels[c], cmap="viridis", origin="lower", vmin=dose_vmin, vmax=dose_vmax)
            axs[r, c].set_title(dose_titles[c])
            axs[r, c].set_xticks([])
            axs[r, c].set_yticks([])
            if c == 2:
                cb = fig.colorbar(im, ax=axs[r, c], fraction=0.046, pad=0.02)
                cb.ax.tick_params(labelsize=8)

        for c, arr, ttl in [
            (3, diff_noisy, "Noisy - GT"),
            (4, diff_deepmc, "DeepMC - GT"),
        ]:
            imd = axs[r, c].imshow(arr, cmap="RdBu_r", origin="lower", vmin=-ad, vmax=ad)
            axs[r, c].set_title(ttl)
            axs[r, c].set_xticks([])
            axs[r, c].set_yticks([])
            if c == 4:
                cbd = fig.colorbar(imd, ax=axs[r, c], fraction=0.046, pad=0.02)
                cbd.ax.tick_params(labelsize=8)

        for c, gm, ttl in [
            (5, g_noisy, "1%/1mm: Noisy"),
            (6, g_deepmc, "1%/1mm: DeepMC"),
        ]:
            # Show pass in blue and fail in red by plotting gamma clipped to [0,2].
            gshow = np.clip(gm, 0.0, 2.0)
            img = axs[r, c].imshow(gshow, cmap="RdBu_r", origin="lower", vmin=0.0, vmax=2.0)
            axs[r, c].set_title(ttl)
            axs[r, c].set_xticks([])
            axs[r, c].set_yticks([])
            if c == 6:
                cbg = fig.colorbar(img, ax=axs[r, c], fraction=0.046, pad=0.02)
                cbg.ax.tick_params(labelsize=8)

        axs[r, 0].set_ylabel(f"{pid}\n{row['slice_label']}", fontsize=11)

    fig.suptitle("Deliverable Dose Washes and 1%/1mm Gamma Maps", fontsize=14)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    _log(f"Saving figure: {args.out}")
    fig.savefig(args.out, dpi=200)
    plt.close(fig)

    meta = {
        "rows": [
            {
                "patient_id": str(r["patient_id"]),
                "z_slice": int(r["z_slice"]),
                "y_slice": int(r["y_slice"]),
                "x_slice": int(r["x_slice"]),
                "slice_mode": str(r["slice_mode"]),
                "slice_label": str(r["slice_label"]),
            }
            for r in rows_data
        ],
        "gamma_criterion": {"dd_percent": float(args.gamma_dd), "dta_mm": float(args.gamma_dta), "threshold_percent": float(args.gamma_threshold)},
    }
    with open(args.out.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    _log("Saved JSON metadata")

    print(args.out)
    print(args.out.with_suffix(".json"))


if __name__ == "__main__":
    main()
