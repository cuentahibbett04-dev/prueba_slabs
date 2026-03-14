#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader

from proton_denoise.data import ProtonDoseDataset
from proton_denoise.losses import PhysicsWeightedMSELoss
from proton_denoise.model import ALLOWED_ARCHS, build_model


def one_train_step(
    model: torch.nn.Module,
    batch: dict,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    amp: bool,
    amp_dtype: torch.dtype,
    scaler: GradScaler,
) -> float:
    x = batch["input"].to(device, non_blocking=True)
    y = batch["target"].to(device, non_blocking=True)

    use_amp = amp and device.type == "cuda"
    optimizer.zero_grad(set_to_none=True)
    with autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
        yhat = model(x)
        loss = criterion(yhat, y)

    if use_amp and scaler.is_enabled():
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()

    return float(loss.item())


def run_benchmark_for_batch(
    args: argparse.Namespace,
    batch_size: int,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> dict:
    ds = ProtonDoseDataset(
        Path(args.data_root) / "train",
        normalize_target=not args.no_normalize_target,
        input_norm_mode=args.input_norm_mode,
        input_dose_scale=args.input_dose_scale,
        low_events_allow=args.low_events_allow,
    )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=args.workers > 0,
    )

    model = build_model(
        args.arch,
        in_channels=2,
        out_channels=1,
        base_channels=args.base_channels,
        output_activation=args.output_activation,
    ).to(device)
    model.train(True)

    criterion = PhysicsWeightedMSELoss(
        alpha=args.loss_alpha,
        min_weight=args.loss_min_weight,
        background_threshold=args.background_threshold,
        background_lambda=args.background_lambda,
    )
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scaler = GradScaler(device.type, enabled=(device.type == "cuda" and args.amp and args.amp_dtype == "fp16"))

    it = itertools.cycle(loader)

    # Warmup steps (excluded from metrics): absorbs MIOpen search / graph setup.
    for _ in range(args.warmup_steps):
        batch = next(it)
        _ = one_train_step(model, batch, criterion, optimizer, device, args.amp, amp_dtype, scaler)

    losses = []
    start = time.perf_counter()
    for _ in range(args.measure_steps):
        batch = next(it)
        losses.append(one_train_step(model, batch, criterion, optimizer, device, args.amp, amp_dtype, scaler))
    end = time.perf_counter()

    elapsed = end - start
    steps_per_sec = args.measure_steps / max(elapsed, 1e-12)
    samples_per_sec = steps_per_sec * batch_size

    return {
        "batch_size": batch_size,
        "warmup_steps": args.warmup_steps,
        "measure_steps": args.measure_steps,
        "elapsed_sec": elapsed,
        "steps_per_sec": steps_per_sec,
        "samples_per_sec": samples_per_sec,
        "mean_loss": float(np.mean(losses)) if losses else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark batch sizes excluding startup warmup")
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument("--arch", type=str, choices=list(ALLOWED_ARCHS), default="resunet3d")
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[2, 4, 8, 12])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--amp-dtype", type=str, choices=["fp16", "bf16"], default="bf16")
    ap.add_argument("--warmup-steps", type=int, default=80)
    ap.add_argument("--measure-steps", type=int, default=200)
    ap.add_argument("--out-json", type=str, default="artifacts/batch_benchmark/results.json")

    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--base-channels", type=int, default=16)
    ap.add_argument("--output-activation", type=str, choices=["identity", "relu", "softplus"], default="relu")
    ap.add_argument("--loss-alpha", type=float, default=3.0)
    ap.add_argument("--loss-min-weight", type=float, default=0.05)
    ap.add_argument("--background-threshold", type=float, default=0.02)
    ap.add_argument("--background-lambda", type=float, default=0.2)
    ap.add_argument(
        "--input-norm-mode",
        type=str,
        choices=["none", "per_channel_max", "global_max", "coupled_target_max"],
        default="per_channel_max",
    )
    ap.add_argument("--input-dose-scale", type=float, default=1.0)
    ap.add_argument("--no-normalize-target", action="store_true")
    ap.add_argument("--low-events-allow", type=int, nargs="*", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16

    print(f"Using device={device}, arch={args.arch}, amp={args.amp}, amp_dtype={args.amp_dtype}")

    results = []
    for bs in args.batch_sizes:
        print(f"\n=== Benchmark batch_size={bs} (warmup {args.warmup_steps}, measure {args.measure_steps}) ===")
        r = run_benchmark_for_batch(args, batch_size=int(bs), device=device, amp_dtype=amp_dtype)
        results.append(r)
        print(
            f"batch={r['batch_size']} steps/s={r['steps_per_sec']:.3f} "
            f"samples/s={r['samples_per_sec']:.3f} elapsed={r['elapsed_sec']:.2f}s"
        )

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    best = max(results, key=lambda d: d["samples_per_sec"])
    print("\nBest batch size by samples/s:")
    print(best)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
