#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from proton_denoise.config import TrainConfig
from proton_denoise.data import ProtonDoseDataset
from proton_denoise.losses import PhysicsWeightedMSELoss
from proton_denoise.model import ResUNet3D


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, optimizer, device, train: bool, scaler: GradScaler | None) -> float:
    model.train(train)
    losses = []
    use_amp = scaler is not None and scaler.is_enabled() and device.type == "cuda"
    for batch in tqdm(loader, leave=False):
        x = batch["input"].to(device, non_blocking=True)
        y = batch["target"].to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            with autocast(device_type=device.type, enabled=use_amp):
                yhat = model(x)
                loss = criterion(yhat, y)

            if train:
                optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else float("nan")


def main(args: argparse.Namespace) -> None:
    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        out_dir=Path(args.out_dir),
        seed=args.seed,
    )
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU device")

    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = ProtonDoseDataset(
        Path(args.data_root) / "train",
        normalize_target=not args.no_normalize_target,
        input_norm_mode=args.input_norm_mode,
        input_dose_scale=args.input_dose_scale,
        low_events_allow=args.low_events_allow,
    )
    val_ds = ProtonDoseDataset(
        Path(args.data_root) / "val",
        normalize_target=not args.no_normalize_target,
        input_norm_mode=args.input_norm_mode,
        input_dose_scale=args.input_dose_scale,
        low_events_allow=args.low_events_allow,
    )

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=pin,
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin,
        persistent_workers=args.workers > 0,
    )

    model = ResUNet3D(
        in_channels=2,
        out_channels=1,
        base_channels=args.base_channels,
        output_activation=args.output_activation,
    ).to(device)
    criterion = PhysicsWeightedMSELoss(
        alpha=args.loss_alpha,
        min_weight=args.loss_min_weight,
        background_threshold=args.background_threshold,
        background_lambda=args.background_lambda,
    )
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=TrainConfig().weight_decay)
    scaler = GradScaler(device.type, enabled=(device.type == "cuda" and args.amp))

    history = []
    start_epoch = int(args.start_epoch)

    if args.resume_checkpoint is not None:
        ckpt_path = Path(args.resume_checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        if start_epoch <= 0:
            start_epoch = int(ckpt.get("epoch", 0))
        print(f"Resumed model from: {ckpt_path} (start_epoch={start_epoch})")

    history_path = cfg.out_dir / "history.json"
    if args.resume_history and history_path.exists():
        with open(history_path, "r", encoding="utf-8") as f:
            prev = json.load(f)
            if isinstance(prev, list):
                history = prev

    best_val = float("inf")
    best_epoch = 0
    epochs_without_improve = 0

    if history:
        best_row = min(history, key=lambda r: float(r.get("val_loss", float("inf"))))
        best_val = float(best_row.get("val_loss", float("inf")))
        best_epoch = int(best_row.get("epoch", 0))

    save_epoch_set = set(args.save_epochs or [])
    save_every = int(args.save_every)

    for epoch in range(start_epoch + 1, start_epoch + cfg.epochs + 1):
        tr = run_epoch(model, train_loader, criterion, optimizer, device, train=True, scaler=scaler)
        va = run_epoch(model, val_loader, criterion, optimizer, device, train=False, scaler=scaler)

        row = {"epoch": epoch, "train_loss": tr, "val_loss": va}
        history.append(row)
        print(row)

        if (save_every > 0 and epoch % save_every == 0) or (epoch in save_epoch_set):
            ckpt_epoch = {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_loss": va,
                "base_channels": args.base_channels,
                "output_activation": args.output_activation,
            }
            torch.save(ckpt_epoch, cfg.out_dir / f"checkpoint_epoch_{epoch:03d}.pt")

        if va < (best_val - args.min_delta):
            best_val = va
            best_epoch = epoch
            epochs_without_improve = 0
            ckpt = {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_loss": va,
                "base_channels": args.base_channels,
                "output_activation": args.output_activation,
            }
            torch.save(ckpt, cfg.out_dir / "best_model.pt")
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= args.patience:
                print(
                    f"Early stopping at epoch {epoch}: no val improvement for {args.patience} epochs. "
                    f"Best epoch={best_epoch}, best_val={best_val:.6g}"
                )
                break

    with open(cfg.out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 3D ResUNet for proton dose denoising")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--out-dir", type=str, default="artifacts")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision on CUDA")
    parser.add_argument("--loss-alpha", type=float, default=3.0)
    parser.add_argument("--loss-min-weight", type=float, default=None)
    parser.add_argument(
        "--background-threshold",
        type=float,
        default=None,
        help="Optional threshold on normalized target to define low-dose voxels for background penalty",
    )
    parser.add_argument(
        "--background-lambda",
        type=float,
        default=0.0,
        help="Weight of background residual penalty term (0 disables)",
    )
    parser.add_argument(
        "--input-norm-mode",
        type=str,
        choices=["none", "per_channel_max", "global_max"],
        default="none",
        help="Input normalization strategy applied by ProtonDoseDataset",
    )
    parser.add_argument(
        "--input-dose-scale",
        type=float,
        default=1.0,
        help="Additional scale factor for input channel 0 (low-dose channel)",
    )
    parser.add_argument(
        "--no-normalize-target",
        action="store_true",
        help="Disable per-sample target/max(target) normalization in dataset loader",
    )
    parser.add_argument(
        "--low-events-allow",
        type=int,
        nargs="*",
        default=None,
        help="Optional list of low_events values to include (requires low_events in .npz)",
    )
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--save-every", type=int, default=0, help="Save checkpoint every N epochs (0 disables)")
    parser.add_argument("--save-epochs", type=int, nargs="*", default=None, help="Explicit epochs to save")
    parser.add_argument("--resume-checkpoint", type=str, default=None, help="Checkpoint path to resume model weights")
    parser.add_argument("--start-epoch", type=int, default=0, help="Starting epoch number for resumed run")
    parser.add_argument("--resume-history", action="store_true", help="Append to existing history.json in out-dir")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument(
        "--output-activation",
        type=str,
        choices=["identity", "relu", "softplus"],
        default="identity",
        help="Final output activation after model head",
    )
    main(parser.parse_args())
