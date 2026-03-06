#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from proton_denoise.physics import build_multilayer_phantom, normalize_spr_to_01


def _load_optional_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_array(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path).astype(np.float32)


def _save_npz(
    out_path: Path,
    noisy: np.ndarray,
    target: np.ndarray,
    spr01: np.ndarray,
    energy: float,
    events_low: int,
    events_high: int,
) -> None:
    inp = np.stack([noisy, spr01], axis=0).astype(np.float32)
    np.savez_compressed(
        out_path,
        input=inp,
        target=target.astype(np.float32),
        spr=spr01.astype(np.float32),
        energy_mev=np.float32(energy),
        low_events=np.int32(events_low),
        high_events=np.int32(events_high),
    )


def _split_indices(n: int, train_ratio: float, val_ratio: float, seed: int) -> dict[str, list[int]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)

    train_idx = idx[:n_train].tolist()
    val_idx = idx[n_train : n_train + n_val].tolist()
    test_idx = idx[n_train + n_val :].tolist()
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def main(args: argparse.Namespace) -> None:
    mc_root = Path(args.mc_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    sample_dirs = sorted([p for p in mc_root.iterdir() if p.is_dir()])
    if not sample_dirs:
        raise RuntimeError(f"No sample folders found in {mc_root}")

    splits = _split_indices(len(sample_dirs), args.train_ratio, args.val_ratio, args.seed)

    phantom = build_multilayer_phantom()
    spr_default = normalize_spr_to_01(phantom.spr_map)

    split_lookup = {}
    for split_name, indices in splits.items():
        for i in indices:
            split_lookup[i] = split_name
            (out_root / split_name).mkdir(parents=True, exist_ok=True)

    for i, sample_dir in enumerate(sample_dirs):
        split = split_lookup[i]

        low_path = sample_dir / "low" / args.dose_filename
        high_path = sample_dir / "high" / args.dose_filename

        low = _load_array(low_path)
        high = _load_array(high_path)

        if low.shape != high.shape:
            raise ValueError(f"Shape mismatch in {sample_dir}: low={low.shape} high={high.shape}")

        spr_path = sample_dir / "high" / args.spr_filename
        if spr_path.exists():
            spr = _load_array(spr_path)
            spr01 = normalize_spr_to_01(spr)
        else:
            # Fallback to canonical slab SPR if simulation did not export SPR map.
            if spr_default.shape != high.shape:
                raise ValueError(
                    f"Missing SPR and default slab shape mismatch for {sample_dir}: "
                    f"default={spr_default.shape} dose={high.shape}"
                )
            spr01 = spr_default

        meta = _load_optional_json(sample_dir / "meta.json")
        energy = float(meta.get("energy_mev", -1.0))
        events_low = int(meta.get("events_low", args.default_events_low))
        events_high = int(meta.get("events_high", args.default_events_high))
        if energy < 0:
            print(f"Warning: missing energy in {sample_dir}/meta.json, setting -1")

        target_max = float(np.max(high))
        if target_max <= 0:
            raise ValueError(f"Target max dose is zero in {sample_dir}")

        target = high / target_max
        noisy = low / target_max
        if args.rescale_low_by_history_ratio:
            # Make low-dose input comparable across different history levels.
            # E[dose] scales linearly with particle histories in MC.
            if events_low <= 0 or events_high <= 0:
                raise ValueError(f"Invalid event counts for {sample_dir}: low={events_low}, high={events_high}")
            noisy = noisy * (float(events_high) / float(events_low))

        out_name = f"sample_{i:04d}.npz"
        out_path = out_root / split / out_name
        _save_npz(out_path, noisy, target, spr01, energy, events_low=events_low, events_high=events_high)

    print("Dataset build complete")
    for split in ["train", "val", "test"]:
        n = len(list((out_root / split).glob("*.npz")))
        print(f"{split}: {n}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build training dataset from real Monte Carlo outputs")
    parser.add_argument("--mc-root", type=str, default="mc_runs")
    parser.add_argument("--out-root", type=str, default="data")
    parser.add_argument("--dose-filename", type=str, default="dose.npy")
    parser.add_argument("--spr-filename", type=str, default="spr.npy")
    parser.add_argument("--train-ratio", type=float, default=0.75)
    parser.add_argument("--val-ratio", type=float, default=0.125)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--rescale-low-by-history-ratio",
        action="store_true",
        help="Multiply normalized low-dose input by (events_high / events_low) using meta.json counts",
    )
    parser.add_argument("--default-events-low", type=int, default=2000)
    parser.add_argument("--default-events-high", type=int, default=100000)
    main(parser.parse_args())
