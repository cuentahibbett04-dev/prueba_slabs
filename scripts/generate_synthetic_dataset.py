#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from proton_denoise.config import DataConfig, GeometryConfig, MaterialSPR
from proton_denoise.physics import (
    add_monte_carlo_noise,
    build_multilayer_phantom,
    normalize_spr_to_01,
    simulate_reference_dose,
)


def _save_sample(path: Path, noisy: np.ndarray, target: np.ndarray, spr01: np.ndarray, energy: float) -> None:
    # Input is 2 channels: [noisy dose, SPR map]
    inp = np.stack([noisy, spr01], axis=0).astype(np.float32)
    np.savez_compressed(
        path,
        input=inp,
        target=target.astype(np.float32),
        spr=spr01.astype(np.float32),
        energy_mev=np.float32(energy),
    )


def generate_dataset(root: Path, seed: int = 42) -> None:
    np.random.seed(seed)

    data_cfg = DataConfig(root=root)
    phantom = build_multilayer_phantom(GeometryConfig(), MaterialSPR())
    spr01 = normalize_spr_to_01(phantom.spr_map)

    splits = {
        "train": data_cfg.train_count,
        "val": data_cfg.val_count,
        "test": data_cfg.test_count,
    }

    for split, n in splits.items():
        out_dir = root / split
        out_dir.mkdir(parents=True, exist_ok=True)
        for i in tqdm(range(n), desc=f"Generating {split}"):
            energy = np.random.uniform(data_cfg.energy_min_mev, data_cfg.energy_max_mev)

            target_raw = simulate_reference_dose(energy, phantom.spr_map, phantom.voxel_mm)
            # Normalize both by max of target, as requested.
            target_max = float(np.max(target_raw))
            target = target_raw / max(target_max, 1e-8)

            noisy_raw = add_monte_carlo_noise(target_raw, data_cfg.low_stat_events)
            noisy = noisy_raw / max(target_max, 1e-8)

            _save_sample(out_dir / f"sample_{i:04d}.npz", noisy, target, spr01, energy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic slab-based proton dose dataset")
    parser.add_argument("--root", type=Path, default=Path("data"), help="Dataset root directory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_dataset(args.root, seed=args.seed)
