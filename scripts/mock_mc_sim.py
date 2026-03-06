#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from proton_denoise.physics import (
    add_monte_carlo_noise,
    build_multilayer_phantom,
    simulate_reference_dose,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mock Monte Carlo simulator output generator")
    parser.add_argument("--energy", type=float, required=True)
    parser.add_argument("--events", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    np.random.seed(args.seed)

    phantom = build_multilayer_phantom()
    target = simulate_reference_dose(args.energy, phantom.spr_map, phantom.voxel_mm)
    dose = add_monte_carlo_noise(target, args.events)

    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "dose.npy", dose.astype(np.float32))
    np.save(args.out / "spr.npy", phantom.spr_map.astype(np.float32))


if __name__ == "__main__":
    main()
