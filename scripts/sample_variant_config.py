#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import random


LAYER_PRESETS = [
    ("water,air,water", "60,80,60"),
    ("water,lung,water", "60,80,60"),
    ("water,bone,water", "60,80,60"),
]

ANGLE_PRESETS_DEG = [0.0]


def stable_seed(sample_id: str, global_seed: int) -> int:
    h = hashlib.sha256(f"{sample_id}|{global_seed}".encode("utf-8")).hexdigest()
    return int(h[:12], 16)


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic per-sample geometry/source variation")
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--global-seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(stable_seed(args.sample_id, args.global_seed))

    layer_order, layer_thickness = rng.choice(LAYER_PRESETS)
    source_x_mm = 0.0
    source_y_mm = 0.0
    source_z_cm = -30.0
    incidence_angle_deg = rng.choice(ANGLE_PRESETS_DEG)

    # Output format is easy to parse in shell with read -r.
    print(layer_order, layer_thickness, source_x_mm, source_y_mm, source_z_cm, incidence_angle_deg)


if __name__ == "__main__":
    main()
