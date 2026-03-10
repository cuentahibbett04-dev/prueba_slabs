#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import random


LAYER_PRESETS = [
    ("water,bone,lung,water", "40,30,50,80"),
    ("water,lung,bone,water", "40,50,30,80"),
    ("bone,water,lung,water", "30,40,50,80"),
    ("lung,water,bone,water", "50,40,30,80"),
    ("water,bone,water,lung", "40,30,80,50"),
    ("water,lung,water,bone", "40,50,80,30"),
    ("water,air,bone,water", "40,20,40,100"),
    ("water,lung,air,water", "45,35,20,100"),
    ("bone,air,water,lung", "30,20,60,90"),
]

ANGLE_PRESETS_DEG = [-66.0, -45.0, -30.0, -15.0, 0.0, 15.0, 30.0, 45.0, 66.0]


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
    source_x_mm = rng.choice([-6.0, -3.0, 0.0, 3.0, 6.0])
    source_y_mm = rng.choice([-6.0, -3.0, 0.0, 3.0, 6.0])
    source_z_cm = rng.choice([-28.0, -30.0, -32.0, -35.0])
    incidence_angle_deg = rng.choice(ANGLE_PRESETS_DEG)

    # Output format is easy to parse in shell with read -r.
    print(layer_order, layer_thickness, source_x_mm, source_y_mm, source_z_cm, incidence_angle_deg)


if __name__ == "__main__":
    main()
