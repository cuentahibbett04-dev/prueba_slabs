#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shlex
import subprocess
from pathlib import Path

import numpy as np


def stable_seed(text: str, salt: str) -> int:
    h = hashlib.sha256(f"{salt}|{text}".encode("utf-8")).hexdigest()
    return (int(h[:8], 16) % 2_147_483_647) + 1


def run(cmd: str) -> None:
    print(f"[CMD] {cmd}")
    p = subprocess.run(shlex.split(cmd), capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout)
        print(p.stderr)
        raise RuntimeError(f"Command failed: {cmd}")


def is_valid_npy(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        _ = np.load(path, mmap_mode="r")
        return True
    except Exception:
        return False


def parse_sample(sample_name: str) -> tuple[float, int]:
    m = re.match(r"E(\d+)_r(\d+)$", sample_name)
    if not m:
        raise ValueError(f"Invalid sample name: {sample_name}")
    energy = float(int(m.group(1)))
    rep = int(m.group(2))
    return energy, rep


def main() -> None:
    ap = argparse.ArgumentParser(description="Complete missing low/high outputs and meta.json in MC runs")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    cmd_template = str(cfg["simulator_command_template"])
    events_low = int(cfg.get("events_low", 2000))
    events_high = int(cfg.get("events_high", 100000))

    sample_dirs = sorted([p for p in args.root.iterdir() if p.is_dir()])

    fixed = 0
    for sdir in sample_dirs:
        low = sdir / "low" / "dose.npy"
        high = sdir / "high" / "dose.npy"
        meta = sdir / "meta.json"

        need_low = not is_valid_npy(low)
        need_high = not is_valid_npy(high)

        if not need_low and not need_high and meta.exists():
            continue

        energy, rep = parse_sample(sdir.name)

        low_seed = stable_seed(sdir.name, "low")
        high_seed = stable_seed(sdir.name, "high")

        if need_low:
            cmd = cmd_template.format(
                energy_mev=energy,
                events=events_low,
                output_dir=sdir / "low",
                seed=low_seed,
            )
            run(cmd)

        if need_high:
            cmd = cmd_template.format(
                energy_mev=energy,
                events=events_high,
                output_dir=sdir / "high",
                seed=high_seed,
            )
            run(cmd)

        payload = {
            "energy_mev": energy,
            "replica": rep,
            "events_low": events_low,
            "events_high": events_high,
            "seed_low": low_seed,
            "seed_high": high_seed,
            "backfilled": True,
        }
        with open(meta, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        fixed += 1

    print(f"Backfill complete. Updated samples: {fixed}")


if __name__ == "__main__":
    main()
