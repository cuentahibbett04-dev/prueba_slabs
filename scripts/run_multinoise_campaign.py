#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import shlex
import shutil
import subprocess
from pathlib import Path


def run_cmd(cmd: str) -> None:
    print(f"[CMD] {cmd}")
    proc = subprocess.run(shlex.split(cmd), capture_output=True, text=True)
    if proc.returncode != 0:
        if proc.stdout:
            print(proc.stdout)
        if proc.stderr:
            print(proc.stderr)
        raise RuntimeError(f"Command failed ({proc.returncode}): {cmd}")


def ensure(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def copy_pair(low_dir: Path, high_dir: Path, out_pair: Path, meta: dict) -> None:
    (out_pair / "low").mkdir(parents=True, exist_ok=True)
    (out_pair / "high").mkdir(parents=True, exist_ok=True)

    for name in ["dose.npy", "spr.npy"]:
        low_src = low_dir / name
        high_src = high_dir / name
        ensure(low_src)
        ensure(high_src)
        shutil.copy2(low_src, out_pair / "low" / name)
        shutil.copy2(high_src, out_pair / "high" / name)

    variant_src = low_dir.parent / "variant.json"
    if variant_src.exists():
        shutil.copy2(variant_src, out_pair / "variant.json")

    with open(out_pair / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def main(args: argparse.Namespace) -> None:
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    seed = int(cfg.get("seed", 42))
    rng = random.Random(seed)

    energies = [float(v) for v in cfg.get("energies_mev", [1.0, 2.0, 4.0, 6.0])]
    low_levels = [int(v) for v in cfg.get("low_events_levels", [2000, 5000, 10000, 20000])]
    events_high = int(cfg.get("events_high", 100000))
    n_geometries = int(cfg.get("n_geometries", 8))
    repeats = int(cfg.get("noisy_repeats_per_level", 1))

    sim_tmpl = str(
        cfg.get(
            "simulator_command_template",
            "./mc/run_photon_sim.sh --energy {energy_mev} --events {events} --out {output_dir} --seed {seed}",
        )
    )
    output_root = Path(cfg.get("output_root", "multinoise_deepmc"))
    raw_root = output_root / "raw"
    pairs_root = output_root / "pairs"

    if args.clean and output_root.exists():
        shutil.rmtree(output_root)
    raw_root.mkdir(parents=True, exist_ok=True)
    pairs_root.mkdir(parents=True, exist_ok=True)

    n_pairs = 0
    for gi in range(n_geometries):
        geom_id = f"G{gi:03d}"
        energy = float(rng.choice(energies))
        geom_root = raw_root / geom_id

        high_dir = geom_root / "high_ref"
        high_dir.mkdir(parents=True, exist_ok=True)
        high_seed = rng.randint(1, 2_147_483_647)
        run_cmd(
            sim_tmpl.format(
                energy_mev=energy,
                events=events_high,
                output_dir=high_dir,
                seed=high_seed,
            )
        )

        for low_events in low_levels:
            for rep in range(repeats):
                low_dir = geom_root / f"low_e{low_events:05d}_r{rep:03d}"
                low_dir.mkdir(parents=True, exist_ok=True)
                low_seed = rng.randint(1, 2_147_483_647)
                run_cmd(
                    sim_tmpl.format(
                        energy_mev=energy,
                        events=low_events,
                        output_dir=low_dir,
                        seed=low_seed,
                    )
                )

                pair_name = f"{geom_id}_e{low_events:05d}_r{rep:03d}"
                pair_dir = pairs_root / pair_name
                copy_pair(
                    low_dir=low_dir,
                    high_dir=high_dir,
                    out_pair=pair_dir,
                    meta={
                        "energy_mev": energy,
                        "events_low": low_events,
                        "events_high": events_high,
                        "geometry_id": geom_id,
                        "replica": rep,
                    },
                )
                n_pairs += 1

    print(f"Generated pair folders: {n_pairs}")
    print(f"Pairs root: {pairs_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate multi-noise MC pairs: [2k,5k,10k,20k] -> 100k")
    parser.add_argument("--config", type=Path, default=Path("mc/campaign.opengate.photon.multinoise.small.json"))
    parser.add_argument("--clean", action="store_true")
    main(parser.parse_args())
