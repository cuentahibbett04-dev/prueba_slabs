#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import shlex
import shutil
import subprocess
import sys
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


def ensure_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def copy_pair_files(low_dir: Path, high_ref_dir: Path, pair_dir: Path, energy_mev: float, geom_id: str, rep: int) -> None:
    pair_low = pair_dir / "low"
    pair_high = pair_dir / "high"
    pair_low.mkdir(parents=True, exist_ok=True)
    pair_high.mkdir(parents=True, exist_ok=True)

    for name in ["dose.npy", "spr.npy"]:
        src_low = low_dir / name
        src_high = high_ref_dir / name
        ensure_file(src_low)
        ensure_file(src_high)
        shutil.copy2(src_low, pair_low / name)
        shutil.copy2(src_high, pair_high / name)

    variant_src = low_dir.parent / "variant.json"
    if variant_src.exists():
        shutil.copy2(variant_src, pair_dir / "variant.json")

    meta = {
        "energy_mev": float(energy_mev),
        "geometry_id": geom_id,
        "noisy_rep": int(rep),
    }
    with open(pair_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def build_dataset(mc_pairs_root: Path, out_root: Path, train_ratio: float, val_ratio: float, seed: int) -> None:
    cmd = (
        f"{shlex.quote(sys.executable)} scripts/build_dataset_from_mc.py "
        f"--mc-root {shlex.quote(str(mc_pairs_root))} "
        f"--out-root {shlex.quote(str(out_root))} "
        f"--train-ratio {train_ratio} "
        f"--val-ratio {val_ratio} "
        f"--seed {seed}"
    )
    run_cmd(cmd)


def main(args: argparse.Namespace) -> None:
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    seed = int(cfg.get("seed", 42))
    rng = random.Random(seed)

    energies = [float(e) for e in cfg.get("energies_mev", [1.0, 2.0, 4.0, 6.0])]
    n_geometries = int(cfg.get("n_geometries", 8))
    noisy_repeats = int(cfg.get("noisy_repeats_per_geometry", 3))
    events_low = int(cfg.get("events_low", 2000))
    events_high = int(cfg.get("events_high", 100000))

    sim_tmpl = str(
        cfg.get(
            "simulator_command_template",
            "./mc/run_photon_sim.sh --energy {energy_mev} --events {events} --out {output_dir} --seed {seed}",
        )
    )

    output_root = Path(cfg.get("output_root", "mini_deepmc"))
    raw_root = output_root / "raw"
    pairs_root = output_root / "pairs"
    dataset_root = output_root / "dataset"

    raw_root.mkdir(parents=True, exist_ok=True)
    pairs_root.mkdir(parents=True, exist_ok=True)

    if args.clean:
        if pairs_root.exists():
            shutil.rmtree(pairs_root)
        if raw_root.exists():
            shutil.rmtree(raw_root)
        if dataset_root.exists():
            shutil.rmtree(dataset_root)
        raw_root.mkdir(parents=True, exist_ok=True)
        pairs_root.mkdir(parents=True, exist_ok=True)

    total_pairs = 0
    for gi in range(n_geometries):
        geom_id = f"G{gi:03d}"
        energy = float(rng.choice(energies))

        geom_root = raw_root / geom_id
        high_ref_dir = geom_root / "high_ref"
        high_ref_dir.mkdir(parents=True, exist_ok=True)

        high_seed = rng.randint(1, 2_147_483_647)
        high_cmd = sim_tmpl.format(
            energy_mev=energy,
            events=events_high,
            output_dir=high_ref_dir,
            seed=high_seed,
        )
        run_cmd(high_cmd)

        for rep in range(noisy_repeats):
            low_dir = geom_root / f"low_r{rep:03d}"
            low_dir.mkdir(parents=True, exist_ok=True)

            low_seed = rng.randint(1, 2_147_483_647)
            low_cmd = sim_tmpl.format(
                energy_mev=energy,
                events=events_low,
                output_dir=low_dir,
                seed=low_seed,
            )
            run_cmd(low_cmd)

            pair_dir = pairs_root / f"{geom_id}_n{rep:03d}"
            copy_pair_files(low_dir=low_dir, high_ref_dir=high_ref_dir, pair_dir=pair_dir, energy_mev=energy, geom_id=geom_id, rep=rep)
            total_pairs += 1

    print(f"Generated pair folders: {total_pairs}")

    if bool(cfg.get("build_dataset", True)):
        train_ratio = float(cfg.get("train_ratio", 0.75))
        val_ratio = float(cfg.get("val_ratio", 0.125))
        build_dataset(
            mc_pairs_root=pairs_root,
            out_root=dataset_root,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
        )
        print(f"Dataset ready: {dataset_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mini DeepMC-style campaign: one high-dose target + repeated low-dose noise")
    parser.add_argument("--config", type=Path, default=Path("mc/campaign.opengate.photon.mini_deepmc.json"))
    parser.add_argument("--clean", action="store_true", help="Remove previous output_root before running")
    main(parser.parse_args())
