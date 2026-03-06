#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import random
import shlex
import subprocess
from pathlib import Path


def _run(cmd: str, cwd: Path | None = None) -> None:
    print(f"[CMD] {cmd}")
    proc = subprocess.run(shlex.split(cmd), cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
        raise RuntimeError(f"Command failed with code {proc.returncode}: {cmd}")


def _run_one_sample(task: dict[str, object]) -> tuple[str, bool, str]:
    sample_name = str(task["sample_name"])
    sample_dir = Path(task["sample_dir"])
    low_dir = Path(task["low_dir"])
    high_dir = Path(task["high_dir"])

    low_cmd = str(task["low_cmd"])
    high_cmd = str(task["high_cmd"])

    meta = {
        "energy_mev": float(task["energy_mev"]),
        "replica": int(task["replica"]),
        "events_low": int(task["events_low"]),
        "events_high": int(task["events_high"]),
        "seed_low": int(task["seed_low"]),
        "seed_high": int(task["seed_high"]),
    }

    try:
        low_dir.mkdir(parents=True, exist_ok=True)
        high_dir.mkdir(parents=True, exist_ok=True)
        print(f"\\n=== Sample {sample_name} ===")
        _run(low_cmd)
        _run(high_cmd)

        with open(sample_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        return sample_name, True, ""
    except Exception as exc:  # pylint: disable=broad-except
        return sample_name, False, str(exc)


def main(args: argparse.Namespace) -> None:
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    seed = int(cfg.get("seed", 42))
    random.seed(seed)

    energies = [float(e) for e in cfg["energies_mev"]]
    replicas = int(cfg.get("replicas_per_energy", 1))
    events_low = int(cfg.get("events_low", 2000))
    events_high = int(cfg.get("events_high", 100000))
    cmd_template = str(cfg["simulator_command_template"])
    output_root = Path(cfg.get("output_root", "mc_runs"))
    max_parallel_cfg = int(cfg.get("max_parallel", 1))
    max_parallel = int(args.max_parallel) if args.max_parallel is not None else max_parallel_cfg
    if max_parallel < 1:
        raise ValueError("max_parallel must be >= 1")

    output_root.mkdir(parents=True, exist_ok=True)

    tasks: list[dict[str, object]] = []
    for energy in energies:
        for rep in range(replicas):
            sample_name = f"E{int(round(energy)):03d}_r{rep:03d}"
            sample_dir = output_root / sample_name
            low_dir = sample_dir / "low"
            high_dir = sample_dir / "high"
            low_dir.mkdir(parents=True, exist_ok=True)
            high_dir.mkdir(parents=True, exist_ok=True)

            low_seed = random.randint(1, 2_147_483_647)
            high_seed = random.randint(1, 2_147_483_647)

            low_cmd = cmd_template.format(
                energy_mev=energy,
                events=events_low,
                output_dir=low_dir,
                seed=low_seed,
            )
            high_cmd = cmd_template.format(
                energy_mev=energy,
                events=events_high,
                output_dir=high_dir,
                seed=high_seed,
            )

            tasks.append(
                {
                    "sample_name": sample_name,
                    "sample_dir": str(sample_dir),
                    "low_dir": str(low_dir),
                    "high_dir": str(high_dir),
                    "energy_mev": energy,
                    "replica": rep,
                    "events_low": events_low,
                    "events_high": events_high,
                    "seed_low": low_seed,
                    "seed_high": high_seed,
                    "low_cmd": low_cmd,
                    "high_cmd": high_cmd,
                }
            )

    if max_parallel == 1:
        print(f"Running sequential campaign for {len(tasks)} samples")
    else:
        print(
            f"Running parallel campaign for {len(tasks)} samples with "
            f"max_parallel={max_parallel} (cpu_count={os.cpu_count()})"
        )

    completed = 0
    failed: list[tuple[str, str]] = []

    if max_parallel == 1:
        for task in tasks:
            name, ok, err = _run_one_sample(task)
            if ok:
                completed += 1
            else:
                failed.append((name, err))
                if args.fail_fast:
                    break
    else:
        with cf.ThreadPoolExecutor(max_workers=max_parallel) as pool:
            futures = [pool.submit(_run_one_sample, task) for task in tasks]
            for fut in cf.as_completed(futures):
                name, ok, err = fut.result()
                if ok:
                    completed += 1
                else:
                    failed.append((name, err))
                    if args.fail_fast:
                        for pending in futures:
                            pending.cancel()
                        break

    print(f"\\nCompleted samples: {completed}")
    if failed:
        print(f"Failed samples: {len(failed)}")
        for name, err in failed[:10]:
            print(f"  - {name}: {err}")
    print(f"Output root: {output_root}")

    if failed:
        raise RuntimeError("Campaign finished with failures")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Monte Carlo campaign for low/high statistics proton doses")
    parser.add_argument("--config", type=Path, default=Path("mc/campaign.example.json"))
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=None,
        help="Maximum number of samples to run concurrently (overrides config max_parallel)",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop campaign as soon as one sample fails",
    )
    main(parser.parse_args())
