#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path

import numpy as np


def run_cmd(cmd: str) -> None:
    proc = subprocess.run(shlex.split(cmd), capture_output=True, text=True)
    if proc.returncode != 0:
        if proc.stdout:
            print(proc.stdout)
        if proc.stderr:
            print(proc.stderr)
        raise RuntimeError(f"Command failed ({proc.returncode}): {cmd}")


def infer_energy_mev(sample_dir: Path) -> float:
    meta_path = sample_dir / "meta.json"
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if "energy_mev" in meta:
                return float(meta["energy_mev"])
        except Exception:  # pylint: disable=broad-except
            pass

    name = sample_dir.name
    if name.startswith("E") and "_" in name:
        try:
            return float(int(name[1:].split("_", 1)[0]))
        except Exception as exc:  # pylint: disable=broad-except
            raise ValueError(f"Cannot parse energy from sample name {name}") from exc
    raise ValueError(f"Cannot infer energy for {sample_dir}")


def make_task(sample_dir: Path, low_events: int, template: str) -> tuple[str, Path, str]:
    if low_events == 2000:
        low_dir = sample_dir / "low"
    else:
        low_dir = sample_dir / f"low_e{low_events:05d}"

    dose_file = low_dir / "dose.npy"
    if dose_file.exists():
        try:
            _ = np.load(dose_file)
            return sample_dir.name, low_dir, ""
        except Exception:  # pylint: disable=broad-except
            # Corrupt file: rerun this simulation.
            pass

    energy = infer_energy_mev(sample_dir)
    h = hashlib.sha256(f"{sample_dir.name}|{low_events}".encode("utf-8")).hexdigest()
    seed = int(h[:12], 16) % 2_147_483_647
    cmd = template.format(
        energy_mev=energy,
        events=low_events,
        output_dir=str(low_dir),
        seed=seed,
    )
    return sample_dir.name, low_dir, cmd


def run_task(task: tuple[str, Path, str]) -> tuple[str, bool, str]:
    sample_name, low_dir, cmd = task
    try:
        low_dir.mkdir(parents=True, exist_ok=True)
        if cmd:
            print(f"[RUN] {sample_name} -> {low_dir.name}")
            run_cmd(cmd)
        return sample_name, True, ""
    except Exception as exc:  # pylint: disable=broad-except
        return sample_name, False, str(exc)


def symlink_file(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())


def build_pairs(
    sample_dirs: list[Path],
    pairs_root: Path,
    low_levels: list[int],
    events_high: int,
    clean_pairs: bool,
) -> None:
    if clean_pairs and pairs_root.exists():
        shutil.rmtree(pairs_root)
    pairs_root.mkdir(parents=True, exist_ok=True)

    for sample_dir in sample_dirs:
        energy = infer_energy_mev(sample_dir)
        high_dir = sample_dir / "high"
        if not (high_dir / "dose.npy").exists():
            raise FileNotFoundError(f"Missing high dose for {sample_dir}")

        for low_events in low_levels:
            low_dir = sample_dir / ("low" if low_events == 2000 else f"low_e{low_events:05d}")
            if not (low_dir / "dose.npy").exists():
                raise FileNotFoundError(f"Missing low dose ({low_events}) for {sample_dir}")

            pair_name = f"{sample_dir.name}_L{low_events:05d}"
            pair_dir = pairs_root / pair_name
            (pair_dir / "low").mkdir(parents=True, exist_ok=True)
            (pair_dir / "high").mkdir(parents=True, exist_ok=True)

            for name in ["dose.npy", "spr.npy"]:
                symlink_file(low_dir / name, pair_dir / "low" / name)
                symlink_file(high_dir / name, pair_dir / "high" / name)

            meta = {
                "energy_mev": energy,
                "events_low": int(low_events),
                "events_high": int(events_high),
                "source_sample": sample_dir.name,
            }
            with open(pair_dir / "meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)


def main(args: argparse.Namespace) -> None:
    mc_root = Path(args.mc_root)
    if not mc_root.exists():
        raise FileNotFoundError(mc_root)

    low_levels = [int(v) for v in args.low_levels]
    sample_dirs = sorted([p for p in mc_root.iterdir() if p.is_dir()])
    if args.max_samples is not None:
        sample_dirs = sample_dirs[: args.max_samples]

    tasks: list[tuple[str, Path, str]] = []
    for sample_dir in sample_dirs:
        for low_events in low_levels:
            tasks.append(make_task(sample_dir, low_events, args.simulator_command_template))

    missing_tasks = [t for t in tasks if t[2]]
    print(f"Samples selected: {len(sample_dirs)}")
    print(f"Requested low levels: {low_levels}")
    print(f"Missing low simulations to run: {len(missing_tasks)}")

    failures: list[tuple[str, str]] = []
    if missing_tasks:
        max_parallel = max(1, int(args.max_parallel))
        if max_parallel == 1:
            for t in missing_tasks:
                name, ok, err = run_task(t)
                if not ok:
                    failures.append((name, err))
                    if args.fail_fast:
                        break
        else:
            with cf.ThreadPoolExecutor(max_workers=max_parallel) as pool:
                futures = [pool.submit(run_task, t) for t in missing_tasks]
                for fut in cf.as_completed(futures):
                    name, ok, err = fut.result()
                    if not ok:
                        failures.append((name, err))
                        if args.fail_fast:
                            for p in futures:
                                p.cancel()
                            break

    if failures:
        print(f"Failures: {len(failures)}")
        for name, err in failures[:20]:
            print(f"  - {name}: {err}")
        raise RuntimeError("Extension failed for some samples")

    pairs_root = Path(args.pairs_root)
    build_pairs(
        sample_dirs=sample_dirs,
        pairs_root=pairs_root,
        low_levels=low_levels,
        events_high=args.events_high,
        clean_pairs=args.clean_pairs,
    )
    print(f"Pairs root ready: {pairs_root}")

    if args.dataset_out is not None:
        dataset_out = Path(args.dataset_out)
        cmd = (
            f"{shlex.quote(args.python_executable)} scripts/build_dataset_from_mc.py "
            f"--mc-root {shlex.quote(str(pairs_root))} "
            f"--out-root {shlex.quote(str(dataset_out))} "
            f"--train-ratio {args.train_ratio} --val-ratio {args.val_ratio} --seed {args.seed} "
            f"--rescale-low-by-history-ratio "
            f"--default-events-low 2000 --default-events-high {args.events_high}"
        )
        print(f"[CMD] {cmd}")
        run_cmd(cmd)
        print(f"Dataset ready: {dataset_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extend existing 2k/100k MC root with 5k/10k/20k lows and build multinoise pairs")
    parser.add_argument("--mc-root", type=str, default="mc_runs_opengate_photon_1000")
    parser.add_argument("--pairs-root", type=str, default="mc_runs_opengate_photon_1000_multinoise_pairs")
    parser.add_argument("--dataset-out", type=str, default="data_real_photon_1000_multinoise")
    parser.add_argument("--events-high", type=int, default=100000)
    parser.add_argument("--low-levels", type=int, nargs="+", default=[2000, 5000, 10000, 20000])
    parser.add_argument("--max-parallel", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.75)
    parser.add_argument("--val-ratio", type=float, default=0.125)
    parser.add_argument("--clean-pairs", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--simulator-command-template",
        type=str,
        default="./mc/run_photon_sim.sh --energy {energy_mev} --events {events} --out {output_dir} --seed {seed}",
    )
    parser.add_argument("--python-executable", type=str, default=os.environ.get("PYTHON", "python3"))
    main(parser.parse_args())
