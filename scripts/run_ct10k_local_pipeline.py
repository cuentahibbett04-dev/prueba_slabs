#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shlex
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class CaseEntry:
    group: str
    case_id: str
    ct_mhd: Path

    @property
    def tag(self) -> str:
        return f"{self.group}_{self.case_id}"


@dataclass(frozen=True)
class GeomSpec:
    idx: int
    x_mm: float
    y_mm: float
    angle_deg: float


@dataclass(frozen=True)
class TaskSpec:
    case: CaseEntry
    geom: GeomSpec


REQUIRED_FILES = ("dose.npy", "spr.npy")


def _run(cmd: list[str], cwd: Path) -> None:
    pretty = " ".join(shlex.quote(x) for x in cmd)
    print(f"[CMD] {pretty}")
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) % 2_147_483_647) + 1


def _load_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"Empty manifest: {path}")
    return rows


def _choose_cases(rows: list[dict[str, str]], max_each: int) -> list[CaseEntry]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for r in rows:
        grouped.setdefault(r["group"], []).append(r)

    chosen: list[CaseEntry] = []
    for group in sorted(grouped.keys()):
        for r in grouped[group][:max_each]:
            chosen.append(
                CaseEntry(
                    group=r["group"],
                    case_id=r["case_id"],
                    ct_mhd=Path(r["ct_mhd"]),
                )
            )
    if not chosen:
        raise RuntimeError("No cases selected from manifest")
    return chosen


def _build_geometry(grid_x: int, grid_y: int, angles: list[float], x_range_mm: tuple[float, float], y_range_mm: tuple[float, float]) -> list[GeomSpec]:
    if grid_x < 2 or grid_y < 2:
        raise ValueError("grid_x and grid_y must be >= 2")
    if not angles:
        raise ValueError("angles list is empty")

    x0, x1 = x_range_mm
    y0, y1 = y_range_mm

    geoms: list[GeomSpec] = []
    idx = 0
    for ia, angle in enumerate(angles):
        for iy in range(grid_y):
            for ix in range(grid_x):
                x_mm = x0 + (x1 - x0) * (ix / float(grid_x - 1))
                y_mm = y0 + (y1 - y0) * (iy / float(grid_y - 1))
                geoms.append(GeomSpec(idx=idx, x_mm=float(x_mm), y_mm=float(y_mm), angle_deg=float(angle)))
                idx += 1
    return geoms


def _is_complete(sample_dir: Path) -> bool:
    needed = [
        sample_dir / "low" / "dose.npy",
        sample_dir / "low" / "spr.npy",
        sample_dir / "high" / "dose.npy",
        sample_dir / "high" / "spr.npy",
        sample_dir / "meta.json",
        sample_dir / "variant.json",
    ]
    return all(p.exists() for p in needed)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _simulate_one(
    task: TaskSpec,
    repo_root: Path,
    out_pairs_root: Path,
    py_cmd_prefix: list[str],
    particle: str,
    energy_mev: float,
    events_low: int,
    events_high: int,
    source_z_cm: float,
    force: bool,
) -> str:
    case = task.case
    geom = task.geom
    sample_name = f"{case.tag}_g{geom.idx:04d}"
    sample_dir = out_pairs_root / sample_name
    low_dir = sample_dir / "low"
    high_dir = sample_dir / "high"

    if not force and _is_complete(sample_dir):
        return f"SKIP {sample_name}"

    low_dir.mkdir(parents=True, exist_ok=True)
    high_dir.mkdir(parents=True, exist_ok=True)

    seed_high = _stable_seed(case.tag, f"geom{geom.idx}", "high")
    seed_low = _stable_seed(case.tag, f"geom{geom.idx}", "low")

    gate_script = repo_root / "scripts" / "gate_voxelized_ct_beam.py"
    post_script = repo_root / "scripts" / "postprocess_gate_output.py"

    high_sim_cmd = py_cmd_prefix + [
        str(gate_script),
        "--ct-mhd",
        str(case.ct_mhd),
        "--output-dir",
        str(high_dir),
        "--particle",
        particle,
        "--energy-mev",
        str(energy_mev),
        "--n-events",
        str(events_high),
        "--seed",
        str(seed_high),
        "--source-z-cm",
        str(source_z_cm),
        "--source-x-mm",
        str(geom.x_mm),
        "--source-y-mm",
        str(geom.y_mm),
        "--incidence-angle-deg",
        str(geom.angle_deg),
    ]
    _run(high_sim_cmd, cwd=repo_root)

    high_post_cmd = py_cmd_prefix + [
        str(post_script),
        "--dose-mhd",
        str(high_dir / "dose_voxelized_ct_edep.mhd"),
        "--ct-mhd",
        str(case.ct_mhd),
        "--out-dir",
        str(high_dir),
    ]
    _run(high_post_cmd, cwd=repo_root)

    low_sim_cmd = py_cmd_prefix + [
        str(gate_script),
        "--ct-mhd",
        str(case.ct_mhd),
        "--output-dir",
        str(low_dir),
        "--particle",
        particle,
        "--energy-mev",
        str(energy_mev),
        "--n-events",
        str(events_low),
        "--seed",
        str(seed_low),
        "--source-z-cm",
        str(source_z_cm),
        "--source-x-mm",
        str(geom.x_mm),
        "--source-y-mm",
        str(geom.y_mm),
        "--incidence-angle-deg",
        str(geom.angle_deg),
    ]
    _run(low_sim_cmd, cwd=repo_root)

    low_post_cmd = py_cmd_prefix + [
        str(post_script),
        "--dose-mhd",
        str(low_dir / "dose_voxelized_ct_edep.mhd"),
        "--ct-mhd",
        str(case.ct_mhd),
        "--out-dir",
        str(low_dir),
    ]
    _run(low_post_cmd, cwd=repo_root)

    meta = {
        "group": case.group,
        "case_id": case.case_id,
        "energy_mev": float(energy_mev),
        "events_low": int(events_low),
        "events_high": int(events_high),
        "particle": particle,
        "seed_low": int(seed_low),
        "seed_high": int(seed_high),
    }
    variant = {
        "source_x_mm": float(geom.x_mm),
        "source_y_mm": float(geom.y_mm),
        "incidence_angle_deg": float(geom.angle_deg),
        "source_z_cm": float(source_z_cm),
        "geometry_idx": int(geom.idx),
    }
    _write_json(sample_dir / "meta.json", meta)
    _write_json(sample_dir / "variant.json", variant)

    return f"DONE {sample_name}"


def _iter_tasks(cases: Iterable[CaseEntry], geoms: Iterable[GeomSpec]) -> Iterable[TaskSpec]:
    for c in cases:
        for g in geoms:
            yield TaskSpec(case=c, geom=g)


def _count_dirs(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(1 for p in root.iterdir() if p.is_dir())


def main() -> None:
    ap = argparse.ArgumentParser(description="Local pipeline: generate 10k CT low/high pairs with geometry variation")
    ap.add_argument("--manifest", type=Path, default=Path("mc/ct_for_gate/manifest.csv"))
    ap.add_argument("--out-root", type=Path, default=Path("ct10k_local"))
    ap.add_argument("--max-cases-each", type=int, default=4, help="Cases per group (lung, colorectal)")
    ap.add_argument("--grid-x", type=int, default=25)
    ap.add_argument("--grid-y", type=int, default=25)
    ap.add_argument("--angles-deg", type=str, default="-7,7", help='Comma-separated list, e.g. "-7,7"')
    ap.add_argument("--x-range-mm", type=str, default="-60,60", help='Min,max in mm')
    ap.add_argument("--y-range-mm", type=str, default="-60,60", help='Min,max in mm')
    ap.add_argument("--source-z-cm", type=float, default=-30.0)
    ap.add_argument("--particle", type=str, default="gamma", choices=["gamma", "proton", "e-"])
    ap.add_argument("--energy-mev", type=float, default=6.0)
    ap.add_argument("--events-low", type=int, default=2000)
    ap.add_argument("--events-high", type=int, default=100000)
    ap.add_argument("--python-cmd", type=str, default="python3", help='Python command with OpenGATE, e.g. "python3" or "conda run -n gate python"')
    ap.add_argument("--max-workers", type=int, default=4)
    ap.add_argument("--force", action="store_true", help="Recompute samples even if already complete")
    ap.add_argument("--build-dataset", action="store_true")
    ap.add_argument("--dataset-out", type=Path, default=Path("data_ct10k_geom"))
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--holdout-val-by-angle", action="store_true")
    ap.add_argument("--val-angle-ratio", type=float, default=0.2)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    manifest_path = (repo_root / args.manifest).resolve() if not args.manifest.is_absolute() else args.manifest
    out_root = (repo_root / args.out_root).resolve() if not args.out_root.is_absolute() else args.out_root
    out_pairs_root = out_root / "pairs"
    out_pairs_root.mkdir(parents=True, exist_ok=True)

    angles = [float(x.strip()) for x in args.angles_deg.split(",") if x.strip()]
    xvals = [float(x.strip()) for x in args.x_range_mm.split(",") if x.strip()]
    yvals = [float(y.strip()) for y in args.y_range_mm.split(",") if y.strip()]
    if len(xvals) != 2 or len(yvals) != 2:
        raise ValueError("x-range-mm and y-range-mm must be 'min,max'")

    rows = _load_manifest(manifest_path)
    cases = _choose_cases(rows, max_each=int(args.max_cases_each))
    geoms = _build_geometry(
        grid_x=int(args.grid_x),
        grid_y=int(args.grid_y),
        angles=angles,
        x_range_mm=(xvals[0], xvals[1]),
        y_range_mm=(yvals[0], yvals[1]),
    )

    total_tasks = len(cases) * len(geoms)
    print(f"Selected cases: {len(cases)}")
    print(f"Geometries per case: {len(geoms)}")
    print(f"Total tasks: {total_tasks}")
    print(f"Output pairs root: {out_pairs_root}")

    py_cmd_prefix = shlex.split(args.python_cmd)
    if not py_cmd_prefix:
        raise ValueError("Invalid --python-cmd")

    failures = 0
    done = 0
    skipped = 0

    with ThreadPoolExecutor(max_workers=int(args.max_workers)) as ex:
        futures = [
            ex.submit(
                _simulate_one,
                t,
                repo_root,
                out_pairs_root,
                py_cmd_prefix,
                args.particle,
                args.energy_mev,
                args.events_low,
                args.events_high,
                args.source_z_cm,
                args.force,
            )
            for t in _iter_tasks(cases, geoms)
        ]

        for fut in as_completed(futures):
            try:
                msg = fut.result()
                print(msg)
                if msg.startswith("SKIP"):
                    skipped += 1
                else:
                    done += 1
            except Exception as exc:  # pylint: disable=broad-except
                failures += 1
                print(f"ERROR: {exc}")

    print("Generation summary")
    print(f"Done: {done}")
    print(f"Skipped: {skipped}")
    print(f"Failures: {failures}")
    print(f"Folders present in pairs: {_count_dirs(out_pairs_root)}")

    if failures > 0:
        raise RuntimeError(f"Generation finished with failures={failures}")

    if args.build_dataset:
        dataset_out = (repo_root / args.dataset_out).resolve() if not args.dataset_out.is_absolute() else args.dataset_out
        build_cmd = py_cmd_prefix + [
            str(repo_root / "scripts" / "build_dataset_from_mc.py"),
            "--mc-root",
            str(out_pairs_root),
            "--out-root",
            str(dataset_out),
            "--train-ratio",
            str(args.train_ratio),
            "--val-ratio",
            str(args.val_ratio),
            "--seed",
            str(args.seed),
            "--rescale-low-by-history-ratio",
            "--skip-invalid",
        ]
        if args.holdout_val_by_angle:
            build_cmd += [
                "--holdout-val-by-angle",
                "--val-angle-ratio",
                str(args.val_angle_ratio),
            ]

        env_cmd = ["env", f"PYTHONPATH={repo_root / 'src'}"] + build_cmd
        _run(env_cmd, cwd=repo_root)
        print(f"Dataset built at: {dataset_out}")


if __name__ == "__main__":
    main()
