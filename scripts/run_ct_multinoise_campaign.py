#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shlex
import subprocess
from pathlib import Path


def run_cmd(cmd: str, cwd: Path) -> None:
    print(f"[CMD] {cmd}")
    proc = subprocess.run(shlex.split(cmd), cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        if proc.stdout:
            print(proc.stdout)
        if proc.stderr:
            print(proc.stderr)
        raise RuntimeError(f"Command failed ({proc.returncode}): {cmd}")


def stable_seed(*parts: str) -> int:
    h = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return (int(h[:8], 16) % 2_147_483_647) + 1


def symlink_or_copy(src: Path, dst: Path, copy_files: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy_files:
        import shutil

        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def select_cases(rows: list[dict[str, str]], max_each: int) -> list[dict[str, str]]:
    by_group: dict[str, list[dict[str, str]]] = {}
    for r in rows:
        by_group.setdefault(r["group"], []).append(r)
    out: list[dict[str, str]] = []
    for g in sorted(by_group.keys()):
        out.extend(by_group[g][:max_each])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate CT multinoise low/high pair folders from CT manifest")
    ap.add_argument("--manifest", type=Path, default=Path("mc/ct_for_gate/manifest.csv"))
    ap.add_argument("--out-root", type=Path, default=Path("mc_runs_ct_multinoise"))
    ap.add_argument("--energy-mev", type=float, default=6.0)
    ap.add_argument("--events-low", type=int, default=2000)
    ap.add_argument("--events-high", type=int, default=100000)
    ap.add_argument("--low-repeats", type=int, default=5)
    ap.add_argument("--max-cases-each", type=int, default=4)
    ap.add_argument("--particle", type=str, default="gamma", choices=["gamma", "proton", "e-"])
    ap.add_argument("--copy-files", action="store_true", help="Copy high files into pairs instead of symlinks")
    ap.add_argument("--clean", action="store_true")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    with open(args.manifest, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No cases in manifest: {args.manifest}")

    chosen = select_cases(rows, max_each=int(args.max_cases_each))
    if not chosen:
        raise RuntimeError("No selected cases after filtering")

    out_root = args.out_root
    raw_root = out_root / "raw"
    pairs_root = out_root / "pairs"

    if args.clean and out_root.exists():
        import shutil

        shutil.rmtree(out_root)

    raw_root.mkdir(parents=True, exist_ok=True)
    pairs_root.mkdir(parents=True, exist_ok=True)

    total_pairs = 0
    for r in chosen:
        group = r["group"]
        case_id = r["case_id"]
        ct_mhd = Path(r["ct_mhd"])
        case_tag = f"{group}_{case_id}"

        case_raw = raw_root / case_tag
        high_ref = case_raw / "high_ref"
        high_ref.mkdir(parents=True, exist_ok=True)

        high_seed = stable_seed(case_tag, "high")
        run_cmd(
            (
                f"python3 {repo / 'scripts/gate_voxelized_ct_beam.py'} "
                f"--ct-mhd {shlex.quote(str(ct_mhd))} "
                f"--output-dir {shlex.quote(str(high_ref))} "
                f"--particle {args.particle} --energy-mev {args.energy_mev} "
                f"--n-events {args.events_high} --seed {high_seed} "
                f"--source-z-cm -30 --source-x-mm 0 --source-y-mm 0 --incidence-angle-deg 0"
            ),
            cwd=repo,
        )
        run_cmd(
            (
                f"python3 {repo / 'scripts/postprocess_gate_output.py'} "
                f"--dose-mhd {shlex.quote(str(high_ref / 'dose_voxelized_ct_edep.mhd'))} "
                f"--ct-mhd {shlex.quote(str(ct_mhd))} --out-dir {shlex.quote(str(high_ref))}"
            ),
            cwd=repo,
        )

        for rep in range(int(args.low_repeats)):
            pair_name = f"{case_tag}_r{rep:03d}"
            pair_dir = pairs_root / pair_name
            low_dir = pair_dir / "low"
            high_dir = pair_dir / "high"
            low_dir.mkdir(parents=True, exist_ok=True)
            high_dir.mkdir(parents=True, exist_ok=True)

            low_seed = stable_seed(case_tag, f"low{rep}")
            run_cmd(
                (
                    f"python3 {repo / 'scripts/gate_voxelized_ct_beam.py'} "
                    f"--ct-mhd {shlex.quote(str(ct_mhd))} "
                    f"--output-dir {shlex.quote(str(low_dir))} "
                    f"--particle {args.particle} --energy-mev {args.energy_mev} "
                    f"--n-events {args.events_low} --seed {low_seed} "
                    f"--source-z-cm -30 --source-x-mm 0 --source-y-mm 0 --incidence-angle-deg 0"
                ),
                cwd=repo,
            )
            run_cmd(
                (
                    f"python3 {repo / 'scripts/postprocess_gate_output.py'} "
                    f"--dose-mhd {shlex.quote(str(low_dir / 'dose_voxelized_ct_edep.mhd'))} "
                    f"--ct-mhd {shlex.quote(str(ct_mhd))} --out-dir {shlex.quote(str(low_dir))}"
                ),
                cwd=repo,
            )

            for name in ["dose.npy", "spr.npy"]:
                symlink_or_copy(high_ref / name, high_dir / name, copy_files=bool(args.copy_files))

            meta = {
                "group": group,
                "case_id": case_id,
                "energy_mev": float(args.energy_mev),
                "events_low": int(args.events_low),
                "events_high": int(args.events_high),
                "particle": args.particle,
                "seed_low": int(low_seed),
                "seed_high": int(high_seed),
                "source_series_dir": r.get("source_series_dir", ""),
                "ct_mhd": str(ct_mhd),
            }
            with open(pair_dir / "meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            total_pairs += 1

    print(f"Selected CT cases: {len(chosen)}")
    print(f"Generated pair folders: {total_pairs}")
    print(f"Pairs root: {pairs_root}")


if __name__ == "__main__":
    main()
