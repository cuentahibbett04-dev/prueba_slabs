#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import opengate as gate
import SimpleITK as sitk


def add_point_source(
    sim: gate.Simulation,
    particle: str,
    energy_mev: float,
    n_events: int,
    x_mm: float,
    y_mm: float,
    z_cm: float,
    incidence_angle_deg: float,
) -> None:
    mm = gate.g4_units.mm
    cm = gate.g4_units.cm
    MeV = gate.g4_units.MeV

    source = sim.add_source("GenericSource", "beam_source")
    source.particle = particle
    source.energy.mono = float(energy_mev) * MeV
    source.position.type = "point"
    source.position.translation = [float(x_mm) * mm, float(y_mm) * mm, float(z_cm) * cm]
    source.direction.type = "momentum"
    theta = math.radians(float(incidence_angle_deg))
    source.direction.momentum = [math.sin(theta), 0.0, math.cos(theta)]
    source.n = int(n_events)


def build_sim(
    ct_mhd: Path,
    output_dir: Path,
    particle: str,
    energy_mev: float,
    n_events: int,
    seed: int,
    source_z_cm: float,
    source_x_mm: float,
    source_y_mm: float,
    incidence_angle_deg: float,
) -> gate.Simulation:
    sim = gate.Simulation()

    mm = gate.g4_units.mm
    m = gate.g4_units.m

    sim.output_dir = str(output_dir)
    sim.visu = False
    sim.random_seed = int(seed)
    sim.g4_commands_after_init.append("/run/verbose 0")
    sim.g4_commands_after_init.append("/event/verbose 0")

    world = sim.world
    world.size = [2.0 * m, 2.0 * m, 2.0 * m]
    world.material = "G4_AIR"

    patient = sim.add_volume("Image", "patient")
    patient.image = str(ct_mhd)
    patient.material = "G4_AIR"
    patient.voxel_materials = [
        (-2000, -950, "G4_AIR"),
        (-949, -500, "G4_LUNG_ICRP"),
        (-499, -50, "G4_ADIPOSE_TISSUE_ICRP"),
        (-49, 20, "G4_WATER"),
        (21, 100, "G4_MUSCLE_SKELETAL_ICRP"),
        # G4_CARTILAGE_ICRP is not available in some Geant4 builds.
        # Keep the HU bin but map it to a broadly available soft-tissue material.
        (101, 300, "G4_MUSCLE_SKELETAL_ICRP"),
        (301, 2000, "G4_BONE_COMPACT_ICRU"),
    ]

    add_point_source(
        sim=sim,
        particle=particle,
        energy_mev=energy_mev,
        n_events=n_events,
        x_mm=source_x_mm,
        y_mm=source_y_mm,
        z_cm=source_z_cm,
        incidence_angle_deg=incidence_angle_deg,
    )

    ct_img = sitk.ReadImage(str(ct_mhd))
    ct_size = ct_img.GetSize()
    ct_spacing = ct_img.GetSpacing()

    dose = sim.add_actor("DoseActor", "dose")
    dose.attached_to = "patient"
    # Match dose grid to voxelized CT geometry.
    dose.size = [int(ct_size[0]), int(ct_size[1]), int(ct_size[2])]
    dose.spacing = [float(ct_spacing[0]) * mm, float(ct_spacing[1]) * mm, float(ct_spacing[2]) * mm]
    dose.output_filename = "dose_voxelized_ct.mhd"
    dose.write_to_disk = True

    return sim


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenGATE voxelized slab simulation for proton/photon beams")
    parser.add_argument("--ct-mhd", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--particle", type=str, default="proton", choices=["proton", "gamma", "e-"])
    parser.add_argument("--energy-mev", type=float, required=True)
    parser.add_argument("--n-events", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source-z-cm", type=float, default=-30.0)
    parser.add_argument("--source-x-mm", type=float, default=0.0)
    parser.add_argument("--source-y-mm", type=float, default=0.0)
    parser.add_argument("--incidence-angle-deg", type=float, default=0.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sim = build_sim(
        ct_mhd=args.ct_mhd,
        output_dir=args.output_dir,
        particle=args.particle,
        energy_mev=args.energy_mev,
        n_events=args.n_events,
        seed=args.seed,
        source_z_cm=args.source_z_cm,
        source_x_mm=args.source_x_mm,
        source_y_mm=args.source_y_mm,
        incidence_angle_deg=args.incidence_angle_deg,
    )
    sim.run()


if __name__ == "__main__":
    main()
