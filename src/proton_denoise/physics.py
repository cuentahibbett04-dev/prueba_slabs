from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import GeometryConfig, MaterialSPR


@dataclass(frozen=True)
class Phantom:
    spr_map: np.ndarray  # [D, H, W]
    voxel_mm: tuple[float, float, float]


def build_multilayer_phantom(
    geometry: GeometryConfig = GeometryConfig(),
    materials: MaterialSPR = MaterialSPR(),
) -> Phantom:
    """Create a slab phantom with water-bone-lung-water layers."""
    sx_mm, sy_mm, sz_mm = geometry.size_mm
    vx, vy, vz = geometry.voxel_mm

    nx = int(round(sx_mm / vx))
    ny = int(round(sy_mm / vy))
    nz = int(round(sz_mm / vz))

    z_water_1 = int(round(geometry.water_1_mm / vz))
    z_bone = int(round(geometry.bone_mm / vz))
    z_lung = int(round(geometry.lung_mm / vz))

    spr = np.full((nz, ny, nx), materials.water, dtype=np.float32)
    z0 = 0
    z1 = min(nz, z0 + z_water_1)
    z2 = min(nz, z1 + z_bone)
    z3 = min(nz, z2 + z_lung)

    spr[z1:z2, :, :] = materials.cortical_bone
    spr[z2:z3, :, :] = materials.lung

    return Phantom(spr_map=spr, voxel_mm=geometry.voxel_mm)


def normalize_spr_to_01(spr: np.ndarray) -> np.ndarray:
    min_v = float(np.min(spr))
    max_v = float(np.max(spr))
    if math.isclose(max_v, min_v):
        return np.zeros_like(spr, dtype=np.float32)
    return ((spr - min_v) / (max_v - min_v)).astype(np.float32)


def _range_in_water_mm(energy_mev: float) -> float:
    """Simple empirical proton range in water approximation."""
    # R(cm) ~ 0.0022 * E^1.77 -> mm
    return 10.0 * (0.0022 * (energy_mev ** 1.77))


def _wepl_mm(spr_map: np.ndarray, vz_mm: float) -> np.ndarray:
    """Water equivalent path length accumulated along z."""
    # center-of-voxel integration along z
    cumsum = np.cumsum(spr_map, axis=0) * vz_mm
    return cumsum.astype(np.float32)


def simulate_reference_dose(
    energy_mev: float,
    spr_map: np.ndarray,
    voxel_mm: tuple[float, float, float],
    lateral_sigma0_mm: float = 2.0,
) -> np.ndarray:
    """Generate a smooth synthetic dose that mimics Bragg peak and lateral scattering."""
    vx, vy, vz = voxel_mm
    nz, ny, nx = spr_map.shape

    range_mm = _range_in_water_mm(energy_mev)
    wepl = _wepl_mm(spr_map, vz)

    # Bragg-like depth profile in WEPL domain.
    sigma_r = 3.0
    distal = np.exp(-0.5 * ((wepl - range_mm) / sigma_r) ** 2)
    proximal = np.clip(wepl / max(range_mm, 1e-6), 0.0, 1.0) ** 1.3
    depth = (0.25 * proximal + 0.85 * distal).astype(np.float32)

    # Zero out post-range tail rapidly.
    depth = depth * np.exp(-np.clip((wepl - range_mm) / 2.0, 0.0, None))

    # Lateral profile with depth/material dependent broadening.
    yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    cx, cy = nx / 2.0, ny / 2.0
    rx_mm = (xx - cx) * vx
    ry_mm = (yy - cy) * vy
    r2 = (rx_mm ** 2 + ry_mm ** 2).astype(np.float32)

    dose = np.zeros((nz, ny, nx), dtype=np.float32)
    for z in range(nz):
        local_spr = float(np.mean(spr_map[z]))
        sigma_mm = lateral_sigma0_mm + 0.03 * z * vz * (1.0 + 0.6 * local_spr)
        lateral = np.exp(-0.5 * r2 / (sigma_mm ** 2 + 1e-6))
        dose[z] = lateral * depth[z]

    max_v = float(np.max(dose))
    if max_v > 0:
        dose /= max_v
    return dose.astype(np.float32)


def add_monte_carlo_noise(dose: np.ndarray, events: int) -> np.ndarray:
    """Poisson-like noise model for low-statistics simulations."""
    lam = np.clip(dose, 0.0, None) * events
    noisy = np.random.poisson(lam).astype(np.float32) / max(events, 1)
    max_v = float(np.max(noisy))
    if max_v > 0:
        noisy /= max_v
    return noisy
