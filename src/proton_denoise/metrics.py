from __future__ import annotations

import numpy as np


def central_axis_profile(dose: np.ndarray) -> np.ndarray:
    """Return 1D profile along z-axis through beam center."""
    nz, ny, nx = dose.shape
    cy = ny // 2
    cx = nx // 2
    return dose[:, cy, cx]


def bragg_peak_index(profile: np.ndarray) -> int:
    return int(np.argmax(profile))


def lateral_penumbra_width_mm(
    dose: np.ndarray,
    voxel_mm: tuple[float, float, float],
    z_index: int,
    rel_level_low: float = 0.2,
    rel_level_high: float = 0.8,
) -> float:
    """Estimate average penumbra width from x and y centerline profiles."""
    vx, vy, _ = voxel_mm
    plane = dose[z_index]
    ny, nx = plane.shape
    cy = ny // 2
    cx = nx // 2

    px = plane[cy, :]
    py = plane[:, cx]

    def width(profile: np.ndarray, v_mm: float) -> float:
        pmax = float(np.max(profile))
        if pmax <= 0:
            return 0.0
        norm = profile / pmax
        idx_low = np.where(norm >= rel_level_low)[0]
        idx_high = np.where(norm >= rel_level_high)[0]
        if len(idx_low) == 0 or len(idx_high) == 0:
            return 0.0
        full_low = (idx_low[-1] - idx_low[0]) * v_mm
        full_high = (idx_high[-1] - idx_high[0]) * v_mm
        return max(full_low - full_high, 0.0)

    return 0.5 * (width(px, vx) + width(py, vy))


def gamma_pass_rate(
    pred: np.ndarray,
    target: np.ndarray,
    voxel_mm: tuple[float, float, float],
    dose_diff_percent: float = 2.0,
    distance_mm: float = 2.0,
    dose_threshold_percent: float = 10.0,
) -> float:
    """
    Brute-force local gamma with finite neighborhood.
    Suitable for small/medium synthetic volumes.
    """
    if pred.shape != target.shape:
        raise ValueError("pred and target must have identical shape")

    max_ref = float(np.max(target))
    if max_ref <= 0:
        return 0.0

    dd_crit = (dose_diff_percent / 100.0) * max_ref
    thr = (dose_threshold_percent / 100.0) * max_ref

    vz, vy, vx = voxel_mm[2], voxel_mm[1], voxel_mm[0]
    rz = int(np.ceil(distance_mm / max(vz, 1e-6)))
    ry = int(np.ceil(distance_mm / max(vy, 1e-6)))
    rx = int(np.ceil(distance_mm / max(vx, 1e-6)))

    nz, ny, nx = target.shape
    mask = target >= thr
    eval_indices = np.argwhere(mask)
    if len(eval_indices) == 0:
        return 0.0

    passed = 0
    for z, y, x in eval_indices:
        p = pred[z, y, x]
        gmin = np.inf

        z0, z1 = max(0, z - rz), min(nz, z + rz + 1)
        y0, y1 = max(0, y - ry), min(ny, y + ry + 1)
        x0, x1 = max(0, x - rx), min(nx, x + rx + 1)

        for zz in range(z0, z1):
            dz_mm = abs(zz - z) * vz
            for yy in range(y0, y1):
                dy_mm = abs(yy - y) * vy
                for xx in range(x0, x1):
                    dx_mm = abs(xx - x) * vx
                    dist_term = (np.sqrt(dx_mm * dx_mm + dy_mm * dy_mm + dz_mm * dz_mm) / distance_mm) ** 2
                    dose_term = ((p - target[zz, yy, xx]) / max(dd_crit, 1e-8)) ** 2
                    g = np.sqrt(dist_term + dose_term)
                    if g < gmin:
                        gmin = g

        if gmin <= 1.0:
            passed += 1

    return 100.0 * passed / len(eval_indices)
