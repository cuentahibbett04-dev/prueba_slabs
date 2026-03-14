from __future__ import annotations

from pathlib import Path
from typing import Iterable
import zipfile

import numpy as np
import torch
from torch.utils.data import Dataset


class ProtonDoseDataset(Dataset):
    """Loads .npz samples with keys: input, target, spr, energy_mev."""

    def __init__(
        self,
        split_dir: Path,
        *,
        normalize_target: bool = True,
        input_norm_mode: str = "none",
        input_dose_scale: float = 1.0,
        eps: float = 1e-8,
        low_events_allow: Iterable[int] | None = None,
        crop_shape: tuple[int, int, int] | None = None,
        crop_focus: str = "center",
    ):
        self.split_dir = Path(split_dir)
        all_files = sorted(self.split_dir.glob("*.npz"))
        if not all_files:
            raise FileNotFoundError(f"No .npz files found in {self.split_dir}")
        allowed = {"none", "per_channel_max", "global_max", "coupled_target_max"}
        if input_norm_mode not in allowed:
            raise ValueError(f"input_norm_mode must be one of {sorted(allowed)}, got {input_norm_mode!r}")
        self.normalize_target = bool(normalize_target)
        self.input_norm_mode = input_norm_mode
        self.input_dose_scale = float(input_dose_scale)
        self.eps = float(eps)
        self.low_events_allow = None if low_events_allow is None else {int(v) for v in low_events_allow}
        self.crop_shape = None if crop_shape is None else tuple(int(v) for v in crop_shape)
        self.crop_focus = str(crop_focus)
        self._runtime_bad_files: set[Path] = set()

        if self.crop_shape is not None:
            if len(self.crop_shape) != 3 or any(v <= 0 for v in self.crop_shape):
                raise ValueError(f"crop_shape must be a 3-tuple of positive ints, got {self.crop_shape!r}")
        allowed_focus = {"center", "maxdose"}
        if self.crop_focus not in allowed_focus:
            raise ValueError(f"crop_focus must be one of {sorted(allowed_focus)}, got {self.crop_focus!r}")

        required_keys = {"input", "target", "spr", "energy_mev"}
        valid_files: list[Path] = []
        low_events_by_file: dict[Path, int] = {}
        bad_files: list[Path] = []
        for p in all_files:
            try:
                with np.load(p) as z:
                    if not required_keys.issubset(set(z.files)):
                        raise KeyError(f"missing required keys: {sorted(required_keys - set(z.files))}")
                    low_events_by_file[p] = int(z["low_events"].item()) if "low_events" in z else -1
                valid_files.append(p)
            except Exception:  # pylint: disable=broad-except
                bad_files.append(p)

        if bad_files:
            preview = ", ".join(str(p.name) for p in bad_files[:5])
            tail = " ..." if len(bad_files) > 5 else ""
            print(
                f"WARNING: skipped {len(bad_files)} invalid .npz files in {self.split_dir}: {preview}{tail}",
                flush=True,
            )

        if not valid_files:
            raise FileNotFoundError(f"No valid .npz files found in {self.split_dir}")

        if self.low_events_allow is None:
            self.files = valid_files
        else:
            self.files = []
            for p in valid_files:
                low_events = low_events_by_file.get(p, -1)
                if low_events in self.low_events_allow:
                    self.files.append(p)
            if not self.files:
                raise FileNotFoundError(
                    f"No .npz files in {self.split_dir} after low_events filter={sorted(self.low_events_allow)}"
                )

    def __len__(self) -> int:
        return len(self.files)

    @staticmethod
    def _crop_or_pad_3d(arr: np.ndarray, crop_shape: tuple[int, int, int], center: tuple[int, int, int]) -> np.ndarray:
        """Return fixed-size [D,H,W] crop; pads with zeros if needed."""
        out = arr
        cz, cy, cx = center
        tz, ty, tx = crop_shape

        # Pad each axis when source is smaller than target crop.
        dz, dy, dx = out.shape
        pad_z = max(0, tz - dz)
        pad_y = max(0, ty - dy)
        pad_x = max(0, tx - dx)
        if pad_z or pad_y or pad_x:
            pz0 = pad_z // 2
            pz1 = pad_z - pz0
            py0 = pad_y // 2
            py1 = pad_y - py0
            px0 = pad_x // 2
            px1 = pad_x - px0
            out = np.pad(out, ((pz0, pz1), (py0, py1), (px0, px1)), mode="constant", constant_values=0.0)
            cz += pz0
            cy += py0
            cx += px0

        dz, dy, dx = out.shape
        sz = max(0, min(cz - tz // 2, dz - tz))
        sy = max(0, min(cy - ty // 2, dy - ty))
        sx = max(0, min(cx - tx // 2, dx - tx))
        return out[sz : sz + tz, sy : sy + ty, sx : sx + tx]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        p = self.files[idx]
        load_exc: Exception | None = None

        # If one file becomes corrupted after startup checks, try nearby samples
        # instead of crashing the entire training job.
        x = None
        chosen_path = p
        for off in range(len(self.files)):
            candidate = self.files[(idx + off) % len(self.files)]
            try:
                x = np.load(candidate)
                chosen_path = candidate
                break
            except (zipfile.BadZipFile, OSError, ValueError) as exc:
                load_exc = exc
                if candidate not in self._runtime_bad_files:
                    self._runtime_bad_files.add(candidate)
                    print(f"WARNING: runtime-invalid npz skipped: {candidate}", flush=True)
                continue

        if x is None:
            raise RuntimeError(
                f"Failed to load any sample around index {idx} in {self.split_dir}. "
                f"Last error: {load_exc}"
            )

        with x:
            inp = x["input"].astype(np.float32)  # [2, D, H, W]
            target = x["target"].astype(np.float32)  # [D, H, W]
            spr = x["spr"].astype(np.float32)[None, ...]
            energy = np.array([x["energy_mev"].item()], dtype=np.float32)
            low_events_val = int(x["low_events"].item()) if "low_events" in x else -1
            high_events_val = int(x["high_events"].item()) if "high_events" in x else -1
            low_events = np.array([low_events_val], dtype=np.float32)
            high_events = np.array([high_events_val], dtype=np.float32)

        if self.crop_shape is not None:
            if self.crop_focus == "maxdose":
                center = tuple(int(v) for v in np.unravel_index(np.argmax(target), target.shape))
            else:
                d, h, w = target.shape
                center = (d // 2, h // 2, w // 2)

            target = self._crop_or_pad_3d(target, self.crop_shape, center)
            spr_3d = self._crop_or_pad_3d(spr[0], self.crop_shape, center)
            inp0 = self._crop_or_pad_3d(inp[0], self.crop_shape, center)
            inp1 = self._crop_or_pad_3d(inp[1], self.crop_shape, center)
            inp = np.stack([inp0, inp1], axis=0)
            spr = spr_3d[None, ...]

        if self.input_norm_mode == "coupled_target_max":
            # Keep low-dose and target in the same normalized physical space.
            if high_events_val > 0 and low_events_val > 0:
                ratio = float(high_events_val) / float(low_events_val)
            else:
                ratio = self.input_dose_scale

            inp[0] = inp[0] * ratio

            tmax = float(np.max(target))
            if tmax > self.eps:
                target = target / tmax
                inp[0] = inp[0] / tmax

            cmax_ct = float(np.max(inp[1]))
            if cmax_ct > self.eps:
                inp[1] = inp[1] / cmax_ct
        else:
            if self.normalize_target:
                tmax = float(np.max(target))
                if tmax > self.eps:
                    target = target / tmax

        if self.input_norm_mode == "per_channel_max":
            for c in range(inp.shape[0]):
                cmax = float(np.max(inp[c]))
                if cmax > self.eps:
                    inp[c] = inp[c] / cmax
        elif self.input_norm_mode == "global_max":
            gmax = float(np.max(inp))
            if gmax > self.eps:
                inp = inp / gmax

        # Channel 0 is low-MC dose; this optional scaling can inject known history ratio (e.g., x50).
        if self.input_norm_mode != "coupled_target_max":
            inp[0] = inp[0] * self.input_dose_scale
        target = target[None, ...]  # [1, D, H, W]

        return {
            "input": torch.from_numpy(inp),
            "target": torch.from_numpy(target),
            "spr": torch.from_numpy(spr),
            "energy_mev": torch.from_numpy(energy),
            "low_events": torch.from_numpy(low_events),
            "high_events": torch.from_numpy(high_events),
            "path": str(chosen_path),
        }
