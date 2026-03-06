from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GeometryConfig:
    """Phantom geometry in millimeters."""

    size_mm: tuple[int, int, int] = (150, 150, 200)
    voxel_mm: tuple[float, float, float] = (2.0, 2.0, 2.0)

    # slab thicknesses along depth axis (z)
    water_1_mm: int = 40
    bone_mm: int = 30
    lung_mm: int = 50
    water_2_mm: int = 80


@dataclass(frozen=True)
class MaterialSPR:
    """Typical relative stopping power values."""

    water: float = 1.0
    cortical_bone: float = 1.65
    lung: float = 0.3


@dataclass(frozen=True)
class DataConfig:
    root: Path = Path("data")
    train_count: int = 120
    val_count: int = 20
    test_count: int = 20
    energy_min_mev: float = 70.0
    energy_max_mev: float = 150.0
    low_stat_events: int = 2_000
    high_stat_events: int = 100_000


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 2
    epochs: int = 30
    lr: float = 2e-4
    weight_decay: float = 1e-5
    device: str = "cuda"
    seed: int = 42
    out_dir: Path = Path("artifacts")


@dataclass(frozen=True)
class LossConfig:
    alpha: float = 6.0
    min_weight: float = 0.05
