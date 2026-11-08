"""config/ocean.yaml → 데이터클래스."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class DomainConfig:
    size_x: float = 50.0
    size_y: float = 50.0
    surface_res: int = 128


@dataclass
class WaveConfig:
    hs: float = 0.2              # 유의파고 [m]
    tp: float = 3.0              # 첨두주기 [s]
    gamma: float = 3.3           # JONSWAP 첨예도
    n_components: int = 32
    direction_deg: float = 180.0
    spreading_deg: float = 30.0
    seed: int = 0


@dataclass
class BathymetryConfig:
    deep_depth: float = 5.0
    shore_x: float = -10.0
    slope_start_x: float = 15.0
    shallow_water: bool = False


@dataclass
class CurrentConfig:
    velocity: tuple[float, float] = (0.0, 0.0)


@dataclass
class VizConfig:
    surface_fps: float = 25.0


@dataclass
class OceanConfig:
    domain: DomainConfig = field(default_factory=DomainConfig)
    waves: WaveConfig = field(default_factory=WaveConfig)
    bathymetry: BathymetryConfig = field(default_factory=BathymetryConfig)
    current: CurrentConfig = field(default_factory=CurrentConfig)
    viz: VizConfig = field(default_factory=VizConfig)

    @classmethod
    def from_yaml(cls, path: str | Path, preset: str | None = None) -> "OceanConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        w = dict(raw.get("waves", {}))
        presets = w.pop("presets", {})
        name = preset or w.pop("preset", None)
        w.pop("preset", None)
        if name:
            w.update(presets[name])
        cur = raw.get("current", {})
        return cls(
            domain=DomainConfig(**raw.get("domain", {})),
            waves=WaveConfig(**w),
            bathymetry=BathymetryConfig(**raw.get("bathymetry", {})),
            current=CurrentConfig(velocity=tuple(cur.get("velocity", (0.0, 0.0)))),
            viz=VizConfig(**raw.get("viz", {})),
        )
