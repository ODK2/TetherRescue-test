"""[1-3] 해저 지형: 위치별 수심·바닥 높이.

모양 (옆에서 본 단면, x 축은 해안 ← → 먼바다)

    z
    +1 ─┐ 모래사장 (beach_height 까지)
     0 ──\──────────────────────────────── 정지 수면
         \ ← shore_x (수심 0, 물가)
          \   경사면
           \______________________________ -deep_depth (먼바다)
                        ↑ slope_start_x

가정: 지형은 x 방향으로만 변한다 (해안선이 y 축과 나란한 직선 해변).
      → 파도 계산(waves.py)이 이 가정을 써서 얕은 물 효과를 싸게 계산한다.

MuJoCo 는 여기서 모른다. hfield 로 옮기는 건 sim/world.py 가 한다.
"""
from __future__ import annotations

import numpy as np

from .config import BathymetryConfig


class Bathymetry:
    def __init__(self, cfg: BathymetryConfig):
        if not cfg.shore_x < cfg.slope_start_x:
            raise ValueError("shore_x 는 slope_start_x 보다 해안 쪽(작은 x)이어야 함")
        self.cfg = cfg
        self.slope = cfg.deep_depth / (cfg.slope_start_x - cfg.shore_x)   # 수심 증가율 [m/m]

    def depth(self, x: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        """수심 h [m] ≥ 0. 물가(shore_x)에서 0, 해안 너머(땅)도 0, 먼바다는 deep_depth."""
        x = np.asarray(x, dtype=float)
        h = np.clip((x - self.cfg.shore_x) * self.slope, 0.0, self.cfg.deep_depth)
        if y is not None:
            h = np.broadcast_to(h, np.broadcast_shapes(x.shape, np.shape(y))).copy()
        return h

    def seabed_z(self, x: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        """바닥 높이 z [m] (정지 수면 = 0). 물속은 -depth, 땅은 같은 경사로 올라가다 beach_height 에서 평평."""
        x = np.asarray(x, dtype=float)
        z = np.clip(-(x - self.cfg.shore_x) * self.slope, -self.cfg.deep_depth, self.cfg.beach_height)
        if y is not None:
            z = np.broadcast_to(z, np.broadcast_shapes(x.shape, np.shape(y))).copy()
        return z

    @property
    def z_range(self) -> tuple[float, float]:
        """바닥이 가질 수 있는 z 범위 (hfield 높이 범위 설정용)."""
        return -self.cfg.deep_depth, self.cfg.beach_height
