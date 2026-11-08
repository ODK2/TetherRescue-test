"""[1-4] 해류: 위치별 흐름 속도.

지금은 영역 전체에 일정한 해류. 땅(수심 0) 위에서는 0.
"""
from __future__ import annotations

import numpy as np

from .bathymetry import Bathymetry
from .config import CurrentConfig

FADE_DEPTH = 0.3   # 이 수심보다 얕으면 해류를 선형으로 줄인다 [m] (물가에서 0)


class Current:
    def __init__(self, cfg: CurrentConfig, bathymetry: Bathymetry | None = None):
        self.cfg = cfg
        self.bathy = bathymetry
        self._v = np.asarray(cfg.velocity, dtype=float)

    def velocity(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """(u, v) [m/s]. 결과 shape = x 의 shape + (2,)."""
        x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        out = np.broadcast_to(self._v, x.shape + (2,)).copy()
        if self.bathy is not None:
            out *= np.clip(self.bathy.depth(x) / FADE_DEPTH, 0.0, 1.0)[..., None]
        return out
