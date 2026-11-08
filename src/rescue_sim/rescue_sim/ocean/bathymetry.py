"""[1-3] 해저 지형: 위치별 수심, MuJoCo 해저 hfield 데이터 생성.

할 일
  - depth(x, y)          : 수심 h [m] (해안선 너머는 0)
  - seabed_hfield(nx, ny): MuJoCo hfield 용 0~1 정규화 높이 배열
"""
from __future__ import annotations

import numpy as np

from .config import BathymetryConfig


class Bathymetry:
    def __init__(self, cfg: BathymetryConfig):
        self.cfg = cfg

    def depth(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError("1-3에서 같이 구현")

    def seabed_hfield(self, nx: int, ny: int) -> np.ndarray:
        raise NotImplementedError("1-3에서 같이 구현")
