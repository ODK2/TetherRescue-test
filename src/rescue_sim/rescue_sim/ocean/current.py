"""[1-4] 해류: 위치별 흐름 속도.

할 일
  - velocity(x, y): (u, v) [m/s]. 처음엔 일정 해류, 필요하면 해안 근처 변화 추가
"""
from __future__ import annotations

import numpy as np

from .config import CurrentConfig


class Current:
    def __init__(self, cfg: CurrentConfig):
        self.cfg = cfg

    def velocity(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError("1-4에서 같이 구현")
