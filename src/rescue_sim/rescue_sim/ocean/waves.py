"""[1-1] 파도장(wave field): 성분파를 합성해 임의 위치·시각의 수면 상태를 계산.

다른 모듈(부력, 시각화, ROS 노드)은 이 인터페이스만 사용한다.
모든 함수는 x, y 를 배열로 받아 한 번에(numpy 벡터화) 계산한다.

  η(x, y, t) = Σ_i a_i · cos( k_i (x cosθ_i + y sinθ_i) − ω_i t + φ_i )

할 일
  - elevation(x, y, t)  : 수면 높이 η [m]                    ✅ 1-1
  - velocity(x, y, z, t): 물 입자 속도 (u, v, w) [m/s]       ⏳ 2단계 항력 전에
"""
from __future__ import annotations

import numpy as np

from .config import WaveConfig
from .spectrum import WaveComponents, sample_components


class WaveField:
    def __init__(
        self,
        cfg: WaveConfig,
        components: WaveComponents | None = None,
        depth: float | None = None,
    ):
        self.cfg = cfg
        self.comp = components if components is not None else sample_components(cfg, depth)
        c = self.comp
        # 매 호출마다 다시 계산하지 않도록 파수 벡터 (kx, ky) 를 미리 만들어 둔다
        self._kx = c.k * np.cos(c.theta)
        self._ky = c.k * np.sin(c.theta)

    def _phase(self, x: np.ndarray, y: np.ndarray, t: float) -> tuple[np.ndarray, tuple]:
        """(점 개수 N, 성분 n) 위상 행렬과 원래 shape 를 돌려준다."""
        x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        shape = x.shape
        xf, yf = x.reshape(-1, 1), y.reshape(-1, 1)
        c = self.comp
        ph = xf * self._kx + yf * self._ky - c.omega * t + c.phase   # (N, n)
        return ph, shape

    def elevation(self, x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
        """수면 높이 η [m]. x, y 는 같은 shape(또는 브로드캐스트 가능), 결과도 그 shape."""
        ph, shape = self._phase(x, y, t)
        return (np.cos(ph) @ self.comp.amp).reshape(shape)

    def velocity(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, t: float) -> np.ndarray:
        raise NotImplementedError("2단계 전에 구현")
