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

    def max_elevation(self) -> float:
        """이론상 가능한 최대 |η| = 모든 성분파 마루가 한 점에 겹칠 때 = Σ a_i."""
        return float(np.sum(self.comp.amp))

    def fixed_points(self, x: np.ndarray, y: np.ndarray) -> "FixedPointElevation":
        """위치가 고정된 점들(예: 수면 격자)용 빠른 계산기를 만든다."""
        return FixedPointElevation(self, x, y)


class FixedPointElevation:
    """위치가 안 바뀌는 점들의 η(t) 를 빠르게 계산.

    elevation() 은 매번 점 N개 × 성분 n개 만큼 cos 를 계산한다 (128² 격자면 52만 번).
    그런데 격자 위치는 고정이라 '위치에 따른 부분'은 미리 계산해 둘 수 있다:

      cos(A − ωt) = cos A · cos ωt + sin A · sin ωt      (A = 위치로 정해지는 값)

    → 미리 C = a·cos A, S = a·sin A (N×n 행렬) 를 저장해 두면
      매 프레임은 cos/sin 을 성분 개수(n=32)만큼만 계산하고 행렬곱 두 번으로 끝난다.
    대가: 행렬 2개만큼 메모리 (128² × 32 × 8B × 2 ≈ 8 MB).
    """

    def __init__(self, waves: WaveField, x: np.ndarray, y: np.ndarray):
        ph0, self.shape = waves._phase(x, y, 0.0)        # (N, n) — t=0 위상 = A
        a = waves.comp.amp
        self._c = np.cos(ph0) * a
        self._s = np.sin(ph0) * a
        self._omega = waves.comp.omega

    def at(self, t: float) -> np.ndarray:
        wt = self._omega * t
        return (self._c @ np.cos(wt) + self._s @ np.sin(wt)).reshape(self.shape)
