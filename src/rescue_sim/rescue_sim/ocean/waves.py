"""[1-1, 1-3] 파도장(wave field): 성분파를 합성해 임의 위치·시각의 수면 상태를 계산.

다른 모듈(부력, 시각화, ROS 노드)은 이 인터페이스만 사용한다.
모든 함수는 x, y 를 배열로 받아 한 번에(numpy 벡터화) 계산한다.

깊이가 일정한 바다 (bathymetry 없음)
  η(x, y, t) = Σ_i a_i · cos( kx_i·x + ky_i·y − ω_i t + φ_i )

얕은 물 효과 (bathymetry + shallow_water=True) — 1-3
  η(x, y, t) = Σ_i a_i(x) · cos( Sx_i(x) + ky_i·y − ω_i t + φ_i )

  해안 쪽으로 갈수록 성분파마다 세 가지가 바뀐다:
    1) 간격이 촘촘해진다  : 수심이 얕으면 파수 k 가 커진다 (분산관계)
                           → x 방향 위상 Sx(x) = ∫ kx dx  (k 가 일정하면 kx·x 와 같음)
    2) 방향이 해안 쪽으로 꺾인다 (굴절): 해안선 방향 성분 ky 는 그대로 유지 (스넬 법칙)
                           → kx = −sqrt(k² − ky²)
    3) 높이가 커지다가 부서진다
       - 천수(shoaling): 에너지 전달 속도가 느려지는 만큼 파도가 높아짐 a ∝ sqrt(Cg0 cosθ0 / Cg cosθ)
       - 쇄파(breaking): 유의파고가 수심의 BREAK_RATIO 배를 넘으면 그만큼 깎는다
  지형이 x 로만 변한다고 가정했으므로 Sx_i(x), a_i(x) 를 x 표로 미리 만들어 두고 보간한다.

할 일
  - elevation(x, y, t)  : 수면 높이 η [m]                    ✅
  - velocity(x, y, z, t): 물 입자 속도 (u, v, w) [m/s]       ⏳ 2단계 항력 전에
"""
from __future__ import annotations

import numpy as np

from .bathymetry import Bathymetry
from .config import WaveConfig
from .spectrum import G, WaveComponents, dispersion_k, sample_components

BREAK_RATIO = 0.6     # 유의파고 Hs ≤ 0.6 × 수심 (불규칙파 쇄파 한계, 경험값)
MIN_DEPTH = 0.05      # 수심 0 에서 k → ∞ 가 되는 걸 막는 최소 수심 [m]
TABLE_DX = 0.02       # 얕은 물 표 간격 [m]


def group_velocity(omega: np.ndarray, k: np.ndarray, h: np.ndarray) -> np.ndarray:
    """군속도 Cg: 파도 '에너지'가 이동하는 속도 [m/s]. 얕아질수록 느려진다."""
    kh = np.minimum(k * h, 50.0)        # 깊은 물에서 sinh 오버플로 방지
    return 0.5 * (omega / k) * (1.0 + 2.0 * kh / np.sinh(2.0 * kh))


class WaveField:
    def __init__(
        self,
        cfg: WaveConfig,
        components: WaveComponents | None = None,
        depth: float | None = None,
        bathymetry: Bathymetry | None = None,
    ):
        """
        depth      : 일정 수심으로 계산 (bathymetry 없을 때). None = 깊은 물
        bathymetry : 주면 성분파는 먼바다 수심(deep_depth) 기준으로 만들고,
                     shallow_water=True 면 해안 쪽 변화(간격·굴절·천수·쇄파)를 적용
        """
        self.cfg = cfg
        self.bathy = bathymetry
        if bathymetry is not None:
            depth = bathymetry.cfg.deep_depth
        self.depth = depth
        self.comp = components if components is not None else sample_components(cfg, depth)
        c = self.comp
        # 매 호출마다 다시 계산하지 않도록 파수 벡터 (kx, ky) 를 미리 만들어 둔다
        self._kx = c.k * np.cos(c.theta)
        self._ky = c.k * np.sin(c.theta)

        self._shallow = bathymetry is not None and bathymetry.cfg.shallow_water
        if self._shallow:
            self._build_shallow_table()

    # ---------------------------------------------------------------- 얕은 물 표
    def _build_shallow_table(self):
        b, c = self.bathy, self.comp
        x0, x1 = b.cfg.shore_x, b.cfg.slope_start_x       # 이 구간만 수심이 변한다
        xs = np.linspace(x0, x1, int(np.ceil((x1 - x0) / TABLE_DX)) + 1)
        h = np.maximum(b.depth(xs), MIN_DEPTH)[:, None]  # (m, 1)

        k = dispersion_k(c.omega, h)                     # (m, n)
        ky = self._ky
        kx = np.sign(self._kx) * np.sqrt(np.maximum(k**2 - ky**2, 0.0))   # 굴절

        # Sx(x) = kx0·x1 + ∫_{x1}^{x} kx dx'  (x ≥ x1 의 깊은 쪽에서는 그대로 kx0·x)
        seg = 0.5 * (kx[1:] + kx[:-1]) * np.diff(xs)[:, None]
        cum = np.concatenate([np.cumsum(seg[::-1], axis=0)[::-1], np.zeros((1, c.n))])
        sx = self._kx * x1 - cum

        # 천수 계수
        h0 = b.cfg.deep_depth
        cg0 = group_velocity(c.omega, c.k, h0)
        cg = group_velocity(c.omega, k, h)
        cos0 = np.abs(self._kx) / c.k
        cos = np.maximum(np.abs(kx) / k, 1e-6)
        amp = c.amp * np.sqrt((cg0 * cos0) / (cg * cos))

        # 쇄파: 지점별 유의파고가 한계를 넘으면 모든 성분을 같은 비율로 줄인다
        hs_local = 4.0 * np.sqrt(0.5 * np.sum(amp**2, axis=1, keepdims=True))
        amp *= np.minimum(1.0, BREAK_RATIO * b.depth(xs)[:, None] / hs_local)

        self._tab_x, self._tab_sx, self._tab_amp = xs, sx, amp

    def _shallow_lookup(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """x (N,) → (Sx (N,n), amp (N,n)). 깊은 쪽은 원래 공식, 땅 쪽은 진폭 0."""
        xs = self._tab_x
        f = (np.clip(x, xs[0], xs[-1]) - xs[0]) / (xs[1] - xs[0])
        i = np.minimum(f.astype(int), len(xs) - 2)
        w = (f - i)[:, None]
        sx = self._tab_sx[i] * (1 - w) + self._tab_sx[i + 1] * w
        amp = self._tab_amp[i] * (1 - w) + self._tab_amp[i + 1] * w
        deep = x >= xs[-1]
        sx[deep] = x[deep, None] * self._kx
        amp[deep] = self.comp.amp
        return sx, amp

    # ---------------------------------------------------------------- 공용 계산
    def _spatial(self, x: np.ndarray, y: np.ndarray):
        """위치로만 정해지는 부분: 위상 A (N,n), 진폭 (N,n) 또는 (n,), 원래 shape."""
        x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        shape = x.shape
        xf, yf = x.reshape(-1), y.reshape(-1)
        c = self.comp
        if self._shallow:
            sx, amp = self._shallow_lookup(xf)
        else:
            sx, amp = xf[:, None] * self._kx, c.amp
        return sx + yf[:, None] * self._ky + c.phase, amp, shape

    def elevation(self, x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
        """수면 높이 η [m]. x, y 는 같은 shape(또는 브로드캐스트 가능), 결과도 그 shape."""
        a0, amp, shape = self._spatial(x, y)
        return np.sum(amp * np.cos(a0 - self.comp.omega * t), axis=-1).reshape(shape)

    def velocity(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, t: float) -> np.ndarray:
        raise NotImplementedError("2단계 전에 구현")

    def max_elevation(self) -> float:
        """이론상 가능한 최대 |η| = 모든 성분파 마루가 한 점에 겹칠 때 = Σ a_i (얕은 물 포함)."""
        if self._shallow:
            return float(np.max(np.sum(self._tab_amp, axis=1)))
        return float(np.sum(self.comp.amp))

    def fixed_points(self, x: np.ndarray, y: np.ndarray) -> "FixedPointElevation":
        """위치가 고정된 점들(예: 수면 격자)용 빠른 계산기를 만든다."""
        return FixedPointElevation(self, x, y)


class FixedPointElevation:
    """위치가 안 바뀌는 점들의 η(t) 를 빠르게 계산.

    elevation() 은 매번 점 N개 × 성분 n개 만큼 cos 를 계산한다 (128² 격자면 52만 번).
    그런데 격자 위치는 고정이라 '위치에 따른 부분'은 미리 계산해 둘 수 있다:

      a·cos(A − ωt) = (a·cos A) · cos ωt + (a·sin A) · sin ωt      (A, a = 위치로 정해지는 값)

    → 미리 C = a·cos A, S = a·sin A (N×n 행렬) 를 저장해 두면
      매 프레임은 cos/sin 을 성분 개수(n=32)만큼만 계산하고 행렬곱 두 번으로 끝난다.
      얕은 물 효과도 위치로만 정해지므로 여기 다 녹아 들어가서, 매 프레임 비용은 그대로다.
    대가: 행렬 2개만큼 메모리 (128² × 32 × 8B × 2 ≈ 8 MB).
    """

    def __init__(self, waves: WaveField, x: np.ndarray, y: np.ndarray):
        a0, amp, self.shape = waves._spatial(x, y)
        self._c = np.cos(a0) * amp
        self._s = np.sin(a0) * amp
        self._omega = waves.comp.omega

    def at(self, t: float) -> np.ndarray:
        wt = self._omega * t
        return (self._c @ np.cos(wt) + self._s @ np.sin(wt)).reshape(self.shape)
