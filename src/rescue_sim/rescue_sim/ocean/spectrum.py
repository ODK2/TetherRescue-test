"""[1-1] 파도 스펙트럼: Hs, Tp → 성분파(진폭·주파수·방향·위상) 목록.

핵심 아이디어
  실제 바다 = 서로 다른 주기·방향의 사인파 수십 개를 랜덤 위상으로 겹친 것.
  "어떤 주기의 파도가 얼마나 센가"를 정해 주는 게 스펙트럼 S(ω)이고,
  여기서는 해양공학 표준인 JONSWAP 스펙트럼을 쓴다.

흐름
  1) jonswap()            : S(ω) [m²·s]  — Tp 근처에 봉우리, γ 가 클수록 뾰족
  2) sample_components()  : ω 를 n 구간으로 쪼개 a_i = sqrt(2 S(ω_i) Δω)
                            방향은 cos^2s 퍼짐, 위상은 0~2π 균일 랜덤
  3) dispersion_k()       : ω → k (분산관계 ω² = g k tanh(k h))

검증 포인트
  Σ a_i² / 2 = m0 = (Hs/4)²
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import WaveConfig

G = 9.81

# 성분파를 뽑는 주파수 범위 (첨두 주파수 ωp 의 배수).
# 0.5ωp 아래는 에너지가 사실상 0, 3ωp 위는 꼬리라 전체 에너지의 ~1% 수준.
OMEGA_MIN_RATIO = 0.5
OMEGA_MAX_RATIO = 3.0


@dataclass
class WaveComponents:
    amp: np.ndarray     # (n,) 진폭 [m]
    omega: np.ndarray   # (n,) 각주파수 [rad/s]
    k: np.ndarray       # (n,) 파수 [rad/m] — 분산관계로 계산
    theta: np.ndarray   # (n,) 진행 방향 [rad] (파도가 '가는' 방향, 0 = +x)
    phase: np.ndarray   # (n,) 초기 위상 [rad]

    @property
    def n(self) -> int:
        return len(self.amp)

    def hs(self) -> float:
        """성분파들로부터 다시 계산한 유의파고 4·sqrt(m0)."""
        return float(4.0 * np.sqrt(np.sum(self.amp**2) / 2.0))


def jonswap(omega: np.ndarray, hs: float, tp: float, gamma: float = 3.3) -> np.ndarray:
    """JONSWAP 스펙트럼 밀도 S(ω) [m²·s] (DNV-RP-C205 형태).

    S(ω) = A_γ · S_PM(ω) · γ^r
      S_PM : Pierson-Moskowitz (충분히 발달한 바다)
      γ^r  : 첨두 근처만 γ 배 부풀리는 항 (바람 불어온 거리가 짧은 바다 = 더 뾰족)
      A_γ  : γ 로 부풀린 만큼 전체 에너지를 원래대로 맞추는 보정
    """
    omega = np.asarray(omega, dtype=float)
    wp = 2.0 * np.pi / tp
    s = np.zeros_like(omega)
    pos = omega > 0
    w = omega[pos]

    s_pm = (5.0 / 16.0) * hs**2 * wp**4 * w**-5 * np.exp(-1.25 * (w / wp) ** -4)
    sigma = np.where(w <= wp, 0.07, 0.09)
    r = np.exp(-0.5 * ((w - wp) / (sigma * wp)) ** 2)
    a_gamma = 1.0 - 0.287 * np.log(gamma)
    s[pos] = a_gamma * s_pm * gamma**r
    return s


def dispersion_k(omega: np.ndarray, depth: float | np.ndarray | None = None) -> np.ndarray:
    """분산관계 ω² = g k tanh(k h) 를 k 에 대해 푼다.

    depth=None 이면 깊은 물(tanh → 1): k = ω²/g.
    유한 수심이면 같은 ω 라도 k 가 커진다 = 파장이 짧아진다 (해안 쪽에서 파도가 촘촘해지는 이유).
    depth 가 배열이면 omega 와 브로드캐스트한다 (예: depth (m,1) × omega (n,) → (m,n)).
    """
    omega = np.asarray(omega, dtype=float)
    k_deep = omega**2 / G
    if depth is None:
        return k_deep
    h = np.asarray(depth, dtype=float)
    k_deep = np.broadcast_to(k_deep, np.broadcast_shapes(k_deep.shape, h.shape))
    # 초기값: 얕은 물/깊은 물을 잇는 근사, 이후 뉴턴법 몇 번이면 기계 정밀도 수렴
    k = k_deep / np.sqrt(np.tanh(k_deep * h))
    for _ in range(20):
        th = np.tanh(k * h)
        f = G * k * th - omega**2
        df = G * th + G * k * h * (1.0 - th**2)
        dk = f / df
        k = k - dk
        if np.all(np.abs(dk) < 1e-12 * k):
            break
    return k


def spreading_s(spreading_deg: float) -> float:
    """방향 퍼짐 각(표준편차, 도) → cos^2s(Δθ/2) 의 지수 s.

    cos^2s 분포의 표준편차 ≈ sqrt(2/(s+1)) [rad] 관계를 뒤집은 것.
    30° → s ≈ 6.3
    """
    sd = np.deg2rad(max(spreading_deg, 1e-3))
    return max(2.0 / sd**2 - 1.0, 0.0)


def _sample_directions(rng: np.random.Generator, n: int, theta0: float, s: float) -> np.ndarray:
    """D(Δθ) ∝ cos^2s(Δθ/2) 에서 n 개 샘플 (기각 샘플링)."""
    out = np.empty(0)
    while out.size < n:
        d = rng.uniform(-np.pi, np.pi, size=4 * n)
        keep = rng.uniform(0.0, 1.0, size=d.size) < np.cos(d / 2.0) ** (2.0 * s)
        out = np.concatenate([out, d[keep]])
    return theta0 + out[:n]


def sample_components(cfg: WaveConfig, depth: float | None = None) -> WaveComponents:
    """WaveConfig → 성분파 n 개.

    - ω 구간 [0.5ωp, 3ωp] 를 n 등분하고, 각 구간 안에서 ω 를 랜덤으로 살짝 흔든다.
      (등간격 그대로면 2π/Δω 초마다 바다가 똑같이 반복된다 — harbor 기준 약 38초)
    - 마지막에 진폭을 미세 조정해 4·sqrt(m0) 이 정확히 cfg.hs 가 되게 한다.
      (범위 밖 꼬리 에너지 ~1% 와 A_γ 근사 오차 보정)
    - 성분 하나당 방향 하나 (single-summation). 같은 ω 에 여러 방향을 두면
      특정 지점에서 위상이 고정돼 비현실적인 '정상파'가 생길 수 있다.
    """
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_components
    wp = 2.0 * np.pi / cfg.tp

    edges = np.linspace(OMEGA_MIN_RATIO * wp, OMEGA_MAX_RATIO * wp, n + 1)
    d_omega = np.diff(edges)
    omega = edges[:-1] + rng.uniform(0.0, 1.0, size=n) * d_omega

    amp = np.sqrt(2.0 * jonswap(omega, cfg.hs, cfg.tp, cfg.gamma) * d_omega)
    amp *= cfg.hs / (4.0 * np.sqrt(np.sum(amp**2) / 2.0))

    theta = _sample_directions(rng, n, np.deg2rad(cfg.direction_deg), spreading_s(cfg.spreading_deg))
    phase = rng.uniform(0.0, 2.0 * np.pi, size=n)

    return WaveComponents(amp=amp, omega=omega, k=dispersion_k(omega, depth), theta=theta, phase=phase)
