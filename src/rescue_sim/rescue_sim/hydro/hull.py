"""[2단계] 선체 유체력: 부력 + 항력 (MuJoCo 비의존, 순수 numpy).

부력 — 기둥(column) 방식
  선체 바닥을 nx × ny 개의 세로 기둥으로 나눈다. 기둥마다
    잠긴 높이 h = clip(수면 η(x,y,t) − 기둥 바닥 z, 0, 선체 높이)
    부력     F = ρ g · (기둥 단면적) · h        (위쪽)
  를 기둥 중심에 가한다. 기둥마다 잠긴 정도가 다르면 자연히 롤·피치 복원 토크가 생기고,
  수면이 파도로 움직이면 그대로 파도에 흔들리는 힘이 된다.

항력
  물에 대한 상대속도(배 속도 − 해류)를 배 좌표계로 바꿔서
    F = −(선형 계수 · v + 2차 계수 · v|v|)   (전진 / 옆 / 위아래 계수가 다름)
  잠긴 비율만큼 곱한다. 회전도 같은 식.

알려진 한계 (bench/한계 노트에 기록)
  - 부가질량(added mass) 없음 → 위아래 출렁임이 실제보다 빠르다
  - 기둥은 기울어도 '수직'으로 간주 (작은 롤·피치 가정)
  - 물 속도(해류 + 파도 입자 속도)는 선체 중심 한 점에서만 샘플 → 배 길이보다 짧은 파도에선 부정확
  - 선체는 직육면체 근사
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..ocean.current import Current
from ..ocean.waves import WaveField

RHO = 1025.0   # 바닷물 밀도 [kg/m³]
G = 9.81


@dataclass
class HullParams:
    length: float = 2.4            # [m]
    beam: float = 1.0              # 폭 [m]
    height: float = 0.4            # 선체 높이 [m]
    nx: int = 6                    # 부력 기둥 개수 (길이 방향)
    ny: int = 3                    # (폭 방향)
    # 항력 계수: 배 좌표계 (전진 x, 옆 y, 위아래 z)
    lin_drag: tuple = (40.0, 400.0, 2000.0)     # [N/(m/s)]
    quad_drag: tuple = (60.0, 600.0, 600.0)     # [N/(m/s)²]
    ang_lin_drag: tuple = (300.0, 600.0, 150.0)   # 롤, 피치, 요 [N·m/(rad/s)]
    ang_quad_drag: tuple = (100.0, 300.0, 100.0)


@dataclass
class HullState:
    """한 순간 선체 상태 (모두 월드 좌표계)."""
    pos: np.ndarray        # 몸체 원점 = 선체 중심 (3,)
    rot: np.ndarray        # 회전행렬 (3,3): 몸체 → 월드
    com: np.ndarray        # 무게중심 (3,)
    vel: np.ndarray        # 무게중심 선속도 (3,)
    angvel: np.ndarray     # 각속도 (3,)


@dataclass
class HydroResult:
    force: np.ndarray      # 무게중심 기준 합력 (3,)
    torque: np.ndarray     # 무게중심 기준 합토크 (3,)
    wet: float             # 잠긴 비율 0~1 (설계 흘수 기준이 아니라 선체 전체 부피 기준)
    buoyancy: float        # 부력 크기 [N]
    column_depth: np.ndarray = field(repr=False, default=None)   # 기둥별 잠긴 높이 (디버그)
    extra_eta: np.ndarray = field(repr=False, default=None)      # 추가 지점 수면 높이


class HullHydro:
    def __init__(self, p: HullParams, waves: WaveField, current: Current | None = None):
        self.p = p
        self.waves = waves
        self.current = current
        dx, dy = p.length / p.nx, p.beam / p.ny
        xs = -p.length / 2 + dx * (np.arange(p.nx) + 0.5)
        ys = -p.beam / 2 + dy * (np.arange(p.ny) + 0.5)
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        # 기둥 바닥점 (몸체 좌표계)
        self.col_local = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, -p.height / 2)], axis=1)
        self.col_area = dx * dy
        self.volume = p.length * p.beam * p.height
        self._lin, self._quad = np.array(p.lin_drag), np.array(p.quad_drag)
        self._alin, self._aquad = np.array(p.ang_lin_drag), np.array(p.ang_quad_drag)

    def compute(self, s: HullState, t: float, extra_xy: np.ndarray | None = None) -> HydroResult:
        """extra_xy (k,2): 같이 수면 높이를 구할 추가 지점 (예: 추진기). 결과 extra_eta.

        수면 높이를 한 번의 호출로 모아서 구하는 이유: 매 물리 스텝(500Hz) 불리므로
        numpy 호출 횟수 자체가 비용이다 (점 개수보다 호출 횟수가 더 비쌈).
        """
        p = self.p
        bottoms = s.pos + self.col_local @ s.rot.T                     # (n, 3) 월드
        qx, qy = bottoms[:, 0], bottoms[:, 1]
        if extra_xy is not None:
            qx, qy = np.concatenate([qx, extra_xy[:, 0]]), np.concatenate([qy, extra_xy[:, 1]])
        eta_all = self.waves.elevation(qx, qy, t)
        eta, extra_eta = eta_all[: len(bottoms)], eta_all[len(bottoms):]
        h = np.clip(eta - bottoms[:, 2], 0.0, p.height)                # 잠긴 높이

        # 부력: 기둥마다 위쪽 힘, 작용점 = 잠긴 부분의 가운데
        fz = RHO * G * self.col_area * h
        points = bottoms.copy()
        points[:, 2] += h / 2
        r = points - s.com
        force = np.array([0.0, 0.0, fz.sum()])
        torque = np.array([np.sum(r[:, 1] * fz), -np.sum(r[:, 0] * fz), 0.0])   # r × (0,0,fz)

        wet = float(h.sum() * self.col_area / self.volume)
        if wet > 0.0:
            k = min(1.0, wet * 4.0)       # 흘수 정도(선체 1/4)만 잠겨도 항력은 최대
            # 물에 대한 상대속도: 물 자체가 파도(입자 운동)와 해류로 움직이고 있다.
            # 이걸 빼먹으면 '멈춘 물'을 기준으로 저항이 생겨 배가 엉뚱하게 떠밀려 간다.
            z_sample = min(s.com[2], 0.0) - 0.5 * p.height * 0.25
            v_water = self.waves.velocity(s.com[0], s.com[1], z_sample, t)
            if self.current is not None:
                v_water[:2] += self.current.velocity(s.com[0], s.com[1])
            vb = s.rot.T @ (s.vel - v_water)                           # 배 좌표계 상대속도
            fb = -(self._lin * vb + self._quad * vb * np.abs(vb))
            wb = s.rot.T @ s.angvel
            tb = -(self._alin * wb + self._aquad * wb * np.abs(wb))
            force += k * (s.rot @ fb)
            torque += k * (s.rot @ tb)

        return HydroResult(force, torque, wet, float(fz.sum()), h, extra_eta)
