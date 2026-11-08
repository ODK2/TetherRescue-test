"""로프 = MuJoCo spatial tendon 의 길이 제한(limit).

MuJoCo 에서 알아 둘 것
  - tendon range 상한을 넘으면 MuJoCo 가 '제약(constraint)' 으로 끌어당긴다 → 팽팽한 로프.
    짧을 때는 아무 힘 없음 → 느슨한 로프. (스프링이 아니라 늘어나지 않는 끈에 가깝다)
  - model.tendon_range / tendon_rgba 는 컴파일 후에도 바꿀 수 있다 → 실행 중 '연결/해제'.
  - 장력 = 그 제약의 힘: data.efc_force 중 efc_type == LIMIT_TENDON 이고 efc_id == 텐던 id 인 것.
  - 화면: 느슨한(길이 < 상한) limited 텐던은 MuJoCo 가 처진 곡선(현수선)으로 그려 준다 — 그림만.
  - 한계: 물리에선 두 점 사이 직선 거리만 본다. 처짐·로프 질량·물속 저항·뜨는 로프 없음.
"""
from __future__ import annotations

import mujoco
import numpy as np

from ..sim.world import World

DETACHED_RANGE = 1000.0


class Rope:
    def __init__(self, world: World, name: str = "rope", length: float = 8.0):
        self.world = world
        self.id = world.model.tendon(name).id
        self.length = length
        self.detach()

    @property
    def attached(self) -> bool:
        return self.world.model.tendon_range[self.id, 1] < DETACHED_RANGE

    def attach(self, length: float | None = None):
        m = self.world.model
        m.tendon_range[self.id] = (0.0, length or self.length)
        m.tendon_rgba[self.id, 3] = 1.0

    def detach(self):
        m = self.world.model
        m.tendon_range[self.id] = (0.0, DETACHED_RANGE)
        m.tendon_rgba[self.id, 3] = 0.0

    def span(self) -> float:
        """양 끝 직선 거리 [m] (현재 텐던 길이)."""
        return float(self.world.data.ten_length[self.id])

    def tension(self) -> float:
        d = self.world.data
        mask = (d.efc_type == mujoco.mjtConstraint.mjCNSTR_LIMIT_TENDON) & (d.efc_id == self.id)
        return float(np.abs(d.efc_force[mask]).sum())
