"""MuJoCo 월드 래퍼: 씬 로딩, 스텝 루프, 성능 측정의 공통 틀.

단계가 진행되면 여기에 '매 스텝 외력 계산 훅'(부력·항력·로프 등)이 붙는다.
"""
from __future__ import annotations

import time
from pathlib import Path

import mujoco

from .. import ASSETS_DIR

DEFAULT_SCENE = ASSETS_DIR / "ocean_scene.xml"


class World:
    def __init__(self, scene: str | Path = DEFAULT_SCENE):
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.step_wall_time = 0.0   # 누적 스텝 소요 시간 (실시간 배속 계산용)
        self.n_steps = 0

    @property
    def time(self) -> float:
        return float(self.data.time)

    def step(self, n: int = 1):
        t0 = time.perf_counter()
        for _ in range(n):
            # TODO(2단계~): 외력 훅 — qfrc_applied 초기화 후 부력/항력 등 적용
            mujoco.mj_step(self.model, self.data)
        self.step_wall_time += time.perf_counter() - t0
        self.n_steps += n

    def realtime_factor(self) -> float:
        """시뮬레이션 시간 / 실제 걸린 시간. 1.0 이상이면 실시간보다 빠름."""
        if self.step_wall_time == 0:
            return float("nan")
        return self.n_steps * self.model.opt.timestep / self.step_wall_time
