"""MuJoCo 월드 래퍼: 씬 로딩, 스텝 루프, 성능 측정의 공통 틀.

단계가 진행되면 여기에 '매 스텝 외력 계산 훅'(부력·항력·로프 등)이 붙는다.
"""
from __future__ import annotations

import time
from pathlib import Path

import mujoco

from .. import ASSETS_DIR
from ..ocean.config import OceanConfig

DEFAULT_SCENE = ASSETS_DIR / "ocean_scene.xml"


class World:
    def __init__(self, scene: str | Path = DEFAULT_SCENE, ocean: OceanConfig | None = None):
        """ocean 을 주면 yaml 값으로 씬을 고친 뒤 컴파일한다 (yaml 이 기준, xml 은 기본값).

        MjSpec: xml 을 '컴파일 전' 편집 가능한 객체로 읽는 MuJoCo API.
        hfield 칸 수처럼 컴파일 후엔 못 바꾸는 값을 여기서 바꾼다.
        """
        spec = mujoco.MjSpec.from_file(str(scene))
        if ocean is not None:
            self._apply_ocean(spec, ocean)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.step_wall_time = 0.0   # 누적 스텝 소요 시간 (실시간 배속 계산용)
        self.n_steps = 0

    @staticmethod
    def _apply_ocean(spec: mujoco.MjSpec, ocean: OceanConfig):
        d = ocean.domain
        for name in ("water", "seabed"):
            hf = spec.hfield(name)
            size = list(hf.size)
            size[0], size[1] = d.size_x / 2.0, d.size_y / 2.0   # MuJoCo size 는 '반' 길이
            hf.size = size
        water = spec.hfield("water")
        water.nrow = water.ncol = d.surface_res
        # xml 로딩 시 nrow*ncol 길이의 빈 데이터가 붙어 있어서, 칸 수를 바꾸면 길이도 맞춰야 한다
        water.userdata = [0.0] * (d.surface_res * d.surface_res)

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
