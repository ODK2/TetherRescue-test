"""MuJoCo 월드 래퍼: 씬 로딩, 스텝 루프, 성능 측정의 공통 틀.

단계가 진행되면 여기에 '매 스텝 외력 계산 훅'(부력·항력·로프 등)이 붙는다.
"""
from __future__ import annotations

import time
from pathlib import Path

import mujoco
import numpy as np

from .. import ASSETS_DIR
from ..ocean.bathymetry import Bathymetry
from ..ocean.config import OceanConfig

DEFAULT_SCENE = ASSETS_DIR / "ocean_scene.xml"


class World:
    def __init__(self, scene: str | Path = DEFAULT_SCENE, ocean: OceanConfig | None = None):
        """ocean 을 주면 yaml 값으로 씬을 고친 뒤 컴파일한다 (yaml 이 기준, xml 은 기본값).

        MjSpec: xml 을 '컴파일 전' 편집 가능한 객체로 읽는 MuJoCo API.
        hfield 칸 수처럼 컴파일 후엔 못 바꾸는 값을 여기서 바꾼다.
        """
        spec = mujoco.MjSpec.from_file(str(scene))
        self.bathymetry = Bathymetry(ocean.bathymetry) if ocean is not None else None
        if ocean is not None:
            self._apply_ocean(spec, ocean, self.bathymetry)
        self.model = spec.compile()
        if self.bathymetry is not None:
            self._write_seabed(self.model, self.bathymetry)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.step_wall_time = 0.0   # 누적 스텝 소요 시간 (실시간 배속 계산용)
        self.n_steps = 0

    @staticmethod
    def _apply_ocean(spec: mujoco.MjSpec, ocean: OceanConfig, bathy: Bathymetry):
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

        # 해저: 높이 범위와 위치는 '컴파일 전에' 정한다.
        # 바닥은 충돌하는 geom 이라, MuJoCo 가 컴파일 때 계산하는 충돌 경계(AABB)와 맞아야 한다.
        z_min, z_max = bathy.z_range
        seabed = spec.hfield("seabed")
        size = list(seabed.size)
        size[2] = z_max - z_min
        seabed.size = size
        geom = spec.geom("seabed")
        geom.pos = [geom.pos[0], geom.pos[1], z_min]

    @staticmethod
    def _write_seabed(model: mujoco.MjModel, bathy: Bathymetry):
        """해저 hfield 데이터 = Bathymetry.seabed_z 를 0~1 로 바꾼 것 (행=y, 열=x).

        컴파일 후에 쓰는 이유: MjSpec 의 userdata 는 컴파일 때 최소~최대로 다시 늘려 버려서
        (영역 안에 깊은 곳·모래사장 끝이 다 안 들어오면) 높이가 틀어진다. 컴파일 후 직접 쓰면 그대로 들어간다.
        """
        hid = model.hfield("seabed").id
        gid = model.geom("seabed").id
        nrow, ncol, adr = int(model.hfield_nrow[hid]), int(model.hfield_ncol[hid]), int(model.hfield_adr[hid])
        sx, sy, size_z = model.hfield_size[hid, :3]
        gx, gy, gz = model.geom_pos[gid]
        X, _ = np.meshgrid(np.linspace(-sx, sx, ncol) + gx, np.linspace(-sy, sy, nrow) + gy)
        z = bathy.seabed_z(X)
        model.hfield_data[adr: adr + nrow * ncol] = ((z - gz) / size_z).ravel()

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
