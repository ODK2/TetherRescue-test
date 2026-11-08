"""[3단계] 로봇 구조 보트: MuJoCo 몸체 + 유체력(hydro) + 추진기.

  world = World(BOAT_SCENE, ocean=cfg)
  boat  = Boat(world, waves, current)     # world.pre_step 에 유체력 계산을 등록
  boat.set_cmd(thrust=0.8, steer=0.2)    # [-1, 1]
  world.step(10)

추진 방식: 좌우 추진기 2개의 추력 차이로 방향 전환 (differential thrust, 소형 USV 에서 흔함)
  왼쪽 = thrust − steer,  오른쪽 = thrust + steer   (steer > 0 → 왼쪽으로 회전)
추진기가 물 밖으로 나오면(파도에 선미가 들리면) 그만큼 추력이 줄어든다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

from ..hydro.hull import HullHydro, HullParams, HullState
from ..ocean.current import Current
from ..ocean.waves import WaveField
from ..sim.world import World


@dataclass
class BoatParams:
    hull: HullParams = field(default_factory=HullParams)
    max_thrust: float = 300.0       # 추진기 1개 최대 전진 추력 [N]
    max_reverse: float = 150.0      # 후진 최대 [N]
    thruster_depth: float = 0.08    # 추진기가 이만큼 잠겨야 추력 100% [m]


@dataclass
class BoatState:
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float
    surge: float        # 전진 속도 [m/s] (배 좌표계 x)
    sway: float         # 옆 속도
    yaw_rate: float     # [rad/s]


class Boat:
    def __init__(self, world: World, waves: WaveField, current: Current | None = None,
                 params: BoatParams | None = None, name: str = "boat"):
        self.world = world
        self.p = params or BoatParams()
        m = world.model
        self.body_id = m.body(name).id
        self.hydro = HullHydro(self.p.hull, waves, current)
        self.waves = waves
        self._act = [m.actuator("thr_l").id, m.actuator("thr_r").id]
        self._thr_sites = [m.site("thr_l").id, m.site("thr_r").id]
        self.cmd = (0.0, 0.0)
        self.last = None            # 마지막 HydroResult (디버그/텔레메트리)
        self._check_hull_matches_geom(m, name)
        world.pre_step.append(self._pre_step)

    def _check_hull_matches_geom(self, m: mujoco.MjModel, name: str):
        """xml 선체 크기와 부력 계산 크기가 다르면 바로 알려준다 (흔한 실수)."""
        half = m.geom_size[m.geom("hull").id]
        h = self.p.hull
        if not np.allclose(2 * half, [h.length, h.beam, h.height]):
            raise ValueError(f"xml 선체 {2 * half} ≠ HullParams {(h.length, h.beam, h.height)}")

    # ------------------------------------------------------------- 명령
    def set_cmd(self, thrust: float, steer: float):
        """thrust, steer ∈ [-1, 1]."""
        self.cmd = (float(np.clip(thrust, -1, 1)), float(np.clip(steer, -1, 1)))

    def _thruster_force(self, u: float) -> float:
        return u * (self.p.max_thrust if u >= 0 else self.p.max_reverse)

    # ------------------------------------------------------------- 매 물리 스텝
    def _pre_step(self, model: mujoco.MjModel, data: mujoco.MjData):
        b = self.body_id
        t = float(data.time)
        rot = data.xmat[b].reshape(3, 3)
        # freejoint qvel: [선속도(월드, 몸체 원점 기준), 각속도(몸체 좌표계)]
        # → mj_objectVelocity 로 무게중심 기준 월드 속도를 받는다
        vel6 = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, b, vel6, 0)
        s = HullState(pos=data.xpos[b], rot=rot, com=data.xipos[b], vel=vel6[3:], angvel=vel6[:3])
        thr = data.site_xpos[self._thr_sites]                          # (2, 3)
        r = self.hydro.compute(s, t, extra_xy=thr[:, :2])
        data.xfrc_applied[b, :3] = r.force
        data.xfrc_applied[b, 3:] = r.torque
        self.last = r

        thrust, steer = self.cmd
        for i, (side, aid) in enumerate(zip((-1, 1), self._act)):
            u = min(max(thrust + side * steer, -1.0), 1.0)
            wet = min(max((r.extra_eta[i] - thr[i, 2]) / self.p.thruster_depth, 0.0), 1.0)
            data.ctrl[aid] = self._thruster_force(u) * wet

    # ------------------------------------------------------------- 상태
    def state(self) -> BoatState:
        d = self.world.data
        b = self.body_id
        R = d.xmat[b].reshape(3, 3)
        x, y, z = d.xpos[b]
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = -math.asin(float(np.clip(R[2, 0], -1, 1)))
        yaw = math.atan2(R[1, 0], R[0, 0])
        vel6 = np.zeros(6)
        mujoco.mj_objectVelocity(self.world.model, d, mujoco.mjtObj.mjOBJ_BODY, b, vel6, 1)  # 몸체 좌표계
        return BoatState(float(x), float(y), float(z), roll, pitch, yaw,
                         float(vel6[3]), float(vel6[4]), float(vel6[2]))

    def thrust_now(self) -> tuple[float, float]:
        d = self.world.data
        return float(d.ctrl[self._act[0]]), float(d.ctrl[self._act[1]])
