"""물에 빠진 사람: 떠 있기(물리) → 일어서기 → 걸어서 뭍으로(애니메이션).

  FLOAT : 보트와 같은 기둥 부력 + 항력 (hydro/hull.py 재사용).
          구명조끼 입고 등을 대고 누운 자세의 직육면체 근사. 스스로 헤엄치지 않는다.
  STAND : stand_up() 호출 후 STAND_TIME 동안 누운 자세 → 선 자세로 부드럽게 바꾼다.
  WALK  : 발이 바닥에 닿은 채 목표점까지 걷는다. 물이 깊으면 느리게(물 헤치기), 땅에선 빠르게.
  ARRIVED : 도착, 서 있음.

MuJoCo 에서 알아 둘 것
  - STAND/WALK 는 물리가 아니라 '운동학적(kinematic)' 제어: 매 스텝 qpos/qvel 을 직접 써 넣는다.
    freejoint 몸체를 이렇게 움직이면 물리 엔진은 그 위치를 그대로 받아들인다.
    (충돌하면 서로 밀어내며 싸우므로 걷는 동안은 사람 충돌을 끈다: geom_contype = 0)
  - 다리·팔 흔들기 = model.geom_pos / geom_quat 를 실행 중에 바꿈 (관절 없이 겉모양만).
    mjModel 의 geom 위치는 컴파일 후에도 바꿀 수 있고, 다음 mj_kinematics 부터 화면에 반영된다.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from ..hydro.hull import HullHydro, HullParams, HullState
from ..ocean.current import Current
from ..ocean.waves import WaveField
from ..sim.world import World

PERSON_HULL = HullParams(
    length=1.7, beam=0.4, height=0.25, nx=4, ny=4,   # ny=2 면 옆으로 굴러 엎어짐
    lin_drag=(20.0, 80.0, 600.0), quad_drag=(40.0, 200.0, 200.0),
    ang_lin_drag=(15.0, 40.0, 30.0), ang_quad_drag=(5.0, 20.0, 15.0),
)

FOOT = 0.87            # 몸 중심 → 발바닥 (다리 끝 x=-0.8 + 반지름 0.07)
STAND_DEPTH = 1.4      # 이보다 얕으면 발이 닿아 설 수 있다 [m] (키 1.7 m, 턱 높이)
STAND_TIME = 2.0       # 일어서는 데 걸리는 시간 [s]
WADE_SPEED = 0.45      # 무릎 넘는 물속 걷기 [m/s]
WALK_SPEED = 1.1       # 땅 위 걷기
STRIDE = 0.7           # 한 걸음 [m]
SWING = math.radians(28)


def _slerp(q0: np.ndarray, q1: np.ndarray, a: float) -> np.ndarray:
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + a * (q1 - q0)
    else:
        th = math.acos(d)
        q = (math.sin((1 - a) * th) * q0 + math.sin(a * th) * q1) / math.sin(th)
    return q / np.linalg.norm(q)


def _upright_quat(heading: float) -> np.ndarray:
    """선 자세: 몸 x(머리) = 위, 몸 z(가슴) = 걷는 방향."""
    bx = np.array([0.0, 0.0, 1.0])
    bz = np.array([math.cos(heading), math.sin(heading), 0.0])
    by = np.cross(bz, bx)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.column_stack([bx, by, bz]).ravel())
    return q


def _quat_y(a: float) -> np.ndarray:
    return np.array([math.cos(a / 2), 0.0, math.sin(a / 2), 0.0])


class Person:
    FLOAT, STAND, WALK, ARRIVED = "FLOAT", "STAND", "WALK", "ARRIVED"

    def __init__(self, world: World, waves: WaveField, current: Current | None = None,
                 params: HullParams = PERSON_HULL, name: str = "person"):
        self.world = world
        m = world.model
        self.body_id = m.body(name).id
        self.harness_site = m.site("harness").id
        self._qadr = m.jnt_qposadr[m.body_jntadr[self.body_id]]
        self._dadr = m.jnt_dofadr[m.body_jntadr[self.body_id]]
        self._collide_geom = m.geom("person_body").id
        self.hydro = HullHydro(params, waves, current)
        self.bathy = world.bathymetry
        self.mode = self.FLOAT
        # 팔다리: (geom id, 회전 중심(엉덩이/어깨), 원래 pos, 원래 quat, 위상 부호)
        self._limbs = []
        for gname, pivot, sign in (("leg_l", (0.0, 0.09, -0.02), 1), ("leg_r", (0.0, -0.09, -0.02), -1),
                                   ("arm_l", (0.45, 0.2, 0.0), -1), ("arm_r", (0.45, -0.2, 0.0), 1)):
            g = m.geom(gname).id
            self._limbs.append((g, np.array(pivot), m.geom_pos[g].copy(), m.geom_quat[g].copy(), sign))
        world.pre_step.append(self._pre_step)

    # ------------------------------------------------------------ 외부 명령
    def place(self, x: float, y: float, yaw: float = 0.0):
        d = self.world.data
        a = self._qadr
        d.qpos[a: a + 3] = (x, y, 0.0)
        d.qpos[a + 3: a + 7] = (np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))
        mujoco.mj_forward(self.world.model, d)

    def ground_z(self, x: float) -> float:
        return float(self.bathy.seabed_z(np.array([x]))[0])

    def can_stand(self) -> bool:
        return -self.ground_z(self.pos[0]) < STAND_DEPTH

    def stand_up(self, goal_xy):
        """지금 자리에서 일어나 goal_xy 까지 걸어간다."""
        d = self.world.data
        a = self._qadr
        self.goal = np.asarray(goal_xy, dtype=float)
        p = d.qpos[a: a + 3].copy()
        self.heading = math.atan2(self.goal[1] - p[1], self.goal[0] - p[0])
        self._from = (p, d.qpos[a + 3: a + 7].copy())
        self._t0 = float(d.time)
        self._xy = p[:2].copy()
        self._phase = 0.0
        self.world.model.geom_contype[self._collide_geom] = 0
        self.world.model.geom_conaffinity[self._collide_geom] = 0
        self.mode = self.STAND

    # ------------------------------------------------------------ 매 물리 스텝
    def _pre_step(self, model: mujoco.MjModel, data: mujoco.MjData):
        b = self.body_id
        if self.mode == self.FLOAT:
            vel6 = np.zeros(6)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, b, vel6, 0)
            s = HullState(pos=data.xpos[b], rot=data.xmat[b].reshape(3, 3), com=data.xipos[b],
                          vel=vel6[3:], angvel=vel6[:3])
            r = self.hydro.compute(s, float(data.time))
            data.xfrc_applied[b, :3] = r.force
            data.xfrc_applied[b, 3:] = r.torque
            return

        data.xfrc_applied[b] = 0.0
        dt = model.opt.timestep
        speed = 0.0
        if self.mode == self.STAND:
            a = min(1.0, (float(data.time) - self._t0) / STAND_TIME)
            a = a * a * (3 - 2 * a)                                   # 부드럽게 시작·끝
            p0, q0 = self._from
            p1 = np.array([p0[0], p0[1], self.ground_z(p0[0]) + FOOT])
            pos, quat = p0 + a * (p1 - p0), _slerp(q0, _upright_quat(self.heading), a)
            if a >= 1.0:
                self.mode = self.WALK
        else:
            to_goal = self.goal - self._xy
            dist = float(np.linalg.norm(to_goal))
            if self.mode == self.WALK:
                water = max(0.0, -self.ground_z(self._xy[0]))           # 발밑 수심
                speed = WADE_SPEED if water > 0.3 else WALK_SPEED
                if dist < 0.05:
                    self.mode, speed = self.ARRIVED, 0.0
                else:
                    self._xy = self._xy + to_goal / dist * min(speed * dt, dist)
                    self._phase += speed * dt / STRIDE * math.pi
            bob = 0.02 * abs(math.sin(self._phase)) if speed > 0 else 0.0
            pos = np.array([self._xy[0], self._xy[1], self.ground_z(self._xy[0]) + FOOT + bob])
            quat = _upright_quat(self.heading)

        a = self._qadr
        data.qpos[a: a + 3] = pos
        data.qpos[a + 3: a + 7] = quat
        v = np.zeros(6)
        if self.mode == self.WALK and speed > 0:
            v[:2] = speed * np.array([math.cos(self.heading), math.sin(self.heading)])
        data.qvel[self._dadr: self._dadr + 6] = v
        self._swing(model, math.sin(self._phase) * SWING if self.mode == self.WALK else 0.0)

    def _swing(self, model: mujoco.MjModel, angle: float):
        """팔다리를 몸 y 축(좌우 축) 기준으로 흔든다. 다리 좌우 반대, 팔은 다리와 반대."""
        for g, pivot, pos0, quat0, sign in self._limbs:
            q = _quat_y(sign * angle)
            rel = np.zeros(3)
            mujoco.mju_rotVecQuat(rel, pos0 - pivot, q)
            model.geom_pos[g] = pivot + rel
            out = np.zeros(4)
            mujoco.mju_mulQuat(out, q, quat0)
            model.geom_quat[g] = out

    @property
    def pos(self) -> np.ndarray:
        return self.world.data.xpos[self.body_id].copy()

    @property
    def on_land(self) -> bool:
        return self.ground_z(self.pos[0]) > 0.0
