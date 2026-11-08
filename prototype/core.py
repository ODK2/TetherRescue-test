"""MuJoCo 물가 구조 시뮬레이션 코어 (ROS 의존성 없음).

MuJoCo는 기본적으로 '물'이라는 개념이 없으므로, 이 파일에서 매 물리 스텝마다
  1) 부력   : 몸체를 몇 개의 '슬라이스'로 나눠 잠긴 부피만큼 위로 힘
  2) 항력   : 물에 잠긴 비율만큼 선형+2차 저항 (보트는 전후/좌우 저항이 다름)
  3) 파도   : 수면 높이를 사인파로 바꿔 부력 계산에 반영
  4) 로프   : 보트 tow_point ↔ 익수자 harness 사이 '당길 때만 힘이 생기는' 스프링-댐퍼
  5) 패닉   : 익수자 FSM(PANIC → EXHAUSTED → TOWED → RESCUED)에 따른 허우적임 힘
을 계산해서 qfrc_applied 에 넣는다.

데이터 필드 이름은 TetherRescue SRS 5장(RescueeStatus, TetherData)을 따른다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

MODEL_PATH = Path(__file__).parent / "models" / "rescue_scene.xml"

RHO_WATER = 1000.0
G = 9.81


@dataclass
class SimParams:
    # 파도
    wave_amp: float = 0.08          # m
    wave_length: float = 6.0        # m
    wave_period: float = 3.0        # s
    # 로프 (Tether)
    rope_length: float = 3.0        # m, 이 길이까지는 느슨함
    rope_k: float = 1500.0          # N/m
    rope_c: float = 150.0           # N·s/m (늘어나는 방향일 때만)
    attach_dist: float = 2.2        # m, 보트 중심-익수자 수평거리가 이 안이면 접촉 성공
    attach_max_speed: float = 1.5   # m/s, 너무 빠르게 지나가면 접촉 실패
    rescue_radius: float = 4.0      # m, 익수자가 rescue_point 이 안에 오면 구조 완료
    # 익수자
    panic_force: float = 80.0       # N, 수평 허우적임 최대
    panic_sink_force: float = 140.0 # N, 패닉 시 몸을 물속으로 누르는 주기적 힘
    stamina_drain: float = 0.012    # 1/s (panic=1 기준)
    victim_buoyancy: float = 0.92   # 구명조끼/폐 부피 보정 계수
    seed: int = 0


@dataclass
class RescueeStatus:
    position: tuple[float, float, float]
    fsm_state: str
    panic_level: float
    stamina: float
    is_conscious: bool
    airway_submerged: bool


@dataclass
class TetherData:
    timestamp: float
    tension_force: float
    robot_accel_x: float
    robot_accel_y: float
    robot_accel_z: float
    thrust_cmd: float
    steer_cmd: float

    def as_list(self) -> list[float]:
        return [self.timestamp, self.tension_force, self.robot_accel_x,
                self.robot_accel_y, self.robot_accel_z, self.thrust_cmd, self.steer_cmd]


@dataclass
class _Slices:
    """부력 계산용 슬라이스: 몸체 좌표계 위치, 부피, 세로 높이."""
    body_id: int
    local_pos: np.ndarray   # (n, 3)
    volume: np.ndarray      # (n,)
    height: np.ndarray      # (n,)
    scale: float = 1.0


class RescueSim:
    PHASES = ("APPROACH", "TOW", "RESCUED")

    def __init__(self, params: SimParams | None = None, model_path: Path = MODEL_PATH):
        self.p = params or SimParams()
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        m = self.model

        self.boat_id = m.body("boat").id
        self.victim_id = m.body("victim").id
        self.tow_site = m.site("tow_point").id
        self.harness_site = m.site("harness").id
        self.airway_site = m.site("airway").id
        self.rescue_site = m.site("rescue_point").id
        self.accel_adr = m.sensor_adr[m.sensor("boat_accel").id]
        self.gyro_adr = m.sensor_adr[m.sensor("boat_gyro").id]

        # 보트: 선체 박스(2.4 x 1.0 x 0.36)를 네 모서리 슬라이스로 → 롤/피치 복원력
        hx, hy, hz = 0.9, 0.35, 0.18
        hull_vol = 2.4 * 1.0 * 0.36
        self.boat_slices = _Slices(
            self.boat_id,
            np.array([[hx, hy, 0], [hx, -hy, 0], [-hx, hy, 0], [-hx, -hy, 0]], float),
            np.full(4, hull_vol / 4), np.full(4, 2 * hz))
        # 익수자: 몸통 3조각 + 머리. 부력 중심이 무게중심보다 위 → 서 있는 자세 유지
        torso_vol = math.pi * 0.18**2 * 0.64 + 4 / 3 * math.pi * 0.18**3
        head_vol = 4 / 3 * math.pi * 0.12**3
        self.victim_slices = _Slices(
            self.victim_id,
            np.array([[0, 0, -0.333], [0, 0, 0.0], [0, 0, 0.333], [0, 0, 0.58]], float),
            np.array([torso_vol / 3] * 3 + [head_vol]),
            np.array([0.333, 0.333, 0.333, 0.24]),
            scale=self.p.victim_buoyancy)

        self.rng = np.random.default_rng(self.p.seed)
        self.reset()

    # ------------------------------------------------------------------ API
    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.phase = "APPROACH"
        self.tension = 0.0
        self.attached = False
        self.panic_level = 1.0
        self.stamina = 1.0
        self.fsm_state = "PANIC"
        self._panic_vec = np.zeros(2)
        self._panic_phase = self.rng.uniform(0, 2 * math.pi)
        self.cmd = (0.0, 0.0)
        self.events: list[tuple[float, str]] = []

    @property
    def time(self) -> float:
        return float(self.data.time)

    def set_cmd(self, thrust: float, steer: float):
        """thrust, steer ∈ [-1, 1]. ROS2에서는 /rescue/cmd(Twist) 가 여기로 들어온다."""
        self.cmd = (float(np.clip(thrust, -1, 1)), float(np.clip(steer, -1, 1)))

    def step(self, n: int = 1):
        """물리 스텝 n번 (dt=0.002s). 50Hz 제어라면 n=10."""
        for _ in range(n):
            self._physics_substep()
        self._update_logic()

    # --------------------------------------------------------------- 상태 조회
    def boat_pose(self):
        """(x, y, yaw, 전진속도, yaw rate)"""
        d = self.data
        x, y = d.xpos[self.boat_id][:2]
        R = d.xmat[self.boat_id].reshape(3, 3)
        yaw = math.atan2(R[1, 0], R[0, 0])
        v_world = d.qvel[0:3]
        v_fwd = float(R[:, 0] @ v_world)
        yaw_rate = float(d.sensordata[self.gyro_adr + 2])
        return float(x), float(y), yaw, v_fwd, yaw_rate

    def victim_position(self) -> np.ndarray:
        return self.data.xpos[self.victim_id].copy()

    def rescue_point(self) -> np.ndarray:
        return self.data.site_xpos[self.rescue_site].copy()

    def rescuee_status(self) -> RescueeStatus:
        airway_z = self.data.site_xpos[self.airway_site][2]
        surf = self.surface_height(*self.data.site_xpos[self.airway_site][:2])
        return RescueeStatus(
            position=tuple(float(v) for v in self.victim_position()),
            fsm_state=self.fsm_state,
            panic_level=round(self.panic_level, 3),
            stamina=round(self.stamina, 3),
            is_conscious=self.stamina > 0.0,
            airway_submerged=bool(airway_z < surf + 0.02))

    def tether_data(self) -> TetherData:
        a = self.data.sensordata[self.accel_adr:self.accel_adr + 3]
        return TetherData(self.time, float(self.tension), float(a[0]), float(a[1]),
                          float(a[2]), self.cmd[0], self.cmd[1])

    def surface_height(self, x: float, y: float) -> float:
        p = self.p
        k = 2 * math.pi / p.wave_length
        w = 2 * math.pi / p.wave_period
        t = self.time
        return p.wave_amp * (math.sin(k * x - w * t) + 0.5 * math.sin(0.7 * k * y + 0.6 * w * t))

    # -------------------------------------------------------------- 내부 물리
    def _physics_substep(self):
        m, d = self.model, self.data
        d.qfrc_applied[:] = 0.0
        d.ctrl[0], d.ctrl[1] = self.cmd

        sub_boat = self._apply_buoyancy(self.boat_slices)
        sub_vic = self._apply_buoyancy(self.victim_slices)
        self._apply_boat_drag(sub_boat)
        self._apply_victim_drag(sub_vic)
        self._apply_panic()
        self._apply_tether()
        mujoco.mj_step(m, d)

    def _apply_buoyancy(self, s: _Slices) -> float:
        """잠긴 부피에 비례한 부력을 슬라이스 위치에 가한다. 반환: 잠긴 비율(0~1)."""
        d = self.data
        R = d.xmat[s.body_id].reshape(3, 3)
        origin = d.xpos[s.body_id]
        sub_vol = 0.0
        for lp, vol, h in zip(s.local_pos, s.volume, s.height):
            wp = origin + R @ lp
            depth = self.surface_height(wp[0], wp[1]) - (wp[2] - h / 2)
            frac = min(max(depth / h, 0.0), 1.0)
            if frac <= 0:
                continue
            v = vol * frac * s.scale
            sub_vol += vol * frac
            self._apply_force(np.array([0, 0, RHO_WATER * G * v]), wp, s.body_id)
        return sub_vol / float(s.volume.sum())

    def _apply_boat_drag(self, sub: float):
        if sub <= 0:
            return
        d = self.data
        R = d.xmat[self.boat_id].reshape(3, 3)
        v_b = R.T @ d.qvel[0:3]                     # 보트 좌표계 속도
        lin = np.array([40.0, 400.0, 900.0])        # 전진은 잘 미끄러지고, 옆/위아래는 저항 큼
        quad = np.array([56.0, 600.0, 300.0])
        f_b = -(lin * v_b + quad * v_b * np.abs(v_b))
        w_b = d.qvel[3:6]                           # freejoint 각속도는 이미 몸체 좌표계
        t_b = -np.array([600.0, 600.0, 400.0]) * w_b
        k = min(1.0, sub * 4)                       # 흘수(약 25%)에서 이미 최대 저항
        self._apply_force(R @ f_b * k, d.xipos[self.boat_id], self.boat_id, R @ t_b * k)

    def _apply_victim_drag(self, sub: float):
        if sub <= 0:
            return
        d = self.data
        v = d.qvel[6:9]
        f = -(np.array([60.0, 60.0, 250.0]) * v + np.array([120.0, 120.0, 150.0]) * v * np.abs(v))
        R = d.xmat[self.victim_id].reshape(3, 3)
        t = -30.0 * (R @ d.qvel[9:12])              # 몸체 좌표계 각속도 → 월드
        self._apply_force(f * sub, d.xipos[self.victim_id], self.victim_id, t * sub)

    def _apply_panic(self):
        """허우적임: 수평으로는 랜덤 드리프트(OU 노이즈), 수직으로는 주기적으로 몸을 누름."""
        if self.panic_level <= 0:
            return
        dt = self.model.opt.timestep
        self._panic_vec += -1.5 * self._panic_vec * dt + self.rng.normal(0, 1.2, 2) * math.sqrt(dt)
        f_xy = self.p.panic_force * self.panic_level * np.clip(self._panic_vec, -1, 1)
        bob = max(0.0, math.sin(2 * math.pi * self.time / 1.4 + self._panic_phase))
        f_z = -self.p.panic_sink_force * self.panic_level * bob
        d = self.data
        self._apply_force(np.array([f_xy[0], f_xy[1], f_z]), d.xipos[self.victim_id], self.victim_id)

    def _apply_tether(self):
        """한쪽 방향 스프링-댐퍼 로프: 길이 L 보다 늘어날 때만 장력 발생."""
        if not self.attached:
            self.tension = 0.0
            return
        m, d = self.model, self.data
        pa = d.site_xpos[self.tow_site]
        pb = d.site_xpos[self.harness_site]
        diff = pb - pa
        dist = float(np.linalg.norm(diff))
        stretch = dist - self.p.rope_length
        if stretch <= 0 or dist < 1e-6:
            self.tension = 0.0
            return
        u = diff / dist
        # 두 점의 상대속도(로프 방향 성분)
        va = np.zeros(6); vb = np.zeros(6)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_SITE, self.tow_site, va, 0)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_SITE, self.harness_site, vb, 0)
        rel_v = float((vb[3:] - va[3:]) @ u)
        T = max(0.0, self.p.rope_k * stretch + self.p.rope_c * rel_v)
        self.tension = T
        self._apply_force(T * u, pa.copy(), self.boat_id)       # 보트는 익수자 쪽으로 끌림
        self._apply_force(-T * u, pb.copy(), self.victim_id)    # 익수자는 보트 쪽으로 끌림

    def _apply_force(self, force, point, body, torque=None):
        if torque is None:
            torque = np.zeros(3)
        mujoco.mj_applyFT(self.model, self.data, np.asarray(force, float),
                          np.asarray(torque, float), np.asarray(point, float),
                          body, self.data.qfrc_applied)

    # ---------------------------------------------------------- 시나리오 로직
    def _update_logic(self):
        dt = 0.02  # step() 1회 ≈ 50Hz 기준으로 FSM 갱신
        bx, by, _, v_fwd, _ = self.boat_pose()
        vic = self.victim_position()

        # 익수자 FSM
        if self.fsm_state == "PANIC":
            self.stamina = max(0.0, self.stamina - self.p.stamina_drain * self.panic_level * dt)
            if self.stamina < 0.25:
                self.fsm_state = "EXHAUSTED"
        elif self.fsm_state == "EXHAUSTED":
            self.panic_level = max(0.2, self.panic_level - 0.05 * dt)
            self.stamina = max(0.0, self.stamina - 0.004 * dt)
            self.victim_slices.scale = self.p.victim_buoyancy * 0.9   # 기운이 빠져 더 가라앉음
        elif self.fsm_state == "TOWED":
            self.panic_level = max(0.0, self.panic_level - 0.15 * dt)  # 로프를 잡고 진정

        # 접촉 → 로프 연결
        if self.phase == "APPROACH":
            dist = math.hypot(vic[0] - bx, vic[1] - by)
            if dist < self.p.attach_dist and abs(v_fwd) < self.p.attach_max_speed:
                self.attached = True
                self.phase = "TOW"
                self.fsm_state = "TOWED"
                self.events.append((self.time, "CONTACT: tether attached"))
        elif self.phase == "TOW":
            rp = self.rescue_point()
            if math.hypot(vic[0] - rp[0], vic[1] - rp[1]) < self.p.rescue_radius:
                self.phase = "RESCUED"
                self.fsm_state = "RESCUED"
                self.panic_level = 0.0
                self.events.append((self.time, "RESCUED: victim at dock"))
