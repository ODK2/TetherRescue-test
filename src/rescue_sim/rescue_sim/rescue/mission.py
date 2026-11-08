"""자율 구조 미션: 접근 → 로프 연결 → 물러남 → 해안으로 끌기 → 로프 감기 → 풀어 줌 → 사람이 걸어서 뭍으로.

  APPROACH : 사람 앞 3.5 m 까지 접근 (선수가 사람을 향함)
  CONNECT  : 로프 끝이 reach 안에 들어오면 1초 동안 로프를 건다
  CLEAR    : 후진으로 clear_dist 만큼 떨어진다 — 바로 U턴하면 선수가 그리는 원에 사람이 걸린다
             (로프 길이 > reach 라서 후진해도 로프는 아직 느슨)
  TOW      : 제자리 회전 후 해안으로. 보트가 바닥에 안 닿는 수심(hold_x)까지만 간다
  WINCH    : 보트는 그 자리를 지키고(추진기로 버팀) 로프를 감아 사람을 선미 가까이 끌어온다
             = 로프 길이 상한을 조금씩 줄임. 그래도 발이 안 닿으면 아주 천천히 더 들어간다
  RELEASE  : 로프를 풀고, 보트는 옆으로 비켜 사람이 걸어갈 길을 비운다
  WALK     : 사람이 일어서서 모래사장(walk_goal_x)까지 걸어간다 (person.py 애니메이션)

단순화
  - 사람 위치는 정확히 안다고 가정 (실제로는 카메라·GPS 추정 + 오차)
  - '로프 연결' = 로프 끝(tow_point)이 사람에게 충분히 가깝고 보트가 느리면 즉시 연결
    (실제로는 사람이 구조링을 잡거나 로봇 팔/그물이 필요)
  - 장애물·다른 배·통신 지연 없음

조향: 목표 방향과 현재 방향의 차이(heading error)에 비례 + 회전 속도로 제동 (PD).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from ..boat import Boat
from .person import Person
from .rope import Rope


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


@dataclass
class Autopilot:
    kp: float = 1.2           # 방향 오차 1 rad → steer 1.2
    kd: float = 0.8           # 회전 속도 제동
    def heading_cmd(self, boat: Boat, target_xy, max_thrust: float, slow_radius: float,
                    stop_radius: float = 0.0, closing: float | None = None) -> tuple[float, float, float]:
        """목표점으로 향하는 (thrust, steer) 와 거리.

        closing: 목표에 가까워지는 속도 [m/s]. 목표(사람)도 해류에 떠내려가므로
        땅 기준 배 속도(surge) 대신 이걸로 제동을 판단한다. None 이면 surge 사용.
        """
        s = boat.state()
        dx, dy = target_xy[0] - s.x, target_xy[1] - s.y
        dist = math.hypot(dx, dy)
        err = wrap(math.atan2(dy, dx) - s.yaw)
        steer = float(np.clip(self.kp * err - self.kd * s.yaw_rate, -1, 1))
        # 멀면 최대, 가까우면 감속, 방향이 크게 틀리면 제자리 회전 위주
        speed = max_thrust * min(1.0, max(0.0, dist - stop_radius) / slow_radius)
        thrust = speed * max(0.0, math.cos(err)) ** 2
        # 너무 빠르게 다가가면 역추진으로 제동 (추진 0 이어도 배는 관성으로 몇 m 더 간다)
        approach = s.surge if closing is None else closing
        if dist < stop_radius + 2.0 and approach > 0.3:
            thrust = -0.5
        return balance_thrust(boat, thrust, steer), steer, dist


def balance_thrust(boat: Boat, thrust: float, steer: float) -> float:
    """조향해도 '앞으로 미는 합력'이 thrust 가 뜻하는 만큼만 나오게 thrust 를 고친다.

    추진기는 후진 추력(150 N)이 전진(300 N)의 절반이라, 제자리 회전(한쪽 +1, 한쪽 −1)을 하면
    합력이 +150 N 남아 배가 앞으로 밀려 나간다 (해안을 보고 돌면 모래톱에 올라감).
    합력은 thrust 에 대해 단조 증가 → 이분법으로 맞는 값을 찾는다.
    """
    def net(t: float) -> float:
        return sum(boat._thruster_force(min(max(t + k * steer, -1.0), 1.0)) for k in (-1, 1))

    want = 2 * boat._thruster_force(thrust)
    lo, hi = -1.0, 1.0
    if net(lo) >= want:
        return lo
    if net(hi) <= want:
        return hi
    for _ in range(20):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if net(mid) < want else (lo, mid)
    return 0.5 * (lo + hi)


class RescueMission:
    APPROACH, CONNECT, CLEAR, TOW, DONE = "APPROACH", "CONNECT", "CLEAR", "TOW", "DONE"
    WINCH, RELEASE, WALK = "WINCH", "RELEASE", "WALK"

    def __init__(self, boat: Boat, person: Person, rope: Rope, hold_x: float = -7.0,
                 reach: float = 5.0, clear_dist: float = 4.5, pilot: Autopilot | None = None,
                 winch_speed: float = 0.4, winch_gap: float = 2.0, min_keel_depth: float = 0.35,
                 walk_goal_x: float = -16.0):
        self.boat, self.person, self.rope = boat, person, rope
        self.hold_x = hold_x                # 끌고 갈 때 보트가 멈출 x (수심 ~0.6 m)
        self.winch_speed = winch_speed      # 로프 감는 속도 [m/s]
        self.winch_gap = winch_gap          # 로프 끝 ↔ 사람 이만큼까지 감는다 [m]
        self.min_keel_depth = min_keel_depth  # 보트 밑 최소 수심 (이보다 얕은 곳엔 안 들어감)
        self.walk_goal_x = walk_goal_x      # 사람이 걸어갈 마른 모래 위 x
        # 해변 hfield 는 y 로 ±half 까지만 있다 → 걷기 목표·보트 비킬 곳을 그 안으로 제한
        m = boat.world.model
        self._y_lim = float(m.hfield_size[m.hfield("seabed").id, 1]) - 2.0
        self.reach = reach              # 로프를 걸 수 있는 거리 (로프 끝 ↔ 사람)
        self.clear_dist = clear_dist    # 회전 전에 확보할 거리 (선체 중심 ↔ 사람)
        m = boat.world.model
        self._hull_geom, self._person_geom = m.geom("hull").id, m.geom("person_body").id
        self._tow_site = m.site("tow_point").id
        self.min_clearance = math.inf   # 미션 동안 선체 ↔ 사람 최소 간격 [m]
        self._prev = None               # (t, 사람까지 거리) — 가까워지는 속도 계산용
        self.closing = 0.0
        self.pilot = pilot or Autopilot()
        self.state = self.APPROACH
        self.t_state = 0.0
        self.log: list[tuple[float, str]] = []

    def _go(self, state: str, t: float):
        self.state, self.t_state = state, t
        self.log.append((t, state))

    def tow_gap(self) -> float:
        d = self.boat.world.data
        return float(np.linalg.norm(d.site_xpos[self._tow_site] - d.site_xpos[self.person.harness_site]))

    def clearance(self) -> float:
        """선체 ↔ 사람 표면 사이 최단 거리 (MuJoCo 충돌 검사기 사용, 2 m 넘으면 2 m 로 자름)."""
        w = self.boat.world
        return float(mujoco.mj_geomDistance(w.model, w.data, self._hull_geom, self._person_geom, 2.0, None))

    def update(self, t: float):
        """제어 주기마다 호출 (예: 25 Hz)."""
        p = self.person.pos
        s = self.boat.state()
        self.min_clearance = min(self.min_clearance, self.clearance())
        dist = math.hypot(p[0] - s.x, p[1] - s.y)
        if self._prev is not None and t > self._prev[0]:
            raw = -(dist - self._prev[1]) / (t - self._prev[0])
            self.closing += 0.3 * (raw - self.closing)        # 파도 흔들림 때문에 살짝 거른다
        self._prev = (t, dist)
        if self.state == self.APPROACH:
            # 사람 앞 3.5 m 에 멈추도록 (선체 중심 기준). 선수 1.2 m + 사람 반 길이 0.85 m = 2.05 m 면 닿음
            th, st, dist = self.pilot.heading_cmd(self.boat, p[:2], 0.8, slow_radius=8.0, stop_radius=3.5,
                                                    closing=self.closing)
            self.boat.set_cmd(th, st)
            if self.tow_gap() < self.reach and abs(self.closing) < 0.3:
                self._go(self.CONNECT, t)
        elif self.state == self.CONNECT:
            # 1초 = 로프 거는 시간. 그동안 제자리 유지 (앞으로 미끄러지면 역추진)
            c = self.closing
            self.boat.set_cmd(-0.4 if c > 0.05 else (0.2 if c < -0.2 else 0.0), 0.0)
            if t - self.t_state > 1.0:
                self.rope.attach()
                self._go(self.CLEAR, t)
        elif self.state == self.CLEAR:
            # 사람 쪽을 바라본 채로 후진 (방향 유지는 같은 PD 조향)
            _, st, _ = self.pilot.heading_cmd(self.boat, p[:2], 0.0, slow_radius=1.0)
            self.boat.set_cmd(balance_thrust(self.boat, -0.4 if s.surge > -0.6 else 0.0, st), st)
            if dist > self.clear_dist or t - self.t_state > 10.0:
                self._go(self.TOW, t)
        elif self.state == self.TOW:
            # 멈출 선(hold_x)까지 남은 거리로 감속 — 전속으로 가면 관성으로 얕은 데까지 미끄러진다
            th, st, _ = self.pilot.heading_cmd(self.boat, (self.hold_x - 3.0, p[1]), 0.7, slow_radius=4.0)
            to_go = s.x - self.hold_x
            th = min(th, 0.7 * max(0.15, min(1.0, to_go / 8.0)))
            if to_go < 3.0 and s.surge > 0.5:
                th = -0.4
            # 로프가 느슨한 동안엔 살살 (팽팽해지는 순간의 충격 장력 줄이기)
            if self.rope.span() < self.rope.length - 0.2:
                th = min(th, 0.2)
            self.boat.set_cmd(balance_thrust(self.boat, th, st), st)
            if to_go < 0.3:
                self._winch_len = min(self.rope.length, self.rope.span())   # 느슨한 만큼은 건너뛴다
                self._go(self.WINCH, t)
        elif self.state == self.WINCH:
            dt = t - self._prev_t if hasattr(self, "_prev_t") else 0.04
            if self.tow_gap() > self.winch_gap:
                self._winch_len = max(self.winch_gap, self._winch_len - self.winch_speed * dt)
                self.rope.attach(self._winch_len)
                x_goal = self.hold_x
            else:
                # 다 감았는데 아직 발이 안 닿으면 보트째 조금 더 얕은 곳으로 (바닥 닿기 직전까지)
                keel_ok = self.boat_depth(s.x - 0.5) > self.min_keel_depth
                x_goal = s.x - 0.5 if keel_ok and not self.person.can_stand() else s.x
                if self.person.can_stand() or not keel_ok:
                    self._go(self.RELEASE, t)
                    self.rope.detach()
                    self._aside_y = float(np.clip(p[1] + (-1.0 if p[1] > 0 else 1.0) * 8.0,
                                                  -self._y_lim, self._y_lim))
            self.boat.set_cmd(*self._hold(s, x_goal))
        elif self.state == self.RELEASE:
            # 사람 앞(해안 쪽)에 있는 보트가 옆으로 비킨다. 다 비키면 사람이 일어선다
            th, st, _ = self.pilot.heading_cmd(self.boat, (self.hold_x, self._aside_y), 0.5,
                                               slow_radius=3.0, stop_radius=0.5)
            self.boat.set_cmd(th, st)
            if abs(s.y - p[1]) > 2.5:
                self.person.stand_up((self.walk_goal_x, float(np.clip(p[1], -self._y_lim, self._y_lim))))
                self._go(self.WALK, t)
        elif self.state == self.WALK:
            th, st, _ = self.pilot.heading_cmd(self.boat, (self.hold_x, self._aside_y), 0.5,
                                               slow_radius=3.0, stop_radius=0.5)
            self.boat.set_cmd(th, st)
            if self.person.mode == self.person.ARRIVED:
                self._go(self.DONE, t)
        else:
            self.boat.set_cmd(0.0, 0.0)
        self._prev_t = t

    def boat_depth(self, x: float) -> float:
        return float(self.boat.world.bathymetry.depth(np.array([x]))[0])

    def _hold(self, s, x_goal: float) -> tuple[float, float]:
        """해안을 바라본 채 x_goal 에 버티기. 로프 장력만큼 미리 앞으로 밀어 준다(feedforward)."""
        _, st, _ = self.pilot.heading_cmd(self.boat, (x_goal - 10.0, s.y), 0.0, slow_radius=1.0)
        e = s.x - x_goal                                   # +: 바다 쪽으로 밀려남 → 앞으로
        th = 0.6 * e - 0.6 * s.surge + self.rope.tension() / 600.0
        return balance_thrust(self.boat, float(np.clip(th, -0.5, 0.8)), st), st
