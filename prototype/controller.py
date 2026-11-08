"""구조 보트 제어기 (ROS 의존성 없음 — 노드와 단독 실행 스크립트가 같이 사용).

APPROACH : 익수자 쪽으로 방향을 맞추고, 가까워질수록 감속 (패닉 익수자에게 천천히 접근)
TOW      : 부두 옆 목표점으로 견인. 로프 장력이 한계를 넘으면 추진을 줄임
           (TetherSense '감속' 되먹임을 아주 단순화한 버전)
RESCUED  : 정지
"""
from __future__ import annotations

import math
from dataclasses import dataclass

TOW_TARGET = (0.5, 2.8)   # 부두 옆. 익수자가 로프 길이만큼 뒤따라와 부두 끝(rescue_point)에 닿음


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def clip(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


@dataclass
class ControllerGains:
    kp_yaw: float = 1.6
    kd_yaw: float = 0.8
    kp_speed: float = 0.8
    approach_vmax: float = 2.5     # m/s
    approach_stop: float = 1.2     # m, 익수자 앞 이 거리에서 정지 목표
    tow_vmax: float = 1.0          # m/s, 견인은 천천히 (기도 보호)
    tension_limit: float = 450.0   # N


class RescueController:
    def __init__(self, gains: ControllerGains | None = None):
        self.g = gains or ControllerGains()

    @staticmethod
    def thrust_feedforward(v: float) -> float:
        # core.py 보트 전진 항력(40v + 56v|v|)을 상쇄하는 추진 (최대 900N)
        return (40.0 * v + 56.0 * v * abs(v)) / 900.0

    def compute(self, phase: str, boat: tuple, victim_xy: tuple, tension: float = 0.0):
        """boat = (x, y, yaw, v_fwd, yaw_rate). 반환: (thrust, steer) ∈ [-1, 1]."""
        x, y, yaw, v_fwd, yaw_rate = boat
        g = self.g
        if phase == "RESCUED":
            return clip(-0.8 * v_fwd, -1, 1), clip(-g.kd_yaw * yaw_rate, -1, 1)

        if phase == "APPROACH":
            tx, ty = victim_xy
            dist = math.hypot(tx - x, ty - y)
            v_des = clip(0.6 * (dist - g.approach_stop), 0.0, g.approach_vmax)
        else:  # TOW
            tx, ty = TOW_TARGET
            dist = math.hypot(tx - x, ty - y)
            v_des = clip(0.5 * dist, 0.0, g.tow_vmax)

        err = wrap(math.atan2(ty - y, tx - x) - yaw)
        steer = clip(g.kp_yaw * err - g.kd_yaw * yaw_rate, -1, 1)
        # 방향이 많이 틀어져 있으면 먼저 돌고 나서 전진
        v_des *= max(0.0, math.cos(err)) ** 2
        thrust = self.thrust_feedforward(v_des) + g.kp_speed * (v_des - v_fwd)

        if phase == "TOW" and tension > g.tension_limit:
            thrust *= g.tension_limit / tension
        return clip(thrust, -1, 1), steer
