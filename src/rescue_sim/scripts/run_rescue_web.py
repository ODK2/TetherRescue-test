#!/usr/bin/env python3
"""자율 구조 미션 — 브라우저 뷰어 (Ubuntu + CPU 렌더링).

  python run_rescue_web.py                       # → http://localhost:8000
  python run_rescue_web.py --preset rough --current 0.0 0.3 --person 25 -6

보트가 스스로: 사람에게 접근 → 로프 연결 → 해안(-x) 으로 끌고 감 → 로프 감기
→ 로프 풀고 비켜 줌 → 사람이 일어나 모래사장까지 걸어감.
키보드 (브라우저): R = 처음부터 다시,  C = 카메라 (보트 추적 ↔ 전체 보기)
마우스: 드래그 = 카메라 회전, 휠 = 줌
"""
from rescue_sim.sim.web_viewer import WebViewer, setup_cpu_gl

setup_cpu_gl()   # mujoco import 전에

import argparse
import math
import time

import mujoco

from rescue_sim import CONFIG_DIR
from rescue_sim.boat import Boat
from rescue_sim.ocean import Current, OceanConfig, WaveField
from rescue_sim.rescue import Person, RescueMission, Rope
from rescue_sim.sim import RESCUE_SCENE, World
from rescue_sim.sim.water_surface import WaterSurface, upload_hfield


class Sim:
    """월드 + 보트 + 사람 + 로프 + 미션 한 벌 (R 키로 새로 만든다)."""

    def __init__(self, args):
        cfg = OceanConfig.from_yaml(CONFIG_DIR / "ocean.yaml", preset=args.preset)
        if args.current is not None:
            cfg.current.velocity = tuple(args.current)
        self.world = World(RESCUE_SCENE, ocean=cfg)
        waves = WaveField(cfg.waves, bathymetry=self.world.bathymetry)
        current = Current(cfg.current, self.world.bathymetry)
        self.boat = Boat(self.world, waves, current)
        self.person = Person(self.world, waves, current)
        self.person.place(*args.person)
        self.rope = Rope(self.world, length=args.rope)
        self.mission = RescueMission(self.boat, self.person, self.rope)
        self.surface = WaterSurface(self.world.model, waves)
        self.fps = cfg.viz.surface_fps
        self.steps_per_frame = max(1, round(1.0 / (self.fps * self.world.model.opt.timestep)))
        self.max_tension = 0.0

    def step(self):
        self.world.step(self.steps_per_frame)
        self.mission.update(self.world.time)       # 제어 주기 = 화면 주기 (25 Hz)
        self.max_tension = max(self.max_tension, self.rope.tension())

    def status(self) -> str:
        s, p, m = self.boat.state(), self.person.pos, self.mission
        log = "  ".join(f"{st}@{t:.0f}s" for t, st in m.log)
        return (f"R: restart  C: camera  |  drag: rotate  wheel: zoom\n\n"
                f"t={self.world.time:6.1f}s   MISSION: {m.state}\n{log}\n\n"
                f"boat   x={s.x:6.1f} y={s.y:6.1f}  speed={s.surge:+.2f} m/s  heading={math.degrees(s.yaw):5.0f}°\n"
                f"cmd    thrust={self.boat.cmd[0]:+.2f} steer={self.boat.cmd[1]:+.2f}\n"
                f"person x={p[0]:6.1f} y={p[1]:6.1f}  {self.person.mode}  water depth={max(0.0, -self.person.ground_z(p[0])):.2f} m\n"
                f"clearance hull↔person={m.clearance():4.2f} m  (min {m.min_clearance:4.2f} m)\n"
                f"rope   {f'ATTACHED  length={self.world.model.tendon_range[self.rope.id, 1]:.1f} m' if self.rope.attached else '-'}"
                f"  end gap={m.tow_gap():4.1f} m  tension={self.rope.tension():5.0f} N"
                f"  (max {self.max_tension:5.0f} N)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default=None, help="harbor | rough")
    ap.add_argument("--current", type=float, nargs=2, default=None, metavar=("U", "V"))
    ap.add_argument("--person", type=float, nargs=2, default=(22.0, 8.0), metavar=("X", "Y"))
    ap.add_argument("--rope", type=float, default=8.0, help="로프 길이 [m] (걸 수 있는 거리 reach=5 m 보다 길어야 함)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--shadows", action="store_true", help="그림자 (CPU 렌더링에서 ~40%% 느려짐)")
    args = ap.parse_args()

    sim = Sim(args)
    renderer = mujoco.Renderer(sim.world.model, args.height, args.width)
    cam = mujoco.MjvCamera()

    def track(on: bool):
        if on:
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = sim.boat.body_id
            cam.distance, cam.elevation, cam.azimuth = 14.0, -25.0, 135.0
        else:
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = (8.0, 5.0, 0.0)
            cam.distance, cam.elevation, cam.azimuth = 45.0, -50.0, 160.0

    track(True)
    tracking = [True]
    pending_reset = [False]

    def on_key(k: str):
        if k == "reset":
            pending_reset[0] = True
        elif k == "cam":
            tracking[0] = not tracking[0]
            track(tracking[0])

    viewer = WebViewer(cam, on_key, args.port, title="rescue_sim rescue")
    # 물리는 '벽시계'에 맞춰 진행하고, 화면은 그릴 수 있는 만큼만 그린다.
    # CPU 렌더링(그리기 + 수면 업로드)이 한 장에 ~50 ms 라 25 fps 를 못 맞추기 때문.
    t_start, frames_done = time.perf_counter(), 0
    try:
        while True:
            t0 = time.perf_counter()
            with viewer.lock:
                if pending_reset[0]:
                    # 새 MjModel 이라 GL 자원(hfield·텍스처)도 새로 올려야 한다 → renderer 재생성
                    sim = Sim(args)
                    renderer.close()
                    renderer = mujoco.Renderer(sim.world.model, args.height, args.width)
                    track(tracking[0])
                    pending_reset[0] = False
                    t_start, frames_done = time.perf_counter(), 0
                due = int((time.perf_counter() - t_start) * sim.fps) - frames_done
                if due > 5:                       # 너무 밀리면 따라잡기 포기 (느려짐 허용)
                    t_start += (due - 5) / sim.fps
                    due = 5
                for _ in range(max(due, 1)):
                    sim.step()
                frames_done += max(due, 1)
                sim.surface.update(sim.world.time)
                upload_hfield(renderer, sim.surface.hfield_id)
                renderer.update_scene(sim.world.data, cam)
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = args.shadows
                text = sim.status()
            viewer.publish(renderer.render(), text)
            time.sleep(max(0.0, 1.0 / sim.fps - (time.perf_counter() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()
        print(f"\n물리+유체력 : {sim.world.realtime_factor():.1f}x realtime (수면 갱신·렌더링 제외)")


if __name__ == "__main__":
    main()
