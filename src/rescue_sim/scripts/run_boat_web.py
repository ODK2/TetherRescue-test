#!/usr/bin/env python3
"""로봇 보트 — 브라우저 뷰어 (WSLg 창이 안 보일 때, Ubuntu + CPU 렌더링).

  python run_boat_web.py                  # → Windows 브라우저에서 http://localhost:8000
  python run_boat_web.py --preset rough --port 8001

키보드 (브라우저 페이지에서)
  ↑ / ↓   : 추력 +/− 0.1      ← / → : 왼쪽 / 오른쪽 회전 0.1
  Home 또는 S : 정지
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
from rescue_sim.sim import BOAT_SCENE, World
from rescue_sim.sim.water_surface import WaterSurface, upload_hfield


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default=None, help="harbor | rough")
    ap.add_argument("--current", type=float, nargs=2, default=None, metavar=("U", "V"))
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--shadows", action="store_true", help="그림자 (CPU 렌더링에서 ~40%% 느려짐)")
    args = ap.parse_args()

    cfg = OceanConfig.from_yaml(CONFIG_DIR / "ocean.yaml", preset=args.preset)
    if args.current is not None:
        cfg.current.velocity = tuple(args.current)
    world = World(BOAT_SCENE, ocean=cfg)
    waves = WaveField(cfg.waves, bathymetry=world.bathymetry)
    boat = Boat(world, waves, Current(cfg.current, world.bathymetry))
    surface = WaterSurface(world.model, waves)

    fps = cfg.viz.surface_fps
    steps_per_frame = max(1, round(1.0 / (fps * world.model.opt.timestep)))
    renderer = mujoco.Renderer(world.model, args.height, args.width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = boat.body_id
    cam.distance, cam.elevation, cam.azimuth = 12.0, -20.0, 135.0

    def status() -> str:
        s = boat.state()
        tl, tr = boat.thrust_now()
        return (f"↑↓ thrust  ←→ steer  Home/S stop  |  drag: rotate  wheel: zoom\n\n"
                f"t={world.time:6.1f}s\ncmd thrust={boat.cmd[0]:+.1f} steer={boat.cmd[1]:+.1f}\n"
                f"speed={s.surge:+.2f} m/s  heading={math.degrees(s.yaw):6.1f}°\n"
                f"roll={math.degrees(s.roll):+5.1f}° pitch={math.degrees(s.pitch):+5.1f}°\n"
                f"thr L/R={tl:4.0f}/{tr:4.0f} N")

    def on_key(k: str):
        th, st = boat.cmd
        th += {"up": 0.1, "down": -0.1}.get(k, 0.0)
        st += {"left": 0.1, "right": -0.1}.get(k, 0.0)
        if k == "stop":
            th, st = 0.0, 0.0
        boat.set_cmd(round(th, 1), round(st, 1))

    viewer = WebViewer(cam, on_key, args.port, title="rescue_sim boat")
    try:
        while True:
            t0 = time.perf_counter()
            with viewer.lock:
                world.step(steps_per_frame)
                surface.update(world.time)
                upload_hfield(renderer, surface.hfield_id)
                renderer.update_scene(world.data, cam)
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = args.shadows
                text = status()
            viewer.publish(renderer.render(), text)
            time.sleep(max(0.0, 1.0 / fps - (time.perf_counter() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()
        print(f"\n물리+유체력 : {world.realtime_factor():.1f}x realtime (수면 갱신·렌더링 제외)")


if __name__ == "__main__":
    main()
