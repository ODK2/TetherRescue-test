#!/usr/bin/env python3
"""로봇 보트 띄우기 (ROS2 없이).

  python run_boat.py                       # 뷰어 + 키보드 조종 (WSLg)
  python run_boat.py --preset rough
  python run_boat.py --current 0.3 0.0     # 해류 0.3 m/s (+x, 먼바다 쪽)
  MUJOCO_GL=egl python run_boat.py --video boat.mp4   # 정해진 조종 시나리오 녹화

키보드 (뷰어 창 클릭 후)
  ↑ / ↓   : 추력 +/− 0.1   (−1 ~ 1, 음수 = 후진)
  ← / →   : 왼쪽 / 오른쪽 회전 0.1
  Home    : 정지 (추력·조향 0)
  (스페이스, 백스페이스는 MuJoCo 뷰어가 쓰는 키라 쓰지 않음)
"""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # view_ocean.py 참고

import argparse
import math
import time

import mujoco
import mujoco.viewer
import numpy as np

from rescue_sim import CONFIG_DIR
from rescue_sim.boat import Boat
from rescue_sim.ocean import Current, OceanConfig, WaveField
from rescue_sim.sim import BOAT_SCENE, World
from rescue_sim.sim.water_surface import WaterSurface, upload_hfield

KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT, KEY_HOME = 265, 264, 263, 262, 268


def scripted_cmd(t: float) -> tuple[float, float]:
    """녹화용 조종: 해안 쪽으로 직진 → 왼쪽 U턴 → 먼바다로 직진 → 정지."""
    if t < 4:
        return 0.8, 0.0
    if t < 10:
        return 0.4, 0.6
    if t < 17:
        return 0.8, 0.0
    return 0.0, 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default=None, help="harbor | rough")
    ap.add_argument("--current", type=float, nargs=2, default=None, metavar=("U", "V"))
    ap.add_argument("--duration", type=float, default=600.0)
    ap.add_argument("--video", default=None)
    args = ap.parse_args()

    cfg = OceanConfig.from_yaml(CONFIG_DIR / "ocean.yaml", preset=args.preset)
    if args.current is not None:
        cfg.current.velocity = tuple(args.current)
    world = World(BOAT_SCENE, ocean=cfg)
    waves = WaveField(cfg.waves, bathymetry=world.bathymetry)
    current = Current(cfg.current, world.bathymetry)
    boat = Boat(world, waves, current)
    surface = WaterSurface(world.model, waves)

    fps = cfg.viz.surface_fps
    steps_per_frame = max(1, round(1.0 / (fps * world.model.opt.timestep)))

    def status() -> str:
        s = boat.state()
        tl, tr = boat.thrust_now()
        return (f"t={world.time:6.1f}s  cmd thrust={boat.cmd[0]:+.1f} steer={boat.cmd[1]:+.1f}  "
                f"speed={s.surge:+.2f} m/s  heading={math.degrees(s.yaw):6.1f}°  "
                f"roll={math.degrees(s.roll):+5.1f}° pitch={math.degrees(s.pitch):+5.1f}°  "
                f"thr L/R={tl:4.0f}/{tr:4.0f} N")

    if args.video:
        import imageio.v2 as imageio
        duration = min(args.duration, 22.0)
        r = mujoco.Renderer(world.model, 480, 864)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = boat.body_id
        cam.distance, cam.elevation, cam.azimuth = 12.0, -20.0, 135.0
        frames = []
        while world.time < duration:
            boat.set_cmd(*scripted_cmd(world.time))
            world.step(steps_per_frame)
            surface.update(world.time)
            upload_hfield(r, surface.hfield_id)
            r.update_scene(world.data, cam)
            frames.append(r.render())
            if len(frames) % int(2 * fps) == 0:
                print(status())
        imageio.mimsave(args.video, frames, fps=fps, macro_block_size=1)
        print(f"saved {args.video}")
    else:
        def on_key(key: int):
            th, st = boat.cmd
            if key == KEY_UP:
                th += 0.1
            elif key == KEY_DOWN:
                th -= 0.1
            elif key == KEY_LEFT:
                st += 0.1
            elif key == KEY_RIGHT:
                st -= 0.1
            elif key == KEY_HOME:
                th, st = 0.0, 0.0
            boat.set_cmd(round(th, 1), round(st, 1))

        with mujoco.viewer.launch_passive(world.model, world.data, key_callback=on_key) as v:
            v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            v.cam.trackbodyid = boat.body_id
            v.cam.distance, v.cam.elevation, v.cam.azimuth = 12.0, -20.0, 135.0
            while v.is_running() and world.time < args.duration:
                t0 = time.perf_counter()
                with v.lock():
                    world.step(steps_per_frame)
                    surface.update(world.time)
                    v.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                 "↑↓ thrust  ←→ steer  Home stop", status().replace("  ", "\n")))
                v.update_hfield(surface.hfield_id)
                v.sync()
                time.sleep(max(0.0, 1.0 / fps - (time.perf_counter() - t0)))

    print(f"\n물리+유체력 : {world.realtime_factor():.1f}x realtime (수면 갱신·렌더링 제외)")


if __name__ == "__main__":
    main()
