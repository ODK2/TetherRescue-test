#!/usr/bin/env python3
"""바다 씬 보기 (ROS2 없이).

  python view_ocean.py                 # 뷰어 창 (WSLg)
  python view_ocean.py --video o.mp4   # 헤드리스 녹화
  python view_ocean.py --preset rough

지금(뼈대)은 평평한 씬만 뜬다. 1-2 구현 후 WaterSurface.update() 를 루프에 연결.
"""
import argparse
import time

import mujoco
import mujoco.viewer

from rescue_sim import CONFIG_DIR
from rescue_sim.ocean import OceanConfig
from rescue_sim.sim import World


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default=None, help="harbor | rough")
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--video", default=None)
    args = ap.parse_args()

    cfg = OceanConfig.from_yaml(CONFIG_DIR / "ocean.yaml", preset=args.preset)
    world = World()
    print(f"waves: Hs={cfg.waves.hs} m, Tp={cfg.waves.tp} s")

    # TODO(1-2): surface = WaterSurface(world.model, WaveField(cfg.waves))

    if args.video:
        import imageio.v2 as imageio
        r = mujoco.Renderer(world.model, 480, 854)
        cam = mujoco.MjvCamera()
        cam.distance, cam.elevation, cam.azimuth = 45.0, -30.0, 150.0
        frames = []
        while world.time < args.duration:
            world.step(20)  # 25fps
            r.update_scene(world.data, cam)
            frames.append(r.render())
        imageio.mimsave(args.video, frames, fps=25)
        print(f"saved {args.video}")
        return

    with mujoco.viewer.launch_passive(world.model, world.data) as v:
        while v.is_running() and world.time < args.duration:
            t0 = time.perf_counter()
            world.step(20)
            v.sync()
            time.sleep(max(0.0, 0.04 - (time.perf_counter() - t0)))
    print(f"realtime factor: {world.realtime_factor():.1f}x")


if __name__ == "__main__":
    main()
