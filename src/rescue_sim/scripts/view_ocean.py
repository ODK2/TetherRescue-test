#!/usr/bin/env python3
"""바다 씬 보기 (ROS2 없이).

  python view_ocean.py                 # 뷰어 창 (WSLg)
  python view_ocean.py --preset rough
  MUJOCO_GL=egl python view_ocean.py --video o.mp4   # 헤드리스 녹화
"""
import os

# numpy(OpenBLAS) 스레드를 1개로: 수면 계산처럼 작은 행렬곱은 1스레드가 더 빠르고,
# WSL 소프트웨어 렌더러(llvmpipe)와 CPU 코어를 뺏고 빼앗기며 15배 느려지는 걸 막는다.
# numpy 가 import 되기 '전에' 설정해야 효과가 있다.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import time

import mujoco
import mujoco.viewer

from rescue_sim import CONFIG_DIR
from rescue_sim.ocean import OceanConfig, WaveField
from rescue_sim.sim import World
from rescue_sim.sim.water_surface import WaterSurface, upload_hfield


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default=None, help="harbor | rough")
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--video", default=None)
    args = ap.parse_args()

    cfg = OceanConfig.from_yaml(CONFIG_DIR / "ocean.yaml", preset=args.preset)
    world = World(ocean=cfg)
    waves = WaveField(cfg.waves, depth=cfg.bathymetry.deep_depth)
    surface = WaterSurface(world.model, waves)
    print(f"waves: Hs={cfg.waves.hs} m, Tp={cfg.waves.tp} s, 수면 표현 범위 ±{surface.half_range:.2f} m")

    # 화면 1프레임(= 수면 갱신 1번) 당 물리 스텝 수
    fps = cfg.viz.surface_fps
    steps_per_frame = max(1, round(1.0 / (fps * world.model.opt.timestep)))
    surface_time = 0.0

    def advance():
        nonlocal surface_time
        world.step(steps_per_frame)
        t0 = time.perf_counter()
        surface.update(world.time)
        surface_time += time.perf_counter() - t0

    if args.video:
        import imageio.v2 as imageio
        r = mujoco.Renderer(world.model, 480, 854)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = (0.0, 0.0, 0.0)
        cam.distance, cam.elevation, cam.azimuth = 55.0, -22.0, 200.0
        frames = []
        while world.time < args.duration:
            advance()
            upload_hfield(r, surface.hfield_id)
            r.update_scene(world.data, cam)
            frames.append(r.render())
        imageio.mimsave(args.video, frames, fps=fps)
        print(f"saved {args.video}")
    else:
        with mujoco.viewer.launch_passive(world.model, world.data) as v:
            while v.is_running() and world.time < args.duration:
                t0 = time.perf_counter()
                with v.lock():
                    advance()
                v.update_hfield(surface.hfield_id)
                v.sync()
                time.sleep(max(0.0, 1.0 / fps - (time.perf_counter() - t0)))

    sim_t = world.time
    print(f"물리 스텝 : {world.step_wall_time:.3f} s  ({world.realtime_factor():.0f}x realtime)")
    print(f"수면 갱신 : {surface_time:.3f} s  (sim {sim_t:.1f}s 동안, {sim_t * fps:.0f}회)")
    print(f"둘 합친 배속 : {sim_t / (world.step_wall_time + surface_time):.0f}x realtime (렌더링 제외)")


if __name__ == "__main__":
    main()
