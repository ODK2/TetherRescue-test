#!/usr/bin/env python3
"""ROS2 없이 MuJoCo만으로 시나리오 실행.

  python run_standalone.py              # 헤드리스, 로그만
  python run_standalone.py --video out.mp4
  python run_standalone.py --viewer     # 인터랙티브 창 (WSLg 필요)
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/rescue_sim

from rescue_sim.controller import RescueController  # noqa: E402
from rescue_sim.core import RescueSim, SimParams  # noqa: E402

CTRL_SUBSTEPS = 10  # 0.002s * 10 = 50Hz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=90.0)
    ap.add_argument("--video", type=str, default=None)
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sim = RescueSim(SimParams(seed=args.seed))
    ctrl = RescueController()

    renderer = cam = viewer = None
    frames = []
    if args.video:
        import mujoco
        renderer = mujoco.Renderer(sim.model, 480, 854)
        cam = mujoco.MjvCamera()
        cam.distance, cam.elevation, cam.azimuth = 26.0, -35.0, -120.0
    if args.viewer:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(sim.model, sim.data)

    last_phase = None
    k = 0
    while sim.time < args.duration:
        t0 = time.perf_counter()
        thrust, steer = ctrl.compute(sim.phase, sim.boat_pose(),
                                     tuple(sim.victim_position()[:2]), sim.tension)
        sim.set_cmd(thrust, steer)
        sim.step(CTRL_SUBSTEPS)
        k += 1

        if sim.phase != last_phase:
            print(f"[{sim.time:6.2f}s] phase = {sim.phase}")
            last_phase = sim.phase
        if k % 100 == 0:  # 2초마다
            st, td = sim.rescuee_status(), sim.tether_data()
            bx, by, *_ = sim.boat_pose()
            print(f"  t={sim.time:5.1f} boat=({bx:5.1f},{by:5.1f}) victim={st.fsm_state:9s} "
                  f"panic={st.panic_level:.2f} stamina={st.stamina:.2f} "
                  f"airway_sub={st.airway_submerged!s:5s} tension={td.tension_force:6.1f}N")

        if renderer is not None and k % 2 == 0:  # 25fps
            mid = (sim.data.xpos[sim.boat_id] + sim.victim_position()) / 2
            cam.lookat[:] = mid
            renderer.update_scene(sim.data, cam)
            frames.append(renderer.render())
        if viewer is not None:
            if not viewer.is_running():
                break
            viewer.sync()
            time.sleep(max(0.0, 0.02 - (time.perf_counter() - t0)))

        if sim.phase == "RESCUED" and sim.time > sim.events[-1][0] + 4.0:
            break

    for t, e in sim.events:
        print(f"EVENT {t:6.2f}s  {e}")
    if frames:
        import imageio.v2 as imageio
        imageio.mimsave(args.video, frames, fps=25)
        print(f"saved {args.video} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
