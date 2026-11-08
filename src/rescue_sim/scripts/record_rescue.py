#!/usr/bin/env python3
"""자율 구조 미션을 mp4 로 녹화 (화면 없이, CPU 렌더링).

  python record_rescue.py rescue.mp4                    # 2배속, 960x540
  python record_rescue.py rescue_rough.mp4 --preset rough --speed 1 --bitrate 2M

카메라: 보트를 따라가다가, 사람이 일어서면 사람을 따라간다. 왼쪽 위에 미션 단계·장력 표시.
"""
from rescue_sim.sim.web_viewer import setup_cpu_gl

setup_cpu_gl()   # mujoco import 전에

import argparse
import importlib.util
import time
from pathlib import Path
from types import SimpleNamespace

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from rescue_sim.sim.water_surface import upload_hfield

# run_rescue_web.py 의 Sim (월드 + 보트 + 사람 + 로프 + 미션 한 벌) 재사용
_spec = importlib.util.spec_from_file_location("run_rescue_web", Path(__file__).with_name("run_rescue_web.py"))
_web = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_web)

CAPTION = {
    "APPROACH": "1. Approach the person",
    "CONNECT": "2. Attach the rope",
    "CLEAR": "3. Back off before turning",
    "TOW": "4. Tow toward the shore",
    "WINCH": "5. Winch the rope in (shallow water)",
    "RELEASE": "6. Release rope, boat moves aside",
    "WALK": "7. Person stands up and walks ashore",
    "DONE": "Rescue complete",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--preset", default=None)
    ap.add_argument("--current", type=float, nargs=2, default=None)
    ap.add_argument("--person", type=float, nargs=2, default=(22.0, 8.0))
    ap.add_argument("--rope", type=float, default=8.0)
    ap.add_argument("--speed", type=float, default=2.0, help="재생 배속")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--bitrate", default="800k")
    ap.add_argument("--tail", type=float, default=3.0, help="DONE 뒤에 더 찍을 시뮬 시간 [s]")
    args = ap.parse_args()

    sim = _web.Sim(SimpleNamespace(preset=args.preset, current=args.current,
                                   person=tuple(args.person), rope=args.rope))
    fps = sim.fps
    frames_per_out = max(1, round(args.speed))         # 출력 한 장당 시뮬 프레임(25 Hz) 개수
    r = mujoco.Renderer(sim.world.model, args.height, args.width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = sim.boat.body_id
    cam.distance, cam.elevation, cam.azimuth = 16.0, -25.0, 135.0
    font = ImageFont.load_default(size=18)
    small = ImageFont.load_default(size=14)

    writer = imageio.get_writer(args.out, fps=fps, codec="libx264", bitrate=args.bitrate,
                                macro_block_size=8, ffmpeg_params=["-pix_fmt", "yuv420p"])
    t_done, n, t0 = None, 0, time.perf_counter()
    while sim.world.time < 200:
        for _ in range(frames_per_out):
            sim.step()
        m, p = sim.mission, sim.person
        if p.mode != p.FLOAT and cam.trackbodyid != p.body_id:   # 사람이 일어서면 사람을 따라간다
            cam.trackbodyid = p.body_id
            cam.distance, cam.azimuth = 12.0, 60.0
        sim.surface.update(sim.world.time)
        upload_hfield(r, sim.surface.hfield_id)
        r.update_scene(sim.world.data, cam)
        r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        img = Image.fromarray(r.render())

        d = ImageDraw.Draw(img, "RGBA")
        s = sim.boat.state()
        rope = (f"rope {sim.world.model.tendon_range[sim.rope.id, 1]:.1f} m  tension {sim.rope.tension():4.0f} N"
                if sim.rope.attached else "rope -")
        lines = [f"{CAPTION[m.state]}", f"t = {sim.world.time:5.1f} s   (x{args.speed:g} speed)",
                 f"boat speed {s.surge:+.1f} m/s   {rope}",
                 f"person: {p.mode.lower()}   water depth {max(0.0, -p.ground_z(p.pos[0])):.1f} m"]
        d.rectangle((8, 8, 470, 104), fill=(0, 0, 0, 150))
        d.text((16, 12), lines[0], font=font, fill=(255, 230, 80))
        for i, line in enumerate(lines[1:]):
            d.text((16, 40 + 20 * i), line, font=small, fill=(235, 235, 235))
        writer.append_data(np.asarray(img))
        n += 1
        if n % 100 == 0:
            print(f"  t={sim.world.time:5.1f}s {m.state:8s} frames={n}  ({time.perf_counter() - t0:.0f}s)", flush=True)
        if m.state == m.DONE:
            t_done = t_done or sim.world.time
            if sim.world.time - t_done > args.tail:
                break
    writer.close()
    size = Path(args.out).stat().st_size / 1e6
    print(f"saved {args.out}: {n} frames, {n / fps:.0f} s video, {size:.1f} MB, "
          f"mission {' '.join(f'{st}@{t:.0f}s' for t, st in sim.mission.log)}")


if __name__ == "__main__":
    main()
