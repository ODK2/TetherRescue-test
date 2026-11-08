#!/usr/bin/env python3
"""보트 성능 측정: 어디에 시간이 드는지 나눠서 본다.

  python bench_boat.py              # 화면 출력
  python bench_boat.py --save 2-3   # bench/results/2-3.txt 로도 저장
"""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # view_ocean.py 참고

import argparse
import datetime
import time
from pathlib import Path

from rescue_sim import CONFIG_DIR
from rescue_sim.boat import Boat
from rescue_sim.ocean import Current, OceanConfig, WaveField
from rescue_sim.sim import BOAT_SCENE, World
from rescue_sim.sim.water_surface import WaterSurface

RESULTS_DIR = Path(__file__).resolve().parents[3] / "bench" / "results"
SIM_SECONDS = 20.0


def run(preset: str, with_boat: bool, with_surface: bool) -> tuple[float, float, float]:
    """(실시간 배속, 1스텝당 유체력 µs, 1스텝당 mj_step µs)"""
    cfg = OceanConfig.from_yaml(CONFIG_DIR / "ocean.yaml", preset=preset)
    w = World(BOAT_SCENE, ocean=cfg)
    waves = WaveField(cfg.waves, bathymetry=w.bathymetry)
    hydro_t = 0.0
    if with_boat:
        boat = Boat(w, waves, Current(cfg.current, w.bathymetry))
        boat.set_cmd(0.5, 0.2)
        inner = w.pre_step[-1]

        def timed(m, d):
            nonlocal hydro_t
            t0 = time.perf_counter()
            inner(m, d)
            hydro_t += time.perf_counter() - t0
        w.pre_step[-1] = timed
    surf = WaterSurface(w.model, waves) if with_surface else None
    steps = round(1.0 / (cfg.viz.surface_fps * w.model.opt.timestep))
    t0 = time.perf_counter()
    while w.time < SIM_SECONDS:
        w.step(steps)
        if surf is not None:
            surf.update(w.time)
    wall = time.perf_counter() - t0
    n = w.n_steps
    return SIM_SECONDS / wall, hydro_t / n * 1e6, (w.step_wall_time - hydro_t) / n * 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", metavar="STAGE")
    args = ap.parse_args()
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out(f"# {datetime.date.today()}  boat_scene, dt=2ms (500 스텝/초), 렌더링 제외")
    out(f"{'구성':<28}{'배속':>8}{'유체력 µs/스텝':>16}{'mj_step µs/스텝':>17}")
    for preset in ["harbor", "rough"]:
        for label, boat, surf in [("MuJoCo 만", False, False), ("+ 보트 유체력", True, False),
                                  ("+ 보트 유체력 + 수면 128²", True, True)]:
            rtf, hy, mj = run(preset, boat, surf)
            out(f"{preset + ' ' + label:<28}{rtf:7.1f}x{hy:14.1f}{mj:17.1f}")

    if args.save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        path = RESULTS_DIR / f"{args.save}.txt"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
