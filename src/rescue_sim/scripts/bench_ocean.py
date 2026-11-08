#!/usr/bin/env python3
"""성능 측정: 단계마다 같은 조건으로 돌려 bench/results/ 에 기록.

  python bench_ocean.py                  # 화면 출력만
  python bench_ocean.py --save 1-2       # bench/results/1-2.txt 로도 저장

측정 항목 (1단계)
  - mj_step 만          : 순수 MuJoCo 스텝 속도
  - elevation(N점)      : 파도 계산 비용 (점 개수별)                 ✅ 1-1
  - 수면 hfield 갱신     : 해상도별 갱신 비용 + 물리와 합친 실시간 배속  ✅ 1-2
  (화면 렌더링 비용은 제외 — 뷰어를 끈 학습/헤드리스 실행 기준)
"""
import os

# numpy(OpenBLAS) 스레드를 1개로: 수면 계산처럼 작은 행렬곱은 1스레드가 더 빠르고,
# WSL 소프트웨어 렌더러(llvmpipe)와 CPU 코어를 뺏고 빼앗기며 15배 느려지는 걸 막는다.
# numpy 가 import 되기 '전에' 설정해야 효과가 있다.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import datetime
import platform
import time
from copy import deepcopy
from pathlib import Path

import numpy as np

from rescue_sim import CONFIG_DIR
from rescue_sim.ocean import OceanConfig, WaveField
from rescue_sim.sim import World
from rescue_sim.sim.water_surface import WaterSurface

RESULTS_DIR = Path(__file__).resolve().parents[3] / "bench" / "results"


def bench_mj_step(n: int = 50_000) -> float:
    w = World()
    w.step(n)
    return w.realtime_factor()


def bench_elevation(wf: WaveField, n_points: int, reps: int = 50) -> float:
    """elevation() 한 번 호출 시간 [ms]."""
    x = np.random.default_rng(0).uniform(-25, 25, size=(2, n_points))
    wf.elevation(x[0], x[1], 0.0)  # 워밍업
    t0 = time.perf_counter()
    for i in range(reps):
        wf.elevation(x[0], x[1], i * 0.04)
    return (time.perf_counter() - t0) / reps * 1e3


def bench_surface(cfg: OceanConfig, res: int, sim_seconds: float = 20.0) -> tuple[float, float]:
    """물리 스텝 + surface_fps 마다 수면 갱신. (갱신 1회 ms, 합친 실시간 배속)."""
    cfg = deepcopy(cfg)
    cfg.domain.surface_res = res
    world = World(ocean=cfg)
    surf = WaterSurface(world.model, WaveField(cfg.waves, bathymetry=world.bathymetry))
    steps = max(1, round(1.0 / (cfg.viz.surface_fps * world.model.opt.timestep)))
    surf_time, n_upd = 0.0, 0
    while world.time < sim_seconds:
        world.step(steps)
        t0 = time.perf_counter()
        surf.update(world.time)
        surf_time += time.perf_counter() - t0
        n_upd += 1
    return surf_time / n_upd * 1e3, world.time / (world.step_wall_time + surf_time)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", metavar="STAGE", help="bench/results/STAGE.txt 로 저장")
    args = ap.parse_args()
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    cfg = OceanConfig.from_yaml(CONFIG_DIR / "ocean.yaml")
    out(f"# {datetime.date.today()}  {platform.processor() or platform.machine()}  "
        f"OPENBLAS_NUM_THREADS={os.environ.get('OPENBLAS_NUM_THREADS')}")
    out(f"mj_step only: {bench_mj_step():.0f}x realtime")

    wf = WaveField(cfg.waves)
    out(f"\nelevation() 일반 계산, n_components={cfg.waves.n_components}")
    for n in [1, 100, 64**2, 128**2, 256**2]:
        out(f"  {n:>6} pts: {bench_elevation(wf, n):8.3f} ms/call")

    out(f"\n수면 hfield 갱신 (fixed_points), {cfg.viz.surface_fps:.0f}fps, 렌더링 제외")
    for res in [64, 128, 256, 512]:
        ms, rtf = bench_surface(cfg, res)
        out(f"  {res:>3}×{res:<3}: {ms:7.3f} ms/update  → 물리+수면 {rtf:6.0f}x realtime")

    if args.save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        path = RESULTS_DIR / f"{args.save}.txt"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
