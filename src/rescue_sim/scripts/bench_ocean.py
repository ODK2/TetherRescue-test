#!/usr/bin/env python3
"""성능 측정: 단계마다 같은 조건으로 돌려 bench/results/ 에 기록.

측정 항목 (1단계)
  - mj_step 만        : 순수 MuJoCo 스텝 속도
  - elevation(N점)    : 파도 계산 비용 (점 개수별)        ✅ 1-1
  - hfield 갱신       : 수면 해상도별 갱신 비용           ⏳ 1-2
"""
import time

import numpy as np

from rescue_sim import CONFIG_DIR
from rescue_sim.ocean import OceanConfig, WaveField
from rescue_sim.sim import World


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


def main():
    print(f"mj_step only: {bench_mj_step():.0f}x realtime")

    cfg = OceanConfig.from_yaml(CONFIG_DIR / "ocean.yaml")
    wf = WaveField(cfg.waves)
    print(f"\nelevation(), n_components={cfg.waves.n_components}")
    for n in [1, 10, 100, 64**2, 128**2, 256**2]:
        ms = bench_elevation(wf, n)
        # 수면을 surface_fps 로 갱신할 때, 시뮬레이션 1초당 파도 계산에만 드는 실제 시간
        per_sim_sec = ms * cfg.viz.surface_fps / 1e3
        print(f"  {n:>6} pts: {ms:8.3f} ms/call   @{cfg.viz.surface_fps:.0f}fps → {per_sim_sec:6.3f} s per sim-second")


if __name__ == "__main__":
    main()
