"""2~3단계 완료 기준: 보트가 뜨고, 파도에 안 뒤집히고, 명령대로 움직인다."""
import math
from dataclasses import replace

import numpy as np
import pytest

from rescue_sim import CONFIG_DIR
from rescue_sim.boat import Boat
from rescue_sim.hydro.hull import RHO
from rescue_sim.ocean import Current, OceanConfig, WaveField
from rescue_sim.sim import BOAT_SCENE, World


def make(preset=None, calm=False, current=None):
    cfg = OceanConfig.from_yaml(CONFIG_DIR / "ocean.yaml", preset=preset)
    if calm:
        cfg.waves = replace(cfg.waves, hs=1e-6)
    if current is not None:
        cfg.current.velocity = current
    w = World(BOAT_SCENE, ocean=cfg)
    waves = WaveField(cfg.waves, bathymetry=w.bathymetry)
    return w, Boat(w, waves, Current(cfg.current, w.bathymetry))


def test_floats_at_design_draft():
    """잔잔한 물: 부력 = 무게, 잠긴 깊이 = 질량 / (ρ · 바닥 면적)."""
    w, b = make(calm=True)
    w.step(3000)
    mass = float(w.model.body_subtreemass[b.body_id])
    h = b.p.hull
    draft = mass / (RHO * h.length * h.beam)
    s = b.state()
    assert s.z == pytest.approx(h.height / 2 - draft, abs=2e-3)   # 몸체 원점 = 선체 중심
    assert b.last.buoyancy == pytest.approx(mass * 9.81, rel=0.01)
    assert abs(s.roll) < 1e-3 and abs(s.pitch) < 1e-3


def test_survives_rough_sea():
    """거친 바다에서 30초 방치: 안 뒤집히고, 안 가라앉고, 멀리 떠밀리지 않는다."""
    w, b = make(preset="rough")
    worst = 0.0
    for _ in range(300):
        w.step(50)
        s = b.state()
        worst = max(worst, abs(s.roll), abs(s.pitch))
        assert -1.0 < s.z < 1.5
    assert math.degrees(worst) < 30.0
    assert math.hypot(s.x - 10.0, s.y) < 2.0


def test_forward_thrust_speed():
    """추력 0.8 로 6초: 2~3 m/s, 방향 유지."""
    w, b = make(calm=True)
    b.set_cmd(0.8, 0.0)
    w.step(3000)
    s = b.state()
    assert 2.0 < s.surge < 3.0
    assert abs(math.degrees(s.yaw)) == pytest.approx(180.0, abs=2.0)


def test_steer_turns_left():
    """steer > 0 → 왼쪽(반시계) 회전 = yaw rate > 0."""
    w, b = make(calm=True)
    b.set_cmd(0.4, 0.6)
    w.step(1500)
    assert b.state().yaw_rate > math.radians(10)


def test_current_drifts_boat():
    """추력 0, 해류 +x 0.3 m/s → 20초 뒤 +x 로 4 m 이상 떠내려감."""
    w, b = make(calm=True, current=(0.3, 0.0))
    w.step(10_000)
    assert b.state().x - 10.0 > 4.0


def test_thruster_out_of_water_no_thrust():
    """보트를 물 위로 들어 올리면 추진기 추력 0 (공중에서 프로펠러는 힘을 못 냄)."""
    w, b = make(calm=True)
    import mujoco
    w.data.qpos[2] = 3.0
    mujoco.mj_forward(w.model, w.data)     # qpos 를 바꾼 뒤엔 위치(site_xpos 등)를 다시 계산해야 한다
    b.set_cmd(1.0, 0.0)
    w.step(1)
    assert b.thrust_now() == (0.0, 0.0)


def test_current_zero_on_land():
    cfg = OceanConfig.from_yaml(CONFIG_DIR / "ocean.yaml")
    cfg.current.velocity = (0.5, 0.2)
    w = World(ocean=cfg)
    c = Current(cfg.current, w.bathymetry)
    np.testing.assert_allclose(c.velocity(-20.0, 0.0), 0.0)
    np.testing.assert_allclose(c.velocity(10.0, 0.0), (0.5, 0.2))


def test_wave_velocity_matches_surface_motion():
    """수면에서 물의 위아래 속도 = 수면 높이 변화율 (얕은 곳 포함)."""
    cfg = OceanConfig.from_yaml(CONFIG_DIR / "ocean.yaml", preset="rough")
    w = World(ocean=cfg)
    wf = WaveField(cfg.waves, bathymetry=w.bathymetry)
    x = np.array([20.0, 5.0, -5.0])
    y = np.array([1.0, -3.0, 2.0])
    t, dt = 5.0, 1e-4
    deta = (wf.elevation(x, y, t + dt) - wf.elevation(x, y, t - dt)) / (2 * dt)
    np.testing.assert_allclose(wf.velocity(x, y, 0.0, t)[:, 2], deta, rtol=1e-5, atol=1e-6)
