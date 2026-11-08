"""1단계 완료 기준을 테스트로 고정. 구현하면서 skip 을 하나씩 푼다."""
import numpy as np
import pytest

from rescue_sim import CONFIG_DIR
from rescue_sim.ocean import Bathymetry, OceanConfig, WaveField
from rescue_sim.ocean.waves import BREAK_RATIO
from rescue_sim.ocean.spectrum import dispersion_k, jonswap, sample_components
from rescue_sim.sim import World
from rescue_sim.sim.water_surface import WaterSurface


def test_config_loads_presets():
    harbor = OceanConfig.from_yaml(CONFIG_DIR / "ocean.yaml", preset="harbor")
    rough = OceanConfig.from_yaml(CONFIG_DIR / "ocean.yaml", preset="rough")
    assert harbor.waves.hs < rough.waves.hs


def test_scene_loads_and_steps():
    w = World()
    w.step(100)
    assert w.time == pytest.approx(0.2)
    assert w.model.hfield("water").id >= 0


@pytest.fixture(params=["harbor", "rough"])
def wave_cfg(request):
    return OceanConfig.from_yaml(CONFIG_DIR / "ocean.yaml", preset=request.param).waves


def test_spectrum_energy_matches_hs(wave_cfg):
    """성분파 에너지 합 → 4*sqrt(m0) ≈ Hs (±5%).

    정규화 전 원본 스펙트럼 적분도 확인: 정규화가 큰 오차를 덮고 있지 않은지.
    """
    comp = sample_components(wave_cfg)
    assert comp.hs() == pytest.approx(wave_cfg.hs, rel=0.05)

    wp = 2 * np.pi / wave_cfg.tp
    w = np.linspace(1e-3, 10 * wp, 100_000)
    m0 = np.trapezoid(jonswap(w, wave_cfg.hs, wave_cfg.tp, wave_cfg.gamma), w)
    assert 4 * np.sqrt(m0) == pytest.approx(wave_cfg.hs, rel=0.05)


def test_dispersion_relation():
    """ω² = g k tanh(kh) 만족, 깊은 물 극한에서 k = ω²/g."""
    omega = np.linspace(0.5, 5.0, 20)
    k = dispersion_k(omega, depth=5.0)
    np.testing.assert_allclose(9.81 * k * np.tanh(k * 5.0), omega**2, rtol=1e-10)
    np.testing.assert_allclose(dispersion_k(omega, depth=1e4), omega**2 / 9.81, rtol=1e-10)


def test_elevation_statistics(wave_cfg):
    """한 점에서 긴 시간 샘플링한 η 의 4*std ≈ Hs, 평균 ≈ 0."""
    wf = WaveField(wave_cfg, depth=5.0)
    t = np.arange(0.0, 1800.0, 0.1)
    eta = np.array([wf.elevation(3.0, -2.0, ti) for ti in t])
    assert 4 * eta.std() == pytest.approx(wave_cfg.hs, rel=0.05)
    assert abs(eta.mean()) < 0.02 * wave_cfg.hs


def test_elevation_vectorized(wave_cfg):
    """배열 입력/출력 shape 일치, 스칼라 루프 결과와 동일."""
    wf = WaveField(wave_cfg)
    x, y = np.meshgrid(np.linspace(-25, 25, 7), np.linspace(-25, 25, 5))
    eta = wf.elevation(x, y, 1.23)
    assert eta.shape == x.shape
    loop = np.array([[wf.elevation(x[i, j], y[i, j], 1.23) for j in range(7)] for i in range(5)])
    np.testing.assert_allclose(eta, loop, rtol=1e-12, atol=1e-14)


def test_fixed_points_matches_elevation(wave_cfg):
    """격자 전용 빠른 계산기 = elevation() 결과와 동일."""
    wf = WaveField(wave_cfg)
    x, y = np.meshgrid(np.linspace(-25, 25, 9), np.linspace(-25, 25, 6))
    fast = wf.fixed_points(x, y)
    for t in [0.0, 3.7, 125.0]:
        np.testing.assert_allclose(fast.at(t), wf.elevation(x, y, t), atol=1e-12)


def test_world_uses_yaml_resolution():
    """yaml 의 surface_res / 영역 크기가 MuJoCo 모델에 반영된다 (MjSpec)."""
    cfg = OceanConfig.from_yaml(CONFIG_DIR / "ocean.yaml")
    cfg.domain.surface_res = 40
    cfg.domain.size_x = 60.0
    m = World(ocean=cfg).model
    h = m.hfield("water").id
    assert m.hfield_nrow[h] == m.hfield_ncol[h] == 40
    assert m.hfield_size[h, 0] == pytest.approx(30.0)


WATER_GROUP, SEABED_GROUP = 1, 0


def _ray_z(world, x, y, group=WATER_GROUP):
    """MuJoCo 가 '실제로' 보는 높이: 위에서 아래로 광선을 쏴서 맞은 높이 (geom group 으로 수면/해저 선택)."""
    import mujoco
    mujoco.mj_forward(world.model, world.data)
    geomid = np.zeros(1, np.int32)
    groups = np.zeros(6, np.uint8)
    groups[group] = 1
    dist = mujoco.mj_ray(world.model, world.data, np.array([x, y, 20.0]), np.array([0, 0, -1.0]),
                         groups, 1, -1, geomid)
    return 20.0 - dist, int(geomid[0])


def test_water_surface_matches_waves(wave_cfg):
    """MuJoCo hfield 에 쓴 수면 = WaveField 가 계산한 수면 (방향·축척·높이 기준점 확인).

    격자점 위에서 광선을 쏴 비교한다. 행/열(x/y)을 뒤집거나, 0~1 변환을 잘못하면 여기서 걸린다.
    """
    cfg = OceanConfig.from_yaml(CONFIG_DIR / "ocean.yaml")
    cfg.waves = wave_cfg
    w = World(ocean=cfg)
    wf = WaveField(wave_cfg, bathymetry=w.bathymetry)   # 얕은 물 효과 포함
    surf = WaterSurface(w.model, wf)
    surf.update(7.3)
    n = cfg.domain.surface_res
    xs = np.linspace(-25, 25, n)
    for i, j in [(10, 100), (100, 10), (64, 64), (5, 120)]:   # (행=y, 열=x)
        z, g = _ray_z(w, xs[j], xs[i])
        assert g == surf.geom_id
        assert z == pytest.approx(wf.elevation(xs[j], xs[i], 7.3), abs=1e-4)


def test_water_surface_never_clips(wave_cfg):
    """긴 시간 동안 hfield 값이 0~1 을 벗어나지 않는다 = 화면에서 파도가 잘리지 않는다."""
    cfg = OceanConfig.from_yaml(CONFIG_DIR / "ocean.yaml")
    cfg.waves = wave_cfg
    w = World(ocean=cfg)
    surf = WaterSurface(w.model, WaveField(wave_cfg, bathymetry=w.bathymetry))
    lo, hi = 1.0, 0.0
    for t in np.arange(0.0, 600.0, 1.7):
        surf.update(t)
        lo, hi = min(lo, surf._data.min()), max(hi, surf._data.max())
    assert 0.0 <= lo and hi <= 1.0


# ------------------------------------------------------------------ 1-3 해저 지형
@pytest.fixture
def bathy():
    return Bathymetry(OceanConfig.from_yaml(CONFIG_DIR / "ocean.yaml").bathymetry)


def test_depth_zero_beyond_shore(bathy):
    """해안선 너머 수심 0, 먼바다 수심 = deep_depth, 그 사이는 직선 경사."""
    c = bathy.cfg
    assert bathy.depth(c.shore_x - 5.0) == 0.0
    assert bathy.depth(c.shore_x) == 0.0
    assert bathy.depth(c.slope_start_x + 5.0) == pytest.approx(c.deep_depth)
    mid = 0.5 * (c.shore_x + c.slope_start_x)
    assert bathy.depth(mid) == pytest.approx(c.deep_depth / 2)
    assert bathy.seabed_z(-1e3) == pytest.approx(c.beach_height)   # 모래사장은 beach_height 에서 평평


def test_seabed_in_mujoco(bathy):
    """MuJoCo 해저 hfield = Bathymetry.seabed_z (광선으로 확인, 수면은 제외)."""
    w = World(ocean=OceanConfig.from_yaml(CONFIG_DIR / "ocean.yaml"))
    n = w.model.hfield_ncol[w.model.hfield("seabed").id]
    xs = np.linspace(-25, 25, n)
    for j in [0, 20, 40, 60, 90, 127]:                     # 모래사장 ~ 경사 ~ 먼바다
        z, g = _ray_z(w, xs[j], 3.1, group=SEABED_GROUP)
        assert g == w.model.geom("seabed").id
        assert z == pytest.approx(bathy.seabed_z(xs[j]), abs=1e-4)


def test_shallow_water_effects(wave_cfg, bathy):
    """얕은 물: 깊은 쪽은 그대로, 땅 위는 0, 얕은 곳은 쇄파 한계 이하, 경계에서 끊김 없음."""
    wf = WaveField(wave_cfg, bathymetry=bathy)
    deep = WaveField(wave_cfg, depth=bathy.cfg.deep_depth)
    x0 = bathy.cfg.slope_start_x
    y = np.linspace(-20, 20, 9)
    for t in [0.0, 11.0]:
        np.testing.assert_allclose(wf.elevation(x0 + 3.0, y, t), deep.elevation(x0 + 3.0, y, t), atol=1e-12)
        np.testing.assert_array_equal(wf.elevation(bathy.cfg.shore_x - 2.0, y, t), 0.0)
        assert np.max(np.abs(wf.elevation(x0 + 1e-4, y, t) - wf.elevation(x0 - 1e-4, y, t))) < 1e-3

    t = np.arange(0.0, 600.0, 0.3)
    x_shallow = bathy.cfg.shore_x + 2.0                    # 수심 0.4 m
    eta = np.array([wf.elevation(x_shallow, 0.0, ti) for ti in t])
    assert 4 * eta.std() <= 1.1 * BREAK_RATIO * bathy.depth(x_shallow)


def test_shallow_water_off_is_constant_depth(wave_cfg, bathy):
    """shallow_water=False 면 수심 변화를 무시하고 먼바다 수심으로 계산."""
    from dataclasses import replace
    off = Bathymetry(replace(bathy.cfg, shallow_water=False))
    x, y = np.meshgrid(np.linspace(-25, 25, 7), np.linspace(-25, 25, 5))
    np.testing.assert_allclose(WaveField(wave_cfg, bathymetry=off).elevation(x, y, 4.0),
                               WaveField(wave_cfg, depth=bathy.cfg.deep_depth).elevation(x, y, 4.0))
