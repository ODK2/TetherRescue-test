"""1단계 완료 기준을 테스트로 고정. 구현하면서 skip 을 하나씩 푼다."""
import numpy as np
import pytest

from rescue_sim import CONFIG_DIR
from rescue_sim.ocean import OceanConfig, WaveField
from rescue_sim.ocean.spectrum import dispersion_k, jonswap, sample_components
from rescue_sim.sim import World


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


@pytest.mark.skip(reason="1-3 구현 후")
def test_depth_zero_beyond_shore():
    """해안선 너머 수심 0, 먼바다 수심 = deep_depth."""
