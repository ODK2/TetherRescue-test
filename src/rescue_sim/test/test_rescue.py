"""4단계 완료 기준: 사람이 엎어지지 않고 뜨고, 자율 미션이 사람을 끌고 와서 사람이 뭍까지 걸어간다."""
import math

import pytest

from rescue_sim import CONFIG_DIR
from rescue_sim.boat import Boat
from rescue_sim.ocean import Current, OceanConfig, WaveField
from rescue_sim.rescue import Person, RescueMission, Rope
from rescue_sim.sim import RESCUE_SCENE, World


def make(preset=None, current=None):
    cfg = OceanConfig.from_yaml(CONFIG_DIR / "ocean.yaml", preset=preset)
    if current is not None:
        cfg.current.velocity = current
    w = World(RESCUE_SCENE, ocean=cfg)
    waves = WaveField(cfg.waves, bathymetry=w.bathymetry)
    c = Current(cfg.current, w.bathymetry)
    b, p = Boat(w, waves, c), Person(w, waves, c)
    r = Rope(w)
    return w, b, p, r, RescueMission(b, p, r)


def roll_deg(w, p):
    R = w.data.xmat[p.body_id].reshape(3, 3)
    return abs(math.degrees(math.atan2(R[2, 1], R[2, 2])))


def test_person_stays_face_up_in_rough_sea():
    w, b, p, r, m = make(preset="rough")
    worst = 0.0
    for _ in range(750):
        w.step(20)
        worst = max(worst, roll_deg(w, p))
    assert worst < 30.0


def test_rope_is_slack_until_attached():
    w, b, p, r, m = make()
    w.step(500)
    assert not r.attached and r.tension() == 0.0


@pytest.mark.parametrize("preset,current", [(None, None), ("rough", (0.0, -0.3)), (None, (0.3, 0.0))])
def test_mission_delivers_person_to_land(preset, current):
    w, b, p, r, m = make(preset, current)
    max_tension, min_keel = 0.0, 9.0
    while w.time < 150 and m.state != m.DONE:
        w.step(20)
        m.update(w.time)
        max_tension = max(max_tension, r.tension())
        min_keel = min(min_keel, m.boat_depth(b.state().x))
        if p.mode == p.FLOAT:
            assert roll_deg(w, p) < 60.0
    assert m.state == m.DONE
    assert [s for _, s in m.log] == [m.CONNECT, m.CLEAR, m.TOW, m.WINCH, m.RELEASE, m.WALK, m.DONE]
    assert m.min_clearance > 0.5                       # 선체가 사람에게 0.5 m 안으로 안 들어감
    assert p.mode == p.ARRIVED and p.on_land           # 사람이 마른 땅 위에 서 있다
    assert max_tension < 1500.0                        # 로프 충격 장력 (탄성 없으면 ~4.8 kN)
    assert min_keel > 0.2                              # 보트가 모래톱에 안 올라감
