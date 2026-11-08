#!/usr/bin/env python3
"""MuJoCo 자체의 한계 vs 우리 구현의 단순화 — 직접 돌려서 확인하는 실험 모음.

  python check_mujoco_limits.py        # 결과: bench/results/mujoco_limits.txt 에 기록해 둔 것과 같아야 함

각 실험은 '이건 MuJoCo 가 못 하는가?' 를 최소 모델로 확인한다.
"""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import time

import mujoco
import numpy as np

from rescue_sim.sim import RESCUE_SCENE

G = 9.81


def run(xml: str, seconds: float, dt: float | None = None):
    m = mujoco.MjModel.from_xml_string(xml)
    if dt:
        m.opt.timestep = dt
    d = mujoco.MjData(m)
    n = int(seconds / m.opt.timestep)
    return m, d, n


def fluid_buoyancy():
    """내장 유체(option density): 물보다 가벼운 상자가 뜨는가?"""
    print("[1] 내장 유체에 부력·수면이 있는가")
    for opt in ("", 'density="1000" viscosity="0.001"'):
        xml = f"""<mujoco><option {opt}/><worldbody><body><freejoint/>
        <geom type="box" size=".5 .5 .1" density="500"/></body></worldbody></mujoco>"""
        m, d, n = run(xml, 1.0)
        for _ in range(n):
            mujoco.mj_step(m, d)
        print(f"    {opt or '유체 없음':34s} 밀도 500 상자, 1초 뒤 z = {d.qpos[2]:+.2f} m")
    print("    → 저항은 생기지만(덜 떨어짐) 부력이 없어 가라앉는다. 수면 개념도 없음 = MuJoCo 한계")


def constraint_stiffness():
    """제약(solref)에 강성을 직접 넣으면 실제 N/m 로 동작하는가? / 텐던 스프링은?"""
    print("[2] 80 kg 을 1 m 줄에 매달기 — 기대 늘어남 = 무게 / 강성")
    body = """<worldbody><site name="a"/><body pos="0 0 -1"><freejoint/>
      <geom type="sphere" size=".1" mass="80"/><site name="b"/></body></worldbody>"""
    k = 20000.0
    cases = [
        ("제약 solreflimit='-20000 -2500' dt=2ms",
         f'<spatial limited="true" range="0 1" solreflimit="-{k:.0f} -2500">', 0.002),
        ("제약 solreflimit='-20000 -2500' dt=0.5ms",
         f'<spatial limited="true" range="0 1" solreflimit="-{k:.0f} -2500">', 0.0005),
        ("텐던 스프링 stiffness=20000 springlength='0 1'",
         f'<spatial stiffness="{k:.0f}" damping="2500" springlength="0 1">', 0.002),
    ]
    for name, tendon, dt in cases:
        xml = f"<mujoco>{body}<tendon>{tendon}<site site='a'/><site site='b'/></spatial></tendon></mujoco>"
        m, d, n = run(xml, 20.0, dt)
        L = []
        for i in range(n):
            mujoco.mj_step(m, d)
            if i > 0.8 * n:
                L.append(float(d.ten_length[0]))
        print(f"    {name:44s} 늘어남 {np.mean(L) - 1:+.4f} m (흔들림 {np.std(L):.4f})  기대 {80 * G / k:.4f} m")
    print("    → 제약의 '부드러움'은 실제 강성과 다르다 = MuJoCo 한계. 텐던 스프링은 정확 = 로프 탄성은 구현 선택")


def flex_rope():
    """휘어지는 로프(1차원 변형체)를 만들 수 있는가?"""
    print("[3] 휘어지는 로프 (flexcomp dim=1)")
    xml = """<mujoco><worldbody><flexcomp name="rope" type="grid" count="21 1 1" spacing=".2 1 1"
      dim="1" radius=".01" mass="2"><edge equality="true"/></flexcomp></worldbody></mujoco>"""
    m, d, n = run(xml, 0.5)
    for _ in range(n):
        mujoco.mj_step(m, d)
    print(f"    마디 {m.nflexvert} 개 로프 생성·시뮬 OK (매단 곳 없어서 떨어짐: 최저 z {d.flexvert_xpos[:, 2].min():.2f} m)")
    print("    → 로프는 만들 수 있다. 직선 '거리 제한' 텐던을 쓴 건 구현 선택")


def runtime_add():
    """실행 중에 물체를 추가할 수 있는가? (상태 유지)"""
    print("[4] 실행 중 물체 추가 (MjSpec.recompile)")
    spec = mujoco.MjSpec.from_file(str(RESCUE_SCENE))
    m = spec.compile()
    d = mujoco.MjData(m)
    for _ in range(100):
        mujoco.mj_step(m, d)
    qpos0, t0 = d.qpos.copy(), d.time
    b = spec.worldbody.add_body(name="ring", pos=[5, 0, 0.3])
    b.add_freejoint()
    b.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.2, 0, 0])
    t = time.perf_counter()
    m2, d2 = spec.recompile(m, d)
    ms = (time.perf_counter() - t) * 1000
    kept = np.allclose(d2.qpos[: len(qpos0)], qpos0) and d2.time == t0
    print(f"    구조 씬에 구조링 추가: {ms:.0f} ms, body {m.nbody}→{m2.nbody}, 기존 상태 유지 {kept}")
    print("    → 가능하다. 로프를 미리 숨겨 둔 건 구현 선택")


def muscle():
    print("[5] 근육 액추에이터")
    xml = """<mujoco><worldbody><body><joint type="hinge"/><geom type="capsule" fromto="0 0 0 .3 0 0" size=".03"/>
      <site name="s1" pos=".1 0 0"/></body><site name="s0" pos=".1 .2 0"/></worldbody>
      <tendon><spatial name="t"><site site="s0"/><site site="s1"/></spatial></tendon>
      <actuator><muscle tendon="t"/></actuator></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    print(f"    생성 OK (dyntype = {mujoco.mjtDyn(m.actuator_dyntype[0]).name})")
    print("    → 사람 몸은 관절·근육으로 만들 수 있다. 걷기를 애니메이션으로 한 건 구현 선택")


if __name__ == "__main__":
    print(f"MuJoCo {mujoco.__version__}\n")
    for f in (fluid_buoyancy, constraint_stiffness, flex_rope, runtime_add, muscle):
        f()
        print()
