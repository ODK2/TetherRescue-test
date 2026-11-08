"""[1-2] 수면 시각화: WaveField → MuJoCo water hfield 데이터 갱신.

할 일
  - update(t): 격자점에서 elevation 계산 → 0~1 정규화 → model.hfield_data 에 기록
  - 뷰어 사용 시 viewer.update_hfield(hfield_id) 호출해 GPU에 다시 올림
주의
  - hfield 데이터는 행(row)=y, 열(col)=x 순서
  - 정규화 규칙은 assets/ocean_scene.xml 주석 참고 (data 0.5 = 정지 수면)
"""
from __future__ import annotations

import mujoco

from ..ocean.waves import WaveField


class WaterSurface:
    def __init__(self, model: mujoco.MjModel, waves: WaveField, hfield_name: str = "water"):
        self.model = model
        self.waves = waves
        self.hfield_id = model.hfield(hfield_name).id

    def update(self, t: float):
        raise NotImplementedError("1-2에서 같이 구현")
