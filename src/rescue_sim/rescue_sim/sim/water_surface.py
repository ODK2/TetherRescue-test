"""[1-2] 수면 시각화: WaveField → MuJoCo water hfield 데이터 갱신.

MuJoCo 에서 알아 둘 것
  - hfield(height field) = 바둑판 격자 높이 지도. 칸마다 0~1 값 하나.
    실제 높이 z = geom 위치 z + 값 × hfield_size[2]
  - 데이터 배치: 행(row) = y 방향(-y → +y), 열(col) = x 방향(-x → +x)
  - model.hfield_data 를 바꾸면 '물리'(ray, 충돌)에는 즉시 반영되지만,
    '화면'에는 GPU 로 다시 올려야 보인다 → upload_hfield() / viewer.update_hfield()
  - 컴파일된 MjModel 이라도 hfield_size, geom_pos 같은 값은 실행 중에 바꿀 수 있다.
    (단, 격자 칸 수 nrow/ncol 처럼 배열 크기가 바뀌는 건 불가 → World 생성 시 MjSpec 으로)

이 수면은 '보이기만' 한다 (contype=0). 부력은 2단계에서 WaveField 를 직접 쓴다.
"""
from __future__ import annotations

import mujoco
import numpy as np

from ..ocean.waves import WaveField

HEADROOM = 1.05   # 이론상 최대 파고보다 5% 여유


class WaterSurface:
    def __init__(self, model: mujoco.MjModel, waves: WaveField, hfield_name: str = "water"):
        self.model = model
        self.waves = waves
        self.hfield_id = hid = model.hfield(hfield_name).id
        self.geom_id = self._find_geom(model, hid)

        nrow, ncol = int(model.hfield_nrow[hid]), int(model.hfield_ncol[hid])
        adr = int(model.hfield_adr[hid])
        sx, sy = model.hfield_size[hid, :2]
        gx, gy = model.geom_pos[self.geom_id, :2]

        # 격자점의 월드 좌표 (행 = y, 열 = x)
        X, Y = np.meshgrid(np.linspace(-sx, sx, ncol) + gx, np.linspace(-sy, sy, nrow) + gy)
        self._grid = waves.fixed_points(X, Y)

        # 표현 범위를 파도에 맞춘다: z ∈ [-R, +R], 정지 수면(η=0) = 값 0.5
        # 이론상 최대 |η| 로 잡으므로 절대 잘리지 않는다.
        self.half_range = HEADROOM * waves.max_elevation()
        model.hfield_size[hid, 2] = 2.0 * self.half_range
        model.geom_pos[self.geom_id, 2] = -self.half_range

        self._data = model.hfield_data[adr: adr + nrow * ncol]   # model 메모리를 직접 가리키는 view
        self.update(0.0)

    @staticmethod
    def _find_geom(model: mujoco.MjModel, hid: int) -> int:
        for g in range(model.ngeom):
            if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_HFIELD and model.geom_dataid[g] == hid:
                return g
        raise ValueError(f"hfield {hid} 를 쓰는 geom 이 없음")

    def update(self, t: float) -> np.ndarray:
        """시각 t 의 수면을 hfield 에 기록하고 η 격자를 돌려준다 (GPU 업로드는 별도)."""
        eta = self._grid.at(t)
        self._data[:] = ((eta + self.half_range) / (2.0 * self.half_range)).ravel()
        return eta


def upload_hfield(renderer: mujoco.Renderer, hfield_id: int):
    """오프스크린 Renderer 에 바뀐 hfield 를 다시 올린다.

    mujoco.Renderer 에는 viewer 처럼 update_hfield() 가 없어서 내부 GL 컨텍스트를 직접 쓴다.
    (_gl_context / _mjr_context 는 private 속성 — MuJoCo 버전 올릴 때 확인 필요)
    """
    if renderer._gl_context is not None:
        renderer._gl_context.make_current()
    mujoco.mjr_uploadHField(renderer.model, renderer._mjr_context, hfield_id)
