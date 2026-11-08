# rescue_sim_ws — MuJoCo 물가 구조 시뮬레이션 (학습용)

TetherRescue 시나리오(바다 · 익수자 · 구조보트)를 **MuJoCo 단독, CPU 기반**으로
최대한 정교하게 만들어 보며 MuJoCo의 한계를 시험한다. ROS2(Jazzy)는 마지막에 연동한다.

## 단계 로드맵

| 단계 | 내용 | 상태 |
|---|---|---|
| **1. 바다** | 1-1 파도(JONSWAP 합성) · 1-2 수면 시각화(hfield) · 1-3 해저 지형 · 1-4 해류 | 🚧 진행 중 |
| 2. 유체력 | 부력(메시/슬라이스), 항력, 부가질량, 파도 가진력 | ⏳ |
| 3. 보트 | 선체, 추진기·조향, 조종성 검증 | ⏳ |
| 4. 익수자 | 인체 모델, 패닉 FSM, 기도(airway) 판정 | ⏳ |
| 5. 로프 | Tether 장력 모델, TetherData 50Hz | ⏳ |
| 6. 시나리오 | 접근 → 접촉 → 견인 → 구조, 제어기 | ⏳ |
| 7. ROS2 | sim/controller 노드, 토픽, launch | ⏳ |

각 단계는 **구현 → 테스트(test/) → 성능 측정(bench/results/)** 순으로 마무리한다.

## 구조

```
rescue_sim_ws/
├── .venv/                       # Python 3.12 venv (--system-site-packages, ROS2 rclpy 공유용)
├── requirements.txt
├── bench/results/               # 단계별 성능 측정 기록
├── prototype/                   # 초기 프로토타입 (참고용, 사용 안 함)
└── src/rescue_sim/              # ROS2 ament_python 패키지 (colcon 빌드 대상)
    ├── package.xml, setup.py, setup.cfg, resource/
    ├── config/ocean.yaml        # 바다 파라미터 (harbor / rough 프리셋)
    ├── launch/                  # 7단계
    ├── scripts/
    │   ├── view_ocean.py        # 뷰어 / 영상 녹화
    │   └── bench_ocean.py       # 성능 측정
    ├── test/test_ocean.py       # 단계 완료 기준 (skip 을 하나씩 해제)
    └── rescue_sim/
        ├── ocean/               # 1단계 — MuJoCo 비의존 순수 numpy
        │   ├── config.py        #   yaml → dataclass
        │   ├── spectrum.py      #   [1-1] JONSWAP → 성분파
        │   ├── waves.py         #   [1-1] WaveField.elevation / velocity
        │   ├── bathymetry.py    #   [1-3] 수심, 해저 hfield
        │   └── current.py       #   [1-4] 해류
        ├── sim/                 # MuJoCo 연동
        │   ├── world.py         #   씬 로딩, 스텝, 실시간 배속 측정
        │   └── water_surface.py #   [1-2] 파도 → water hfield 갱신
        └── assets/ocean_scene.xml
```

설계 원칙: `ocean/`은 MuJoCo를 모른다 → 단위 테스트가 쉽고, 부력·시각화·ROS 노드가 같은 인터페이스를 공유.

## 실행

```bash
cd ~/study/rescue_sim_ws
source .venv/bin/activate
pip install -e src/rescue_sim            # 최초 1회

pytest -q src/rescue_sim/test
python src/rescue_sim/scripts/view_ocean.py                 # 뷰어 (WSLg)
MUJOCO_GL=egl python src/rescue_sim/scripts/view_ocean.py --video ocean.mp4
python src/rescue_sim/scripts/bench_ocean.py
```

### ROS2 (7단계에서 필요)
Ubuntu 24.04 → ROS2 **Jazzy** (`sudo apt install ros-jazzy-ros-base python3-colcon-common-extensions`).
venv는 `--system-site-packages`로 만들어 두었으므로 `source /opt/ros/jazzy/setup.bash` 후 rclpy를 그대로 쓸 수 있다.
