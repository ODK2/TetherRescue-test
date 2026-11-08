# rescue_sim_ws — MuJoCo 물가 구조 시뮬레이션 (학습용)

TetherRescue 시나리오(바다 · 익수자 · 구조보트)를 **MuJoCo 단독, CPU 기반**으로
최대한 정교하게 만들어 보며 MuJoCo의 한계를 시험한다. ROS2(Jazzy)는 마지막에 연동한다.

## 단계 로드맵

| 단계 | 내용 | 상태 |
|---|---|---|
| 1. 바다 | 1-1 파도(JONSWAP 합성) · 1-2 수면 시각화(hfield) · 1-3 해저 지형·얕은 물 · 1-4 해류 | ✅ |
| 2. 유체력 | 기둥 부력, 항력(물 입자 속도·해류 기준) — 부가질량 없음 | ✅ 기본 |
| 3. 보트 | 직육면체 선체, 좌우 추진기(차동 추력), 키보드 조종 | ✅ 기본 |
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
    │   ├── view_ocean.py        # 바다만 보기 / 녹화
    │   ├── run_boat.py          # 보트 띄우기: 키보드 조종 / 시나리오 녹화
    │   ├── bench_ocean.py       # 바다 성능 측정
    │   └── bench_boat.py        # 보트 성능 측정 (유체력 vs mj_step 시간 분해)
    ├── test/                    # 단계 완료 기준
    └── rescue_sim/
        ├── ocean/               # 1단계 — MuJoCo 비의존 순수 numpy
        │   ├── config.py        #   yaml → dataclass
        │   ├── spectrum.py      #   [1-1] JONSWAP → 성분파
        │   ├── waves.py         #   [1-1,1-3] WaveField.elevation / velocity, 얕은 물 효과
        │   ├── bathymetry.py    #   [1-3] 수심·바닥 높이
        │   └── current.py       #   [1-4] 해류
        ├── hydro/hull.py        # 2단계 — 선체 부력·항력 (MuJoCo 비의존)
        ├── boat/boat.py         # 3단계 — 보트: 유체력·추진기를 MuJoCo 에 연결
        ├── sim/                 # MuJoCo 연동
        │   ├── world.py         #   씬 로딩(MjSpec), pre_step 훅, 스텝, 배속 측정
        │   └── water_surface.py #   [1-2] 파도 → water hfield 갱신
        └── assets/
            ├── ocean_scene.xml  #   수면·해저·조명
            └── boat_scene.xml   #   ocean_scene include + 보트·추진기·센서
```

설계 원칙: `ocean/`은 MuJoCo를 모른다 → 단위 테스트가 쉽고, 부력·시각화·ROS 노드가 같은 인터페이스를 공유.

## 실행

```bash
cd ~/study/rescue_sim_ws
source .venv/bin/activate
pip install -e src/rescue_sim            # 최초 1회

pytest -q src/rescue_sim/test
python src/rescue_sim/scripts/run_boat.py                   # 보트 조종 (↑↓ 추력, ←→ 조향, Home 정지)
python src/rescue_sim/scripts/run_boat.py --preset rough --current 0.3 0
MUJOCO_GL=egl python src/rescue_sim/scripts/run_boat.py --video boat.mp4
python src/rescue_sim/scripts/view_ocean.py                 # 바다만
python src/rescue_sim/scripts/bench_boat.py
```

## 확인된 한계 (MuJoCo + Python 구성)

| 항목 | 측정 / 내용 | 영향 · 대응 |
|---|---|---|
| **Python 유체력이 병목** | 1스텝: mj_step 11 µs vs 유체력 175 µs → 보트 1척 약 10배속 (`bench/results/2-3.txt`) | 물체가 늘면 선형으로 느려짐. 익수자 추가 시 ~5배속 예상. 필요하면 유체력을 2스텝마다 계산하거나 C 플러그인/numba |
| numpy 호출 횟수 | 점 개수보다 호출 횟수가 비용을 좌우 | 수면 높이 질의를 한 번에 모아서 호출 (6.3 → 11.9배속) |
| WSL 소프트웨어 렌더링 | llvmpipe 가 CPU 전 코어 사용 → numpy 멀티스레드와 경합 15배 느려짐 | `OPENBLAS_NUM_THREADS=1` (모든 스크립트·테스트) |
| 수면 hfield | 128² 갱신 0.12 ms(사전계산). 512² 는 3.5 ms → 11배속 | 128 유지. 시각 전용(contype=0) |
| 해저 hfield 충돌 | 컴파일 때 충돌 경계 계산 → 높이 범위는 컴파일 전(MjSpec) 확정 필요. userdata 는 컴파일 때 0~1 로 재정규화됨 | 범위는 MjSpec, 데이터는 컴파일 후 직접 기록 |
| 바닥에 닿은 물체 | 보트가 해저에 얹히면 mj_step 11 → 40 µs (hfield 접촉 계산) | 해안 접근 시나리오에서 배속 하락 감안 |
| 부가질량 없음 | 출렁임 주기가 실제보다 짧음 (보트 위아래 고유주기 ≈ 0.65 s) | 4단계 전 필요하면 추가 |
| 선체 = 직육면체 | 부력 기둥 6×3, 작은 기울기 가정 | 조종성은 계수로 맞춤(직진 2.6 m/s, 선회 ~30°/s) |
| 1스텝 지연 | pre_step 은 직전 스텝의 위치로 힘 계산 (2 ms 지연) | 현재 무시 가능 |


### ROS2 (7단계에서 필요)
Ubuntu 24.04 → ROS2 **Jazzy** (`sudo apt install ros-jazzy-ros-base python3-colcon-common-extensions`).
venv는 `--system-site-packages`로 만들어 두었으므로 `source /opt/ros/jazzy/setup.bash` 후 rclpy를 그대로 쓸 수 있다.
