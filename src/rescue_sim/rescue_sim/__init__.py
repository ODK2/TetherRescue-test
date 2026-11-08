"""rescue_sim — MuJoCo 물가 구조 시뮬레이션.

패키지 구성 (단계별로 채워 나감)
  ocean/  1단계: 파도·해저·해류 (MuJoCo 의존성 없음, 순수 numpy)
  sim/    MuJoCo 월드 로딩·스텝 루프·시각화
  (hydro/ 2단계, boat/ 3단계, ... 는 해당 단계에서 추가)
"""
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
ASSETS_DIR = PKG_DIR / "assets"
CONFIG_DIR = PKG_DIR.parent / "config"   # 소스 트리 기준. ROS 설치 후엔 share/ 경로 사용
