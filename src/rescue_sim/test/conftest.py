import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # scripts/view_ocean.py 참고

import sys
from pathlib import Path

# colcon/pip 설치 없이도 소스 트리에서 테스트 가능하게
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
