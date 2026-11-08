from glob import glob

from setuptools import find_packages, setup

PACKAGE = "rescue_sim"

setup(
    name=PACKAGE,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    package_data={PACKAGE: ["assets/*.xml"]},
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{PACKAGE}"]),
        (f"share/{PACKAGE}", ["package.xml"]),
        (f"share/{PACKAGE}/config", glob("config/*.yaml")),
        (f"share/{PACKAGE}/launch", glob("launch/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="user1",
    description="MuJoCo 기반 물가 익수자 구조 시뮬레이션",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            # 7단계(ROS2 연동)에서 노드 추가
        ],
    },
)
