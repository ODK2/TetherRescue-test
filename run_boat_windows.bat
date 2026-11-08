@echo off
rem WSLg 뷰어 창이 안 보일 때: Windows 네이티브 Python으로 MuJoCo 뷰어 실행.
rem 코드는 WSL 쪽 소스를 그대로 사용 (수정하면 바로 반영됨).
rem 사용: 탐색기에서 더블클릭, 또는  run_boat_windows.bat --preset rough
rem 최초 1회 설정 (이미 완료):
rem   py -3.12 -m venv %USERPROFILE%\rescue_sim_venv
rem   %USERPROFILE%\rescue_sim_venv\Scripts\pip install mujoco==4.0.0 numpy pyyaml imageio[ffmpeg]

rem MuJoCo는 \\wsl.localhost\... (UNC) 경로를 못 읽음 -> pushd 로 임시 드라이브 문자(Z: 등) 할당
pushd "\\wsl.localhost\Ubuntu-24.04\home\user1\study\rescue_sim_ws\src\rescue_sim"
set "SRC=%CD%"
set "PYTHONPATH=%SRC%"
set "PYTHONUTF8=1"
"%USERPROFILE%\rescue_sim_venv\Scripts\python.exe" "%SRC%\scripts\run_boat.py" %*
set "RC=%ERRORLEVEL%"
popd
if not "%RC%"=="0" pause
