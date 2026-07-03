@echo off
setlocal

cd /d "%~dp0"

echo ==========================================
echo   Repair SignLink Python Environment
echo ==========================================
echo.

set "PY_CMD="

where py >nul 2>nul
if not errorlevel 1 (
  set "PY_CMD=py -3"
)

if not defined PY_CMD (
  where python >nul 2>nul
  if not errorlevel 1 (
    set "PY_CMD=python"
  )
)

if not defined PY_CMD (
  echo [ERROR] Khong tim thay Python.
  echo Hay cai Python 3.9-3.11, chon "Add Python to PATH", roi chay lai file nay.
  pause
  exit /b 1
)

echo [1/4] Python:
%PY_CMD% --version
if errorlevel 1 (
  echo [ERROR] Python khong chay duoc.
  pause
  exit /b 1
)

echo.
echo [2/4] Tao lai virtual environment...
if exist venv (
  rmdir /s /q venv
)
%PY_CMD% -m venv venv
if errorlevel 1 (
  echo [ERROR] Khong tao duoc venv.
  pause
  exit /b 1
)

echo.
echo [3/4] Nang cap pip...
venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 (
  echo [ERROR] Khong nang cap duoc pip.
  pause
  exit /b 1
)

echo.
echo [4/4] Cai dependencies...
venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
  echo [ERROR] Cai dependencies that bai.
  echo Neu loi o torch, hay cai thu cong:
  echo venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
  pause
  exit /b 1
)

echo.
echo [OK] Da sua xong venv. Bay gio chay:
echo run_web_local_vsl.bat
pause
