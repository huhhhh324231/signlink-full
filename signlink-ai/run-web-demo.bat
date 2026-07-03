@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

set "NODE_DIR=C:\Program Files\nodejs"
set "NPM_CMD=%NODE_DIR%\npm.cmd"

if not exist "%NPM_CMD%" (
  echo [ERROR] Khong tim thay npm.cmd trong "%NODE_DIR%"
  echo [HINT] Hay kiem tra lai cai dat Node.js.
  pause
  exit /b 1
)

set "PATH=%NODE_DIR%;%PATH%"

echo ==========================================
echo   SignLink Letters - Web Demo Launcher
echo ==========================================
echo.
echo [1/3] Kiem tra dependencies...
if not exist "node_modules" (
  echo [INFO] Chua co node_modules, dang cai dat...
  call "%NPM_CMD%" install
  if errorlevel 1 (
    echo [ERROR] Cai dat dependencies that bai.
    pause
    exit /b 1
  )
) else (
  echo [OK] Dependencies da san sang.
)

set "LAN_IP="
for /f "tokens=2 delims=:" %%A in ('ipconfig ^| findstr /c:"IPv4 Address" /c:"IPv4-address"') do (
  set "LAN_IP=%%A"
  goto :trim_ip
)

:trim_ip
if defined LAN_IP (
  set "LAN_IP=%LAN_IP: =%"
)

echo.
echo [2/3] Link mo tren may tinh:
echo       http://localhost:3000
if defined LAN_IP (
  echo [3/3] Link mo tren dien thoai cung Wi-Fi:
  echo       http://%LAN_IP%:3000
) else (
  echo [3/3] Khong doc duoc IP tu dong. Ban co the chay "ipconfig" de xem IPv4 Address.
)
echo.
echo [NOTE] Khi trinh duyet hoi quyen camera, hay bam Allow.
echo [NOTE] Giu cua so nay mo trong suot qua trinh demo.
echo.

call "%NPM_CMD%" run dev

endlocal
