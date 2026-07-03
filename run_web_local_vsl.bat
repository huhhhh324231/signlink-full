@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

echo ==========================================
echo   SignLink Local Web + VSL Word API
echo ==========================================
echo.
echo Starting local web. npm run dev will auto-start Python API.
cd /d "%~dp0signlink-ai"
call npm run dev

endlocal
