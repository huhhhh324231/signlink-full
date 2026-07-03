@echo off
REM Auto-setup script for gesture recognition project
REM Chỉ cần chạy file này 1 lần là xong

echo [+] Setting up environment...

REM Create virtual environment
python -m venv venv >nul 2>&1

REM Activate and install
call venv\Scripts\activate.bat
echo [+] Installing dependencies (this may take 2-3 minutes)...
pip install -q -r requirements.txt

REM Verify
echo [+] Verifying installation...
python check_setup.py

echo.
echo [✓] Setup complete!
echo [✓] Run: python gesture_recognition.py
echo.
pause
