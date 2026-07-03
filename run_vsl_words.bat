@echo off
REM Run Vietnamese sign-language word recognition with recording/upload support.

call venv\Scripts\activate.bat
venv\Scripts\python.exe vsl_word_video_recognition.py --camera
pause
