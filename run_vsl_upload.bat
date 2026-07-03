@echo off
REM Pick an existing video file and run Vietnamese sign-language word recognition.

call venv\Scripts\activate.bat
venv\Scripts\python.exe vsl_word_video_recognition.py --upload
pause
