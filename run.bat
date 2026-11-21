@echo off
cd /d "%~dp0"
python ensure_dependencies.py
if errorlevel 1 (
    echo Failed to install missing dependencies.
    pause
    exit /b 1
)
python main.py
pause
