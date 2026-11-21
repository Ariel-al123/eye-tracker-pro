@echo off
cd /d "%~dp0"

rem Prefer the project virtualenv (needed by the optional EyeTheia engine),
rem but fall back to the system Python so the default EyeGestures build still runs.
if exist ".venv\Scripts\python.exe" (
    set "PY=.venv\Scripts\python.exe"
) else (
    set "PY=python"
)

"%PY%" ensure_dependencies.py
if errorlevel 1 (
    echo No se pudieron instalar las dependencias.
    pause
    exit /b 1
)
"%PY%" main.py
pause
