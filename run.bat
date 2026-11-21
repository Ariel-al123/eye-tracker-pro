@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo No se encontro el entorno virtual .venv.
    echo Crealo con Python 3.11 y uv antes de ejecutar el rastreador.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" main.py
pause
