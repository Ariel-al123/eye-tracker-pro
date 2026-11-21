@echo off
setlocal
cd /d "%~dp0"

set "UV_EXE="
where uv.exe >nul 2>nul
if not errorlevel 1 set "UV_EXE=uv.exe"
if not defined UV_EXE if exist "%USERPROFILE%\.local\bin\uv.exe" set "UV_EXE=%USERPROFILE%\.local\bin\uv.exe"

if not defined UV_EXE (
    echo No se encontro uv.
    echo Instala uv desde https://docs.astral.sh/uv/getting-started/installation/
    goto :error
)

if not exist ".venv\Scripts\python.exe" (
    echo Creando el entorno virtual con Python 3.11...
    "%UV_EXE%" venv --python 3.11 .venv
    if errorlevel 1 goto :error
) else (
    ".venv\Scripts\python.exe" -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)"
    if errorlevel 1 (
        echo El entorno .venv existe, pero no usa Python 3.11.
        echo Renombra o elimina .venv y vuelve a ejecutar este archivo.
        goto :error
    )
)

echo Instalando dependencias de Eye Tracker Pro y EyeGestures...
"%UV_EXE%" pip install --cache-dir .uv-cache --python ".venv\Scripts\python.exe" -r requirements.txt -r requirements-eyegestures.txt
if errorlevel 1 goto :error

echo.
echo Dependencias instaladas. Ya puedes iniciar el programa con run.bat.
pause
exit /b 0

:error
echo.
echo No se pudieron instalar todas las dependencias.
pause
exit /b 1
