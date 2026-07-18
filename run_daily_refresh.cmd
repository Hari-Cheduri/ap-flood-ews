@echo off
setlocal EnableExtensions

chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

set "PROJECT=C:\Users\user\Desktop\flood_monitor_clean"
set "PYTHON=C:\Users\user\Desktop\flood_monitor\.venv\Scripts\python.exe"
set "LOGDIR=%PROJECT%\outputs\reports"
set "LOGFILE=%LOGDIR%\daily_refresh_task.log"

if not exist "%PROJECT%\main.py" (
    echo ERROR: Project not found: %PROJECT%
    exit /b 1
)

if not exist "%PYTHON%" (
    echo ERROR: Python environment not found: %PYTHON%
    exit /b 1
)

if not exist "%LOGDIR%" mkdir "%LOGDIR%"

cd /d "%PROJECT%"

echo.>> "%LOGFILE%"
echo ============================================================>> "%LOGFILE%"
echo START %DATE% %TIME%>> "%LOGFILE%"

"%PYTHON%" main.py refresh --project ap-flood-monitor >> "%LOGFILE%" 2>&1
set "RESULT=%ERRORLEVEL%"

echo END %DATE% %TIME% EXIT=%RESULT%>> "%LOGFILE%"
echo ============================================================>> "%LOGFILE%"

exit /b %RESULT%
