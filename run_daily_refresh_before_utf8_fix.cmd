@echo off
setlocal

set "PROJECT=C:\Users\user\Desktop\flood_monitor_clean"
set "PYTHON=C:\Users\user\Desktop\flood_monitor\.venv\Scripts\python.exe"
set "LOGDIR=%PROJECT%\outputs\reports"

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

echo.>> "%LOGDIR%\daily_refresh_task.log"
echo ============================================================>> "%LOGDIR%\daily_refresh_task.log"
echo START %DATE% %TIME%>> "%LOGDIR%\daily_refresh_task.log"

"%PYTHON%" main.py refresh --project ap-flood-monitor >> "%LOGDIR%\daily_refresh_task.log" 2>&1
set "RESULT=%ERRORLEVEL%"

echo END %DATE% %TIME% EXIT=%RESULT%>> "%LOGDIR%\daily_refresh_task.log"
echo ============================================================>> "%LOGDIR%\daily_refresh_task.log"

exit /b %RESULT%
