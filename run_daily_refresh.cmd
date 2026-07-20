@echo off
setlocal EnableExtensions

chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

set "PROJECT=C:\Users\user\Desktop\ap-flood-ews_github_update"
set "PYTHON=C:\Users\user\Desktop\ap-flood-ews_github_update\.venv\Scripts\python.exe"
set "LOGDIR=%PROJECT%\outputs\reports"
set "LOGFILE=%LOGDIR%\daily_refresh_task.log"
set "FINAL_RESULT=0"

if not exist "%LOGDIR%" mkdir "%LOGDIR%"
cd /d "%PROJECT%"

echo.>> "%LOGFILE%"
echo ============================================================>> "%LOGFILE%"
echo DAILY REFRESH START %DATE% %TIME%>> "%LOGFILE%"

echo FIRST PASS - REFRESH ALL DISTRICTS>> "%LOGFILE%"
"%PYTHON%" main.py all-districts --project ap-flood-monitor >> "%LOGFILE%" 2>&1
if errorlevel 1 set "FINAL_RESULT=1"

echo SECOND PASS - RETRY FAILED DISTRICTS>> "%LOGFILE%"
"%PYTHON%" main.py all-districts --project ap-flood-monitor --resume >> "%LOGFILE%" 2>&1

echo BUILD STATEWIDE DATA>> "%LOGFILE%"
"%PYTHON%" main.py aggregate >> "%LOGFILE%" 2>&1
if errorlevel 1 set "FINAL_RESULT=1"

echo GENERATE MAPS>> "%LOGFILE%"
"%PYTHON%" main.py map >> "%LOGFILE%" 2>&1
if errorlevel 1 set "FINAL_RESULT=1"

echo EVALUATE ALERTS>> "%LOGFILE%"
"%PYTHON%" main.py alert >> "%LOGFILE%" 2>&1
if errorlevel 1 set "FINAL_RESULT=1"

echo DAILY REFRESH END %DATE% %TIME% EXIT=%FINAL_RESULT%>> "%LOGFILE%"
echo ============================================================>> "%LOGFILE%"

exit /b %FINAL_RESULT%
