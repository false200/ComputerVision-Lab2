@echo off
REM CV Lab 2 launcher — prefers Python 3.11 if present on this machine.
setlocal
set PY=
if exist "%LocalAppData%\Programs\Python\Python311\python.exe" (
  set "PY=%LocalAppData%\Programs\Python\Python311\python.exe"
) else (
  set PY=python
)
"%PY%" -m pip install -r "%~dp0requirements.txt" -q
"%PY%" "%~dp0run_lab.py" %*
echo.
echo Outputs are in: %~dp0outputs
pause
