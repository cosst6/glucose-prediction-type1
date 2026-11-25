@echo off
REM Makes start_background.bat run every time you log in to Windows (Task Scheduler, your user only).
cd /d "%~dp0"
schtasks /Create /F /SC ONLOGON /TN "GlucoseForecast" /TR "\"%~dp0start_background.bat\"" /RL LIMITED
echo Done. To undo:  schtasks /Delete /TN GlucoseForecast /F
pause
