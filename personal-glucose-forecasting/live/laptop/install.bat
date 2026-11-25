@echo off
REM One-time setup on Windows 10. Needs Python 3.13 from python.org (tick "Add python.exe to PATH").
cd /d "%~dp0"
python -m venv .venv || goto :err
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt || goto :err
if not exist config.json copy config.example.json config.json
echo.
echo Installed. Next: edit config.json, then run  login.bat
pause
exit /b 0
:err
echo Something failed above. Copy the red text and send it to Claude.
pause
exit /b 1
