@echo off
REM Plug in the reader, then double-click this to upload once and make a forecast.
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python libre_uploader.py sync && python forecaster.py once
pause
