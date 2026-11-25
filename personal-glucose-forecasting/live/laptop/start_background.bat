@echo off
REM Starts the reader watcher and the 15-min forecaster with no windows. Runs until you log off.
cd /d "%~dp0"
start "" .venv\Scripts\pythonw.exe libre_uploader.py watch
start "" .venv\Scripts\pythonw.exe forecaster.py loop
