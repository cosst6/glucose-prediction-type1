@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python libre_uploader.py login
pause
