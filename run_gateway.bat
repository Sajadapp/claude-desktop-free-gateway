@echo off
REM Start the local inference gateway (opencode free-tier -> Claude Desktop).
REM The gateway auto-spawns `opencode serve` if it is not already running.
cd /d "%~dp0"
call "%~dp0.venv\Scripts\activate.bat"
uvicorn gateway:app --host 127.0.0.1 --port 3457
