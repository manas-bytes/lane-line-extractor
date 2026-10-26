@echo off
cd /d "%~dp0"
echo ============================================================
echo   Starting Lane Line Extractor Web Dashboard...
echo ============================================================
echo.
echo Your browser will open automatically at http://127.0.0.1:5000
echo You can drag and drop any road image into the browser!
echo.
echo Press Ctrl+C in this window anytime to stop the server.
echo.
python web_app.py
pause
