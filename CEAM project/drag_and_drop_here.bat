@echo off
setlocal enabledelayedexpansion

:: Switch to the directory where this batch script lives
cd /d "%~dp0"

echo ============================================================
echo      LANE LINE EXTRACTOR - DRAG & DROP RUNNER
echo ============================================================

set "IMAGE_PATH=%~1"

:: If no file was dragged onto the icon, prompt the user
if "%IMAGE_PATH%"=="" (
    echo.
    echo [TIP] You can drag any road image file from your computer
    echo       and DROP IT right into this window!
    echo.
    set /p "IMAGE_PATH=Drag and drop your road image here, then press Enter: "
)

:: Strip surrounding quotes from the path
set IMAGE_PATH=!IMAGE_PATH:"=!
set IMAGE_PATH=!IMAGE_PATH:'=!

:: Trim trailing spaces if any
for /f "tokens=* delims= " %%a in ("!IMAGE_PATH!") do set IMAGE_PATH=%%a

if "!IMAGE_PATH!"=="" (
    echo [ERROR] No image file provided!
    echo.
    pause
    exit /b 1
)

if not exist "!IMAGE_PATH!" (
    echo.
    echo [ERROR] File not found: "!IMAGE_PATH!"
    echo Please make sure the path is correct.
    echo.
    pause
    exit /b 1
)

echo.
echo [START] Processing image: "!IMAGE_PATH!"
echo.

:: Run lane_extractor.py with the provided image
python lane_extractor.py --input "!IMAGE_PATH!" --show

:: Get image name without path or extension
for %%F in ("!IMAGE_PATH!") do set "IMG_NAME=%%~nF"

echo.
echo ============================================================
echo [SUCCESS] Processing complete!
echo All stages saved into: output\!IMG_NAME!\
echo ============================================================
echo.

:: Open the output folder in Windows Explorer automatically
if exist "output\!IMG_NAME!" (
    echo Opening output folder: output\!IMG_NAME!\
    explorer "output\!IMG_NAME!"
)

echo.
pause
