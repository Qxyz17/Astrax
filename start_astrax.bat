@echo off
chcp 65001 >nul
setlocal
set ROOT=%~dp0
set PORT=8080

REM The local engine and model live outside this repository so the project
REM itself stays free of third-party model files. Point ASTRAX_ENGINE and
REM ASTRAX_MODEL at them, or keep the defaults below.
set ENGINE=%ASTRAX_ENGINE%
set MODEL=%ASTRAX_MODEL%
if "%ENGINE%"=="" set ENGINE=D:\AstraxLocal\build\engine\llama-server.exe
if "%MODEL%"==""  set MODEL=D:\AstraxLocal\data\LFM2-350M-Q4_K_M.gguf

echo Starting Astrax engine...
start "Astrax Engine" /min "%ENGINE%" -m "%MODEL%" -t 4 --port %PORT%

echo Waiting for engine...
timeout /t 6 /nobreak >nul

echo Starting Astrax interface...
cd /d "%ROOT%web"
start "Astrax Interface" cmd /c "npm run dev"

echo.
echo Astrax is starting.
echo Open http://localhost:5173 in your browser.
echo Close the two new windows to stop.
endlocal
