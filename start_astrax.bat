@echo off
chcp 65001 >nul
setlocal
set ROOT=%~dp0
set PORT=8080

echo Starting Astrax engine...
start "Astrax Engine" /min "%ROOT%build\engine\astrax-engine.exe" -m "%ROOT%data\astrax-model.gguf" -t 4 --port %PORT% -a astrax

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
