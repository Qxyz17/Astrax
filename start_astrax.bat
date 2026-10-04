@echo off
chcp 65001 >nul
setlocal
set ROOT=%~dp0

echo Starting Astrax backend (llama-server)...
start "Astrax backend" /min "%ROOT%build\llama-cpu\bin\llama-server.exe" -m "%ROOT%data\LFM2-350M-Q4_K_M.gguf" -t 4 --port 8080

echo Waiting for backend...
timeout /t 6 /nobreak >nul

echo Starting Astrax frontend (Vite)...
cd /d "%ROOT%web"
start "Astrax frontend" cmd /c "npm run dev"

echo.
echo Astrax is starting.
echo Open http://localhost:5173 in your browser.
echo Close the two new windows to stop.
endlocal
