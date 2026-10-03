@echo off
chcp 65001 >nul
setlocal
set ROOT=%~dp0
set MODEL=%ROOT%data\LFM2-350M-Q4_K_M.gguf
set CLI=%ROOT%build\llama-cpu\bin\llama-cli.exe
if not exist "%MODEL%" (
  echo Model not found: %MODEL%
  echo Run: python tools\lfm\download_model.py
  exit /b 1
)
if "%~1"=="" (
  "%CLI%" -m "%MODEL%" -t 4 -st -cnv
) else (
  "%CLI%" -m "%MODEL%" -p "%~1" -n 256 -t 4 -st
)
