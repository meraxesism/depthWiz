@echo off
setlocal
set "PROJECT=%~dp0"
set "PYTHON=%PROJECT%..\.venv\Scripts\python.exe"
set "MODEL=C:\temp\depth-anything-v2-small-quantized.onnx"

if not exist "%PROJECT%package.json" (
  echo ERROR: Run this file from the DepthWizard project folder.
  exit /b 1
)
if not exist "%PYTHON%" (
  echo ERROR: Python environment not found at "%PYTHON%".
  echo Create it or update PYTHON in this file.
  exit /b 1
)
if not exist "%MODEL%" (
  echo WARNING: Depth Anything model not found at "%MODEL%".
  echo The Python backend will start, but /process will return 503 until the model path is set.
)

echo Starting DepthWizard services...
powershell.exe -NoProfile -Command "$c=Get-NetTCPConnection -LocalPort 8787 -State Listen -ErrorAction SilentlyContinue; if(-not $c){exit 1}"
if errorlevel 1 start "DepthWizard TypeScript API" powershell.exe -NoExit -ExecutionPolicy Bypass -Command "Set-Location -LiteralPath '%PROJECT%'; npm run server"
powershell.exe -NoProfile -Command "$c=Get-NetTCPConnection -LocalPort 5173 -State Listen -ErrorAction SilentlyContinue; if(-not $c){exit 1}"
if errorlevel 1 start "DepthWizard Vite frontend" powershell.exe -NoExit -ExecutionPolicy Bypass -Command "Set-Location -LiteralPath '%PROJECT%'; npm run dev"
powershell.exe -NoProfile -Command "$c=Get-NetTCPConnection -LocalPort 8788 -State Listen -ErrorAction SilentlyContinue; if(-not $c){exit 1}"
if errorlevel 1 start "DepthWizard Python API" powershell.exe -NoExit -ExecutionPolicy Bypass -Command "$env:DEPTHWIZ_DEPTH_MODEL_PATH='%MODEL%'; Set-Location -LiteralPath '%PROJECT%'; & '%PYTHON%' -m uvicorn backend.main:app --host 127.0.0.1 --port 8788"

echo.
echo Frontend:    http://localhost:5173/
echo TypeScript:  http://localhost:8787/api/health
echo Python API:  http://localhost:8788/health
echo.
echo Close the three service windows to stop DepthWizard.
endlocal
