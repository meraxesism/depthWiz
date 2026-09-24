@echo off
setlocal
set "SCRIPT_DIR=%~dp0"

where node >nul 2>nul
if errorlevel 1 (
  echo ERROR: Node.js is required to run DepthWizard.
  echo Install Node.js and re-run this launcher.
  exit /b 1
)

node "%SCRIPT_DIR%start-depthwiz.js" %*
exit /b %ERRORLEVEL%
