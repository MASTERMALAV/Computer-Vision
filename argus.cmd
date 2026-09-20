@echo off
setlocal
rem Runs ARGUS with the project virtualenv, so it works no matter which
rem Python happens to be first on PATH.
set "ARGUS_DIR=%~dp0"
set "ARGUS_PY=%ARGUS_DIR%.venv\Scripts\python.exe"
if not exist "%ARGUS_PY%" (
  echo.
  echo   The project virtualenv is missing: %ARGUS_PY%
  echo.
  echo   Create it with:
  echo     py -3.12 -m venv .venv
  echo     .venv\Scripts\python.exe -m pip install -r requirements-core.txt
  echo     .venv\Scripts\python.exe -m argus models pull
  echo.
  exit /b 1
)
pushd "%ARGUS_DIR%"
"%ARGUS_PY%" -m argus %*
set "ARGUS_RC=%ERRORLEVEL%"
popd
exit /b %ARGUS_RC%
