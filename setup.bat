@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ============================================================
echo   stock_factor_lab  Environment Setup  (Python 3.10 / venv)
echo ============================================================
echo.

REM ---------- 0. Check Python 3.10 ----------
py -3.10 --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python 3.10 not found. Install Python 3.10 with the "py launcher" option.
    echo         Then open a new cmd and run this script again.
    goto :fail
)
echo [OK] Python 3.10 detected
echo.

REM ---------- 1. Create virtual environment ----------
echo [Step 1/5] Create .venv ...
if exist ".venv\Scripts\python.exe" (
    echo   .venv already exists, skip.
) else (
    py -3.10 -m venv .venv
    if errorlevel 1 ( echo [ERROR] venv creation failed & goto :fail )
    echo   [OK] .venv created
)
echo.

REM ---------- 2. Activate and upgrade pip ----------
echo [Step 2/5] Activate env and upgrade pip ...
call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
echo.

REM ---------- 3. Install requirements (pins numpy 1.24.4) ----------
echo [Step 3/5] Install requirements_clean.txt ...
pip install -r requirements_clean.txt
if errorlevel 1 ( echo [ERROR] requirements install failed & goto :fail )
echo.

REM ---------- 4. Install TA-Lib (cp310, --no-deps) ----------
echo [Step 4/5] Install TA-Lib (cp310) ...
pip install --no-deps "TA-LIB\ta_lib-0.6.3-cp310-cp310-win_amd64.whl"
if errorlevel 1 ( echo [ERROR] TA-Lib install failed & goto :fail )
echo.

REM ---------- 5. Build Cython backtest core ----------
echo [Step 5/5] Build core\backtest_core (needs VS C++ Build Tools) ...
pushd core
python setup.py build_ext --inplace
set BUILD_RC=!errorlevel!
popd
if not "!BUILD_RC!"=="0" (
    echo   [WARN] Cython build failed, usually missing Microsoft C++ Build Tools.
    echo          Install then re-run setup.bat:
    echo          https://visualstudio.microsoft.com/zh-hant/visual-cpp-build-tools/
    echo          All other packages are installed.
    goto :end
)
echo   [OK] core built
echo.

echo ============================================================
echo   ALL DONE!
echo   - Activate env:  .venv\Scripts\activate
echo   - See README.md / 安裝說明_SETUP.md for how to run the pipeline
echo ============================================================
goto :end

:fail
echo.
echo *** Setup stopped. Fix the issue above and re-run setup.bat ***

:end
echo.
pause
