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

REM ---------- 1. Git LFS pull (optional) ----------
echo [Step 1/6] Fetch Git LFS notebooks (optional)...
where git >nul 2>&1
if errorlevel 1 (
    echo   [SKIP] git not found. example.ipynb is already a full file, so this is fine.
) else (
    git lfs version >nul 2>&1
    if errorlevel 1 (
        echo   [SKIP] git-lfs not installed. example.ipynb is already a full file, so this is fine.
    ) else (
        git lfs install
        git lfs pull
        echo   [OK] git lfs pull done
    )
)
echo.

REM ---------- 2. Create virtual environment ----------
echo [Step 2/6] Create .venv ...
if exist ".venv\Scripts\python.exe" (
    echo   .venv already exists, skip.
) else (
    py -3.10 -m venv .venv
    if errorlevel 1 ( echo [ERROR] venv creation failed & goto :fail )
    echo   [OK] .venv created
)
echo.

REM ---------- 3. Activate and upgrade pip ----------
echo [Step 3/6] Activate env and upgrade pip ...
call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
echo.

REM ---------- 4. Install requirements (pins numpy 1.24.4) ----------
echo [Step 4/6] Install requirements_clean.txt ...
pip install -r requirements_clean.txt
if errorlevel 1 ( echo [ERROR] requirements install failed & goto :fail )
echo.

REM ---------- 5. Install TA-Lib (cp310, --no-deps) ----------
echo [Step 5/6] Install TA-Lib (cp310) ...
pip install --no-deps "TA-LIB\ta_lib-0.6.3-cp310-cp310-win_amd64.whl"
if errorlevel 1 ( echo [ERROR] TA-Lib install failed & goto :fail )
echo.

REM ---------- 6. Build Cython backtest core ----------
echo [Step 6/6] Build core\backtest_core (needs VS C++ Build Tools) ...
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
echo   - Open code\example.ipynb, select the .venv Python as Kernel
echo ============================================================
goto :end

:fail
echo.
echo *** Setup stopped. Fix the issue above and re-run setup.bat ***

:end
echo.
pause
