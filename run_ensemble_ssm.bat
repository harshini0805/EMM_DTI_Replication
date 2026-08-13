@echo off
REM Ensemble Training Launcher (SSM Version) - Windows Batch
REM Runs EMM-DTI with OFFICIAL Mamba-SSM in background
REM Results saved to: results/ensemble_ssm/ (SEPARATE from standard version)

setlocal enabledelayedexpansion

cd /d D:\EMM_DTI_Replication\EMM_DTI_Replication

echo ========================================
echo EMM-DTI Ensemble Training (Mamba-SSM)
echo ========================================
echo.

REM Check if venv exists
if not exist venv (
    echo [ERROR] Virtual environment not found
    echo Please create it first:
    echo   python -m venv venv
    echo   venv\Scripts\activate.bat
    pause
    exit /b 1
)

REM Install mamba-ssm if needed
echo [INFO] Checking mamba-ssm installation...
venv\Scripts\python.exe -c "import mamba_ssm" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing mamba-ssm...
    venv\Scripts\pip.exe install mamba-ssm
)

REM Create results directory
if not exist results\ensemble_ssm (
    mkdir results\ensemble_ssm
)

echo [INFO] Starting ensemble training as background process (Mamba-SSM)...
echo.

REM Run training in background
start "" /B venv\Scripts\python.exe ensemble_train_ssm.py > results\ensemble_ssm\training.log 2>&1

echo ========================================
echo SUCCESS!
echo ========================================
echo.
echo Training started in background (5 seeds, Mamba-SSM)
echo Estimated time: ~150-200 minutes
echo.
echo View logs:
echo   type results\ensemble_ssm\training.log
echo.
echo Or (live):
echo   powershell -Command "Get-Content results\ensemble_ssm\training.log -Tail 50 -Wait"
echo.
echo COMPARE RESULTS:
echo   Standard:  results\ensemble\ensemble_results.json
echo   SSM:       results\ensemble_ssm\ensemble_results.json
echo.
echo YOUR LAPTOP CAN SLEEP - TRAINING CONTINUES!
echo.
pause
