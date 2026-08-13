# Ensemble Training Launcher (SSM Version) - Windows PowerShell
#
# Runs EMM-DTI with OFFICIAL Mamba-SSM in background PowerShell job
# Results saved to: results/ensemble_ssm/ (separate from standard version)
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File run_ensemble_ssm.ps1

param(
    [string]$Action = "start"
)

$ProjectRoot = "D:\EMM_DTI_Replication\EMM_DTI_Replication"
$VenvPath = Join-Path $ProjectRoot "venv"
$LogDir = Join-Path $ProjectRoot "results" "ensemble_ssm"
$LogFile = Join-Path $LogDir "training.log"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "EMM-DTI Ensemble Training (Mamba-SSM)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Project root: $ProjectRoot"
Write-Host "Virtual environment: $VenvPath"
Write-Host "Log file: $LogFile"
Write-Host "Results: $LogDir (SEPARATE from standard version)"
Write-Host ""

# ============================================================
# Check if venv exists
# ============================================================

if (-not (Test-Path $VenvPath)) {
    Write-Host "[ERROR] Virtual environment not found at: $VenvPath" -ForegroundColor Red
    Write-Host "Please create and activate your venv first:" -ForegroundColor Yellow
    Write-Host "  cd $ProjectRoot"
    Write-Host "  python -m venv venv"
    Write-Host "  .\venv\Scripts\Activate.ps1"
    exit 1
}

# ============================================================
# Install mamba-ssm if needed
# ============================================================

Write-Host "[INFO] Checking if mamba-ssm is installed..." -ForegroundColor Green
& "$VenvPath\Scripts\python.exe" -c "import mamba_ssm" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[INFO] Installing mamba-ssm..." -ForegroundColor Yellow
    & "$VenvPath\Scripts\pip.exe" install mamba-ssm
}

# ============================================================
# Ensure log directory exists
# ============================================================

if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

# ============================================================
# Start training as background job
# ============================================================

Write-Host "[INFO] Starting ensemble training as background job (Mamba-SSM)..." -ForegroundColor Green
Write-Host ""

$ScriptBlock = {
    param($ProjectRoot, $VenvPath, $LogFile)

    Set-Location $ProjectRoot
    $env:Path = "$VenvPath\Scripts;$env:Path"

    Write-Output "========================================"
    Write-Output "EMM-DTI Ensemble Training (Mamba-SSM)"
    Write-Output "========================================"
    Write-Output "Start time: $(Get-Date)"
    Write-Output "Model: EMMDTI_SSM (Official Mamba-SSM)"
    Write-Output "Seeds: [42, 123, 2024, 456, 789]"
    Write-Output "Epochs per seed: 200"
    Write-Output "Patience: 30"
    Write-Output ""

    & "$VenvPath\Scripts\python.exe" ensemble_train_ssm.py

    Write-Output ""
    Write-Output "========================================"
    Write-Output "Ensemble Training Complete (Mamba-SSM)!"
    Write-Output "========================================"
    Write-Output "End time: $(Get-Date)"
    Write-Output "Check results in: results/ensemble_ssm/"
    Write-Output ""
}

$Job = Start-Job -ScriptBlock $ScriptBlock -ArgumentList $ProjectRoot, $VenvPath, $LogFile

$JobId = $Job.Id

Write-Host "========================================" -ForegroundColor Green
Write-Host "SUCCESS!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Job ID: $JobId" -ForegroundColor Yellow
Write-Host "Status: Running in background"
Write-Host "Model: Official Mamba-SSM (Selective StateSpaceModel)"
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "MONITOR TRAINING (Mamba-SSM)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. View running jobs:" -ForegroundColor Green
Write-Host "   Get-Job"
Write-Host ""
Write-Host "2. View job output (live):" -ForegroundColor Green
Write-Host "   Receive-Job -Id $JobId -Keep | tail -20"
Write-Host ""
Write-Host "3. View full log file:" -ForegroundColor Green
Write-Host "   Get-Content '$LogFile' -Tail 50 -Wait"
Write-Host ""
Write-Host "4. Check job status:" -ForegroundColor Green
Write-Host "   Get-Job -Id $JobId | Select-Object State, Status"
Write-Host ""
Write-Host "5. Stop the job (when done):" -ForegroundColor Green
Write-Host "   Stop-Job -Id $JobId"
Write-Host "   Remove-Job -Id $JobId"
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "COMPARE RESULTS" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Standard version (simplified Mamba):"
Write-Host "   results/ensemble/ensemble_results.json"
Write-Host ""
Write-Host "SSM version (official Mamba-SSM):"
Write-Host "   results/ensemble_ssm/ensemble_results.json"
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "YOUR LAPTOP CAN SLEEP - TRAINING CONTINUES!" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
