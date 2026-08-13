#!/bin/bash

################################################################################
# Ensemble Training Launcher (SSM Version) - Linux/macOS tmux
#
# Runs EMM-DTI with OFFICIAL Mamba-SSM in tmux session
# Results saved to: results/ensemble_ssm/ (SEPARATE from standard version)
#
# Usage:
#   bash run_ensemble_ssm.sh
#
################################################################################

set -e

SESSION_NAME="emm_ensemble_ssm"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${PROJECT_ROOT}/venv"
LOGFILE="${PROJECT_ROOT}/results/ensemble_ssm/training.log"

echo "========================================"
echo "EMM-DTI Ensemble Training (Mamba-SSM)"
echo "========================================"
echo "Project root: ${PROJECT_ROOT}"
echo "Session name: ${SESSION_NAME}"
echo "Log file: ${LOGFILE}"
echo "Model: Official Mamba-SSM (Selective StateSpaceModel)"
echo ""

# ============================================================
# Check if session already exists
# ============================================================

if tmux has-session -t ${SESSION_NAME} 2>/dev/null; then
    echo "[INFO] Session '${SESSION_NAME}' already running."
    echo ""
    echo "To attach to the running session:"
    echo "  tmux attach -t ${SESSION_NAME}"
    echo ""
    echo "To view the last 50 lines:"
    echo "  tmux capture-pane -t ${SESSION_NAME} -p | tail -50"
    echo ""
    echo "To stop the session:"
    echo "  tmux kill-session -t ${SESSION_NAME}"
    exit 0
fi

# ============================================================
# Verify venv exists
# ============================================================

if [ ! -d "${VENV_PATH}" ]; then
    echo "[ERROR] Virtual environment not found at: ${VENV_PATH}"
    echo "Please activate your venv first:"
    echo "  cd ${PROJECT_ROOT}"
    echo "  source venv/bin/activate"
    exit 1
fi

# ============================================================
# Install mamba-ssm if needed
# ============================================================

echo "[INFO] Checking mamba-ssm installation..."
if ! ${VENV_PATH}/bin/python -c "import mamba_ssm" 2>/dev/null; then
    echo "[INFO] Installing mamba-ssm..."
    ${VENV_PATH}/bin/pip install mamba-ssm
fi

# ============================================================
# Create tmux session and start training
# ============================================================

echo "[INFO] Starting tmux session: ${SESSION_NAME}"

mkdir -p "${PROJECT_ROOT}/results/ensemble_ssm"

tmux new-session -d -s ${SESSION_NAME} -c "${PROJECT_ROOT}" \
    "
    set -e
    source ${VENV_PATH}/bin/activate

    echo '========================================'
    echo 'EMM-DTI Ensemble Training (Mamba-SSM)'
    echo '========================================'
    echo 'Start time:' \$(date)
    echo 'Model: Official Mamba-SSM (Selective StateSpaceModel)'
    echo 'Seeds: [42, 123, 2024, 456, 789]'
    echo 'Epochs per seed: 200'
    echo 'Patience: 30'
    echo ''

    python ensemble_train_ssm.py 2>&1 | tee -a ${LOGFILE}

    echo ''
    echo '========================================'
    echo 'Ensemble Training Complete (Mamba-SSM)!'
    echo '========================================'
    echo 'End time:' \$(date)
    echo 'Check results in: results/ensemble_ssm/'
    echo ''
    echo 'To view this session:'
    echo '  tmux attach -t ${SESSION_NAME}'
    echo '========================================'

    # Keep session open so you can view output
    read -p 'Press ENTER to exit tmux...'
    "

echo "[OK] Session started successfully!"
echo ""
echo "========================================"
echo "NEXT STEPS"
echo "========================================"
echo ""
echo "1. Attach to the running session:"
echo "   tmux attach -t ${SESSION_NAME}"
echo ""
echo "2. View last 50 lines (without attaching):"
echo "   tmux capture-pane -t ${SESSION_NAME} -p | tail -50"
echo ""
echo "3. View full log file:"
echo "   tail -f ${LOGFILE}"
echo ""
echo "4. Stop the session (when done):"
echo "   tmux kill-session -t ${SESSION_NAME}"
echo ""
echo "========================================"
echo "COMPARE RESULTS"
echo "========================================"
echo ""
echo "Standard version (simplified Mamba):"
echo "   results/ensemble/ensemble_results.json"
echo ""
echo "SSM version (official Mamba-SSM):"
echo "   results/ensemble_ssm/ensemble_results.json"
echo ""
echo "========================================"
echo "Training is now running in the background."
echo "Your laptop can sleep — training will continue!"
echo "========================================"
