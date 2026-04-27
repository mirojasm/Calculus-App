#!/bin/bash
# =============================================================================
# CollabMath — One-time environment setup on Sapelo2
#
# Run this ONCE from a LOGIN/SUBMIT node (ss-sub1, ss-sub2, etc.) BEFORE
# submitting any jobs.  It is safe to re-run; conda create is idempotent.
#
# Usage:
#   bash /home/$USER/collabmath/scripts/sapelo_setup.sh
# =============================================================================

set -euo pipefail

# ---------- configurable variables -------------------------------------------
MYID="${MYID:-$USER}"                        # your Sapelo2 MyID
CONDA_ENV_NAME="collabmath"
# Install the env under $HOME so it persists across jobs (HOME = 200 GB, backed up)
CONDA_ENV_PATH="/home/${MYID}/.conda/envs/${CONDA_ENV_NAME}"
SCRATCH_DIR="/scratch/${MYID}/collabmath"
HOME_CODE_DIR="/home/${MYID}/collabmath"
# -----------------------------------------------------------------------------

echo "=== CollabMath Sapelo2 setup ==="
echo "MyID           : ${MYID}"
echo "Conda env path : ${CONDA_ENV_PATH}"
echo "Scratch dir    : ${SCRATCH_DIR}"
echo ""

# ── 1. Load Miniforge3 (the GACRC-recommended conda provider) ─────────────────
# NOTE: Do NOT use module load on login nodes for actual computation, but
#       loading it here for setup purposes is fine.
module load Miniforge3/24.11.3-0

# ── 2. Create isolated conda environment (skips if it already exists) ─────────
if conda env list | grep -q "${CONDA_ENV_PATH}"; then
    echo "[INFO] Conda environment already exists at ${CONDA_ENV_PATH} — skipping create."
else
    echo "[INFO] Creating conda environment '${CONDA_ENV_NAME}' with Python 3.12..."
    conda create -p "${CONDA_ENV_PATH}" python=3.12 -y
fi

# ── 3. Activate and install packages ─────────────────────────────────────────
echo "[INFO] Activating environment and installing packages..."
source activate "${CONDA_ENV_PATH}"

# Install from the project requirements file.
# We install with pip inside the conda env (pip is isolated per env).
pip install --upgrade pip

pip install \
    "openai>=1.30.0" \
    "datasets>=2.19.0" \
    "pandas>=2.2.0" \
    "numpy>=1.26.0" \
    "scipy>=1.13.0" \
    "huggingface_hub>=0.23.0"

conda deactivate

echo "[INFO] Conda environment ready."

# ── 4. Create scratch working directory ──────────────────────────────────────
mkdir -p "${SCRATCH_DIR}/outputs/data"
mkdir -p "${SCRATCH_DIR}/outputs/splits"
mkdir -p "${SCRATCH_DIR}/outputs/conversations"
mkdir -p "${SCRATCH_DIR}/outputs/scores"
mkdir -p "${SCRATCH_DIR}/outputs/results"
echo "[INFO] Scratch directories created at ${SCRATCH_DIR}"

# ── 5. Copy code from $HOME to scratch ───────────────────────────────────────
# GACRC policy: run jobs FROM /scratch, NOT from $HOME.
# $HOME is for scripts/source only; heavy I/O must hit /scratch.
if [ -d "${HOME_CODE_DIR}/research" ]; then
    echo "[INFO] Syncing code from ${HOME_CODE_DIR} to ${SCRATCH_DIR}..."
    rsync -av --exclude='__pycache__' --exclude='*.pyc' \
        --exclude='.venv' --exclude='outputs' --exclude='.git' \
        "${HOME_CODE_DIR}/" "${SCRATCH_DIR}/"
    echo "[INFO] Code synced."
else
    echo "[WARN] ${HOME_CODE_DIR}/research not found."
    echo "       Transfer your code first:  rsync -av ./  ${MYID}@xfer.gacrc.uga.edu:~/collabmath/"
fi

# ── 6. API key setup reminder ─────────────────────────────────────────────────
echo ""
echo "========================================================"
echo "  IMPORTANT: set your OpenAI API key BEFORE submitting"
echo "========================================================"
echo ""
echo "  Create the secrets file (only readable by you):"
echo ""
echo "    cat > /home/${MYID}/.collabmath_secrets << 'EOF'"
echo "    export OPENAI_API_KEY=sk-...your-key-here..."
echo "    EOF"
echo "    chmod 600 /home/${MYID}/.collabmath_secrets"
echo ""
echo "  The job script sources this file at runtime — your key"
echo "  is NEVER stored in the job script or in /scratch."
echo ""
echo "=== Setup complete. ==="
echo "Next: create ~/.collabmath_secrets, then submit the job:"
echo "  sbatch ${SCRATCH_DIR}/scripts/sapelo_job.sh"
