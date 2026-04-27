#!/bin/bash
# =============================================================================
# CollabMath — GPU environment setup for Sapelo2 (run ONCE, interactively)
#
# Creates a dedicated conda env at ~/.conda/envs/collabmath_gpu with:
#   - PyTorch + CUDA  (for vLLM and fine-tuning)
#   - vLLM            (OpenAI-compatible local inference server)
#   - HuggingFace stack (transformers, datasets, peft, trl, bitsandbytes)
#   - collabmath research package
#
# Also downloads the default inference model (Qwen2.5-72B-Instruct-AWQ) to
# /scratch/$USER/collabmath/models/ so SLURM jobs don't need internet.
#
# Run from a Sapelo login node:
#   bash scripts/sapelo_gpu_setup.sh
#
# Prerequisites:
#   - ~/.collabmath_secrets contains OPENAI_API_KEY (needed for splitter/validator)
#   - HuggingFace token set (only if using gated models like Llama)
#     export HF_TOKEN=hf_...
# =============================================================================

set -euo pipefail

MYID="${USER}"
SCRATCH="/scratch/${MYID}/collabmath"
CONDA_ENV="collabmath_gpu"
CONDA_ENV_PATH="/home/${MYID}/.conda/envs/${CONDA_ENV}"

# Default model: Qwen2.5-72B-Instruct-AWQ — Apache-2.0, no gating, ~37 GB
# For 1×A100 40GB use AWQ quantized version; for 2×A100 use fp16 full model.
# Override with: export COLLABMATH_MODEL_ID=Qwen/Qwen2.5-32B-Instruct-AWQ
MODEL_ID="${COLLABMATH_MODEL_ID:-Qwen/Qwen2.5-72B-Instruct-AWQ}"
MODEL_DIR="${SCRATCH}/models/$(echo "${MODEL_ID}" | tr '/' '__')"

echo "============================================================"
echo "  CollabMath GPU setup"
echo "  User    : ${MYID}"
echo "  Env     : ${CONDA_ENV_PATH}"
echo "  Model   : ${MODEL_ID}"
echo "  ModelDir: ${MODEL_DIR}"
echo "============================================================"

# ── 1. Load Miniforge3 ────────────────────────────────────────────────────────
module load Miniforge3/24.11.3-0

# ── 2. Create isolated conda env (Python 3.11 for vLLM compatibility) ─────────
if [ -d "${CONDA_ENV_PATH}" ]; then
    echo "[INFO] Env already exists at ${CONDA_ENV_PATH} — skipping creation"
else
    echo "[INFO] Creating conda env ${CONDA_ENV} ..."
    conda create -y -p "${CONDA_ENV_PATH}" python=3.11
fi

source activate "${CONDA_ENV_PATH}"
echo "[INFO] Python: $(which python3) ($(python3 --version))"

# ── 3. Install PyTorch with CUDA 12.1 (matches Sapelo2 A100 driver) ───────────
echo "[INFO] Installing PyTorch + CUDA ..."
pip install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# ── 4. Install vLLM ──────────────────────────────────────────────────────────
# Pin to a stable release; bump as needed.
echo "[INFO] Installing vLLM ..."
pip install --quiet "vllm==0.6.6.post1"

# ── 5. Install HuggingFace stack for fine-tuning ─────────────────────────────
echo "[INFO] Installing HuggingFace training stack ..."
pip install --quiet \
    "transformers>=4.46" \
    "datasets>=3.0" \
    "peft>=0.13" \
    "trl>=0.12" \
    "bitsandbytes>=0.44" \
    "accelerate>=1.0" \
    "huggingface_hub>=0.26"

# ── 6. Install collabmath research dependencies ───────────────────────────────
echo "[INFO] Installing collabmath research deps ..."
pip install --quiet \
    openai \
    pandas \
    numpy \
    scipy

# ── 7. Install the collabmath package in editable mode ───────────────────────
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
if [ -f "${REPO_DIR}/setup.py" ] || [ -f "${REPO_DIR}/pyproject.toml" ]; then
    pip install --quiet -e "${REPO_DIR}"
    echo "[INFO] collabmath package installed from ${REPO_DIR}"
else
    echo "[WARN] No setup.py/pyproject.toml found — add ${REPO_DIR} to PYTHONPATH manually"
fi

# ── 8. Mirror code to /scratch ────────────────────────────────────────────────
mkdir -p "${SCRATCH}/logs" "${SCRATCH}/outputs" "${SCRATCH}/models"
rsync -av --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' \
    --exclude 'outputs/' \
    "${REPO_DIR}/" "${SCRATCH}/"
echo "[INFO] Code synced to ${SCRATCH}"

# ── 9. Download model to /scratch (HuggingFace hub) ──────────────────────────
# Compute nodes on Sapelo2 may NOT have internet access.
# Download the model from the login node now.
if [ -d "${MODEL_DIR}" ] && [ "$(ls -A "${MODEL_DIR}" 2>/dev/null)" ]; then
    echo "[INFO] Model already cached at ${MODEL_DIR} — skipping download"
else
    echo "[INFO] Downloading ${MODEL_ID} → ${MODEL_DIR} ..."
    echo "       This may take 20–40 min for a 70B AWQ model."
    python3 - <<EOF
from huggingface_hub import snapshot_download
import os
snapshot_download(
    repo_id="${MODEL_ID}",
    local_dir="${MODEL_DIR}",
    ignore_patterns=["*.msgpack", "*.h5", "flax_model*"],
    token=os.environ.get("HF_TOKEN"),
)
print("[INFO] Download complete.")
EOF
fi

conda deactivate

echo ""
echo "============================================================"
echo "  Setup complete."
echo ""
echo "  Next steps:"
echo "  1. Create secrets file (if not done):"
echo "     echo 'export OPENAI_API_KEY=sk-...' > ~/.collabmath_secrets"
echo "     chmod 600 ~/.collabmath_secrets"
echo ""
echo "  2. Submit the inference job:"
echo "     sbatch scripts/sapelo_gpu_job.sh"
echo ""
echo "  3. (Optional) Submit fine-tuning job after inference:"
echo "     sbatch scripts/sapelo_finetune_job.sh"
echo "============================================================"
