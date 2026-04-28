#!/bin/bash
# =============================================================================
# CollabMath — SLURM fine-tuning job for Sapelo2 (Option C: QLoRA splitter)
#
# Fine-tunes Qwen2.5-7B-Instruct on the CollabMath jigsaw split dataset using
# QLoRA (4-bit base + LoRA adapters).  The adapter (~50 MB) is saved to
# outputs/training/splitter_adapter/final_adapter/ for reuse.
#
# GPU requirements:
#   Qwen2.5-7B in 4-bit ≈ 4–5 GB; gradients + activations ≈ 15 GB total
#   → 1×A100-40GB is sufficient.
#
# Run AFTER the main pipeline has finished (splits must exist in outputs/splits/).
# Make sure to run prepare_finetune_data.py first if it hasn't run yet.
#
# Submit:
#   # Step 1 — prepare dataset (run once from login node, takes seconds):
#   cd /scratch/$USER/collabmath
#   source ~/.conda/envs/collabmath_gpu/bin/activate
#   python3 -m research.training.prepare_finetune_data
#
#   # Step 2 — submit fine-tuning job:
#   sbatch scripts/sapelo_finetune_job.sh
# =============================================================================

#SBATCH --job-name=collabmath_finetune
#SBATCH --partition=gpu_p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=40G
#SBATCH --time=0-06:00:00
#SBATCH --output=/scratch/%u/collabmath/logs/collabmath_finetune_%j.out
#SBATCH --error=/scratch/%u/collabmath/logs/collabmath_finetune_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=matias.rojasm@gmail.com

set -euo pipefail

MYID="${USER}"
SCRATCH_JOB="/scratch/${MYID}/collabmath"
CONDA_ENV_PATH="/home/${MYID}/.conda/envs/collabmath_gpu"

# Fine-tuning hyperparameters (override via SBATCH --export or env vars)
BASE_MODEL="${FINETUNE_BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
EPOCHS="${FINETUNE_EPOCHS:-2}"
LORA_R="${FINETUNE_LORA_R:-16}"

echo "============================================================"
echo "  CollabMath fine-tuning — Job ${SLURM_JOB_ID}"
echo "  Node      : $(hostname)"
echo "  Start     : $(date)"
echo "  GPU       : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "  Base model: ${BASE_MODEL}"
echo "  Epochs    : ${EPOCHS}"
echo "  LoRA r    : ${LORA_R}"
echo "============================================================"

mkdir -p "${SCRATCH_JOB}/logs"

# ── 1. Load modules ───────────────────────────────────────────────────────────
module load Miniforge3/24.11.3-0
module load CUDA/12.1.0

source activate "${CONDA_ENV_PATH}"
echo "[INFO] Python: $(which python3) ($(python3 --version))"

# ── 2. Change to scratch working directory ────────────────────────────────────
cd "${SCRATCH_JOB}"
export PYTHONPATH="${SCRATCH_JOB}:${PYTHONPATH:-}"
echo "[INFO] Working directory: $(pwd)"

# ── 3. Verify training data exists ───────────────────────────────────────────
TRAIN_JSONL="outputs/training/splits_instruct.jsonl"
if [ ! -f "${TRAIN_JSONL}" ]; then
    echo "[INFO] Training data not found — running prepare_finetune_data.py ..."
    python3 -m research.training.prepare_finetune_data
fi
N_EXAMPLES=$(wc -l < "${TRAIN_JSONL}")
echo "[INFO] Training examples: ${N_EXAMPLES}"

if [ "${N_EXAMPLES}" -lt 10 ]; then
    echo "[ERROR] Too few training examples (${N_EXAMPLES}). Run stage_split first."
    exit 1
fi

# ── 4. Download base model if not cached ─────────────────────────────────────
BASE_MODEL_DIR="${SCRATCH_JOB}/models/$(echo "${BASE_MODEL}" | tr '/' '__')"
if [ ! -d "${BASE_MODEL_DIR}" ] || [ -z "$(ls -A "${BASE_MODEL_DIR}" 2>/dev/null)" ]; then
    echo "[INFO] Downloading base model ${BASE_MODEL} ..."
    python3 - <<EOF
from huggingface_hub import snapshot_download
import os
snapshot_download(
    repo_id="${BASE_MODEL}",
    local_dir="${BASE_MODEL_DIR}",
    ignore_patterns=["*.msgpack", "*.h5", "flax_model*"],
    token=os.environ.get("HF_TOKEN"),
)
print("[INFO] Base model download complete.")
EOF
fi

# ── 5. Set HuggingFace cache to scratch (avoids filling home quota) ───────────
export HF_HOME="${SCRATCH_JOB}/hf_cache"
export TRANSFORMERS_CACHE="${SCRATCH_JOB}/hf_cache"
mkdir -p "${HF_HOME}"

# ── 6. Run fine-tuning ────────────────────────────────────────────────────────
echo "[INFO] Starting fine-tuning at $(date) ..."

python3 -m research.training.finetune_splitter \
    --base-model "${BASE_MODEL_DIR}" \
    --epochs     "${EPOCHS}" \
    --lora-r     "${LORA_R}"

EXIT_CODE=$?

echo "============================================================"
echo "  Fine-tuning finished — exit code: ${EXIT_CODE}"
echo "  End: $(date)"
echo "============================================================"

if [ "${EXIT_CODE}" -eq 0 ]; then
    echo "[INFO] Adapter saved to:"
    ls -lh "${SCRATCH_JOB}/outputs/training/splitter_adapter/final_adapter/" 2>/dev/null || true
fi

conda deactivate
exit ${EXIT_CODE}
