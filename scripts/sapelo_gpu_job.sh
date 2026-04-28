#!/bin/bash
# =============================================================================
# CollabMath — SLURM GPU batch job for Sapelo2 (Option B: local vLLM inference)
#
# Architecture:
#   1. Starts a vLLM server on the compute node (background process, port 8000)
#   2. Waits for the server to become healthy
#   3. Runs the full 5-stage research pipeline with LOCAL_MODEL_BASE_URL set
#      → simulator and scorer are routed to vLLM (no OpenAI API cost)
#      → splitter and validator stay on OpenAI API (gpt-4.1 / gpt-4o-mini)
#   4. Shuts down vLLM server cleanly
#
# GPU requirements:
#   Qwen2.5-72B-Instruct-AWQ  ≈ 37 GB  → 1×A100-40GB  (or 2 for headroom)
#   Change --gres and tensor_parallel_size together if you switch models.
#
# Submit:
#   sbatch scripts/sapelo_gpu_job.sh
# =============================================================================

#SBATCH --job-name=collabmath_gpu
#SBATCH --partition=gpu_p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:A100:2
#SBATCH --mem=64G
#SBATCH --time=1-12:00:00
#SBATCH --output=/scratch/%u/collabmath/logs/collabmath_gpu_%j.out
#SBATCH --error=/scratch/%u/collabmath/logs/collabmath_gpu_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=matias.rojasm@gmail.com

set -euo pipefail

MYID="${USER}"
SCRATCH_JOB="/scratch/${MYID}/collabmath"
CONDA_ENV_PATH="/home/${MYID}/.conda/envs/collabmath_gpu"

# ── Model settings ─────────────────────────────────────────────────────────
MODEL_ID="${COLLABMATH_MODEL_ID:-Qwen/Qwen2.5-72B-Instruct-AWQ}"
MODEL_DIR="${SCRATCH_JOB}/models/$(echo "${MODEL_ID}" | tr '/' '__')"
VLLM_PORT=8000
VLLM_BASE_URL="http://localhost:${VLLM_PORT}/v1"
TENSOR_PARALLEL=2   # must equal number of GPUs requested in --gres

echo "============================================================"
echo "  CollabMath GPU inference — Job ${SLURM_JOB_ID}"
echo "  Node     : $(hostname)"
echo "  Start    : $(date)"
echo "  GPUs     : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -4)"
echo "  Model    : ${MODEL_ID}"
echo "  SCRATCH  : ${SCRATCH_JOB}"
echo "  CPUs     : ${SLURM_CPUS_PER_TASK}"
echo "============================================================"

mkdir -p "${SCRATCH_JOB}/logs"

# ── 1. Load modules and activate env ─────────────────────────────────────────
module load Miniforge3/24.11.3-0
module load CUDA/12.1.1

source activate "${CONDA_ENV_PATH}"
echo "[INFO] Python: $(which python3) ($(python3 --version))"

# ── 2. Load API key for splitter / validator (remain on OpenAI) ───────────────
SECRETS_FILE="/home/${MYID}/.collabmath_secrets"
if [ -f "${SECRETS_FILE}" ]; then
    source "${SECRETS_FILE}"
    echo "[INFO] API key loaded (prefix: ${OPENAI_API_KEY:0:8}...)"
else
    echo "[ERROR] ${SECRETS_FILE} not found"
    echo "        echo 'export OPENAI_API_KEY=sk-...' > ${SECRETS_FILE} && chmod 600 ${SECRETS_FILE}"
    exit 1
fi

# ── 3. Set scale ──────────────────────────────────────────────────────────────
export COLLABMATH_PROBLEMS_PER_CELL=20   # 600 problems

# ── 4. Change to scratch working dir ─────────────────────────────────────────
cd "${SCRATCH_JOB}"
export PYTHONPATH="${SCRATCH_JOB}:${PYTHONPATH:-}"
echo "[INFO] Working directory: $(pwd)"
python3 -c "import research; print('[INFO] research package OK')"

# ── 5. Start vLLM server in background ───────────────────────────────────────
echo "[INFO] Starting vLLM server (tp=${TENSOR_PARALLEL}) at $(date) ..."

VLLM_LOG="${SCRATCH_JOB}/logs/vllm_${SLURM_JOB_ID}.log"

python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_DIR}" \
    --served-model-name "${MODEL_ID}" \
    --tensor-parallel-size "${TENSOR_PARALLEL}" \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --port "${VLLM_PORT}" \
    --disable-log-requests \
    > "${VLLM_LOG}" 2>&1 &

VLLM_PID=$!
echo "[INFO] vLLM PID: ${VLLM_PID}"

# ── 6. Wait for vLLM health check ────────────────────────────────────────────
echo "[INFO] Waiting for vLLM server to become ready ..."
MAX_WAIT=300   # 5 minutes
ELAPSED=0
until curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; do
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "[ERROR] vLLM server crashed. Last 30 lines of log:"
        tail -30 "${VLLM_LOG}"
        exit 1
    fi
    if [ "${ELAPSED}" -ge "${MAX_WAIT}" ]; then
        echo "[ERROR] vLLM did not become ready in ${MAX_WAIT}s. Log tail:"
        tail -30 "${VLLM_LOG}"
        kill "${VLLM_PID}" || true
        exit 1
    fi
    sleep 10
    ELAPSED=$((ELAPSED + 10))
    echo "  ... ${ELAPSED}s / ${MAX_WAIT}s"
done
echo "[INFO] vLLM server ready at ${VLLM_BASE_URL}"

# ── 7. Run the full pipeline with local routing ───────────────────────────────
# simulator (gpt-5.4-mini) and scorer (gpt-5.4-mini) → routed to vLLM
# splitter  (gpt-4.1)      and validator (gpt-4o-mini) → stay on OpenAI API
export LOCAL_MODEL_BASE_URL="${VLLM_BASE_URL}"
export LOCAL_MODEL_NAME="${MODEL_ID}"
export KEEP_REMOTE_MODELS="gpt-4o-mini,gpt-4.1"

echo "[INFO] Starting pipeline at $(date)"

python3 -m research.run_experiment \
    --stage all \
    --workers "${SLURM_CPUS_PER_TASK}"

EXIT_CODE=$?

# ── 8. Shut down vLLM ─────────────────────────────────────────────────────────
echo "[INFO] Stopping vLLM server (PID ${VLLM_PID}) ..."
kill "${VLLM_PID}" || true
wait "${VLLM_PID}" 2>/dev/null || true

# ── 9. Report ─────────────────────────────────────────────────────────────────
echo "============================================================"
echo "  Pipeline finished — exit code: ${EXIT_CODE}"
echo "  End: $(date)"
echo "============================================================"
echo "[INFO] Output directory sizes:"
du -sh "${SCRATCH_JOB}/outputs/"* 2>/dev/null || true

# Uncomment to archive to /project:
# PROJECT_DIR="/project/your_lab_name/collabmath_results_$(date +%Y%m%d)"
# mkdir -p "${PROJECT_DIR}"
# cp -r "${SCRATCH_JOB}/outputs/results" "${PROJECT_DIR}/"
# cp -r "${SCRATCH_JOB}/outputs/scores"  "${PROJECT_DIR}/"

conda deactivate
exit ${EXIT_CODE}
