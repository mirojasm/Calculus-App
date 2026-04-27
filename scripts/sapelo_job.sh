#!/bin/bash
# =============================================================================
# CollabMath — SLURM batch job for Sapelo2
#
# Runs the full 5-stage research pipeline (load → split → simulate → score →
# analyse) on the 600-problem corpus (problems_per_cell = 20).
#
# Partition choice: "batch"
#   - Up to 7 days wall time
#   - Up to 250 running jobs / 10,000 submitted
#   - Regular nodes: 64–128 cores, 120–740 GB RAM per node
#   - No GPU needed (pipeline is API-driven, not GPU-compute)
#
# We request 1 node / 1 task / 16 CPUs so ThreadPoolExecutor can saturate
# --workers 16 across the parallelisable stages (split, simulate, score).
# Memory: 32 GB is ample for the JSON data volumes at 600 problems.
#
# Submit from /scratch/MyID/collabmath/:
#   sbatch scripts/sapelo_job.sh
# =============================================================================

#SBATCH --job-name=collabmath_full
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --output=/scratch/%u/collabmath/logs/collabmath_%j.out
#SBATCH --error=/scratch/%u/collabmath/logs/collabmath_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=matias.rojasm@gmail.com

# =============================================================================
# Runtime setup
# =============================================================================

set -euo pipefail

MYID="${USER}"
SCRATCH_JOB="/scratch/${MYID}/collabmath"
CONDA_ENV_PATH="/home/${MYID}/.conda/envs/collabmath"

echo "============================================================"
echo "  CollabMath full pipeline — Job ${SLURM_JOB_ID}"
echo "  Node     : $(hostname)"
echo "  Start    : $(date)"
echo "  SCRATCH  : ${SCRATCH_JOB}"
echo "  CPUs     : ${SLURM_CPUS_PER_TASK}"
echo "============================================================"

# ── 1. Ensure log directory exists ───────────────────────────────────────────
mkdir -p "${SCRATCH_JOB}/logs"

# ── 2. Load Miniforge3 and activate the isolated conda env ───────────────────
# GACRC convention: modules are loaded on compute nodes inside the job script.
# Do NOT use "conda activate" in batch jobs — use "source activate <path>".
module load Miniforge3/24.11.3-0
source activate "${CONDA_ENV_PATH}"

echo "[INFO] Python: $(which python3)"
echo "[INFO] Python version: $(python3 --version)"

# ── 3. Load the OpenAI API key from the secrets file ─────────────────────────
# The key is stored in a chmod-600 file in $HOME, NOT in this script or in
# /scratch (which has a 30-day purge and no guaranteed privacy).
SECRETS_FILE="/home/${MYID}/.collabmath_secrets"
if [ -f "${SECRETS_FILE}" ]; then
    # shellcheck source=/dev/null
    source "${SECRETS_FILE}"
    echo "[INFO] API key loaded from ${SECRETS_FILE}"
else
    echo "[ERROR] Secrets file not found: ${SECRETS_FILE}"
    echo "        Create it with:  echo 'export OPENAI_API_KEY=sk-...' > ${SECRETS_FILE}"
    echo "        Then:            chmod 600 ${SECRETS_FILE}"
    exit 1
fi

# Verify the key is present (print first 8 chars only — never log the full key)
if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "[ERROR] OPENAI_API_KEY is empty after sourcing ${SECRETS_FILE}"
    exit 1
fi
echo "[INFO] OPENAI_API_KEY present (prefix: ${OPENAI_API_KEY:0:8}...)"

# ── 3b. Set experiment scale (600 problems for Sapelo run) ───────────────────
export COLLABMATH_PROBLEMS_PER_CELL=20

# ── 4. Change to scratch working directory ───────────────────────────────────
# GACRC: Run jobs FROM /scratch, not from $HOME.
# All output files will be written under ${SCRATCH_JOB}/outputs/
cd "${SCRATCH_JOB}"
echo "[INFO] Working directory: $(pwd)"

# ── 5. Verify the research package is importable ─────────────────────────────
python3 -c "import research; print('[INFO] research package OK')"

# ── 6. Run the full pipeline ─────────────────────────────────────────────────
# --workers 16 matches --cpus-per-task=16.
# The score stage internally caps workers at 4 (already in run_experiment.py)
# to avoid overwhelming the rate-limited scoring API.
echo "[INFO] Starting pipeline at $(date)"

python3 -m research.run_experiment \
    --stage all \
    --workers "${SLURM_CPUS_PER_TASK}"

EXIT_CODE=$?

echo "============================================================"
echo "  Pipeline finished — exit code: ${EXIT_CODE}"
echo "  End: $(date)"
echo "============================================================"

# ── 7. Copy outputs from /scratch to /project for long-term storage ──────────
# /scratch is subject to a 30-day purge policy.  Copy results to /project
# (if your lab has /project space) or transfer home via xfer nodes.
#
# Uncomment and adjust the path below if you have /project space:
# PROJECT_DIR="/project/your_lab_name/collabmath_results_$(date +%Y%m%d)"
# mkdir -p "${PROJECT_DIR}"
# cp -r "${SCRATCH_JOB}/outputs/results" "${PROJECT_DIR}/"
# cp -r "${SCRATCH_JOB}/outputs/scores"  "${PROJECT_DIR}/"
# echo "[INFO] Results archived to ${PROJECT_DIR}"

# ── 8. Report output sizes ───────────────────────────────────────────────────
echo "[INFO] Output directory sizes:"
du -sh "${SCRATCH_JOB}/outputs/"* 2>/dev/null || true

conda deactivate
exit ${EXIT_CODE}
