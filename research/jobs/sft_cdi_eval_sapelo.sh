#!/bin/bash
#SBATCH --job-name=collabmath_sft_cdi_eval
#SBATCH --partition=gpu_p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/sft_cdi_eval_%j.out
#SBATCH --error=logs/sft_cdi_eval_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=matias.rojasm@gmail.com

cd $SLURM_SUBMIT_DIR
mkdir -p logs outputs/eval

module load CUDA/12.1.1

source .venv/bin/activate
export PYTHONPATH=$VIRTUAL_ENV/lib/python3.11/site-packages

export SFT_BASE_MODEL="/scratch/$USER/llm/models/hf/Mistral-7B-Instruct-v0.3"
export SFT_ADAPTER_PATH="outputs/training/split_adapter/final_adapter"

echo "[INFO] Base model : $SFT_BASE_MODEL"
echo "[INFO] Adapter    : $SFT_ADAPTER_PATH"
echo "[INFO] Date       : $(date)"

# Load OPENAI_API_KEY from .env if present
if [ -f ".env" ]; then
    set -a; source .env; set +a
    echo "[INFO] Loaded .env"
fi

if [ -z "$OPENAI_API_KEY" ]; then
    echo "[ERROR] OPENAI_API_KEY not set — required for C7 simulation"
    exit 1
fi

if [ ! -d "$SFT_ADAPTER_PATH" ]; then
    echo "[ERROR] Adapter not found at $SFT_ADAPTER_PATH"
    exit 1
fi

if [ ! -f "outputs/training/split_dpo_pairs_full.jsonl" ]; then
    echo "[ERROR] DPO pairs file not found — run prepare_split_dpo_data first"
    exit 1
fi

echo "=== CDI EVALUATION: all test-set problems ==="
# --n-problems 87 exceeds typical test-set size, so all available test problems
# are evaluated. Output prefix 'sft_cdi_full' distinguishes from n=29 pilot run.
python -m research.experiments.eval_sft_cdi \
    --base-model    "$SFT_BASE_MODEL" \
    --adapter-path  "$SFT_ADAPTER_PATH" \
    --n-problems    87 \
    --n-samples     3 \
    --output-prefix sft_cdi_full

echo "=== DONE: $(date) ==="
echo "Results → outputs/eval/sft_cdi_full_results.jsonl"
echo "Summary → outputs/eval/sft_cdi_full_summary.json"
