#!/bin/bash
#SBATCH --job-name=collabmath_dpocdi
#SBATCH --partition=gpu_p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/dpocdi_%j.out
#SBATCH --error=logs/dpocdi_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=matias.rojasm@gmail.com

cd $SLURM_SUBMIT_DIR
mkdir -p logs outputs/eval

module load CUDA/12.1.1
source ~/.bashrc 2>/dev/null || true
source .venv/bin/activate
export PYTHONPATH=$VIRTUAL_ENV/lib/python3.11/site-packages

pip install --require-virtualenv --quiet openai peft accelerate

MODEL_PATH="/scratch/$USER/llm/models/hf/Mistral-7B-Instruct-v0.3"
SFT_ADAPTER="outputs/training/split_adapter/final_adapter"
DPO_ADAPTER="outputs/training/sft_dpo_adapter/final_adapter"

echo "=== SFT→DPO CDI Evaluation ==="
echo "[INFO] Base model   : $MODEL_PATH"
echo "[INFO] SFT adapter  : $SFT_ADAPTER"
echo "[INFO] DPO adapter  : $DPO_ADAPTER"

if [ ! -d "$SFT_ADAPTER" ]; then
    echo "[ERROR] SFT adapter not found at $SFT_ADAPTER"
    exit 1
fi

if [ ! -d "$DPO_ADAPTER" ]; then
    echo "[ERROR] DPO adapter not found at $DPO_ADAPTER"
    exit 1
fi

if [ ! -f "outputs/training/split_dpo_pairs_full.jsonl" ]; then
    echo "[ERROR] outputs/training/split_dpo_pairs_full.jsonl not found"
    exit 1
fi

python3 -m research.experiments.eval_sft_cdi \
    --base-model      "$MODEL_PATH" \
    --sft-adapter-path "$SFT_ADAPTER" \
    --adapter-path    "$DPO_ADAPTER" \
    --n-problems      29 \
    --n-samples       3 \
    --output-prefix   "dpo_cdi"

echo ""
echo "=== DONE ==="
echo "Summary: outputs/eval/dpo_cdi_summary.json"
