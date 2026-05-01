#!/bin/bash
#SBATCH --job-name=collabmath_dpo
#SBATCH --partition=gpu_p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/dpo_%j.out
#SBATCH --error=logs/dpo_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=matias.rojasm@gmail.com

# ── environment ───────────────────────────────────────────────────────────────
cd $SLURM_SUBMIT_DIR
mkdir -p logs outputs/training

module load CUDA/12.1.1

source .venv/bin/activate
# Force venv packages to take priority over any system site-packages
export PYTHONPATH=$VIRTUAL_ENV/lib/python3.11/site-packages

# TRL must be installed manually per GACRC docs
pip install --require-virtualenv --quiet "trl==0.11.4" "peft" "accelerate"

# ── model path ────────────────────────────────────────────────────────────────
MODEL_PATH="/scratch/$USER/llm/models/hf/Mistral-7B-Instruct-v0.3"
echo "[INFO] Using model: $MODEL_PATH"

# ── step 1: verify DPO data ───────────────────────────────────────────────────
# Data was prepared locally and transferred — do not regenerate on Sapelo
echo "=== STEP 1: verify DPO pairs ==="
if [ ! -f outputs/training/dpo_train.jsonl ]; then
    echo "[ERROR] outputs/training/dpo_train.jsonl not found."
    echo "Transfer it from your local machine first:"
    echo "  scp outputs/training/dpo_train.jsonl outputs/training/dpo_test.jsonl \\"
    echo "      mir85108@sapelo2.gacrc.uga.edu:/scratch/mir85108/Calculus-App/outputs/training/"
    exit 1
fi
TRAIN_N=$(wc -l < outputs/training/dpo_train.jsonl)
TEST_N=$(wc -l < outputs/training/dpo_test.jsonl)
echo "[OK] dpo_train.jsonl: ${TRAIN_N} pairs"
echo "[OK] dpo_test.jsonl:  ${TEST_N} pairs"

# ── step 2: DPO fine-tuning ───────────────────────────────────────────────────
echo ""
echo "=== STEP 2: DPO fine-tuning ==="
python3 -m research.training.dpo_train \
    --base-model "$MODEL_PATH" \
    --epochs 2 \
    --lr 5e-5 \
    --beta 0.1 \
    --batch-size 1 \
    --grad-accum 8 \
    --max-length 2048 \
    --lora-r 16

echo ""
echo "=== DONE ==="
echo "Adapter saved to outputs/training/dpo_adapter/final_adapter"
