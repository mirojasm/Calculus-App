#!/bin/bash
#SBATCH --job-name=collabmath_splitgen
#SBATCH --partition=gpu_p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/splitgen_%j.out
#SBATCH --error=logs/splitgen_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=matias.rojasm@gmail.com

# ── environment ───────────────────────────────────────────────────────────────
cd $SLURM_SUBMIT_DIR
mkdir -p logs outputs/training

module load CUDA/12.1.1

source .venv/bin/activate
export PYTHONPATH=$VIRTUAL_ENV/lib/python3.11/site-packages

pip install --require-virtualenv --quiet "trl==0.11.4" "peft" "accelerate"

# ── model path ────────────────────────────────────────────────────────────────
MODEL_PATH="/scratch/$USER/llm/models/hf/Mistral-7B-Instruct-v0.3"
echo "[INFO] Using model: $MODEL_PATH"

# ── step 1: verify DPO data ───────────────────────────────────────────────────
# Data was prepared locally — transfer before submitting:
#   scp outputs/training/split_dpo_train.jsonl outputs/training/split_dpo_test.jsonl \
#       mir85108@sapelo2.gacrc.uga.edu:/scratch/mir85108/Calculus-App/outputs/training/
echo "=== STEP 1: verify split DPO pairs ==="
if [ ! -f outputs/training/split_dpo_train.jsonl ]; then
    echo "[ERROR] outputs/training/split_dpo_train.jsonl not found."
    echo "Run locally: python -m research.training.prepare_split_dpo_data"
    echo "Then transfer:  scp outputs/training/split_dpo_*.jsonl \\"
    echo "    mir85108@sapelo2.gacrc.uga.edu:/scratch/mir85108/Calculus-App/outputs/training/"
    exit 1
fi
TRAIN_N=$(wc -l < outputs/training/split_dpo_train.jsonl)
TEST_N=$(wc -l < outputs/training/split_dpo_test.jsonl)
echo "[OK] split_dpo_train.jsonl: ${TRAIN_N} pairs"
echo "[OK] split_dpo_test.jsonl:  ${TEST_N} pairs"

# ── step 2: split generator SFT fine-tuning ───────────────────────────────────
echo ""
echo "=== STEP 2: Split generator SFT fine-tuning ==="
python3 -m research.training.train_split_generator \
    --base-model "$MODEL_PATH" \
    --epochs 3 \
    --lr 2e-5 \
    --batch-size 1 \
    --grad-accum 8 \
    --max-length 2048 \
    --lora-r 16

echo ""
echo "=== DONE ==="
echo "Adapter saved to outputs/training/split_adapter/final_adapter"
