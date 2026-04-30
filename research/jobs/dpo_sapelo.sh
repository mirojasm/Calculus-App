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

module load Transformers
module load datasets
module load CUDA/12.1.1

source .venv/bin/activate

# TRL must be installed manually per GACRC docs
pip install --require-virtualenv --quiet trl peft bitsandbytes accelerate

# ── model path ────────────────────────────────────────────────────────────────
MODEL_PATH="/scratch/$USER/llm/models/hf/Mistral-7B-Instruct-v0.3"
echo "[INFO] Using model: $MODEL_PATH"

# ── step 1: prepare DPO data ──────────────────────────────────────────────────
echo "=== STEP 1: prepare DPO pairs ==="
python3 -m research.training.prepare_dpo_data \
    --pilot-dir outputs/pilot \
    --train-frac 0.8 \
    --seed 42

echo ""
echo "--- DPO dataset stats ---"
cat outputs/training/dpo_stats.json
echo ""

# ── step 2: DPO fine-tuning ───────────────────────────────────────────────────
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
