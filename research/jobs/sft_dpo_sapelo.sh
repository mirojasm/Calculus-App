#!/bin/bash
#SBATCH --job-name=collabmath_sftdpo
#SBATCH --partition=gpu_p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/sftdpo_%j.out
#SBATCH --error=logs/sftdpo_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=matias.rojasm@gmail.com

cd $SLURM_SUBMIT_DIR
mkdir -p logs outputs/training

module load CUDA/12.1.1
source ~/.bashrc 2>/dev/null || true
source .venv/bin/activate
export PYTHONPATH=$VIRTUAL_ENV/lib/python3.11/site-packages

pip install --require-virtualenv --quiet "trl==0.11.4" peft accelerate

MODEL_PATH="/scratch/$USER/llm/models/hf/Mistral-7B-Instruct-v0.3"
SFT_ADAPTER="outputs/training/split_adapter/final_adapter"
OUTPUT_DIR="outputs/training/sft_dpo_adapter"

echo "=== SFT → DPO Fine-tuning ==="
echo "[INFO] Base model  : $MODEL_PATH"
echo "[INFO] SFT adapter : $SFT_ADAPTER"
echo "[INFO] Output dir  : $OUTPUT_DIR"

if [ ! -d "$SFT_ADAPTER" ]; then
    echo "[ERROR] SFT adapter not found at $SFT_ADAPTER"
    echo "Run split_gen_sapelo.sh first to produce the SFT adapter."
    exit 1
fi

if [ ! -f "outputs/training/split_dpo_train.jsonl" ]; then
    echo "[ERROR] split_dpo_train.jsonl not found"
    exit 1
fi

python3 -m research.training.train_sft_dpo \
    --base-model  "$MODEL_PATH" \
    --sft-adapter "$SFT_ADAPTER" \
    --output-dir  "$OUTPUT_DIR" \
    --epochs 1 \
    --lr     5e-6 \
    --beta   0.1 \
    --lora-r 16

echo ""
echo "=== Training done — running structural eval on DPO adapter ==="

python3 -m research.experiments.eval_split_generator \
    --base-model       "$MODEL_PATH" \
    --sft-adapter-path "$SFT_ADAPTER" \
    --adapter-path     "${OUTPUT_DIR}/final_adapter" \
    --n-samples        3 \
    --max-new-tokens   512 \
    --output-prefix    "sft_dpo_eval"

echo ""
echo "=== DONE ==="
echo "Structural eval : outputs/eval/sft_dpo_eval_summary.json"
echo "DPO adapter     : ${OUTPUT_DIR}/final_adapter"
