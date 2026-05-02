#!/bin/bash
#SBATCH --job-name=collabmath_abl
#SBATCH --partition=gpu_p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=02:00:00
#SBATCH --array=100,200
#SBATCH --output=logs/ablation_%a_%j.out
#SBATCH --error=logs/ablation_%a_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=matias.rojasm@gmail.com

# MAX_EXAMPLES is the SLURM array task id (100 or 200).
# n=333 (full dataset) was already trained; results are in split_adapter/.
MAX_EXAMPLES=${SLURM_ARRAY_TASK_ID}

cd $SLURM_SUBMIT_DIR
mkdir -p logs outputs/training

module load CUDA/12.1.1
source ~/.bashrc 2>/dev/null || true
source .venv/bin/activate
export PYTHONPATH=$VIRTUAL_ENV/lib/python3.11/site-packages

MODEL_PATH="/scratch/$USER/llm/models/hf/Mistral-7B-Instruct-v0.3"
OUTPUT_DIR="outputs/training/split_adapter_${MAX_EXAMPLES}"

echo "=== SFT Ablation — N=${MAX_EXAMPLES} ==="
echo "[INFO] Base model  : $MODEL_PATH"
echo "[INFO] Output dir  : $OUTPUT_DIR"
echo "[INFO] Max examples: $MAX_EXAMPLES"

python3 -m research.training.train_split_generator \
    --base-model "$MODEL_PATH" \
    --epochs 3 \
    --lr 2e-5 \
    --max-length 2048 \
    --max-examples "$MAX_EXAMPLES" \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "=== Training done — running structural eval ==="

python3 -m research.experiments.eval_split_generator \
    --base-model "$MODEL_PATH" \
    --adapter-path "${OUTPUT_DIR}/final_adapter" \
    --n-samples 3 \
    --max-new-tokens 512 \
    --output-prefix "ablation_${MAX_EXAMPLES}"

echo ""
echo "=== DONE — N=${MAX_EXAMPLES} ==="
echo "Structural results: outputs/eval/ablation_${MAX_EXAMPLES}_summary.json"
