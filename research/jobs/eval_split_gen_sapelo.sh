#!/bin/bash
#SBATCH --job-name=collabmath_evalsg
#SBATCH --partition=gpu_p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/evalsg_%j.out
#SBATCH --error=logs/evalsg_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=matias.rojasm@gmail.com

cd $SLURM_SUBMIT_DIR
mkdir -p logs outputs/eval

module load CUDA/12.1.1
source .venv/bin/activate
export PYTHONPATH=$VIRTUAL_ENV/lib/python3.11/site-packages

MODEL_PATH="/scratch/$USER/llm/models/hf/Mistral-7B-Instruct-v0.3"
ADAPTER_PATH="outputs/training/split_adapter/final_adapter"

echo "=== Split Generator Evaluation ==="
echo "[INFO] Base model : $MODEL_PATH"
echo "[INFO] Adapter    : $ADAPTER_PATH"

if [ ! -d "$ADAPTER_PATH" ]; then
    echo "[ERROR] Adapter not found at $ADAPTER_PATH"
    exit 1
fi

# n-samples=3: genera 3 splits por problema, queda con el primero válido (sample-then-filter)
python3 -m research.experiments.eval_split_generator \
    --base-model "$MODEL_PATH" \
    --adapter-path "$ADAPTER_PATH" \
    --n-samples 3 \
    --max-new-tokens 512 \
    --show 5

echo ""
echo "=== DONE ==="
echo "Results: outputs/eval/split_eval.jsonl"
echo "Summary: outputs/eval/split_eval_summary.json"
