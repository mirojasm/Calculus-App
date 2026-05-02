#!/bin/bash
#SBATCH --job-name=collabmath_sftcdi
#SBATCH --partition=gpu_p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/sftcdi_%j.out
#SBATCH --error=logs/sftcdi_%j.err
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
ADAPTER_PATH="outputs/training/split_adapter/final_adapter"

echo "=== SFT CDI Evaluation ==="
echo "[INFO] Base model : $MODEL_PATH"
echo "[INFO] Adapter    : $ADAPTER_PATH"

if [ ! -d "$ADAPTER_PATH" ]; then
    echo "[ERROR] Adapter not found at $ADAPTER_PATH"
    exit 1
fi

if [ ! -f "outputs/training/split_dpo_pairs_full.jsonl" ]; then
    echo "[ERROR] outputs/training/split_dpo_pairs_full.jsonl not found"
    echo "Transfer from local: scp outputs/training/split_dpo_pairs_full.jsonl \\"
    echo "    mir85108@sapelo2.gacrc.uga.edu:/scratch/mir85108/Calculus-App/outputs/training/"
    exit 1
fi

python3 -m research.experiments.eval_sft_cdi \
    --base-model "$MODEL_PATH" \
    --adapter-path "$ADAPTER_PATH" \
    --n-problems 50 \
    --n-samples 3

echo ""
echo "=== DONE ==="
echo "Summary: outputs/eval/sft_cdi_summary.json"
