#!/bin/bash
#SBATCH --job-name=collabmath_sft_split
#SBATCH --partition=gpu_p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/sft_split_%j.out
#SBATCH --error=logs/sft_split_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=matias.rojasm@gmail.com

cd $SLURM_SUBMIT_DIR
mkdir -p logs outputs/splits

module load CUDA/12.1.1

source .venv/bin/activate
export PYTHONPATH=$VIRTUAL_ENV/lib/python3.11/site-packages

export SFT_BASE_MODEL="/scratch/$USER/llm/models/hf/Mistral-7B-Instruct-v0.3"
export SFT_ADAPTER_PATH="outputs/training/split_adapter/final_adapter"

echo "[INFO] Base model : $SFT_BASE_MODEL"
echo "[INFO] Adapter    : $SFT_ADAPTER_PATH"

if [ ! -f "outputs/data/math_sample.json" ]; then
    echo "[INFO] math_sample.json not found — running load stage first"
    python -m research.run_experiment --stage load --workers 1
fi

if [ ! -d "$SFT_ADAPTER_PATH" ]; then
    echo "[ERROR] Adapter not found at $SFT_ADAPTER_PATH"
    exit 1
fi

echo "=== STAGE: split (SFT) ==="
# workers=1: model is a singleton, threading doesn't help on a single GPU
python -m research.run_experiment --stage split --splitter sft --workers 1

echo "=== DONE ==="
echo "Splits saved to outputs/splits/"
