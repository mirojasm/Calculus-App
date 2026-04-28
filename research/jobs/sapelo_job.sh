#!/bin/bash
#SBATCH --job-name=collabmath_experiment
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL

# ── environment ───────────────────────────────────────────────────────────────
cd $SLURM_SUBMIT_DIR
module load Python/3.11.3-GCCcore-12.3.0

# Activate virtual env (create once with: python -m venv .venv && pip install -r requirements.txt)
source .venv/bin/activate

# OpenAI API key — set in your Sapelo environment or .bashrc
# export OPENAI_API_KEY="sk-..."

# ── run stages ────────────────────────────────────────────────────────────────
echo "=== STAGE: load ==="
python -m research.run_experiment --stage load --workers 1

echo "=== STAGE: split ==="
python -m research.run_experiment --stage split --workers 8

echo "=== STAGE: simulate ==="
python -m research.run_experiment --stage simulate --workers 16

echo "=== STAGE: score ==="
python -m research.run_experiment --stage score --workers 4

echo "=== STAGE: analyse ==="
python -m research.run_experiment --stage analyse --workers 1

echo "=== DONE ==="
