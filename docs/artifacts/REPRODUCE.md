# Reproduction Guide

This document describes how to reproduce the results in:

> "Epistemic Role Framing Elicits Deep Collaboration in LLM Multi-Agent Systems"
> NeurIPS 2026 Evaluations & Datasets Track

---

## Requirements

- Python 3.10+
- OpenAI API key with access to GPT-4.1
- Dependencies: `pip install -r requirements.txt`

Set your API key:
```bash
export OPENAI_API_KEY="your-key-here"
# or: export $(grep -v '^#' research/.env | xargs)
```

---

## Repository Structure

```
research/
  config.py                    # Global config (max_turns=20, model, temperature)
  splitting/splitter.py        # CIDI split generator
  simulation/simulator.py      # Conversation simulator (C2, C6, C7, CFULL, CEXP)
  scoring/cpp_annotator.py     # CPP rubric annotator (CDI, CQI, PhAQ)
  scoring/atc21s.py            # ATC21S annotator
  experiments/
    run_ablations.py           # CFULL and CEXP ablation experiments
outputs/
  pilot/                       # Phase 1 and Phase 2 conversation logs
  ablations/                   # CFULL and CEXP conversation logs
  aec/                         # AEC solo-run results and summary
```

---

## Phase 1: CIDI Split Generation

The CIDI pipeline decomposes each MATH problem into (shared context, Packet A, Packet B).

```bash
# Generate splits for a list of problems
python3 -m research.splitting.splitter --input data/math_problems.json
```

Cached splits are stored in each pilot JSON file under the `split` key.

---

## Phase 2: Main Experiment (C2, C6, C7)

**Already completed.** Results are in `outputs/pilot/`.

To re-run (warning: ~$50–150 in API costs):

```bash
python3 -m research.experiments.run_phase2 \
  --conditions C2 C6 C7 \
  --workers 6 \
  --reps 3
```

---

## Phase 3: Ablation Conditions (CFULL, CEXP)

**Already completed.** Results in `outputs/ablations/`.
- CFULL: 412 conversations (8 excluded due to content-policy refusals)
- CEXP: 420 conversations
- Total: 832 conversations

To re-run:

```bash
export $(grep -v '^#' research/.env | xargs)
python3 -m research.experiments.run_ablations \
  --conditions CFULL CEXP \
  --workers 6 \
  --reps 3
```

---

## CPP Annotation

Each conversation is annotated by the CPP rubric annotator (LLM-based) with optional
human inter-rater override for reliability conversations.

```bash
# Re-annotate a single file
python3 -m research.scoring.cpp_annotator \
  --input outputs/pilot/math_00001_C7_rep1.json
```

The annotator produces `cpp_vector` (12 binary cells), `cdi`, `cqi`, `phaq`.

---

## AEC Analysis

```bash
# Run solo AEC evaluations (temperature T=0)
python3 -m research.scoring.aec_runner \
  --problems outputs/pilot/phase2_problems.json \
  --output outputs/aec/

# Compute Shapley decomposition
python3 -m research.scoring.aec_summary \
  --input outputs/aec/ \
  --output outputs/aec/aec_summary.json
```

Expected output matches paper Table 3:
- v_A = 0.100, v_B = 0.057, v_AB = 0.357
- Both-solo-zero = 85.0%, EN = 25.0%, CS = +0.207, EB = 0.857

---

## Aggregate Statistics

```bash
# Recompute ablation summary
python3 -m research.experiments.run_ablations \
  --conditions CFULL CEXP \
  --limit 0   # process all cached results only
```

Expected summary matches paper:
- CFULL: CDI=0.296±0.018, TRIVIAL=49%, COUPLING=11%
- CEXP: CDI=0.267±0.017, COLLAPSE=53%, COUPLING=5%

---

## Key Configuration

From `research/config.py`:

| Parameter     | Value      | Description                         |
|---------------|------------|-------------------------------------|
| model         | gpt-4.1    | Model for all conditions            |
| max_turns     | 20         | Max turns per conversation (10/agent)|
| temperature   | 0          | Solo AEC runs (deterministic)       |
| temperature   | varies     | Collaborative conversations (stochastic) |
| n_cells       | 12         | CPP rubric cells (4 phases × 3 competencies) |

---

## Verifying Key Numbers

```python
import json

# CDI means
for cond in ['C2', 'C6', 'C7']:
    files = list(Path(f'outputs/pilot').glob(f'*_{cond}_*.json'))
    cdis = [json.load(open(f))['cdi'] for f in files]
    print(f'{cond}: CDI={sum(cdis)/len(cdis):.3f}')
# Expected: C2=0.362, C6=0.390, C7=0.613

# Ablation summary
summary = json.load(open('outputs/ablations/ablations_summary.json'))
print(summary)
# Expected: CFULL CDI=0.296, CEXP CDI=0.267
```
