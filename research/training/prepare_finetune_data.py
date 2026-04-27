"""
Convert existing jigsaw split files into an instruction-tuning dataset.

Each valid split JSON becomes one training example:
  system  : task description for the splitter
  user    : problem text + n_agents
  assistant: the split JSON output

Output: outputs/training/splits_instruct.jsonl
        outputs/training/splits_instruct_stats.json

Usage:
  python -m research.training.prepare_finetune_data
  python -m research.training.prepare_finetune_data --min-n 2 --max-n 4
"""
import argparse
import json
from pathlib import Path

SPLITS_DIR  = Path("outputs/splits")
TRAIN_DIR   = Path("outputs/training")
OUT_JSONL   = TRAIN_DIR / "splits_instruct.jsonl"
OUT_STATS   = TRAIN_DIR / "splits_instruct_stats.json"

SYSTEM_PROMPT = """You are an expert at designing jigsaw information splits for collaborative problem-solving research.

Given a math problem and a target number of agents N, your task is to split the problem into N information packets such that:
1. Each agent receives unique, non-overlapping information
2. No single agent can solve the problem alone
3. Collaboration is strictly necessary to reach the solution
4. The split reflects one of the canonical patterns: SPLIT-A (Composite Figure), SPLIT-B (Dual Representation), SPLIT-C (Complementary Conditions), SPLIT-D (Multi-Step Chain), SPLIT-E (Objective×Constraints), SPLIT-F (Sample Space×Counting Principle), SPLIT-G (Hypothesis×Key Lemma)

Respond with a single JSON object containing exactly these fields:
{
  "pattern": "SPLIT-X",
  "split_rationale": "brief explanation of why this split pattern applies",
  "shared_context": "information ALL agents receive (problem setup without key facts)",
  "packets": [
    {"agent_id": 1, "information": "..."},
    ...
  ],
  "agent_roles": [
    {"agent_id": 1, "role_name": "...", "role_description": "..."},
    ...
  ]
}"""


def _build_user_message(problem: str, n: int) -> str:
    return f"Problem:\n{problem}\n\nNumber of agents: {n}"


def _split_to_example(split_data: dict, problem_text: str) -> dict:
    assistant_output = {
        "pattern":         split_data.get("pattern", ""),
        "split_rationale": split_data.get("split_rationale", ""),
        "shared_context":  split_data.get("shared_context", ""),
        "packets":         split_data.get("packets", []),
        "agent_roles":     split_data.get("agent_roles", []),
    }
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": _build_user_message(problem_text, split_data["n"])},
            {"role": "assistant", "content": json.dumps(assistant_output, ensure_ascii=False)},
        ]
    }


def prepare(min_n: int = 2, max_n: int = 4, problems_dir: str = "outputs/data") -> None:
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)

    problems_path = Path(problems_dir) / "math_sample.json"
    if not problems_path.exists():
        raise FileNotFoundError(f"Problems file not found: {problems_path}\nRun stage_load first.")

    with open(problems_path) as f:
        problems: list[dict] = json.load(f)
    prob_map = {p["id"]: p["problem"] for p in problems}

    split_files = sorted(SPLITS_DIR.glob("*.json"))
    if not split_files:
        raise FileNotFoundError(f"No split files in {SPLITS_DIR}\nRun stage_split first.")

    stats = {"total": 0, "valid": 0, "skipped_invalid": 0, "skipped_no_problem": 0,
             "by_n": {}, "by_pattern": {}}

    examples = []
    for sf in split_files:
        stats["total"] += 1
        with open(sf) as f:
            split_data = json.load(f)

        if not split_data.get("valid", False):
            stats["skipped_invalid"] += 1
            continue

        n = split_data.get("n", 0)
        if not (min_n <= n <= max_n):
            continue

        pid = split_data.get("problem_id", "")
        if pid not in prob_map:
            stats["skipped_no_problem"] += 1
            continue

        ex = _split_to_example(split_data, prob_map[pid])
        examples.append(ex)
        stats["valid"] += 1
        stats["by_n"][str(n)]  = stats["by_n"].get(str(n), 0) + 1
        pat = split_data.get("pattern", "UNKNOWN")
        stats["by_pattern"][pat] = stats["by_pattern"].get(pat, 0) + 1

    with open(OUT_JSONL, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(OUT_STATS, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Dataset written to {OUT_JSONL}")
    print(f"  Total split files : {stats['total']}")
    print(f"  Training examples : {stats['valid']}")
    print(f"  Skipped (invalid) : {stats['skipped_invalid']}")
    print(f"  By n              : {stats['by_n']}")
    print(f"  By pattern        : {stats['by_pattern']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-n",       type=int, default=2)
    parser.add_argument("--max-n",       type=int, default=4)
    parser.add_argument("--problems-dir", default="outputs/data")
    args = parser.parse_args()
    prepare(min_n=args.min_n, max_n=args.max_n, problems_dir=args.problems_dir)


if __name__ == "__main__":
    main()
