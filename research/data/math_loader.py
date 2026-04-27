"""
Load, filter and stratify-sample from the MATH dataset (Hendrycks et al.).
Outputs a list of dicts ready for the splitter pipeline.
"""
import json, re, random
from pathlib import Path
from typing import List, Dict, Optional
from datasets import load_dataset

from research.config import CFG


# ── helpers ──────────────────────────────────────────────────────────────────

def _clean_latex(text: str) -> str:
    """Light cleanup: collapse whitespace, keep LaTeX intact."""
    return re.sub(r"\s+", " ", text).strip()


def _extract_boxed_answer(solution: str) -> Optional[str]:
    """Pull the \\boxed{...} answer that MATH uses as ground truth."""
    m = re.search(r"\\boxed\{(.+?)\}", solution)
    return m.group(1).strip() if m else None


def _classify_openness(problem: str) -> str:
    """
    Heuristic: problems with 'prove', 'show that', 'explain why'
    are open-ended; otherwise closed-form.
    """
    lower = problem.lower()
    if any(kw in lower for kw in ["prove", "show that", "explain why", "justify"]):
        return "open"
    return "closed"


def _map_subject(raw: str) -> str:
    """Normalise MATH subject names to short keys (matches EleutherAI config names)."""
    table = {
        # EleutherAI config names (already normalised)
        "algebra": "algebra",
        "geometry": "geometry",
        "precalculus": "precalculus",
        "counting_and_probability": "probability",
        "number_theory": "number_theory",
        "prealgebra": "prealgebra",
        "intermediate_algebra": "algebra",
        # Legacy display names
        "Algebra": "algebra",
        "Geometry": "geometry",
        "Precalculus": "precalculus",
        "Counting & Probability": "probability",
        "Number Theory": "number_theory",
        "Prealgebra": "prealgebra",
        "Intermediate Algebra": "algebra",
    }
    return table.get(raw, raw.lower().replace(" ", "_"))


# ── main loader ───────────────────────────────────────────────────────────────

def load_math_dataset(
    problems_per_cell: int = CFG.problems_per_cell,
    seed: int = 42,
    cache_path: Optional[str] = None,
) -> List[Dict]:
    """
    Returns a stratified sample from EleutherAI/hendrycks_math, balanced across
    (subject × difficulty_level) cells.

    Each item:
    {
        "id":         str,
        "problem":    str,
        "solution":   str,
        "answer":     str | None,   # ground truth from \\boxed{}
        "subject":    str,          # normalised short key
        "level":      int,          # 1–5
        "openness":   str,          # "open" | "closed"
    }
    """
    if cache_path and Path(cache_path).exists():
        with open(cache_path) as f:
            return json.load(f)

    rng = random.Random(seed)

    # EleutherAI/hendrycks_math uses per-subject configs
    cells: Dict[tuple, List[Dict]] = {}
    for config_name in CFG.subjects:
        try:
            ds = load_dataset(CFG.dataset_id, config_name,
                              split=CFG.dataset_split)
        except Exception as e:
            print(f"  Warning: could not load config '{config_name}': {e}")
            continue

        subj = _map_subject(config_name)
        for row in ds:
            raw_level = str(row.get("level", "")).replace("Level ", "").strip()
            try:
                level = int(raw_level)
            except ValueError:
                continue
            if level not in CFG.levels:
                continue
            key = (subj, level)
            cells.setdefault(key, []).append(row)

    out: List[Dict] = []
    uid = 0
    for (subj, level), rows in sorted(cells.items()):
        sample = rng.sample(rows, min(problems_per_cell, len(rows)))
        for row in sample:
            problem  = row.get("problem", "") or row.get("question", "")
            solution = row.get("solution", "")
            ans = _extract_boxed_answer(solution)
            out.append({
                "id":       f"math_{uid:05d}",
                "problem":  _clean_latex(problem),
                "solution": _clean_latex(solution),
                "answer":   ans,
                "subject":  subj,
                "level":    level,
                "openness": _classify_openness(problem),
            })
            uid += 1

    rng.shuffle(out)

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)

    return out


def summarize(problems: List[Dict]) -> None:
    from collections import Counter
    subj_counts  = Counter(p["subject"] for p in problems)
    level_counts = Counter(p["level"]   for p in problems)
    open_counts  = Counter(p["openness"] for p in problems)
    print(f"Total problems: {len(problems)}")
    print(f"By subject:  {dict(subj_counts)}")
    print(f"By level:    {dict(sorted(level_counts.items()))}")
    print(f"By openness: {dict(open_counts)}")
    no_ans = sum(1 for p in problems if not p["answer"])
    print(f"Missing ground-truth answer: {no_ans}")
