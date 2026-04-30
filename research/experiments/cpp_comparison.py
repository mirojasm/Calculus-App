"""
Pilot experiment runner: 4 problems × 5 CPP conditions (v4).

2×2 Factorial design + baseline:
  C1  Baseline         — existing corpus splits & conversations (no re-generation)
  C2  CIDI / No-JA     — split_cidi() task-chain split + standard simulation
  C3  Constitutional   — constitutional_split() (36-check critic) + standard simulation
  C4  CIDI / JA        — split_cidi() task-chain split + joint accountability
  C5  Constitutional/JA— constitutional_split() + joint accountability

v4 changes vs v3:
  - goal_anchor uses shared_context (question only), NOT split_result.problem (data leak fix)
  - M4 generates task-role chain packets ("Input/Task/Share/Needs from partner")
  - C4/C5 use joint_accountability instead of Szewkis monitor
  - GROUP VERIFICATION requires independent answer declaration per partner

Usage:
  python3 -m research.experiments.cpp_comparison
  python3 -m research.experiments.cpp_comparison --conditions C2 C5
  python3 -m research.experiments.cpp_comparison --problems math_00042 math_00101
  python3 -m research.experiments.cpp_comparison --target-cpp A1 A2 A3 B1 B2 B3 C1 C2
  python3 -m research.experiments.cpp_comparison --select-only --verbose
"""
from __future__ import annotations
import argparse, json, os, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

_print_lock = threading.Lock()

from research.config import CFG
from research.splitting.splitter import split as standard_split
from research.splitting.constitutional import constitutional_split
from research.simulation.simulator import simulate, simulate_with_monitor
from research.scoring.cpp_annotator import annotate
from research.scoring.atc21s import annotate_conversation as annotate_atc21s
from research.splitting.cidi.pipeline import split_cidi, CPP_DEEP_TARGET

# ── paths ──────────────────────────────────────────────────────────────────────

OUT_DIR    = Path("outputs")
SPLITS_DIR = OUT_DIR / "splits"
CONV_DIR   = OUT_DIR / "conversations"
PILOT_DIR  = OUT_DIR / "pilot"
PILOT_CONV_DIR = PILOT_DIR / "conversations"   # clean conversation archives
MODELS_DIR = OUT_DIR / "models"

PILOT_DIR.mkdir(parents=True, exist_ok=True)
PILOT_CONV_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

ALL_CONDITIONS = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]

# Known correct answers loaded at runtime from corpus conversations (ground_truth field).
# Fallback only — prefer ground_truth from the corpus JSON.
KNOWN_ANSWERS: dict[str, str] = {
    "math_00014": "26",   # x(3x-7)=-3 → x=(7±√13)/6, m+n+p=7+13+6=26
    "math_00050": "8",    # 712n div by 18 → n=8
    "math_00121": "44",   # sec+tan=22/7 → csc+cot=29/15, m+n=44
    "math_00128": "48",   # (6!+7!)/5! = 5760/120 = 48
}

def _get_ground_truth(problem_id: str) -> str | None:
    """Return ground truth from corpus conversation JSON, falling back to KNOWN_ANSWERS."""
    conv_path = CONV_DIR / f"{problem_id}_jigsaw_2.json"
    if conv_path.exists():
        try:
            gt = json.loads(conv_path.read_text()).get("ground_truth", "")
            if gt:
                return str(gt).strip()
        except Exception:
            pass
    return KNOWN_ANSWERS.get(problem_id)

# Redesigned epistemic splits for problems where the CIDI pipeline produces data splits.
# Applied to conditions C2-C5 (not C1 baseline) to isolate condition effects from split quality.
SPLIT_OVERRIDES: dict[str, dict] = {
    # math_00121: information-only split (v5 minimal framing principle).
    # A1 has the given equation; A2 has the target expression and the goal.
    # Neither knows what the other holds. Phase A must emerge from conversation.
    "math_00121": {
        "shared_context": "Find the integer m + n.",
        "agent_1_info": "sec θ + tan θ = 22/7",
        "agent_2_info": "csc θ + cot θ = m/n where gcd(m, n) = 1. Find m + n.",
    },
}


def _check_correctness(final_answer: str | None, problem_id: str,
                        answer_format: dict | None = None) -> str:
    """
    Returns 'correct', 'partial', 'incorrect', 'incomplete', or 'unknown'.
    'partial': answer shows correct mathematical progress but wrong final form
               (e.g., found the intermediate fraction but didn't compute m+n).
    Uses partial_credit_indicators from M1 anatomy when available.
    """
    known = _get_ground_truth(problem_id)
    if not known:
        return "unknown"
    if not final_answer or not final_answer.strip():
        return "incomplete"
    import re
    nums = re.findall(r"-?\d+(?:\.\d+)?", final_answer)
    if not nums:
        return "incomplete"
    if known in nums:
        return "correct"
    # Check for partial credit: correct intermediate value present in answer
    indicators = (answer_format or {}).get("partial_credit_indicators", [])
    answer_lower = final_answer.lower()
    for indicator in indicators:
        if str(indicator).lower() in answer_lower:
            return "partial"
    return "incorrect"


def _compute_cy(cdi: float, correctness: str) -> float:
    """
    Collaborative Yield: combines CDI (process) and correctness (outcome).
    correctness levels: 'correct'=1.0, 'partial'=0.5, else 0.0
    Awards a coupling bonus when deep collaboration produced the right answer,
    and penalizes low-process correct answers (trivial/lucky).
    α=0.35 (process), β=0.45 (outcome), γ=0.20 (coupling bonus).
    """
    corr_weight = {"correct": 1.0, "partial": 0.5}.get(correctness, 0.0)
    if cdi >= 0.5 and correctness == "correct":
        coupling = 1.0
    elif cdi >= 0.5 and correctness == "partial":
        coupling = 0.3   # partial credit for partial answer with deep process
    elif cdi < 0.3 and correctness == "correct":
        coupling = -0.5  # penalize trivial correct
    else:
        coupling = 0.0
    return round(0.35 * cdi + 0.45 * corr_weight + 0.20 * coupling, 3)


def _get_quadrant(cdi: float, correctness: str) -> str:
    """
    2D process-outcome quadrant (Kapur productive-failure taxonomy).
    'partial' is treated as a productive failure variant with correct reasoning.
      COUPLING      — deep collaboration → correct answer (target state)
      PARTIAL_COUP  — deep collaboration → partial answer (correct reasoning, format error)
      PROD_FAIL     — deep collaboration → incorrect (productive failure)
      TRIVIAL       — shallow process → correct (lucky or too easy)
      COLLAPSE      — shallow process → incorrect/incomplete
    """
    high = cdi >= 0.5
    if high and correctness == "correct":   return "COUPLING"
    if high and correctness == "partial":   return "PARTIAL_COUP"
    if high:                                return "PROD_FAIL"
    if correctness == "correct":            return "TRIVIAL"
    return "COLLAPSE"


def _apply_split_override(split_result, problem_id: str):
    """Replace packets for problems with a known better epistemic split."""
    ov = SPLIT_OVERRIDES.get(problem_id)
    if not ov:
        return split_result, False
    from research.splitting.splitter import Packet
    split_result.packets = [
        Packet(agent_id=1, information=ov["agent_1_info"],
               role_name="", role_description=""),
        Packet(agent_id=2, information=ov["agent_2_info"],
               role_name="", role_description=""),
    ]
    split_result.shared_context = ov["shared_context"]
    return split_result, True


# ── Phase 1 split cache ────────────────────────────────────────────────────────
# C2, C6, C7 use the same CIDI split. Phase 2 reuses Phase 1 C7 splits to avoid
# regenerating them (saves ~$0.028/problem and ~15s of gpt-4.1 calls).

_SPLIT_CACHE: dict[str, object] = {}   # problem_id → SplitResult


def _reconstruct_split_result(split_dict: dict):
    """Rebuild a SplitResult from a cached result['split'] dict."""
    from research.splitting.splitter import SplitResult, Packet
    raw_packets = split_dict.get("raw_split", {}).get("packets", [])
    packets = [
        Packet(
            agent_id=p["agent_id"],
            information=p["information"],
            role_name=p.get("role_name", ""),
            role_description=p.get("role_description", ""),
        )
        for p in raw_packets
    ]
    return SplitResult(
        problem_id=split_dict["problem_id"],
        problem=split_dict.get("problem", ""),
        n=split_dict.get("n", 2),
        pattern=split_dict.get("pattern", ""),
        shared_context=split_dict.get("shared_context", ""),
        packets=packets,
        valid=split_dict.get("valid", True),
        validation_log=split_dict.get("validation_log", {}),
        raw_split=split_dict.get("raw_split", {}),
        answer_format=split_dict.get("answer_format", {}),
    )


def load_phase1_split_cache(problem_ids: list[str] | None = None) -> int:
    """
    Scan PILOT_DIR for Phase 1 C7 result files and populate _SPLIT_CACHE.
    Uses the most recent file per problem_id. Returns number of entries loaded.
    """
    pid_set = set(problem_ids) if problem_ids else None
    best: dict[str, tuple[str, dict]] = {}   # pid → (timestamp, split_dict)
    for p in PILOT_DIR.glob("math_*_C7_*.json"):
        parts = p.stem.split("_")
        if len(parts) < 5:
            continue
        pid = f"{parts[0]}_{parts[1]}"
        ts  = f"{parts[3]}_{parts[4]}"
        if pid_set and pid not in pid_set:
            continue
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        if "error" in data or "split" not in data:
            continue
        prev_ts, _ = best.get(pid, ("", {}))
        if ts > prev_ts:
            best[pid] = (ts, data["split"])

    loaded = 0
    for pid, (_, split_dict) in best.items():
        try:
            _SPLIT_CACHE[pid] = _reconstruct_split_result(split_dict)
            loaded += 1
        except Exception:
            pass
    return loaded


# ── corpus loading ─────────────────────────────────────────────────────────────

def _load_existing_splits() -> dict:
    splits = {}
    for p in SPLITS_DIR.glob("*_n2.json"):
        try:
            data = json.loads(p.read_text())
            splits[data["problem_id"]] = data
        except Exception:
            pass
    return splits


def _load_existing_conversations() -> dict:
    convs = {}
    for p in CONV_DIR.glob("*.json"):
        try:
            data = json.loads(p.read_text())
            convs[(data["problem_id"], data["condition"])] = data
        except Exception:
            pass
    return convs


def _load_existing_scores() -> dict:
    scores = {}
    score_dir = OUT_DIR / "scores"
    for p in score_dir.glob("*_jigsaw_2_scores.json"):
        try:
            data = json.loads(p.read_text())
            pid = p.stem.replace("_jigsaw_2_scores", "")
            scores[pid] = data
        except Exception:
            pass
    return scores


def _load_completed_cells() -> dict[tuple[str, str], dict]:
    """
    Scan PILOT_DIR for already-completed (problem_id, condition) pairs.
    Returns dict mapping (pid, cond) → most recent result dict.
    Used by --resume to skip cells and reload their results into the consolidated output.
    """
    completed: dict[tuple[str, str], tuple[str, dict]] = {}  # (pid,cond) → (ts, data)
    for p in PILOT_DIR.glob("math_*_C*_*.json"):
        stem = p.stem  # e.g. math_00121_C7_20260429_042801
        parts = stem.split("_")
        # parts: ['math', 'NNNNN', 'CX', 'YYYYMMDD', 'HHMMSS']
        if len(parts) < 5:
            continue
        pid  = f"{parts[0]}_{parts[1]}"   # math_NNNNN
        cond = parts[2]                    # CX
        ts   = f"{parts[3]}_{parts[4]}"   # YYYYMMDD_HHMMSS
        if not cond.startswith("C"):
            continue
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        if "error" in data:
            continue
        # Keep the most recent timestamp for each (pid, cond)
        prev_ts, _ = completed.get((pid, cond), ("", {}))
        if ts > prev_ts:
            completed[(pid, cond)] = (ts, data)
    return {k: v[1] for k, v in completed.items()}


# ── problem selection ──────────────────────────────────────────────────────────

def select_pilot_problems(n_problems: int = 4, verbose: bool = False) -> list[dict]:
    """
    Select n_problems from corpus with matched splits + conversations + scores,
    ordered by increasing PISA_global (most room for improvement first).
    Falls back to any splits with conversations if no scored examples exist.
    """
    splits  = _load_existing_splits()
    convs   = _load_existing_conversations()
    scores  = _load_existing_scores()

    candidates = []
    for pid, split_data in splits.items():
        conv_data = convs.get((pid, "jigsaw_2"), {})
        if not conv_data:
            continue  # need at least a baseline conversation
        score = scores.get(pid, {})
        pisa_global = score.get("pisa", {}).get("global_index", 99.0)
        turns = conv_data.get("total_turns", 999)
        candidates.append((pisa_global, turns, pid, split_data, conv_data))

    # Sort: ascending PISA_global (most room for improvement), then ascending turns
    candidates.sort(key=lambda x: (x[0], x[1]))

    selected = []
    for pisa_global, turns, pid, split_data, conv_data in candidates[:n_problems]:
        problem_text = conv_data.get("problem", "") or split_data.get("problem", "")
        entry = {
            "problem_id":     pid,
            "problem":        problem_text,
            "subject":        split_data.get("subject", "unknown"),
            "level":          split_data.get("level", 0),
            "c1_pisa_global": pisa_global,
            "c1_turns":       turns,
            **split_data,
        }
        selected.append(entry)
        if verbose:
            print(f"  Selected {pid}: PISA={pisa_global:.3f} turns={turns}")

    return selected


# ── condition runners ──────────────────────────────────────────────────────────

def run_c1(problem_id: str, problem: str) -> dict:
    """C1: load existing baseline outputs."""
    splits = _load_existing_splits()
    convs  = _load_existing_conversations()
    scores = _load_existing_scores()

    split_data = splits.get(problem_id)
    conv_data  = convs.get((problem_id, "jigsaw_2"))
    score_data = scores.get(problem_id, {})

    if not split_data or not conv_data:
        return {"condition": "C1", "problem_id": problem_id,
                "error": "No existing C1 data — run main pipeline first"}

    # Derive CPP vector from PISA code_counts
    from research.splitting.cidi.module2_feasibility import CELL_ORDER
    cc = score_data.get("pisa", {}).get("code_counts", {})
    cpp_vec = [1 if cc.get(c, 0) > 0 else 0 for c in CELL_ORDER]
    cdi     = sum(cpp_vec) / 12

    return {
        "condition":    "C1",
        "problem_id":   problem_id,
        "split":        split_data,
        "conversation": conv_data,
        "cpp_vector":   cpp_vec,
        "cdi":          cdi,
        "cdi_label":    _cdi_label(cdi),
        "pisa_global":  score_data.get("pisa", {}).get("global_index", None),
        "source":       "existing",
    }


def _ensure_answer_format(split_result, problem: str) -> None:
    """
    Ensure split_result.answer_format is populated.
    For conditions that don't use the CIDI pipeline (C3, C4), call M1 lightweight extraction.
    """
    if getattr(split_result, "answer_format", None):
        return
    try:
        from research.splitting.cidi.module1_semantic import extract_answer_format
        split_result.answer_format = extract_answer_format(problem)
    except Exception:
        split_result.answer_format = {
            "type": "other",
            "specification": "State your final answer clearly when both partners agree.",
            "partial_credit_indicators": [],
        }


def _run_with_annotator(
    condition_name: str,
    problem_id: str,
    problem: str,
    split_result,
    sim_fn,
    extra: dict = None,
    apply_override: bool = True,
) -> dict:
    """Common wrapper: optionally apply split override, simulate, annotate, build result dict."""
    overridden = False
    if apply_override:
        split_result, overridden = _apply_split_override(split_result, problem_id)

    _ensure_answer_format(split_result, problem)
    answer_format = getattr(split_result, "answer_format", {}) or {}

    t_sim = time.time()
    conv = sim_fn(split_result)
    sim_sec = round(time.time() - t_sim, 1)

    t_ann = time.time()
    cpp = annotate(conv)
    atc = annotate_atc21s(conv)
    ann_sec = round(time.time() - t_ann, 1)

    correctness = _check_correctness(conv.final_answer, problem_id, answer_format)
    cdi = cpp.cdi

    result = {
        "condition":        condition_name,
        "problem_id":       problem_id,
        "conversation":     conv.to_dict(),
        # CPP / PISA matrix
        "cpp_vector":       cpp.cpp_vector,
        "quality_scores":   cpp.quality_scores,
        "cdi":              cdi,
        "cqi":              cpp.cqi,
        "phaq":             cpp.phaq,
        "cdi_label":        cpp.cdi_label,
        "cpp_rationale":    cpp.rationale,
        # ATC21S
        "atc_dim_scores":   atc.dim_scores,
        "atc_cqi":          atc.atc_cqi,
        "atc_social_qi":    atc.social_qi,
        "atc_cogn_qi":      atc.cogn_qi,
        "atc_rationale":    atc.rationale,
        # Outcome
        "correctness":      correctness,
        "cy":               _compute_cy(cdi, correctness),
        "quadrant":         _get_quadrant(cdi, correctness),
        "final_answer":     conv.final_answer,
        "known_answer":     KNOWN_ANSWERS.get(problem_id),
        "answer_format":    answer_format,
        "split_overridden": overridden,
        "timing":           {"sim_sec": sim_sec, "annot_sec": ann_sec},
    }
    if extra:
        result.update(extra)
    return result


def _cached_split(problem_id: str, n: int):
    """Return cached split if it exists and matches n, else None."""
    sr = _SPLIT_CACHE.get(problem_id)
    return sr if sr is not None and getattr(sr, "n", n) == n else None


def run_c2(
    problem_id: str,
    problem: str,
    n: int = 2,
    target_cpp: list[str] = None,
    skip_validation: bool = False,
) -> dict:
    """C2: CIDI pipeline (6 modules) + standard simulation."""
    target = target_cpp or CPP_DEEP_TARGET
    cached = _cached_split(problem_id, n)
    if cached is not None:
        split  = cached
        timing = {"split_sec": 0.0, "cached": True}
        extra_cidi = {}
    else:
        t0 = time.time()
        cidi_result = split_cidi(
            problem_id, problem, target_cpp=target, n=n,
            skip_validation=skip_validation,
        )
        split  = cidi_result.split
        timing = {**cidi_result.timing_sec, "split_sec": round(time.time() - t0, 1)}
        extra_cidi = {"cidi": cidi_result.to_dict()}

    result = _run_with_annotator(
        "C2", problem_id, problem,
        split,
        lambda sr: simulate(sr, f"cpp_directed_{n}"),
        extra={"split": split.__dict__, "timing": timing, **extra_cidi},
    )
    return result


def run_c3(
    problem_id: str,
    problem: str,
    n: int = 2,
) -> dict:
    """C3: Constitutional pipeline (36-check critic) + standard simulation."""
    t0 = time.time()
    const_result = constitutional_split(problem_id, problem, n)
    split_sec = round(time.time() - t0, 1)

    result = _run_with_annotator(
        "C3", problem_id, problem,
        const_result.split,
        lambda sr: simulate(sr, f"jigsaw_{n}"),
        extra={
            "split": const_result.split.__dict__,
            "constitutional": {
                "final_sqs":     const_result.final_sqs,
                "iterations":    const_result.iterations,
                "approved":      const_result.approved,
                "critique_history": const_result.critique_history[-1:]
                                    if const_result.critique_history else [],
            },
            "timing": {"split_sec": split_sec},
        },
    )
    return result


def run_c4(
    problem_id: str,
    problem: str,
    n: int = 2,
    target_cpp: list[str] = None,
    skip_validation: bool = False,
) -> dict:
    """
    C4: CIDI epistemic split + joint accountability (Roschelle convergence criterion).
    Uses the SAME split as C2 to isolate the effect of joint accountability from split quality.
    Joint accountability: both agents must each independently state the same FINAL ANSWER.
    Grounded in Szewkis (2011) condition 4 (group reward) and Roschelle (1992) convergence.
    Replaces Szewkis monitor (which caused incomplete answers under minimal framing).
    """
    target = target_cpp or CPP_DEEP_TARGET
    t0 = time.time()
    cidi_result = split_cidi(
        problem_id, problem, target_cpp=target, n=n,
        skip_validation=skip_validation,
    )
    split_sec = round(time.time() - t0, 1)

    result = _run_with_annotator(
        "C4", problem_id, problem,
        cidi_result.split,
        lambda sr: simulate(sr, f"joint_jigsaw_{n}", joint_accountability=True),
        extra={
            "split": cidi_result.split.__dict__,
            "cidi": cidi_result.to_dict(),
            "timing": {**cidi_result.timing_sec, "split_sec": split_sec},
            "joint_accountability": True,
        },
    )
    return result


def run_c5(
    problem_id: str,
    problem: str,
    n: int = 2,
) -> dict:
    """
    C5: Constitutional epistemic split + joint accountability.
    Uses constitutional split (same as C3) + joint accountability (same as C4).
    2×2 factorial cell: best split quality × joint accountability.
    Tests whether constitutional split quality and joint accountability compound.
    """
    t0 = time.time()
    const_result = constitutional_split(problem_id, problem, n)
    split_sec = round(time.time() - t0, 1)

    result = _run_with_annotator(
        "C5", problem_id, problem,
        const_result.split,
        lambda sr: simulate(sr, f"joint_jigsaw_{n}", joint_accountability=True),
        extra={
            "split": const_result.split.__dict__,
            "constitutional": {
                "final_sqs":     const_result.final_sqs,
                "iterations":    const_result.iterations,
                "approved":      const_result.approved,
            },
            "timing": {"split_sec": split_sec},
            "joint_accountability": True,
        },
    )
    return result


def run_c6(
    problem_id: str,
    problem: str,
    n: int = 2,
    target_cpp: list[str] = None,
    skip_validation: bool = False,
) -> dict:
    """
    C6: CIDI epistemic split + peer-aware framing.
    Same split as C2. Adds three lines to the system prompt:
      - communication necessity ("you need to communicate to solve this")
      - reasoning rigor ("show all steps, no guessing")
      - solvability ("the problem has a definite solution")
    Does NOT prescribe coordination strategy. Phase A must emerge from conversation.
    Tests H3: peer awareness activates Phase A without scripting it.
    """
    target = target_cpp or CPP_DEEP_TARGET
    cached = _cached_split(problem_id, n)
    if cached is not None:
        split  = cached
        timing = {"split_sec": 0.0, "cached": True}
        extra_cidi = {}
    else:
        t0 = time.time()
        cidi_result = split_cidi(
            problem_id, problem, target_cpp=target, n=n,
            skip_validation=skip_validation,
        )
        split  = cidi_result.split
        timing = {**cidi_result.timing_sec, "split_sec": round(time.time() - t0, 1)}
        extra_cidi = {"cidi": cidi_result.to_dict()}

    result = _run_with_annotator(
        "C6", problem_id, problem,
        split,
        lambda sr: simulate(sr, f"peer_jigsaw_{n}", peer_aware=True),
        extra={"split": split.__dict__, "timing": timing, "peer_aware": True, **extra_cidi},
    )
    return result


def run_c7(
    problem_id: str,
    problem: str,
    n: int = 2,
    target_cpp: list[str] = None,
    skip_validation: bool = False,
) -> dict:
    """
    C7: CIDI epistemic split + student-simulating agent (L3), no joint accountability.
    Same CIDI split as C2/C6. Agent simulates a Calc-1 student's epistemic process:
    tentative reasoning, genuine uncertainty, asks partner when stuck.
    Purpose: proxy for how real students would collaborate on this split.
    A split that forces CDI/CQI with L3 predicts ecological validity with real students.
    Tests H4: student-sim agent produces higher PhAQ than L1/L2 on genuine epistemic splits.
    """
    target = target_cpp or CPP_DEEP_TARGET
    cached = _cached_split(problem_id, n)
    if cached is not None:
        split  = cached
        timing = {"split_sec": 0.0, "cached": True}
        extra_cidi = {}
    else:
        t0 = time.time()
        cidi_result = split_cidi(
            problem_id, problem, target_cpp=target, n=n,
            skip_validation=skip_validation,
        )
        split  = cidi_result.split
        timing = {**cidi_result.timing_sec, "split_sec": round(time.time() - t0, 1)}
        extra_cidi = {"cidi": cidi_result.to_dict()}

    result = _run_with_annotator(
        "C7", problem_id, problem,
        split,
        lambda sr: simulate(sr, f"student_jigsaw_{n}", student_sim=True),
        extra={
            "split": split.__dict__,
            "timing": timing,
            "student_sim": True,
            "agent_type": "L3_student_sim",
            **extra_cidi,
        },
    )
    return result


def run_c8(
    problem_id: str,
    problem: str,
    n: int = 2,
    target_cpp: list[str] = None,
    skip_validation: bool = False,
) -> dict:
    """
    C8: CIDI epistemic split + student-simulating agent (L3) + joint accountability.
    Tests whether JA helps student-sim agents reach convergence without causing
    premature closure (the C4 pathology observed in piloto v5 math_00121: 4 turns).
    """
    target = target_cpp or CPP_DEEP_TARGET
    t0 = time.time()
    cidi_result = split_cidi(
        problem_id, problem, target_cpp=target, n=n,
        skip_validation=skip_validation,
    )
    split_sec = round(time.time() - t0, 1)

    result = _run_with_annotator(
        "C8", problem_id, problem,
        cidi_result.split,
        lambda sr: simulate(sr, f"student_joint_jigsaw_{n}",
                            student_sim=True, joint_accountability=True),
        extra={
            "split": cidi_result.split.__dict__,
            "cidi": cidi_result.to_dict(),
            "timing": {**cidi_result.timing_sec, "split_sec": split_sec},
            "student_sim": True,
            "joint_accountability": True,
            "agent_type": "L3_student_sim",
        },
    )
    return result


CONDITION_RUNNERS = {
    "C1": lambda pid, prob, n, t, sv: run_c1(pid, prob),
    "C2": lambda pid, prob, n, t, sv: run_c2(pid, prob, n, t, sv),
    "C3": lambda pid, prob, n, t, sv: run_c3(pid, prob, n),
    "C4": lambda pid, prob, n, t, sv: run_c4(pid, prob, n, t, sv),
    "C5": lambda pid, prob, n, t, sv: run_c5(pid, prob, n),
    "C6": lambda pid, prob, n, t, sv: run_c6(pid, prob, n, t, sv),
    "C7": lambda pid, prob, n, t, sv: run_c7(pid, prob, n, t, sv),
    "C8": lambda pid, prob, n, t, sv: run_c8(pid, prob, n, t, sv),
}


# ── summary ────────────────────────────────────────────────────────────────────

def _cdi_label(cdi: float) -> str:
    if cdi < 0.08:  return "CPP-Ø"
    if cdi < 0.25:  return "CPP-IC"
    if cdi < 0.42:  return "CPP-CG"
    if cdi < 0.58:  return "CPP-RP"
    if cdi < 0.83:  return "CPP-DEEP"
    return "CPP-FULL"


def _print_summary(results: list[dict]) -> None:
    print("\n" + "="*110)
    print("PILOT v4 RESULTS SUMMARY")
    print("="*110)
    print(f"{'Problem':<22} {'Cond':<4} {'CDI':>6} {'CQI':>6} {'PhAQ':>6} {'CY':>6} "
          f"{'Quadrant':<12} {'Profile':<12} {'Turns':>6}  {'Correct?':<10}")
    print("-"*110)
    for r in results:
        if "error" in r:
            print(f"{r.get('problem_id','?'):<22} {r.get('condition','?'):<4}  ERROR: {r['error']}")
            continue
        conv    = r.get("conversation", {})
        turns   = conv.get("total_turns", "?")
        corr    = r.get("correctness", "?")
        ans     = r.get("final_answer", "") or ""
        ans_s   = ans[:10] if len(ans) <= 10 else ans[:8]+"…"
        ov_flag = "*" if r.get("split_overridden") else " "
        print(
            f"{r['problem_id']:<22} {r['condition']:<4} "
            f"{r.get('cdi', 0):>6.3f} {r.get('cqi', 0):>6.3f} {r.get('phaq', 0):>6.3f} "
            f"{r.get('cy', 0):>6.3f} "
            f"{r.get('quadrant','?'):<12} "
            f"{r.get('cdi_label','?'):<12} "
            f"{turns:>6}{ov_flag} {corr:<10} [{ans_s}]"
        )
    print("="*110)
    print("  * = split override applied (epistemic redesign)")

    # Quadrant distribution summary
    from collections import Counter
    quads = Counter(r.get("quadrant", "?") for r in results if "error" not in r)
    print(f"\n  Quadrant distribution: " +
          " | ".join(f"{q}: {n}" for q, n in sorted(quads.items())))

    # Mean CDI, CQI, PhAQ, ATC, CY per condition
    from collections import defaultdict
    by_cond: dict = defaultdict(list)
    for r in results:
        if "error" not in r:
            by_cond[r["condition"]].append({
                "cdi":        r.get("cdi", 0),
                "cqi":        r.get("cqi", 0),
                "phaq":       r.get("phaq", 0),
                "atc_cqi":    r.get("atc_cqi", 0),
                "social_qi":  r.get("atc_social_qi", 0),
                "cogn_qi":    r.get("atc_cogn_qi", 0),
                "cy":         r.get("cy", 0),
            })
    print("\n  Per-condition means (CDI=binary presence, CQI=PISA quality, ATC=ATC21S quality):")
    hdr = f"    {'Cond':<4}  {'CDI':>6}  {'CQI':>6}  {'PhAQ':>6}  {'ATC_CQI':>8}  {'SocialQI':>9}  {'CognQI':>7}  {'CY':>6}  n"
    print(hdr)
    print("    " + "-"*(len(hdr)-4))
    for cond in sorted(by_cond):
        rows = by_cond[cond]
        n    = len(rows)
        def _m(k): return sum(r[k] for r in rows) / n
        print(
            f"    {cond:<4}  {_m('cdi'):>6.3f}  {_m('cqi'):>6.3f}  {_m('phaq'):>6.3f}  "
            f"{_m('atc_cqi'):>8.3f}  {_m('social_qi'):>9.3f}  {_m('cogn_qi'):>7.3f}  "
            f"{_m('cy'):>6.3f}  {n}"
        )
    print("="*110)


# ── main runner ────────────────────────────────────────────────────────────────

def _run_cell(
    pid: str,
    problem: str,
    cond: str,
    n: int,
    target: list,
    skip_validation: bool,
    timestamp: str,
    verbose: bool = False,
) -> dict:
    """Execute one (problem_id, condition) cell. Thread-safe: writes its own output file."""
    t_start = time.time()
    try:
        result = CONDITION_RUNNERS[cond](pid, problem, n, target, skip_validation)
    except Exception as e:
        result = {"condition": cond, "problem_id": pid, "error": str(e)}
        if verbose:
            import traceback
            with _print_lock:
                traceback.print_exc()

    result.setdefault("cdi_label", _cdi_label(result.get("cdi", 0)))
    elapsed = round(time.time() - t_start, 1)

    # Individual result file (unique filename — no write conflict between threads)
    fname = PILOT_DIR / f"{pid}_{cond}_{timestamp}.json"
    fname.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str))

    # Clean conversation archive
    if "conversation" in result and "error" not in result:
        conv_fname = PILOT_CONV_DIR / f"{pid}_{cond}_{timestamp}.json"
        conv_data  = result["conversation"].copy()
        conv_data.update({"pilot_timestamp": timestamp,
                          "split_overridden": result.get("split_overridden", False)})
        conv_fname.write_text(json.dumps(conv_data, ensure_ascii=False, indent=2, default=str))

    with _print_lock:
        cdi   = result.get("cdi", "N/A")
        cqi   = result.get("cqi", "N/A")
        atc   = result.get("atc_cqi", "N/A")
        label = result.get("cdi_label", "")
        tag   = "[ERR] " if "error" in result else "[DONE]"
        print(f"{tag} {pid} × {cond}  CDI={cdi}  CQI={cqi}  ATC={atc}  {label}  ({elapsed}s)",
              flush=True)
    return result


def run_pilot(
    problems:         Optional[list[dict]] = None,
    conditions:       list[str] = None,
    n:                int = 2,
    target_cpp:       list[str] = None,
    skip_validation:  bool = False,
    verbose:          bool = False,
    resume:           bool = False,
    parallel_workers: int = 1,
    n_problems:       int = 4,
) -> list[dict]:

    if problems is None:
        print(f"[INFO] Selecting up to {n_problems} problems from corpus...")
        problems = select_pilot_problems(n_problems, verbose=verbose)
        if not problems:
            raise RuntimeError(
                "No problems found in outputs/splits/. "
                "Run the main pipeline first to generate the corpus."
            )
        print(f"[INFO] Selected {len(problems)} problems")

    if conditions is None:
        conditions = ALL_CONDITIONS

    # Seed ground-truth answers for corpus-file problems (no jigsaw_2.json on disk).
    for prob in problems:
        pid, ans = prob.get("problem_id", ""), prob.get("answer", "")
        if pid and ans and pid not in KNOWN_ANSWERS:
            KNOWN_ANSWERS[pid] = str(ans).strip()

    # Pre-load Phase 1 CIDI splits to avoid regenerating them for C2/C6/C7 Phase 2 runs.
    if any(c in (conditions or ALL_CONDITIONS) for c in ["C2", "C6", "C7"]):
        pid_list = [p["problem_id"] for p in problems]
        n_cached = load_phase1_split_cache(pid_list)
        if n_cached:
            print(f"[CACHE] {n_cached} Phase 1 CIDI splits loaded (C2/C6/C7 will skip re-generation)")

    target = target_cpp or CPP_DEEP_TARGET
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    completed_cells: dict[tuple[str, str], dict] = {}
    if resume:
        completed_cells = _load_completed_cells()
        n_skip = sum(1 for p in problems for c in conditions if (p["problem_id"], c) in completed_cells)
        if n_skip:
            print(f"[RESUME] {n_skip} cells already completed — will skip them")

    # Build work queue
    work: list[tuple[str, str, str]] = []  # (pid, problem_text, cond)
    skip_results: list[dict] = []
    for prob in problems:
        pid  = prob["problem_id"]
        ptxt = prob.get("problem", "")
        for cond in conditions:
            if resume and (pid, cond) in completed_cells:
                skip_results.append(completed_cells[(pid, cond)])
                with _print_lock:
                    r = completed_cells[(pid, cond)]
                    print(f"[SKIP] {pid} × {cond}  CDI={r.get('cdi','?')}  {r.get('cdi_label','')}")
            else:
                work.append((pid, ptxt, cond))

    print(f"[INFO] {len(work)} cells to run, {len(skip_results)} skipped (resume), "
          f"workers={parallel_workers}")

    new_results: list[dict] = []
    if parallel_workers > 1 and len(work) > 1:
        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            futures = {
                executor.submit(
                    _run_cell, pid, ptxt, cond, n, target, skip_validation, timestamp, verbose
                ): (pid, cond)
                for pid, ptxt, cond in work
            }
            for future in as_completed(futures):
                new_results.append(future.result())
    else:
        for pid, ptxt, cond in work:
            new_results.append(
                _run_cell(pid, ptxt, cond, n, target, skip_validation, timestamp, verbose)
            )

    all_results = skip_results + new_results

    # Save consolidated (all cells, including skipped)
    consolidated = PILOT_DIR / f"pilot_results_{timestamp}.json"
    consolidated.write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2, default=str)
    )
    print(f"\n[DONE] Consolidated → {consolidated}")
    _print_summary(all_results)
    return all_results


def main():
    parser = argparse.ArgumentParser(description="CollabMath CPP pilot experiment")
    parser.add_argument("--problems",    nargs="+", help="Problem IDs (default: auto-select)")
    parser.add_argument("--corpus-file", help="JSON file with list of problem dicts (for Corpus 2+); "
                        "each dict needs 'problem_id' and 'problem' fields")
    parser.add_argument("--conditions",  nargs="+", choices=ALL_CONDITIONS,
                        default=ALL_CONDITIONS)
    parser.add_argument("--n",           type=int, default=2)
    parser.add_argument("--target-cpp",  nargs="+", default=None,
                        help="CPP cells to target (default: CPP-DEEP A1-C2)")
    parser.add_argument("--skip-validation", action="store_true",
                        help="Skip M5 discriminator validation (useful before training)")
    parser.add_argument("--select-only", action="store_true",
                        help="Only print selected problems, do not run")
    parser.add_argument("--resume",      action="store_true",
                        help="Skip already-completed (problem, condition) pairs (checkpoint resume)")
    parser.add_argument("--workers",     type=int, default=1,
                        help="Parallel worker threads (default: 1; use 4-8 for corpus runs)")
    parser.add_argument("--n-problems",  type=int, default=4,
                        help="Max problems to auto-select (default: 4; use 9999 for full corpus)")
    parser.add_argument("--verbose",     action="store_true")
    args = parser.parse_args()

    problems = None
    if args.corpus_file:
        raw = json.loads(Path(args.corpus_file).read_text())
        problems = []
        for item in raw:
            pid = item.get("problem_id") or item.get("id", "")
            if not pid:
                continue
            problems.append({
                "problem_id": pid,
                "problem":    item.get("problem", ""),
                "subject":    item.get("subject", ""),
                "level":      item.get("level", 0),
                "answer":     item.get("answer", ""),
            })
        print(f"[INFO] Loaded {len(problems)} problems from {args.corpus_file}")
    elif args.problems:
        existing_splits = _load_existing_splits()
        existing_convs  = _load_existing_conversations()
        problems = []
        for pid in args.problems:
            if pid not in existing_splits:
                print(f"[WARN] Problem '{pid}' not found in corpus — skipping")
                continue
            split_data   = existing_splits[pid]
            conv_data    = existing_convs.get((pid, "jigsaw_2"), {})
            problem_text = conv_data.get("problem", "") or split_data.get("problem", "")
            if not problem_text:
                print(f"[WARN] Problem text missing for '{pid}' — CIDI will hallucinate. "
                      "Add 'problem' field to split JSON or use select_pilot_problems().")
            problems.append({"problem_id": pid, "problem": problem_text, **split_data})

    if args.select_only:
        probs = problems or select_pilot_problems(args.n_problems, verbose=True)
        print("\nSelected problems:")
        for p in probs:
            print(f"  {p['problem_id']:30s} subject={p.get('subject','')} "
                  f"level={p.get('level','')} pisa={p.get('c1_pisa_global','?')}")
        return

    run_pilot(
        problems=problems,
        conditions=args.conditions,
        n=args.n,
        target_cpp=args.target_cpp,
        skip_validation=args.skip_validation,
        verbose=args.verbose,
        resume=args.resume,
        parallel_workers=args.workers,
        n_problems=args.n_problems,
    )


if __name__ == "__main__":
    main()
