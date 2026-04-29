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
import argparse, json, os, time
from datetime import datetime
from pathlib import Path
from typing import Optional

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

ALL_CONDITIONS = ["C1", "C2", "C3", "C4", "C5", "C6"]

# Known correct answers for pilot problems (AMC/AIME style — numeric)
KNOWN_ANSWERS: dict[str, str] = {
    "math_00014": "26",   # x(3x-7)=-3 → x=(7±√13)/6, m+n+p=7+13+6=26
    "math_00050": "8",    # 712n div by 18 → n=8
    "math_00121": "44",   # sec+tan=22/7 → csc+cot=29/15, m+n=44
    "math_00128": "48",   # (6!+7!)/5! = 5760/120 = 48
}

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
    known = KNOWN_ANSWERS.get(problem_id)
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


def run_c2(
    problem_id: str,
    problem: str,
    n: int = 2,
    target_cpp: list[str] = None,
    skip_validation: bool = False,
) -> dict:
    """C2: CIDI pipeline (6 modules) + standard simulation."""
    target = target_cpp or CPP_DEEP_TARGET
    t0 = time.time()
    cidi_result = split_cidi(
        problem_id, problem, target_cpp=target, n=n,
        skip_validation=skip_validation,
    )
    split_sec = round(time.time() - t0, 1)

    result = _run_with_annotator(
        "C2", problem_id, problem,
        cidi_result.split,
        lambda sr: simulate(sr, f"cpp_directed_{n}"),
        extra={
            "split": cidi_result.split.__dict__,
            "cidi": cidi_result.to_dict(),
            "timing": {**cidi_result.timing_sec, "split_sec": split_sec},
        },
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
    t0 = time.time()
    cidi_result = split_cidi(
        problem_id, problem, target_cpp=target, n=n,
        skip_validation=skip_validation,
    )
    split_sec = round(time.time() - t0, 1)

    result = _run_with_annotator(
        "C6", problem_id, problem,
        cidi_result.split,
        lambda sr: simulate(sr, f"peer_jigsaw_{n}", peer_aware=True),
        extra={
            "split": cidi_result.split.__dict__,
            "cidi": cidi_result.to_dict(),
            "timing": {**cidi_result.timing_sec, "split_sec": split_sec},
            "peer_aware": True,
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

def run_pilot(
    problems:        Optional[list[dict]] = None,
    conditions:      list[str] = None,
    n:               int = 2,
    target_cpp:      list[str] = None,
    skip_validation: bool = False,
    verbose:         bool = False,
) -> list[dict]:

    if problems is None:
        print("[INFO] Selecting pilot problems from corpus...")
        problems = select_pilot_problems(4, verbose=verbose)
        if not problems:
            raise RuntimeError(
                "No problems found in outputs/splits/. "
                "Run the main pipeline first to generate the corpus."
            )
        print(f"[INFO] Selected {len(problems)} problems: "
              f"{[p['problem_id'] for p in problems]}")

    if conditions is None:
        conditions = ALL_CONDITIONS

    target = target_cpp or CPP_DEEP_TARGET
    all_results = []
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")

    for prob in problems:
        pid     = prob["problem_id"]
        problem = prob.get("problem", "")

        for cond in conditions:
            print(f"\n[RUN] {pid} × {cond} ...", flush=True)
            t_start = time.time()
            try:
                result = CONDITION_RUNNERS[cond](pid, problem, n, target, skip_validation)
            except Exception as e:
                import traceback
                result = {"condition": cond, "problem_id": pid, "error": str(e)}
                print(f"  [ERROR] {e}")
                if verbose:
                    traceback.print_exc()

            result.setdefault("cdi_label", _cdi_label(result.get("cdi", 0)))
            all_results.append(result)

            elapsed = round(time.time() - t_start, 1)
            cdi     = result.get("cdi", "N/A")
            cqi     = result.get("cqi", "N/A")
            atc_cqi = result.get("atc_cqi", "N/A")
            label   = result.get("cdi_label", "")
            print(f"  CDI={cdi}  CQI={cqi}  ATC_CQI={atc_cqi}  Profile={label}  ({elapsed}s)")

            # Save individual result (scores + embedded conversation)
            fname = PILOT_DIR / f"{pid}_{cond}_{timestamp}.json"
            fname.write_text(
                json.dumps(result, ensure_ascii=False, indent=2, default=str)
            )

            # Save clean conversation archive (for later re-scoring without re-simulation)
            if "conversation" in result and "error" not in result:
                conv_fname = PILOT_CONV_DIR / f"{pid}_{cond}_{timestamp}.json"
                conv_data  = result["conversation"].copy()
                conv_data["pilot_timestamp"] = timestamp
                conv_data["split_overridden"] = result.get("split_overridden", False)
                conv_fname.write_text(
                    json.dumps(conv_data, ensure_ascii=False, indent=2, default=str)
                )

    # Save consolidated
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
    parser.add_argument("--conditions",  nargs="+", choices=ALL_CONDITIONS,
                        default=ALL_CONDITIONS)
    parser.add_argument("--n",           type=int, default=2)
    parser.add_argument("--target-cpp",  nargs="+", default=None,
                        help="CPP cells to target (default: CPP-DEEP A1-C2)")
    parser.add_argument("--skip-validation", action="store_true",
                        help="Skip M5 discriminator validation (useful before training)")
    parser.add_argument("--select-only", action="store_true",
                        help="Only print selected problems, do not run")
    parser.add_argument("--verbose",     action="store_true")
    args = parser.parse_args()

    problems = None
    if args.problems:
        existing = _load_existing_splits()
        problems = []
        for pid in args.problems:
            if pid in existing:
                problems.append({"problem_id": pid, **existing[pid]})
            else:
                print(f"[WARN] Problem '{pid}' not found in corpus — skipping")

    if args.select_only:
        probs = problems or select_pilot_problems(4, verbose=True)
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
    )


if __name__ == "__main__":
    main()
