"""
Pilot experiment runner: 4 problems × 5 CPP conditions.

Conditions:
  C1  Baseline         — existing splits & conversations (no re-generation)
  C2  CPP-Directed     — split_cpp_targeted() + standard simulation
  C3  Constitutional   — constitutional_split() + standard simulation
  C4  Monitor          — standard split + simulate_with_monitor()
  C5  Integrated       — constitutional_split() + simulate_with_monitor()

Usage:
  # Select 4 pilot problems automatically and run all 5 conditions
  python3 -m research.experiments.cpp_comparison

  # Run specific problems
  python3 -m research.experiments.cpp_comparison --problems algebra_L2_001 geometry_L3_002

  # Run specific conditions only
  python3 -m research.experiments.cpp_comparison --conditions C2 C3 C5
"""
import argparse, json, os, time
from datetime import datetime
from pathlib import Path
from typing import Optional

from research.config import CFG
from research.splitting.splitter import split as standard_split, split_cpp_targeted
from research.splitting.constitutional import constitutional_split
from research.simulation.simulator import simulate, simulate_with_monitor
from research.scoring.cpp_annotator import annotate


# ── paths ──────────────────────────────────────────────────────────────────────

OUT_DIR     = Path("outputs")
SPLITS_DIR  = OUT_DIR / "splits"
CONV_DIR    = OUT_DIR / "conversations"
PILOT_DIR   = OUT_DIR / "pilot"

PILOT_DIR.mkdir(parents=True, exist_ok=True)


# ── problem selection ──────────────────────────────────────────────────────────

def _load_existing_splits() -> dict:
    """Return {problem_id: split_dict} for all saved splits."""
    splits = {}
    for p in SPLITS_DIR.glob("*.json"):
        try:
            data = json.loads(p.read_text())
            splits[data["problem_id"]] = data
        except Exception:
            pass
    return splits


def _load_existing_conversations() -> dict:
    """Return {(problem_id, condition): conv_dict}."""
    convs = {}
    for p in CONV_DIR.glob("*.json"):
        try:
            data = json.loads(p.read_text())
            convs[(data["problem_id"], data["condition"])] = data
        except Exception:
            pass
    return convs


def select_pilot_problems(n_problems: int = 4) -> list[dict]:
    """
    Select pilot problems from the existing corpus:
    - P1: algebra L2, fewest turns in jigsaw_2 (most trivial split)
    - P2: geometry L3, lowest PISA global in jigsaw_2
    - P3: number_theory L4, lowest PISA global
    - P4: counting_and_probability L5, lowest PISA global
    """
    splits = _load_existing_splits()
    convs  = _load_existing_conversations()

    criteria = [
        {"subject": "algebra",                  "level": "2", "sort": "turns_asc"},
        {"subject": "geometry",                 "level": "3", "sort": "turns_asc"},
        {"subject": "number_theory",            "level": "4", "sort": "turns_asc"},
        {"subject": "counting_and_probability", "level": "5", "sort": "turns_asc"},
    ]

    selected = []
    used_ids = set()

    for crit in criteria[:n_problems]:
        candidates = []
        for pid, split_data in splits.items():
            if pid in used_ids:
                continue
            subject = split_data.get("subject", "")
            level   = str(split_data.get("level", ""))
            if subject != crit["subject"] or level != crit["level"]:
                continue
            conv = convs.get((pid, "jigsaw_2"), {})
            turns = conv.get("total_turns", 999)
            candidates.append((pid, split_data, turns))

        if candidates:
            candidates.sort(key=lambda x: x[2])
            pid, split_data, _ = candidates[0]
            selected.append({"problem_id": pid, **split_data})
            used_ids.add(pid)

    return selected


# ── condition runners ──────────────────────────────────────────────────────────

def run_c1(problem_id: str, problem: str) -> dict:
    """C1: load existing outputs, no re-generation."""
    splits = _load_existing_splits()
    convs  = _load_existing_conversations()

    split_data = splits.get(problem_id)
    conv_data  = convs.get((problem_id, "jigsaw_2"))

    if not split_data or not conv_data:
        return {"error": f"No existing data for {problem_id}"}

    return {
        "condition": "C1",
        "problem_id": problem_id,
        "split": split_data,
        "conversation": conv_data,
        "source": "existing",
    }


def _run_condition(
    condition_name: str,
    problem_id: str,
    problem: str,
    split_fn,
    simulate_fn,
    n: int = 2,
) -> dict:
    t0 = time.time()
    split_result = split_fn(problem_id, problem, n)
    t_split = time.time() - t0

    t1 = time.time()
    conv = simulate_fn(split_result)
    t_sim = time.time() - t1

    t2 = time.time()
    cpp = annotate(conv)
    t_ann = time.time() - t2

    return {
        "condition":      condition_name,
        "problem_id":     problem_id,
        "split":          split_result.__dict__ if hasattr(split_result, '__dict__') else {},
        "conversation":   conv.to_dict(),
        "cpp_vector":     cpp.cpp_vector,
        "cdi":            cpp.cdi,
        "cdi_label":      cpp.cdi_label,
        "cpp_rationale":  cpp.rationale,
        "timing": {
            "split_sec":  round(t_split, 1),
            "sim_sec":    round(t_sim, 1),
            "annot_sec":  round(t_ann, 1),
        },
    }


def run_c2(problem_id: str, problem: str, n: int = 2) -> dict:
    def _sim(sr):
        return simulate(sr, f"cpp_directed_{n}")
    return _run_condition("C2", problem_id, problem,
                          lambda pid, p, _n: split_cpp_targeted(pid, p, _n),
                          _sim, n)


def run_c3(problem_id: str, problem: str, n: int = 2) -> dict:
    t0 = time.time()
    result = constitutional_split(problem_id, problem, n)
    t_split = time.time() - t0

    t1 = time.time()
    conv = simulate(result.split, f"jigsaw_{n}")
    t_sim = time.time() - t1

    t2 = time.time()
    cpp = annotate(conv)
    t_ann = time.time() - t2

    return {
        "condition":         "C3",
        "problem_id":        problem_id,
        "split":             result.split.__dict__ if hasattr(result.split, '__dict__') else {},
        "conversation":      conv.to_dict(),
        "cpp_vector":        cpp.cpp_vector,
        "cdi":               cpp.cdi,
        "cdi_label":         cpp.cdi_label,
        "cpp_rationale":     cpp.rationale,
        "constitutional": {
            "final_sqs":     result.final_sqs,
            "iterations":    result.iterations,
            "approved":      result.approved,
            "critique_history": result.critique_history,
            "improvements":  result.improvements_history,
        },
        "timing": {
            "split_sec":  round(t_split, 1),
            "sim_sec":    round(t_sim, 1),
            "annot_sec":  round(t_ann, 1),
        },
    }


def run_c4(problem_id: str, problem: str, n: int = 2) -> dict:
    def _split_fn(pid, p, _n):
        return standard_split(pid, p, _n)
    def _sim(sr):
        return simulate_with_monitor(sr, f"monitored_jigsaw_{n}")
    return _run_condition("C4", problem_id, problem, _split_fn, _sim, n)


def run_c5(problem_id: str, problem: str, n: int = 2) -> dict:
    t0 = time.time()
    result = constitutional_split(problem_id, problem, n)
    t_split = time.time() - t0

    t1 = time.time()
    conv = simulate_with_monitor(result.split, f"integrated_{n}")
    t_sim = time.time() - t1

    t2 = time.time()
    cpp = annotate(conv)
    t_ann = time.time() - t2

    n_interventions = sum(1 for t in conv.turns if t.agent_id == 0)

    return {
        "condition":         "C5",
        "problem_id":        problem_id,
        "split":             result.split.__dict__ if hasattr(result.split, '__dict__') else {},
        "conversation":      conv.to_dict(),
        "cpp_vector":        cpp.cpp_vector,
        "cdi":               cpp.cdi,
        "cdi_label":         cpp.cdi_label,
        "cpp_rationale":     cpp.rationale,
        "constitutional": {
            "final_sqs":     result.final_sqs,
            "iterations":    result.iterations,
            "approved":      result.approved,
        },
        "n_monitor_interventions": n_interventions,
        "timing": {
            "split_sec":  round(t_split, 1),
            "sim_sec":    round(t_sim, 1),
            "annot_sec":  round(t_ann, 1),
        },
    }


# ── summary table ──────────────────────────────────────────────────────────────

def _print_summary(results: list[dict]) -> None:
    print("\n" + "="*80)
    print("PILOT RESULTS SUMMARY")
    print("="*80)
    header = f"{'Problem':<20} {'Cond':<4} {'CDI':>6} {'Profile':<12} {'Turns':>6}"
    print(header)
    print("-"*80)
    for r in results:
        if "error" in r:
            print(f"{r.get('problem_id','?'):<20} {r.get('condition','?'):<4}  ERROR: {r['error']}")
            continue
        conv = r.get("conversation", {})
        turns = conv.get("total_turns", "?")
        print(
            f"{r['problem_id']:<20} {r['condition']:<4} "
            f"{r.get('cdi', 0):>6.3f} {r.get('cdi_label','?'):<12} {turns:>6}"
        )
    print("="*80)


# ── main ───────────────────────────────────────────────────────────────────────

CONDITION_RUNNERS = {
    "C1": run_c1,
    "C2": run_c2,
    "C3": run_c3,
    "C4": run_c4,
    "C5": run_c5,
}

ALL_CONDITIONS = ["C1", "C2", "C3", "C4", "C5"]


def run_pilot(
    problems:   Optional[list[dict]] = None,
    conditions: list[str] = None,
    n: int = 2,
) -> list[dict]:
    if problems is None:
        print("[INFO] Selecting 4 pilot problems from corpus...")
        problems = select_pilot_problems(4)
        if not problems:
            raise RuntimeError("No problems found in outputs/splits/. Run the main pipeline first.")
        print(f"[INFO] Selected: {[p['problem_id'] for p in problems]}")

    if conditions is None:
        conditions = ALL_CONDITIONS

    all_results = []
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")

    for prob in problems:
        pid     = prob["problem_id"]
        problem = prob.get("problem", "")

        for cond in conditions:
            print(f"\n[RUN] {pid} × {cond} ...", flush=True)
            try:
                if cond == "C1":
                    result = CONDITION_RUNNERS[cond](pid, problem)
                else:
                    result = CONDITION_RUNNERS[cond](pid, problem, n)
            except Exception as e:
                result = {"condition": cond, "problem_id": pid, "error": str(e)}
                print(f"  [ERROR] {e}")

            all_results.append(result)

            # Save individual result
            fname = PILOT_DIR / f"{pid}_{cond}_{timestamp}.json"
            fname.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str))
            cdi = result.get("cdi", "N/A")
            label = result.get("cdi_label", "")
            print(f"  CDI={cdi}  Profile={label}  saved → {fname.name}")

    # Save consolidated results
    consolidated = PILOT_DIR / f"pilot_results_{timestamp}.json"
    consolidated.write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2, default=str)
    )
    print(f"\n[DONE] Consolidated results → {consolidated}")

    _print_summary(all_results)
    return all_results


def main():
    parser = argparse.ArgumentParser(description="CPP comparison pilot experiment")
    parser.add_argument("--problems", nargs="+", help="Problem IDs to run (default: auto-select)")
    parser.add_argument("--conditions", nargs="+", choices=ALL_CONDITIONS,
                        default=ALL_CONDITIONS, help="Conditions to run")
    parser.add_argument("--n", type=int, default=2, help="Number of agents per jigsaw (default: 2)")
    args = parser.parse_args()

    problems = None
    if args.problems:
        existing = _load_existing_splits()
        problems = []
        for pid in args.problems:
            if pid in existing:
                problems.append({"problem_id": pid, **existing[pid]})
            else:
                print(f"[WARN] Problem '{pid}' not found in splits — skipping")

    run_pilot(problems=problems, conditions=args.conditions, n=args.n)


if __name__ == "__main__":
    main()
