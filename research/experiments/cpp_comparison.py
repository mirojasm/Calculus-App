"""
Pilot experiment runner: 4 problems × 5 CPP conditions.

Conditions:
  C1  Baseline         — existing splits & conversations (no re-generation)
  C2  CIDI-Directed    — split_cidi() (6-module pipeline) + standard simulation
  C3  Constitutional   — constitutional_split() (36-check critic) + standard simulation
  C4  Monitor          — standard split + simulate_with_monitor()
  C5  Integrated       — split_cidi() + simulate_with_monitor()

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
from research.splitting.cidi.pipeline import split_cidi, CPP_DEEP_TARGET

# ── paths ──────────────────────────────────────────────────────────────────────

OUT_DIR    = Path("outputs")
SPLITS_DIR = OUT_DIR / "splits"
CONV_DIR   = OUT_DIR / "conversations"
PILOT_DIR  = OUT_DIR / "pilot"
MODELS_DIR = OUT_DIR / "models"

PILOT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

ALL_CONDITIONS = ["C1", "C2", "C3", "C4", "C5"]


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


def _run_with_annotator(
    condition_name: str,
    problem_id: str,
    problem: str,
    split_result,
    sim_fn,
    extra: dict = None,
) -> dict:
    """Common wrapper: simulate + annotate + build result dict."""
    t_sim = time.time()
    conv = sim_fn(split_result)
    sim_sec = round(time.time() - t_sim, 1)

    t_ann = time.time()
    cpp = annotate(conv)
    ann_sec = round(time.time() - t_ann, 1)

    result = {
        "condition":     condition_name,
        "problem_id":    problem_id,
        "conversation":  conv.to_dict(),
        "cpp_vector":    cpp.cpp_vector,
        "cdi":           cpp.cdi,
        "cdi_label":     cpp.cdi_label,
        "cpp_rationale": cpp.rationale,
        "timing": {"sim_sec": sim_sec, "annot_sec": ann_sec},
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
) -> dict:
    """C4: Standard split + Szewkis monitor."""
    t0 = time.time()
    split_result = standard_split(problem_id, problem, n)
    split_sec = round(time.time() - t0, 1)

    result = _run_with_annotator(
        "C4", problem_id, problem,
        split_result,
        lambda sr: simulate_with_monitor(sr, f"monitored_jigsaw_{n}"),
        extra={
            "split": split_result.__dict__,
            "timing": {"split_sec": split_sec},
        },
    )
    # Count monitor interventions
    conv = result.get("conversation", {})
    n_int = sum(1 for t in conv.get("turns", []) if t.get("agent_id") == 0)
    result["n_monitor_interventions"] = n_int
    return result


def run_c5(
    problem_id: str,
    problem: str,
    n: int = 2,
    target_cpp: list[str] = None,
    skip_validation: bool = False,
) -> dict:
    """C5: CIDI split + monitor (fully integrated)."""
    target = target_cpp or CPP_DEEP_TARGET
    t0 = time.time()
    cidi_result = split_cidi(
        problem_id, problem, target_cpp=target, n=n,
        skip_validation=skip_validation,
    )
    split_sec = round(time.time() - t0, 1)

    result = _run_with_annotator(
        "C5", problem_id, problem,
        cidi_result.split,
        lambda sr: simulate_with_monitor(sr, f"integrated_{n}"),
        extra={
            "split": cidi_result.split.__dict__,
            "cidi": cidi_result.to_dict(),
            "timing": {**cidi_result.timing_sec, "split_sec": split_sec},
        },
    )
    conv = result.get("conversation", {})
    n_int = sum(1 for t in conv.get("turns", []) if t.get("agent_id") == 0)
    result["n_monitor_interventions"] = n_int
    return result


CONDITION_RUNNERS = {
    "C1": lambda pid, prob, n, t, sv: run_c1(pid, prob),
    "C2": lambda pid, prob, n, t, sv: run_c2(pid, prob, n, t, sv),
    "C3": lambda pid, prob, n, t, sv: run_c3(pid, prob, n),
    "C4": lambda pid, prob, n, t, sv: run_c4(pid, prob, n),
    "C5": lambda pid, prob, n, t, sv: run_c5(pid, prob, n, t, sv),
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
    print("\n" + "="*80)
    print("PILOT RESULTS SUMMARY")
    print("="*80)
    print(f"{'Problem':<22} {'Cond':<4} {'CDI':>6} {'Profile':<12} {'Turns':>6} {'H.err':>6}")
    print("-"*80)
    for r in results:
        if "error" in r:
            print(f"{r.get('problem_id','?'):<22} {r.get('condition','?'):<4}  ERROR: {r['error']}")
            continue
        conv   = r.get("conversation", {})
        turns  = conv.get("total_turns", "?")
        herr   = r.get("cidi", {}).get("targeting_error", "—")
        herr_s = f"{herr:.2f}" if isinstance(herr, float) else "—"
        print(
            f"{r['problem_id']:<22} {r['condition']:<4} "
            f"{r.get('cdi', 0):>6.3f} {r.get('cdi_label','?'):<12} "
            f"{turns:>6} {herr_s:>6}"
        )
    print("="*80)


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
            label   = result.get("cdi_label", "")
            print(f"  CDI={cdi}  Profile={label}  ({elapsed}s)")

            # Save individual result
            fname = PILOT_DIR / f"{pid}_{cond}_{timestamp}.json"
            fname.write_text(
                json.dumps(result, ensure_ascii=False, indent=2, default=str)
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
