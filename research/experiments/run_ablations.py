"""
Ablation experiment: CFULL and CEXP conditions on the 140 phase2 problems.

CFULL: C7 active-learner framing + full problem context (no CIDI split).
  Tests: does high CDI under C7 require structural interdependence,
         or does the active-learner register alone produce collaborative discourse?

CEXP: CIDI split + expert-solver framing (no learner stance).
  Tests: does any strong role instruction improve CDI,
         or is the active-learner communicative stance specifically required?

Each condition runs 3 replications per problem.
Results are saved to outputs/ablations/{condition}/

Usage:
  cd /Users/mir85108/Documents/GitHub/Calculus-App
  python3 -m research.experiments.run_ablations
  python3 -m research.experiments.run_ablations --conditions CFULL
  python3 -m research.experiments.run_ablations --workers 6 --reps 2
"""
from __future__ import annotations
import argparse, json, os, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

from research.config import CFG
from research.splitting.splitter import SplitResult, Packet
from research.simulation.simulator import simulate
from research.scoring.cpp_annotator import annotate
from research.scoring.atc21s import annotate_conversation as annotate_atc21s

# ── paths ─────────────────────────────────────────────────────────────────────

PILOT_DIR   = Path("outputs/pilot")
ABL_DIR     = Path("outputs/ablations")
ABL_DIR.mkdir(parents=True, exist_ok=True)

PHASE2_PROBLEMS = PILOT_DIR / "phase2_problems.json"

# ── helpers ───────────────────────────────────────────────────────────────────

def _load_json(path: Path):
    with open(path) as f:
        return json.load(f)

def _save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _parse_packet(raw) -> Packet:
    """Parse a packet from either a dict or a Python repr string."""
    if isinstance(raw, dict):
        info = raw["information"]
        if isinstance(info, dict):
            info = json.dumps(info, ensure_ascii=False)
        return Packet(
            agent_id=raw["agent_id"],
            information=info,
            role_name=raw.get("role_name", ""),
            role_description=raw.get("role_description", ""),
        )
    # Repr string: eval with Packet in scope
    p = eval(raw, {"Packet": Packet})
    info = p.information
    if isinstance(info, dict):
        info = json.dumps(info, ensure_ascii=False)
    return Packet(agent_id=p.agent_id, information=info,
                  role_name=p.role_name, role_description=p.role_description)


def _load_split(problem_id: str) -> Optional[SplitResult]:
    """Load the cached CIDI split used in phase2 (from C7 pilot files)."""
    for p in sorted(PILOT_DIR.glob(f"{problem_id}_C7_*.json")):
        try:
            data = _load_json(p)
            split_data = data.get("split") or data.get("cidi", {}).get("split")
            if not split_data:
                continue
            packets_raw = split_data.get("packets", [])
            if not packets_raw:
                continue
            packets = [_parse_packet(pk) for pk in packets_raw]
            return SplitResult(
                problem_id=problem_id,
                problem=data.get("problem", ""),
                n=len(packets),
                pattern=split_data.get("pattern", ""),
                shared_context=split_data.get("shared_context", ""),
                packets=packets,
                valid=True,
            )
        except Exception:
            continue
    return None


def _get_quadrant(cdi: float, correctness: str) -> str:
    high = cdi >= 0.5
    correct = correctness in ("correct", "partial")
    if high and correct:     return "COUPLING"
    if high and not correct: return "PROD_FAIL"
    if not high and correct: return "TRIVIAL"
    return "COLLAPSE"


def _check_correctness(final_answer: Optional[str], ground_truth: Optional[str]) -> str:
    if not ground_truth:
        return "unknown"
    if not final_answer or not final_answer.strip():
        return "incomplete"
    import re
    nums = re.findall(r"-?\d+(?:\.\d+)?", str(final_answer))
    return "correct" if str(ground_truth).strip() in nums else "incorrect"


# ── per-problem runner ────────────────────────────────────────────────────────

def run_one(
    problem_id: str,
    problem: str,
    ground_truth: str,
    condition: str,
    rep: int,
    split_result: Optional[SplitResult],
) -> dict:
    """Run one conversation for given condition and rep. Returns result dict."""
    out_path = ABL_DIR / condition / f"{problem_id}_{condition}_rep{rep}.json"
    if out_path.exists():
        return _load_json(out_path)

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")

    # For CFULL we can build a minimal SplitResult carrying only the full problem
    if condition == "CFULL":
        if split_result is None:
            return {"error": "no_split", "problem_id": problem_id, "condition": condition}
        # Build a dummy split that exposes the full problem through .problem
        sr = SplitResult(
            problem_id=problem_id,
            problem=problem,
            n=split_result.n,
            pattern="cfull",
            shared_context=split_result.shared_context,
            packets=split_result.packets,   # packets exist but are not used by cfull prompt
            valid=True,
        )
        sim_cond = f"cfull_{split_result.n}"
    elif condition == "CEXP":
        if split_result is None:
            return {"error": "no_split", "problem_id": problem_id, "condition": condition}
        sr = split_result
        sim_cond = f"cexp_{split_result.n}"
    else:
        raise ValueError(f"Unknown condition: {condition}")

    t0 = time.time()
    conv = simulate(sr, sim_cond)
    sim_sec = round(time.time() - t0, 1)

    t1 = time.time()
    cpp = annotate(conv)
    atc = annotate_atc21s(conv)
    ann_sec = round(time.time() - t1, 1)

    correctness = _check_correctness(conv.final_answer, ground_truth)

    result = {
        "problem_id":    problem_id,
        "condition":     condition,
        "rep":           rep,
        "timestamp":     ts,
        "cdi":           cpp.cdi,
        "cqi":           cpp.cqi,
        "phaq":          cpp.phaq,
        "cdi_label":     cpp.cdi_label,
        "atc_cqi":       atc.atc_cqi,
        "correctness":   correctness,
        "quadrant":      _get_quadrant(cpp.cdi, correctness),
        "final_answer":  conv.final_answer,
        "ground_truth":  ground_truth,
        "total_turns":   conv.total_turns,
        "consensus":     conv.consensus,
        "cpp_vector":    cpp.cpp_vector,
        "quality_scores": cpp.quality_scores,
        "timing":        {"sim_sec": sim_sec, "ann_sec": ann_sec},
    }
    _save_json(out_path, result)
    return result


# ── aggregation ───────────────────────────────────────────────────────────────

def summarise(results: list[dict], condition: str) -> dict:
    """Compute per-problem mean CDI then aggregate statistics."""
    import statistics, math

    # Group by problem
    by_prob: dict[str, list] = {}
    for r in results:
        if "error" in r:
            continue
        by_prob.setdefault(r["problem_id"], []).append(r)

    prob_means: list[float] = []
    prob_cqi:   list[float] = []
    prob_phaq:  list[float] = []
    quadrant_counts = {"COUPLING": 0, "PROD_FAIL": 0, "TRIVIAL": 0, "COLLAPSE": 0}
    n_convs = 0

    for pid, reps in by_prob.items():
        cdis = [r["cdi"] for r in reps]
        prob_means.append(statistics.mean(cdis))
        prob_cqi.append(statistics.mean(r["cqi"] for r in reps))
        prob_phaq.append(statistics.mean(r["phaq"] for r in reps))
        for r in reps:
            quadrant_counts[r["quadrant"]] = quadrant_counts.get(r["quadrant"], 0) + 1
            n_convs += 1

    n_probs = len(prob_means)
    if n_probs == 0:
        return {"condition": condition, "n_problems": 0, "n_conversations": 0}

    mean_cdi = statistics.mean(prob_means)
    std_cdi  = statistics.stdev(prob_means) if n_probs > 1 else 0.0
    se_cdi   = std_cdi / math.sqrt(n_probs)

    total_q = n_convs
    quad_pct = {k: round(100 * v / total_q, 1) for k, v in quadrant_counts.items()}

    return {
        "condition":       condition,
        "n_problems":      n_probs,
        "n_conversations": n_convs,
        "mean_cdi":        round(mean_cdi, 3),
        "std_cdi":         round(std_cdi, 3),
        "se_cdi":          round(se_cdi, 3),
        "mean_cqi":        round(statistics.mean(prob_cqi), 3),
        "mean_phaq":       round(statistics.mean(prob_phaq), 3),
        "quadrant_pct":    quad_pct,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", nargs="+", default=["CFULL", "CEXP"],
                        choices=["CFULL", "CEXP"])
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--reps",    type=int, default=3)
    parser.add_argument("--limit",   type=int, default=None,
                        help="Limit to first N problems (for testing)")
    args = parser.parse_args()

    problems = _load_json(PHASE2_PROBLEMS)
    if args.limit:
        problems = problems[:args.limit]
    print(f"Problems: {len(problems)}, Conditions: {args.conditions}, "
          f"Reps: {args.reps}, Workers: {args.workers}")

    # Pre-load splits (cached from phase2 C7 runs)
    splits: dict[str, SplitResult] = {}
    for prob in problems:
        pid = prob["problem_id"]
        sr = _load_split(pid)
        if sr:
            splits[pid] = sr
        else:
            print(f"  [WARN] no split found for {pid}")
    print(f"Splits loaded: {len(splits)}/{len(problems)}")

    all_tasks = []
    for prob in problems:
        pid = prob["problem_id"]
        for cond in args.conditions:
            for rep in range(1, args.reps + 1):
                out_path = ABL_DIR / cond / f"{pid}_{cond}_rep{rep}.json"
                if out_path.exists():
                    continue
                all_tasks.append((prob, cond, rep))

    print(f"Tasks to run: {len(all_tasks)} "
          f"(skipping {len(problems)*len(args.conditions)*args.reps - len(all_tasks)} cached)")

    all_results: dict[str, list] = {c: [] for c in args.conditions}

    # Load existing cached results
    for prob in problems:
        pid = prob["problem_id"]
        for cond in args.conditions:
            for rep in range(1, args.reps + 1):
                out_path = ABL_DIR / cond / f"{pid}_{cond}_rep{rep}.json"
                if out_path.exists():
                    all_results[cond].append(_load_json(out_path))

    done = failed = 0
    t_start = time.time()

    def _run(task):
        prob, cond, rep = task
        pid = prob["problem_id"]
        return run_one(
            problem_id=pid,
            problem=prob["problem"],
            ground_truth=str(prob.get("answer", "")),
            condition=cond,
            rep=rep,
            split_result=splits.get(pid),
        )

    with ThreadPoolExecutor(max_workers=args.workers) as exe:
        futures = {exe.submit(_run, t): t for t in all_tasks}
        for fut in as_completed(futures):
            prob, cond, rep = futures[fut]
            try:
                r = fut.result()
                all_results[cond].append(r)
                done += 1
                if done % 20 == 0 or done == len(all_tasks):
                    elapsed = (time.time() - t_start) / 60
                    print(f"  {done}/{len(all_tasks)} done  ({elapsed:.1f} min)")
            except Exception as e:
                failed += 1
                print(f"  ERROR {prob['problem_id']} {cond} rep{rep}: {e}")

    print(f"\nDone: {done}, Failed: {failed}")
    print(f"Total time: {(time.time() - t_start)/60:.1f} min")

    # Summarise and print
    summaries = {}
    for cond in args.conditions:
        s = summarise(all_results[cond], cond)
        summaries[cond] = s
        print(f"\n── {cond} ──")
        print(f"  n_problems:  {s.get('n_problems', 0)}")
        print(f"  n_convs:     {s.get('n_conversations', 0)}")
        print(f"  mean CDI:    {s.get('mean_cdi', '?')} ± {s.get('se_cdi', '?')}")
        print(f"  mean CQI:    {s.get('mean_cqi', '?')}")
        print(f"  mean PhAQ:   {s.get('mean_phaq', '?')}")
        print(f"  quadrants:   {s.get('quadrant_pct', {})}")

    # Save combined summary
    summary_path = ABL_DIR / "ablations_summary.json"
    _save_json(summary_path, summaries)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
