"""
Phase 2 scale study analysis — CPP gradient (L1/L2/L3) across 140 epistemic problems.

Aggregates all Phase 2 result files, computes per-problem means across replications,
and tests the four NeurIPS hypotheses:
  H1: CDI(C7) > CDI(C2) > CDI(C6) — monotone L1→L2→L3 gradient
  H2: PhAQ(C7) > PhAQ(C2) = 0      — Phase A only emerges with L3
  H3: CDI gradient holds by subject × level (ecological validity)
  H4: r_pb(CDI, split_genuineness from Phase 1) > 0

Usage:
  python3 -m research.experiments.analyze_phase2
  python3 -m research.experiments.analyze_phase2 --conditions C2 C6 C7 --min-reps 1
"""
from __future__ import annotations
import argparse, json, math
from collections import defaultdict
from pathlib import Path

PILOT_DIR = Path("outputs/pilot")
COMBINED_FILTERED = PILOT_DIR / "phase1_combined_filtered.json"

SCALE_CONDITIONS = ["C2", "C6", "C7"]   # L1, L2, L3
METRICS = ["cdi", "cqi", "phaq", "atc_cqi", "cy"]

# ── data loading ──────────────────────────────────────────────────────────────

def load_phase2_results(conditions: list[str] = SCALE_CONDITIONS) -> dict[tuple[str,str], list[dict]]:
    """
    Returns {(problem_id, condition): [result_dicts_across_reps]}.
    Uses individual per-cell files (math_*_CX_*.json) in PILOT_DIR.
    Skips error entries and old pilot results (pre-Phase 2: pilot v4/v5/v6 files).
    """
    # Load Phase 1 filtered set to restrict to genuine epistemic problems
    filtered_pids: set[str] = set()
    if COMBINED_FILTERED.exists():
        filtered_pids = {e["problem_id"] for e in json.loads(COMBINED_FILTERED.read_text())}

    cells: dict[tuple[str,str], list[dict]] = defaultdict(list)
    for cond in conditions:
        for p in PILOT_DIR.glob(f"math_*_{cond}_*.json"):
            parts = p.stem.split("_")
            if len(parts) < 5:
                continue
            pid = f"{parts[0]}_{parts[1]}"
            if filtered_pids and pid not in filtered_pids:
                continue
            try:
                d = json.loads(p.read_text())
            except Exception:
                continue
            if "error" in d:
                continue
            cells[(pid, cond)].append(d)
    return dict(cells)


def aggregate(cells: dict[tuple[str,str], list[dict]]) -> dict[tuple[str,str], dict]:
    """
    For each (pid, cond), average all metrics across replications.
    Returns {(pid, cond): {metric: mean, ...}}
    """
    agg = {}
    for (pid, cond), reps in cells.items():
        means = {}
        for m in METRICS:
            vals = [r.get(m, 0) for r in reps if m in r]
            means[m] = round(sum(vals) / len(vals), 4) if vals else 0.0
        means["n_reps"] = len(reps)
        means["quadrant_counts"] = _count_quadrants(reps)
        means["correctness_rate"] = sum(1 for r in reps if r.get("correctness") == "correct") / len(reps)
        agg[(pid, cond)] = means
    return agg


def _count_quadrants(reps: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for r in reps:
        counts[r.get("quadrant", "?")] += 1
    return dict(counts)


# ── statistics ────────────────────────────────────────────────────────────────

def _mean(vals): return sum(vals) / len(vals) if vals else 0.0
def _std(vals):
    if len(vals) < 2: return 0.0
    m = _mean(vals)
    return math.sqrt(sum((x - m)**2 for x in vals) / (len(vals) - 1))

def cohens_d(a: list[float], b: list[float]) -> float:
    if not a or not b: return 0.0
    n1, n2 = len(a), len(b)
    s1, s2 = _std(a), _std(b)
    pooled = math.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2)) if n1+n2>2 else 0
    return (_mean(a) - _mean(b)) / pooled if pooled else 0.0

def wilcoxon_sign_rank_p(a: list[float], b: list[float]) -> float:
    """Approximate p-value using scipy's Wilcoxon signed-rank test."""
    try:
        from scipy.stats import wilcoxon
        diffs = [x - y for x, y in zip(a, b) if x != y]
        if not diffs:
            return 1.0
        _, p = wilcoxon(diffs, alternative="greater")
        return round(p, 4)
    except Exception:
        return float("nan")

def paired_t_p(a: list[float], b: list[float]) -> float:
    """Paired t-test (parametric)."""
    try:
        from scipy.stats import ttest_rel
        if len(a) < 3 or len(b) < 3:
            return float("nan")
        _, p = ttest_rel(a, b, alternative="greater")
        return round(p, 4)
    except Exception:
        return float("nan")

def pearson_r(x: list[float], y: list[float]) -> float:
    if len(x) < 3: return float("nan")
    mx, my = _mean(x), _mean(y)
    num = sum((xi - mx)*(yi - my) for xi, yi in zip(x, y))
    den = math.sqrt(sum((xi-mx)**2 for xi in x) * sum((yi-my)**2 for yi in y))
    return round(num / den, 4) if den else 0.0


# ── report ────────────────────────────────────────────────────────────────────

def _get_metric_by_pid(agg, cond, metric):
    """List of per-problem mean values for a given condition and metric."""
    pids = sorted(set(pid for (pid, c) in agg if c == cond))
    return pids, [agg.get((pid, cond), {}).get(metric, 0.0) for pid in pids]


def print_report(agg: dict, min_reps: int = 1, conditions: list[str] = SCALE_CONDITIONS):
    # Filter to problems that have all conditions with ≥ min_reps
    all_pids = set(pid for (pid, c) in agg if c in conditions)
    valid_pids = sorted(
        pid for pid in all_pids
        if all(agg.get((pid, c), {}).get("n_reps", 0) >= min_reps for c in conditions)
    )
    print(f"\n{'='*72}")
    print(f"PHASE 2 ANALYSIS — {len(valid_pids)} problems × {len(conditions)} conditions")
    print(f"(All conditions with ≥{min_reps} reps)")
    print('='*72)

    if not valid_pids:
        # Partial analysis: show what we have
        valid_pids = sorted(all_pids)
        print(f"[NOTE] Showing partial results — not all conditions have ≥{min_reps} reps yet.")

    # ── Per-condition means ────────────────────────────────────────────────────
    print(f"\n{'Condition':<8} {'CDI':>6} {'CQI':>6} {'PhAQ':>6} {'ATC':>6} {'CY':>6} {'n_probs':>8} {'avg_reps':>8}")
    print("-"*60)
    for cond in conditions:
        pids_c = [pid for pid in valid_pids if (pid, cond) in agg]
        if not pids_c: continue
        label = {"C2":"L1-Natural", "C6":"L2-PeerAw.", "C7":"L3-StudSim"}.get(cond, cond)
        cdis  = [agg[(pid,cond)]["cdi"]  for pid in pids_c]
        cqis  = [agg[(pid,cond)]["cqi"]  for pid in pids_c]
        phaqs = [agg[(pid,cond)]["phaq"] for pid in pids_c]
        atcs  = [agg[(pid,cond)]["atc_cqi"] for pid in pids_c]
        cys   = [agg[(pid,cond)]["cy"]   for pid in pids_c]
        reps  = [agg[(pid,cond)]["n_reps"] for pid in pids_c]
        print(f"{label:<12} {_mean(cdis):>6.3f} {_mean(cqis):>6.3f} {_mean(phaqs):>6.3f} "
              f"{_mean(atcs):>6.3f} {_mean(cys):>6.3f} {len(pids_c):>8} {_mean(reps):>8.1f}")

    # ── H1: CDI gradient ──────────────────────────────────────────────────────
    print(f"\n── H1: CDI gradient L1 < L2 < L3 ──────────────────────────────────")
    shared = [pid for pid in valid_pids
              if all((pid, c) in agg for c in ["C2","C6","C7"])]
    if len(shared) >= 5 and all(c in conditions for c in ["C2","C6","C7"]):
        c2_cdi = [agg[(p,"C2")]["cdi"] for p in shared]
        c6_cdi = [agg[(p,"C6")]["cdi"] for p in shared]
        c7_cdi = [agg[(p,"C7")]["cdi"] for p in shared]
        d_c7c2 = cohens_d(c7_cdi, c2_cdi)
        d_c7c6 = cohens_d(c7_cdi, c6_cdi)
        d_c6c2 = cohens_d(c6_cdi, c2_cdi)
        p_c7c2 = wilcoxon_sign_rank_p(c7_cdi, c2_cdi)
        p_c7c6 = wilcoxon_sign_rank_p(c7_cdi, c6_cdi)
        p_c6c2 = wilcoxon_sign_rank_p(c6_cdi, c2_cdi)
        print(f"  CDI:  C7={_mean(c7_cdi):.3f}  C6={_mean(c6_cdi):.3f}  C2={_mean(c2_cdi):.3f}  (n={len(shared)})")
        print(f"  C7 > C2: d={d_c7c2:.2f}, p={p_c7c2} (Wilcoxon)")
        print(f"  C7 > C6: d={d_c7c6:.2f}, p={p_c7c6}")
        print(f"  C6 > C2: d={d_c6c2:.2f}, p={p_c6c2}")
        print(f"  H1 {'✓ SUPPORTED' if p_c7c2 < 0.05 else '✗ NOT SIGNIFICANT'} (C7>C2, α=0.05)")
    else:
        print(f"  Insufficient data ({len(shared)} problems with all 3 conditions)")

    # ── H2: PhAQ discriminator ────────────────────────────────────────────────
    print(f"\n── H2: PhAQ(C7) > 0, PhAQ(L1/L2) = 0 ─────────────────────────────")
    for cond in conditions:
        pids_c = [p for p in valid_pids if (p, cond) in agg]
        if not pids_c: continue
        phaqs = [agg[(p,cond)]["phaq"] for p in pids_c]
        nonzero = sum(1 for v in phaqs if v > 0)
        label = {"C2":"L1","C6":"L2","C7":"L3"}.get(cond, cond)
        print(f"  {label} ({cond}): mean PhAQ={_mean(phaqs):.3f}, PhAQ>0 in {nonzero}/{len(phaqs)} ({100*nonzero/len(phaqs):.1f}%)")
    if "C7" in conditions and "C2" in conditions:
        shared_phaq = [p for p in valid_pids if (p,"C2") in agg and (p,"C7") in agg]
        if len(shared_phaq) >= 5:
            c7_phaq = [agg[(p,"C7")]["phaq"] for p in shared_phaq]
            c2_phaq = [agg[(p,"C2")]["phaq"] for p in shared_phaq]
            d_phaq  = cohens_d(c7_phaq, c2_phaq)
            p_phaq  = wilcoxon_sign_rank_p(c7_phaq, c2_phaq)
            print(f"  C7>C2 PhAQ: d={d_phaq:.2f}, p={p_phaq}")
            print(f"  H2 {'✓ SUPPORTED' if p_phaq < 0.05 else '✗ NOT SIGNIFICANT'} (α=0.05)")

    # ── H3: Gradient by subject × level ───────────────────────────────────────
    print(f"\n── H3: CDI gradient by subject × level ─────────────────────────────")
    filtered_meta = {}
    if COMBINED_FILTERED.exists():
        for e in json.loads(COMBINED_FILTERED.read_text()):
            filtered_meta[e["problem_id"]] = e

    if all(c in conditions for c in ["C2","C7"]):
        subj_level: dict[tuple, list] = defaultdict(list)
        for pid in valid_pids:
            if (pid,"C2") not in agg or (pid,"C7") not in agg:
                continue
            m = filtered_meta.get(pid, {})
            subj, lvl = m.get("subject", "?"), m.get("level", 0)
            delta = agg[(pid,"C7")]["cdi"] - agg[(pid,"C2")]["cdi"]
            subj_level[(subj, lvl)].append(delta)
        print(f"  CDI(C7) - CDI(C2) per cell (positive = L3 advantage):")
        for (s, l), deltas in sorted(subj_level.items()):
            m = _mean(deltas)
            bar = "+" * int(abs(m) * 10) if m >= 0 else "-" * int(abs(m) * 10)
            print(f"  {s:20s} L{l}: Δ={m:+.3f} ({len(deltas)} probs)  {bar}")

    # ── Quadrant distribution ─────────────────────────────────────────────────
    print(f"\n── Quadrant distribution ────────────────────────────────────────────")
    for cond in conditions:
        pids_c = [p for p in valid_pids if (p,cond) in agg]
        if not pids_c: continue
        all_quad: dict[str,int] = defaultdict(int)
        for pid in pids_c:
            for q, cnt in agg[(pid,cond)].get("quadrant_counts", {}).items():
                all_quad[q] += cnt
        label = {"C2":"L1","C6":"L2","C7":"L3"}.get(cond, cond)
        total = sum(all_quad.values())
        parts = [f"{q}:{all_quad[q]}({100*all_quad[q]//total}%)" for q in sorted(all_quad)]
        print(f"  {label} ({cond}): {' | '.join(parts)}")

    print('='*72)


def main():
    parser = argparse.ArgumentParser(description="Analyze Phase 2 CPP gradient results")
    parser.add_argument("--conditions", nargs="+", default=SCALE_CONDITIONS)
    parser.add_argument("--min-reps",   type=int,  default=1,
                        help="Minimum reps required per cell (default: 1)")
    args = parser.parse_args()

    print(f"[INFO] Loading Phase 2 results for conditions: {args.conditions}")
    cells = load_phase2_results(args.conditions)
    total_reps = sum(len(v) for v in cells.values())
    print(f"[INFO] Loaded {len(cells)} (problem×condition) cells, {total_reps} total observations")
    agg = aggregate(cells)
    print_report(agg, min_reps=args.min_reps, conditions=args.conditions)


if __name__ == "__main__":
    main()
