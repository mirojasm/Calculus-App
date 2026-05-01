"""
Partial Information Decomposition (PID) of CIDI information splits.

Theoretical contribution: formalizes WHY the CIDI architecture induces collaboration.

Using AEC v-values as binary information proxies, each problem's outcome is
decomposed into 4 PID atoms (Williams & Beer 2010, I_min axiom):

  Rdn(A,B;Y)  = min(v_A, v_B)            redundant  — both agents had it
  Unique_A    = v_A  - Rdn               unique A   — only agent A had it
  Unique_B    = v_B  - Rdn               unique B   — only agent B had it
  Syn(A,B;Y)  = v_AB - v_A - v_B + Rdn  synergistic — emerges only jointly

Identity: Rdn + Unique_A + Unique_B + Syn = v_AB  (total collaborative info)

Key theoretical connections
---------------------------
  CS (AEC Collaborative Surplus)  ≡  Syn  when Rdn = 0
  EN (AEC Epistemic Necessity)    ≡  Syn > 0  [for the Rdn=0 case]
  AEC balance (EB)                ~ 1 - |Unique_A - Unique_B| / (Unique_A + Unique_B + ε)

Problem taxonomy (6 types)
--------------------------
  PURE_SYN    : v_A=0, v_B=0, v_AB>0  → pure synergy, strongest EN
  UNIQUE_A    : v_A>0, v_B=0, v_AB>0  → agent A holds the key
  UNIQUE_B    : v_A=0, v_B>0, v_AB>0  → agent B holds the key
  REDUNDANT   : v_A>0, v_B>0, v_AB>0  → collaboration unnecessary
  INTERFERENCE: v_AB < max(v_A,v_B)   → collaboration hurt (negative Syn)
  COLLAPSE    : v_AB=0                 → neither solo nor joint succeeded

Implication for CIDI design
----------------------------
Good splits maximize (Syn + Unique_A + Unique_B) and minimize Rdn.
Naive splits (divide the problem arbitrarily) tend toward REDUNDANT or UNIQUE_A.
CIDI splits (epistemic design) tend toward PURE_SYN or balanced UNIQUE_A + UNIQUE_B.

Usage
-----
  python -m research.experiments.compute_pid
  python -m research.experiments.compute_pid --plot   (requires matplotlib)
"""
from __future__ import annotations
import argparse, json
from collections import Counter, defaultdict
from pathlib import Path

AEC_RESULTS = Path("outputs/aec/aec_results.jsonl")
PID_DIR     = Path("outputs/pid")
OUT_RESULTS = PID_DIR / "pid_results.jsonl"
OUT_SUMMARY = PID_DIR / "pid_summary.json"


# ── PID computation ───────────────────────────────────────────────────────────

def compute_pid_atoms(v_a: float, v_b: float, v_ab: float) -> dict:
    """
    I_min PID decomposition (Williams & Beer 2010).
    Works with continuous v ∈ [0,1] (e.g., partial credit 0.5).

    Returns dict with: rdn, unique_a, unique_b, syn, and derived fields.
    """
    rdn      = min(v_a, v_b)
    unique_a = v_a  - rdn
    unique_b = v_b  - rdn
    syn      = v_ab - v_a - v_b + rdn

    # Verify identity: rdn + unique_a + unique_b + syn == v_ab
    # (holds algebraically; floating-point rounding may give tiny deviations)
    total = rdn + unique_a + unique_b + syn

    return {
        "rdn":      round(rdn, 4),
        "unique_a": round(unique_a, 4),
        "unique_b": round(unique_b, 4),
        "syn":      round(syn, 4),
        "total":    round(total, 4),   # should equal v_ab
    }


def classify_problem(v_a: float, v_b: float, v_ab: float, syn: float) -> str:
    """
    Assign one of 6 problem types based on PID atom dominance.
    """
    if v_ab < max(v_a, v_b):
        return "INTERFERENCE"    # collaboration hurt the outcome
    if v_ab == 0:
        return "COLLAPSE"        # nothing worked
    if v_a == 0 and v_b == 0:
        return "PURE_SYN"        # strongest epistemic necessity
    if v_a > 0 and v_b == 0:
        return "UNIQUE_A"        # agent A holds the key
    if v_a == 0 and v_b > 0:
        return "UNIQUE_B"        # agent B holds the key
    # both v_a > 0 and v_b > 0
    if syn > 0:
        return "REDUNDANT_SYN"   # redundant but with some synergy
    return "REDUNDANT"           # collaboration was unnecessary


# ── loading ───────────────────────────────────────────────────────────────────

def load_aec_results() -> list[dict]:
    if not AEC_RESULTS.exists():
        raise FileNotFoundError(
            f"{AEC_RESULTS} not found. Run compute_aec.py first."
        )
    results = []
    with open(AEC_RESULTS, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if "error" not in r:
                results.append(r)
    return results


# ── analysis ──────────────────────────────────────────────────────────────────

def _mean(vals: list) -> float:
    return round(sum(vals) / len(vals), 4) if vals else 0.0


def analyze(results: list[dict]) -> tuple[list[dict], dict]:
    """Compute PID for each problem and aggregate statistics."""
    pid_records = []

    for r in results:
        v_a  = r["v_a"]
        v_b  = r["v_b"]
        v_ab = r["v_ab"]
        atoms = compute_pid_atoms(v_a, v_b, v_ab)
        ptype = classify_problem(v_a, v_b, v_ab, atoms["syn"])

        rec = {
            "problem_id":  r["problem_id"],
            "v_a":         v_a,
            "v_b":         v_b,
            "v_ab":        v_ab,
            "cdi_c7":      r.get("cdi_c7", 0.0),
            "aec_a":       r.get("aec_a", 0.0),
            "aec_b":       r.get("aec_b", 0.0),
            "en":          r.get("en", False),
            "cs":          r.get("cs", 0.0),
            **atoms,
            "problem_type": ptype,
        }
        pid_records.append(rec)

    # ── aggregate stats ───────────────────────────────────────────────────────
    n = len(pid_records)
    type_counts = Counter(r["problem_type"] for r in pid_records)

    # Mean PID atoms per problem type
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in pid_records:
        by_type[r["problem_type"]].append(r)

    type_stats = {}
    for ptype, recs in sorted(by_type.items()):
        type_stats[ptype] = {
            "count":      len(recs),
            "pct":        round(len(recs) / n, 3),
            "syn_mean":   _mean([r["syn"]      for r in recs]),
            "unique_a_mean": _mean([r["unique_a"] for r in recs]),
            "unique_b_mean": _mean([r["unique_b"] for r in recs]),
            "rdn_mean":   _mean([r["rdn"]      for r in recs]),
            "cdi_mean":   _mean([r["cdi_c7"]   for r in recs]),
        }

    # CDI correlation with PID atoms
    def _corr(xs, ys):
        n = len(xs)
        if n < 2: return 0.0
        mx, my = sum(xs)/n, sum(ys)/n
        num = sum((x-mx)*(y-my) for x,y in zip(xs,ys))
        dx  = (sum((x-mx)**2 for x in xs))**0.5
        dy  = (sum((y-my)**2 for y in ys))**0.5
        return round(num / (dx*dy), 4) if dx*dy > 0 else 0.0

    cdis = [r["cdi_c7"] for r in pid_records]
    corr_cdi = {
        "syn_vs_cdi":      _corr([r["syn"]      for r in pid_records], cdis),
        "unique_a_vs_cdi": _corr([r["unique_a"] for r in pid_records], cdis),
        "unique_b_vs_cdi": _corr([r["unique_b"] for r in pid_records], cdis),
        "rdn_vs_cdi":      _corr([r["rdn"]      for r in pid_records], cdis),
    }

    # Key result: fraction of "epistemically well-designed" problems
    # (those where Syn + Unique_A + Unique_B > Rdn)
    epi_useful = sum(1 for r in pid_records
                     if (r["syn"] + r["unique_a"] + r["unique_b"]) > r["rdn"])

    summary = {
        "n":           n,
        "problem_types": {k: v for k, v in type_counts.most_common()},
        "type_stats":  type_stats,
        # Global atom means
        "rdn_mean":      _mean([r["rdn"]      for r in pid_records]),
        "unique_a_mean": _mean([r["unique_a"] for r in pid_records]),
        "unique_b_mean": _mean([r["unique_b"] for r in pid_records]),
        "syn_mean":      _mean([r["syn"]      for r in pid_records]),
        # Derived
        "useful_info_rate": round(epi_useful / n, 4),  # Syn + Unique > Rdn
        "pure_syn_rate":    round(type_counts.get("PURE_SYN", 0) / n, 4),
        "interference_rate": round(type_counts.get("INTERFERENCE", 0) / n, 4),
        # CDI correlations
        "cdi_correlations": corr_cdi,
    }

    return pid_records, summary


def print_report(summary: dict) -> None:
    print("\n" + "="*60)
    print("PID DECOMPOSITION — CollabMath N=2 Splits")
    print("="*60)

    print(f"\nn = {summary['n']} problems")
    print(f"\nGlobal atom means:")
    print(f"  Rdn      (redundant)  : {summary['rdn_mean']:.4f}")
    print(f"  Unique_A              : {summary['unique_a_mean']:.4f}")
    print(f"  Unique_B              : {summary['unique_b_mean']:.4f}")
    print(f"  Syn      (synergistic): {summary['syn_mean']:.4f}")

    print(f"\nProblem type distribution:")
    for ptype, stats in summary["type_stats"].items():
        bar = "█" * int(stats["pct"] * 30)
        print(f"  {ptype:<16} {stats['count']:>3} ({stats['pct']*100:>4.1f}%)  {bar}")
        print(f"                    syn={stats['syn_mean']:.3f}  "
              f"uniq_a={stats['unique_a_mean']:.3f}  "
              f"uniq_b={stats['unique_b_mean']:.3f}  "
              f"rdn={stats['rdn_mean']:.3f}  CDI={stats['cdi_mean']:.3f}")

    print(f"\nKey rates:")
    print(f"  Useful info (Syn+Unique > Rdn): {summary['useful_info_rate']*100:.1f}%")
    print(f"  Pure synergy  (strongest EN)  : {summary['pure_syn_rate']*100:.1f}%")
    print(f"  Interference  (collab hurt)   : {summary['interference_rate']*100:.1f}%")

    print(f"\nCDI correlations with PID atoms:")
    for k, v in summary["cdi_correlations"].items():
        print(f"  r({k}) = {v:+.4f}")

    print(f"\nTheoretical interpretation:")
    print(f"  PURE_SYN problems: the answer only exists at the intersection")
    print(f"    of both packets — maximum epistemic necessity.")
    print(f"  INTERFERENCE: collaboration degraded a capable solo agent —")
    print(f"    split may have introduced noise or conflicting information.")
    print(f"  CDI × Syn correlation shows whether process quality tracks")
    print(f"    information-theoretic synergy (expected: positive).")
    print("="*60)


# ── optional plot ─────────────────────────────────────────────────────────────

def _plot(pid_records: list[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("[WARN] matplotlib not available — skipping plot")
        return

    type_colors = {
        "PURE_SYN":       "#2196F3",
        "UNIQUE_A":       "#4CAF50",
        "UNIQUE_B":       "#8BC34A",
        "REDUNDANT_SYN":  "#FF9800",
        "REDUNDANT":      "#F44336",
        "INTERFERENCE":   "#9C27B0",
        "COLLAPSE":       "#9E9E9E",
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("PID Decomposition of CIDI Splits (N=2)", fontsize=13)

    # Plot 1: CDI vs Syn
    ax = axes[0]
    for r in pid_records:
        c = type_colors.get(r["problem_type"], "gray")
        ax.scatter(r["syn"], r["cdi_c7"], color=c, alpha=0.6, s=30)
    ax.set_xlabel("Synergy (Syn)")
    ax.set_ylabel("CDI (C7)")
    ax.set_title("CDI vs Synergy")
    ax.grid(True, alpha=0.3)

    # Plot 2: Stacked bar — atom composition by problem type
    ax = axes[1]
    types = sorted({r["problem_type"] for r in pid_records})
    by_type: dict[str, list] = defaultdict(list)
    for r in pid_records:
        by_type[r["problem_type"]].append(r)

    means = {t: {a: _mean([r[a] for r in recs])
                 for a in ["rdn", "unique_a", "unique_b", "syn"]}
             for t, recs in by_type.items()}

    x = list(range(len(types)))
    bottom = [0] * len(types)
    atom_colors = {"rdn": "#F44336", "unique_a": "#4CAF50",
                   "unique_b": "#8BC34A", "syn": "#2196F3"}
    for atom, color in atom_colors.items():
        vals = [means[t][atom] for t in types]
        ax.bar(x, vals, bottom=bottom, color=color, label=atom)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_xticks(x)
    ax.set_xticklabels(types, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Mean atom value")
    ax.set_title("PID atoms by problem type")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # Plot 3: Syn + Unique_A + Unique_B vs CDI scatter
    ax = axes[2]
    for r in pid_records:
        useful = r["syn"] + r["unique_a"] + r["unique_b"]
        c = type_colors.get(r["problem_type"], "gray")
        ax.scatter(useful, r["cdi_c7"], color=c, alpha=0.6, s=30)
    ax.set_xlabel("Useful info (Syn + Unique_A + Unique_B)")
    ax.set_ylabel("CDI (C7)")
    ax.set_title("CDI vs Total Useful Info")
    ax.grid(True, alpha=0.3)

    patches = [mpatches.Patch(color=c, label=t) for t, c in type_colors.items()
               if t in {r["problem_type"] for r in pid_records}]
    fig.legend(handles=patches, loc="lower center", ncol=len(patches),
               fontsize=8, bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()
    PID_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(PID_DIR / "pid_plots.png", dpi=150, bbox_inches="tight")
    print(f"[OK] Plot saved → {PID_DIR / 'pid_plots.png'}")
    plt.close()


# ── main ──────────────────────────────────────────────────────────────────────

def run(plot: bool = False) -> tuple[list[dict], dict]:
    PID_DIR.mkdir(parents=True, exist_ok=True)

    aec_results = load_aec_results()
    print(f"[INFO] Loaded {len(aec_results)} AEC results")

    pid_records, summary = analyze(aec_results)

    # Save
    with open(OUT_RESULTS, "w", encoding="utf-8") as f:
        for r in pid_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2))

    print_report(summary)
    print(f"\n[OK] {OUT_RESULTS}  ({len(pid_records)} records)")
    print(f"[OK] {OUT_SUMMARY}")

    if plot:
        _plot(pid_records)

    return pid_records, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true",
                        help="Generate scatter plots (requires matplotlib)")
    args = parser.parse_args()
    run(plot=args.plot)


if __name__ == "__main__":
    main()
