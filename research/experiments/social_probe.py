"""
Social coordination probe: compares jigsaw_2 vs social_jigsaw_2 on 25 matched problems.

Samples 5 problems per difficulty level (1–5), uses the SAME splits already generated,
re-simulates only with the socially-enriched prompt, then scores and reports a paired
comparison on PISA comp1/2/3 and ATC21S Co/CR/SR.

Usage:
  python -m research.experiments.social_probe --workers 4
"""
import argparse, json, random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

from research.config import CFG
from research.splitting.splitter import SplitResult, Packet
from research.simulation.simulator import simulate, Conversation
from research.scoring.pisa import score_conversation_python as pisa_score
from research.scoring.atc21s import score_conversation as atc_score
from research.analysis.metrics import is_correct

PROBE_DIR   = Path("outputs/social_probe")
CONV_DIR    = PROBE_DIR / "conversations"
SCORE_DIR   = PROBE_DIR / "scores"
SPLITS_DIR  = Path("outputs/splits")
DATA_PATH   = Path("outputs/data/math_sample.json")
BASE_SCORES = Path("outputs/scores")

PROBE_DIR.mkdir(parents=True, exist_ok=True)
CONV_DIR.mkdir(parents=True, exist_ok=True)
SCORE_DIR.mkdir(parents=True, exist_ok=True)

N_PER_LEVEL  = None   # None = all available; set to int for a capped sample
RANDOM_SEED  = 42


def _load(path: Path):
    with open(path) as f:
        return json.load(f)

def _save(path: Path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _sample_problems() -> list:
    """
    All problems with valid n=2 splits AND existing jigsaw_2 baseline scores.
    If N_PER_LEVEL is set, sample that many per level (seeded).
    """
    problems = _load(DATA_PATH)
    by_level = defaultdict(list)
    for prob in problems:
        sp = SPLITS_DIR / f"{prob['id']}_n2.json"
        sc = BASE_SCORES / f"{prob['id']}_jigsaw_2_scores.json"
        if sp.exists() and sc.exists():
            s = _load(sp)
            if s.get("valid"):
                by_level[prob["level"]].append(prob)

    rng = random.Random(RANDOM_SEED)
    sample = []
    for level in sorted(by_level):
        pool = by_level[level]
        n = N_PER_LEVEL if N_PER_LEVEL is not None else len(pool)
        sample.extend(rng.sample(pool, min(n, len(pool))))
    return sample


def _run_one(prob: dict) -> dict:
    pid = prob["id"]
    out_path = CONV_DIR / f"{pid}_social_jigsaw_2.json"
    if out_path.exists():
        return _load(out_path)

    split_data = _load(SPLITS_DIR / f"{pid}_n2.json")
    roles = {r["agent_id"]: r for r in split_data.get("agent_roles", [])}
    sr = SplitResult(
        problem_id=split_data["problem_id"],
        problem=prob["problem"],
        n=2,
        pattern=split_data.get("pattern", ""),
        shared_context=split_data["shared_context"],
        packets=[
            Packet(
                agent_id=p["agent_id"],
                information=p["information"],
                role_name=roles.get(p["agent_id"], {}).get("role_name", ""),
                role_description=roles.get(p["agent_id"], {}).get("role_description", ""),
            )
            for p in split_data["packets"]
        ],
        valid=True,
    )

    conv = simulate(sr, "social_jigsaw_2")
    data = conv.to_dict()
    data.update({
        "ground_truth":    prob["answer"],
        "problem":         prob["problem"],
        "subject":         prob["subject"],
        "level":           prob["level"],
        "split_pattern":   split_data.get("pattern", ""),
        "agent_roles":     split_data.get("agent_roles", []),
    })
    _save(out_path, data)
    return data


def _score_one(pid: str) -> dict:
    out_path = SCORE_DIR / f"{pid}_social_jigsaw_2_scores.json"
    if out_path.exists():
        return _load(out_path)

    conv_data = _load(CONV_DIR / f"{pid}_social_jigsaw_2.json")
    from research.simulation.simulator import Turn
    conv = Conversation(
        problem_id=conv_data["problem_id"],
        condition=conv_data["condition"],
        n=conv_data["n"],
        turns=[Turn(agent_id=t["agent_id"], role="assistant", content=t["content"])
               for t in conv_data["turns"]],
        final_answer=conv_data.get("final_answer"),
        consensus=conv_data.get("consensus", False),
        total_turns=conv_data.get("total_turns", 0),
    )
    pisa = pisa_score(conv)
    atc  = atc_score(conv)

    data = {
        "problem_id":   pid,
        "condition":    "social_jigsaw_2",
        "level":        conv_data.get("level"),
        "subject":      conv_data.get("subject"),
        "correct":      is_correct(conv.final_answer, conv_data.get("ground_truth")),
        "total_turns":  conv.total_turns,
        "pisa_global":  pisa.global_cps_index,
        "pisa_comp_1":  pisa.competence_share.get("1", 0),
        "pisa_comp_2":  pisa.competence_share.get("2", 0),
        "pisa_comp_3":  pisa.competence_share.get("3", 0),
        "pisa_process_A": pisa.process_share.get("A", 0),
        "pisa_process_B": pisa.process_share.get("B", 0),
        "pisa_process_C": pisa.process_share.get("C", 0),
        "pisa_process_D": pisa.process_share.get("D", 0),
        "atc_global":   atc.global_atc_index,
        "atc_Co":       atc.dim_means.get("Co", 0),
        "atc_CR":       atc.dim_means.get("CR", 0),
        "atc_SR":       atc.dim_means.get("SR", 0),
        "atc_PC":       atc.dim_means.get("PC", 0),
        "atc_C":        atc.dim_means.get("C", 0),
    }
    _save(out_path, data)
    return data


def _load_baseline(pid: str) -> dict:
    """Load existing jigsaw_2 score and flatten nested pisa/atc21s structure."""
    path = BASE_SCORES / f"{pid}_jigsaw_2_scores.json"
    if not path.exists():
        return {}
    d = _load(path)
    pisa   = d.get("pisa", {})
    atc    = d.get("atc21s", {})
    return {
        "problem_id":    d.get("problem_id"),
        "condition":     d.get("condition"),
        "level":         d.get("level"),
        "subject":       d.get("subject"),
        "correct":       d.get("correct", False),
        "total_turns":   d.get("total_turns", 0),
        "pisa_global":   pisa.get("global_index", 0) or 0,
        "pisa_comp_1":   (pisa.get("competence_share") or {}).get("1", 0),
        "pisa_comp_2":   (pisa.get("competence_share") or {}).get("2", 0),
        "pisa_comp_3":   (pisa.get("competence_share") or {}).get("3", 0),
        "pisa_process_A": (pisa.get("process_share") or {}).get("A", 0),
        "pisa_process_B": (pisa.get("process_share") or {}).get("B", 0),
        "pisa_process_C": (pisa.get("process_share") or {}).get("C", 0),
        "atc_global":    atc.get("global_index", 0) or 0,
        "atc_Co":        (atc.get("dim_means") or {}).get("Co", 0),
        "atc_CR":        (atc.get("dim_means") or {}).get("CR", 0),
        "atc_SR":        (atc.get("dim_means") or {}).get("SR", 0),
        "atc_PC":        (atc.get("dim_means") or {}).get("PC", 0),
        "atc_C":         (atc.get("dim_means") or {}).get("C", 0),
    }


def _report(social_scores: list, pids: list):
    import statistics
    import numpy as np
    from scipy import stats

    keys = ["pisa_global", "pisa_comp_1", "pisa_comp_2", "pisa_comp_3",
            "pisa_process_A", "pisa_process_B", "pisa_process_C",
            "atc_global", "atc_Co", "atc_CR", "atc_SR",
            "correct", "total_turns"]

    social_by_pid = {s["problem_id"]: s for s in social_scores}
    baseline_by_pid = {}
    for pid in pids:
        b = _load_baseline(pid)
        if b:
            baseline_by_pid[pid] = b

    matched = [(social_by_pid[p], baseline_by_pid[p])
               for p in pids if p in social_by_pid and p in baseline_by_pid]

    print(f"\nPaired comparison: n={len(matched)} problems")
    print(f"{'Metric':<22} {'jigsaw_2':>10} {'social':>10} {'Δ':>8} {'d':>6} {'p (Wilcoxon)':>14} {'sig':>4}")
    print("─" * 80)

    results_rows = []
    for key in keys:
        try:
            base_vals   = np.array([b.get(key, 0) for _, b in matched], dtype=float)
            social_vals = np.array([s.get(key, 0) for s, _ in matched], dtype=float)
            b_mean = base_vals.mean()
            s_mean = social_vals.mean()
            delta  = s_mean - b_mean
            diffs  = social_vals - base_vals
            pooled_sd = np.std(np.concatenate([base_vals, social_vals]), ddof=1)
            cohens_d  = delta / pooled_sd if pooled_sd > 0 else 0.0
            _, p = stats.wilcoxon(base_vals, social_vals) if len(set(diffs)) > 1 else (None, 1.0)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"{key:<22} {b_mean:>10.3f} {s_mean:>10.3f} {delta:>+8.3f} "
                  f"{cohens_d:>+6.2f} {p:>14.4f} {sig:>4}")
            results_rows.append({"metric": key, "jigsaw_2": b_mean, "social": s_mean,
                                  "delta": delta, "cohens_d": cohens_d, "p_wilcoxon": p})
        except Exception as e:
            print(f"{key:<22} ERROR: {e}")

    # Save full results
    import csv
    out_csv = PROBE_DIR / "paired_comparison.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metric","jigsaw_2","social","delta","cohens_d","p_wilcoxon"])
        w.writeheader(); w.writerows(results_rows)
    print(f"\nFull results saved to {out_csv}")

    print("\n── By level ──")
    print(f"{'lv':<4} {'n':>4} {'base_pisa':>10} {'soc_pisa':>10} {'Δpisa':>7} "
          f"{'base_comp3':>11} {'soc_comp3':>11} {'Δcomp3':>8} "
          f"{'base_atc':>9} {'soc_atc':>9} {'Δatc':>7}")
    print("─" * 100)
    by_level = defaultdict(lambda: {"base": [], "social": []})
    for s, b in matched:
        lv = s.get("level", b.get("level", "?"))
        by_level[lv]["base"].append(b)
        by_level[lv]["social"].append(s)

    for lv in sorted(by_level):
        bv = by_level[lv]["base"]
        sv = by_level[lv]["social"]
        n  = len(bv)
        def m(lst, k): return statistics.mean(x.get(k, 0) for x in lst)
        print(f"{lv:<4} {n:>4} "
              f"{m(bv,'pisa_global'):>10.2f} {m(sv,'pisa_global'):>10.2f} "
              f"{m(sv,'pisa_global')-m(bv,'pisa_global'):>+7.2f} "
              f"{m(bv,'pisa_comp_3'):>11.2f} {m(sv,'pisa_comp_3'):>11.2f} "
              f"{m(sv,'pisa_comp_3')-m(bv,'pisa_comp_3'):>+8.2f} "
              f"{m(bv,'atc_global'):>9.2f} {m(sv,'atc_global'):>9.2f} "
              f"{m(sv,'atc_global')-m(bv,'atc_global'):>+7.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    print("Sampling problems...")
    problems = _sample_problems()
    pids = [p["id"] for p in problems]
    print(f"Selected {len(problems)} problems: {', '.join(str(p['level']) for p in problems)}")

    print(f"\nSimulating social_jigsaw_2 with {args.workers} workers...")
    done = failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as exe:
        futures = {exe.submit(_run_one, p): p["id"] for p in problems}
        for fut in as_completed(futures):
            pid = futures[fut]
            try:
                fut.result()
                done += 1
                print(f"  simulated {pid}")
            except Exception as e:
                failed += 1
                print(f"  ERROR {pid}: {e}")
    print(f"Simulations: {done} ok, {failed} failed")

    print(f"\nScoring with {args.workers} workers...")
    social_scores = []
    done = failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as exe:
        futures = {exe.submit(_score_one, pid): pid for pid in pids}
        for fut in as_completed(futures):
            pid = futures[fut]
            try:
                social_scores.append(fut.result())
                done += 1
                print(f"  scored {pid}")
            except Exception as e:
                failed += 1
                print(f"  ERROR {pid}: {e}")
    print(f"Scoring: {done} ok, {failed} failed")

    _report(social_scores, pids)


if __name__ == "__main__":
    main()
