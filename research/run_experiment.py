"""
Main experiment runner. Designed to run locally or as a Sapelo batch job.

Usage:
  python -m research.run_experiment --stage all --workers 8

Stages (run in order, or all at once):
  load      Download & cache the MATH dataset sample
  split     Generate N-way jigsaw splits for every problem
  simulate  Run all conversations (solo + unrestricted_pair + jigsaw_N)
  score     Score all conversations with PISA + ATC21S
  analyse   Compute collaborative advantage, CPS necessity, etc.
"""
import argparse, json, os, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict

from research.config import CFG
from research.data.math_loader import load_math_dataset, summarize
from research.splitting.splitter import split, SplitResult
from research.simulation.simulator import simulate, Conversation
from research.scoring.pisa import score_conversation_python as pisa_score
from research.scoring.atc21s import score_conversation as atc_score
from research.analysis.metrics import (
    ExperimentRecord, build_results_df, is_correct,
    collaborative_advantage, cps_necessity, group_size_effect,
    phase_advantage, pisa_vs_atc_correlation,
    problem_type_summary, openness_comparison,
    competence_advantage, competence_by_level,
    split_pattern_analysis,
)

# ── directories ───────────────────────────────────────────────────────────────

OUTPUTS = Path("outputs")
DATA_DIR    = OUTPUTS / "data"
SPLITS_DIR  = OUTPUTS / "splits"
CONVS_DIR   = OUTPUTS / "conversations"
SCORES_DIR  = OUTPUTS / "scores"
RESULTS_DIR = OUTPUTS / "results"

for d in [DATA_DIR, SPLITS_DIR, CONVS_DIR, SCORES_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def _save_json(path: Path, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def _load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def _conditions_for_n(n: int) -> List[str]:
    conds = []
    if n == 1:
        conds.append("solo")
    if n == 2:
        conds += ["unrestricted_pair", "jigsaw_2"]
    if n >= 3:
        conds.append(f"jigsaw_{n}")
    return conds


# ── stage 1: load ─────────────────────────────────────────────────────────────

def stage_load() -> List[dict]:
    cache = DATA_DIR / "math_sample.json"
    problems = load_math_dataset(cache_path=str(cache))
    summarize(problems)
    print(f"Saved to {cache}")
    return problems


# ── stage 2: split ────────────────────────────────────────────────────────────

_SPLITTER = "cidi"   # overridden by --splitter CLI arg


def _split_one(prob: dict, n: int) -> dict:
    out_path = SPLITS_DIR / f"{prob['id']}_n{n}.json"
    if out_path.exists():
        return _load_json(out_path)

    if _SPLITTER == "sft":
        from research.splitting.sft_splitter import generate_split
        result = generate_split(prob["id"], prob["problem"])
    else:
        result = split(prob["id"], prob["problem"], n)
    data = {
        "problem_id": result.problem_id,
        "n": result.n,
        "pattern": result.pattern,
        "shared_context": result.shared_context,
        "agent_roles": [
            {"agent_id": p.agent_id, "role_name": p.role_name,
             "role_description": p.role_description}
            for p in result.packets
        ],
        "packets": [{"agent_id": p.agent_id, "information": p.information}
                    for p in result.packets],
        "valid": result.valid,
        "validation_log": result.validation_log,
        "split_rationale": result.raw_split.get("split_rationale", ""),
    }
    _save_json(out_path, data)
    return data


def stage_split(problems: List[dict], workers: int = 4) -> None:
    tasks = [(prob, n) for prob in problems for n in CFG.n_values]
    print(f"Generating splits: {len(tasks)} tasks with {workers} workers")

    done = failed = 0
    with ThreadPoolExecutor(max_workers=workers) as exe:
        futures = {exe.submit(_split_one, prob, n): (prob["id"], n)
                   for prob, n in tasks}
        for fut in as_completed(futures):
            pid, n = futures[fut]
            try:
                r = fut.result()
                if r.get("valid"):
                    done += 1
                else:
                    failed += 1
                    print(f"  INVALID split: {pid} n={n}")
            except Exception as e:
                failed += 1
                print(f"  ERROR {pid} n={n}: {e}")

    print(f"Splits done: {done} valid, {failed} invalid/failed")


# ── stage 3: simulate ─────────────────────────────────────────────────────────

def _simulate_one(prob: dict, split_data: dict, condition: str) -> dict:
    out_path = CONVS_DIR / f"{prob['id']}_{condition}.json"
    if out_path.exists():
        return _load_json(out_path)

    from research.splitting.splitter import SplitResult, Packet
    roles = {r["agent_id"]: r for r in split_data.get("agent_roles", [])}
    sr = SplitResult(
        problem_id=split_data["problem_id"],
        problem=prob["problem"],
        n=split_data["n"],
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
        valid=split_data["valid"],
    )

    conv = simulate(sr, condition)
    data = conv.to_dict()
    # Embed full context for qualitative analysis — no need to cross-reference files
    data["ground_truth"]   = prob["answer"]
    data["problem"]        = prob["problem"]
    data["subject"]        = prob["subject"]
    data["level"]          = prob["level"]
    data["openness"]       = prob["openness"]
    data["split_pattern"]  = split_data.get("pattern", "")
    data["split_rationale"] = split_data.get("split_rationale", "")
    data["agent_roles"]    = split_data.get("agent_roles", [])
    data["packets"]        = split_data.get("packets", [])
    _save_json(out_path, data)
    return data


def stage_simulate(problems: List[dict], workers: int = 8) -> None:
    tasks = []
    for prob in problems:
        for n in CFG.n_values:
            split_path = SPLITS_DIR / f"{prob['id']}_n{n}.json"
            if not split_path.exists():
                continue
            split_data = _load_json(split_path)
            if not split_data.get("valid") and n > 1:
                continue
            for cond in _conditions_for_n(n):
                tasks.append((prob, split_data, cond))

    print(f"Simulating {len(tasks)} conversations with {workers} workers")
    done = failed = 0
    with ThreadPoolExecutor(max_workers=workers) as exe:
        futures = {exe.submit(_simulate_one, p, s, c): (p["id"], c)
                   for p, s, c in tasks}
        for fut in as_completed(futures):
            pid, cond = futures[fut]
            try:
                fut.result()
                done += 1
                if done % 50 == 0:
                    print(f"  {done}/{len(tasks)} conversations done")
            except Exception as e:
                failed += 1
                print(f"  ERROR {pid} {cond}: {e}")

    print(f"Simulations done: {done} ok, {failed} failed")


# ── stage 4: score ────────────────────────────────────────────────────────────

def _score_one(conv_path: Path) -> dict:
    score_path = SCORES_DIR / conv_path.name.replace(".json", "_scores.json")
    if score_path.exists():
        return _load_json(score_path)

    conv_data = _load_json(conv_path)
    from research.simulation.simulator import Conversation, Turn

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
        "problem_id":  conv.problem_id,
        "condition":   conv.condition,
        "n":           conv.n,
        "subject":     conv_data.get("subject"),
        "level":       conv_data.get("level"),
        "openness":    conv_data.get("openness"),
        "correct":     is_correct(conv.final_answer, conv_data.get("ground_truth")),
        "consensus":   conv.consensus,
        "total_turns": conv.total_turns,
        "final_answer": conv.final_answer,
        "ground_truth": conv_data.get("ground_truth"),
        "pisa": {
            "global_index":    pisa.global_cps_index,
            "richness_entropy": pisa.richness_entropy,
            "process_share":   pisa.process_share,
            "competence_share": pisa.competence_share,
            "code_counts":     pisa.code_counts,
        },
        "atc21s": {
            "global_index":  atc.global_atc_index,
            "social_index":  atc.social_index,
            "cognitive_index": atc.cognitive_index,
            "dim_means":     atc.dim_means,
            "dim_presence":  atc.dim_presence,
        },
    }
    _save_json(score_path, data)
    return data


def stage_score(workers: int = 4) -> List[dict]:
    conv_paths = list(CONVS_DIR.glob("*.json"))
    print(f"Scoring {len(conv_paths)} conversations with {workers} workers")

    all_scores = []
    done = failed = 0
    with ThreadPoolExecutor(max_workers=workers) as exe:
        futures = {exe.submit(_score_one, p): p for p in conv_paths}
        for fut in as_completed(futures):
            p = futures[fut]
            try:
                all_scores.append(fut.result())
                done += 1
                if done % 20 == 0:
                    print(f"  {done}/{len(conv_paths)} scored")
            except Exception as e:
                failed += 1
                print(f"  ERROR {p.name}: {e}")

    print(f"Scoring done: {done} ok, {failed} failed")
    return all_scores


# ── stage 5: analyse ──────────────────────────────────────────────────────────

def stage_analyse() -> None:
    score_files = list(SCORES_DIR.glob("*_scores.json"))
    records = []
    for sf in score_files:
        d = _load_json(sf)
        problem_id = d["problem_id"]
        n_agents   = d["n"]

        # Load split_pattern from the corresponding split file
        split_path = SPLITS_DIR / f"{problem_id}_n{n_agents}.json"
        split_pattern = ""
        if split_path.exists():
            try:
                split_data = _load_json(split_path)
                split_pattern = split_data.get("pattern", "")
            except Exception:
                pass

        r = ExperimentRecord(
            problem_id=problem_id,
            subject=d.get("subject", ""),
            level=d.get("level", 0),
            openness=d.get("openness", "closed"),
            condition=d["condition"],
            n_agents=n_agents,
            correct=d.get("correct", False),
            consensus=d.get("consensus", False),
            total_turns=d.get("total_turns", 0),
            pisa_global=d["pisa"]["global_index"],
            pisa_entropy=d["pisa"]["richness_entropy"],
            pisa_process_A=d["pisa"]["process_share"].get("A", 0),
            pisa_process_B=d["pisa"]["process_share"].get("B", 0),
            pisa_process_C=d["pisa"]["process_share"].get("C", 0),
            pisa_process_D=d["pisa"]["process_share"].get("D", 0),
            pisa_comp_1=d["pisa"]["competence_share"].get("1", 0),
            pisa_comp_2=d["pisa"]["competence_share"].get("2", 0),
            pisa_comp_3=d["pisa"]["competence_share"].get("3", 0),
            atc_global=d["atc21s"]["global_index"],
            atc_social=d["atc21s"]["social_index"],
            atc_cognitive=d["atc21s"]["cognitive_index"],
            atc_PC=d["atc21s"]["dim_means"].get("PC", 0),
            atc_C=d["atc21s"]["dim_means"].get("C", 0),
            atc_Co=d["atc21s"]["dim_means"].get("Co", 0),
            atc_CR=d["atc21s"]["dim_means"].get("CR", 0),
            atc_SR=d["atc21s"]["dim_means"].get("SR", 0),
            split_pattern=split_pattern,
        )
        records.append(r)

    df = build_results_df(records)
    df.to_csv(RESULTS_DIR / "all_results.csv", index=False)
    print(f"Results table: {len(df)} rows saved to {RESULTS_DIR}/all_results.csv")

    # ── key analyses ──
    print("\n── Collaborative Advantage by Complexity ──")
    adv = collaborative_advantage(df)
    print(adv.to_string(index=False))
    adv.to_csv(RESULTS_DIR / "collaborative_advantage.csv", index=False)

    print("\n── CPS Necessity (jigsaw vs unrestricted) ──")
    nec = cps_necessity(df)
    print(nec.to_string(index=False))
    nec.to_csv(RESULTS_DIR / "cps_necessity.csv", index=False)

    print("\n── Group Size Effect ──")
    gs = group_size_effect(df)
    print(gs.to_string(index=False))
    gs.to_csv(RESULTS_DIR / "group_size_effect.csv", index=False)

    print("\n── Phase Advantage ──")
    ph = phase_advantage(df)
    print(ph.to_string())
    ph.to_csv(RESULTS_DIR / "phase_advantage.csv")

    print("\n── PISA vs ATC21S Correlation ──")
    corr = pisa_vs_atc_correlation(df)
    print(corr.to_string(index=False))
    corr.to_csv(RESULTS_DIR / "pisa_atc_correlation.csv", index=False)

    print("\n── Problem Type Summary ──")
    pt = problem_type_summary(df)
    print(pt.to_string(index=False))
    pt.to_csv(RESULTS_DIR / "problem_type.csv", index=False)

    print("\n── Openness Comparison ──")
    op = openness_comparison(df)
    print(op.to_string(index=False))
    op.to_csv(RESULTS_DIR / "openness.csv", index=False)

    print("\n── PISA Competence Advantage (1=shared knowledge, 2=math action, 3=coordination) ──")
    ca = competence_advantage(df)
    print(ca.to_string())
    ca.to_csv(RESULTS_DIR / "competence_advantage.csv")

    print("\n── PISA Competence by Level ──")
    cl = competence_by_level(df)
    print(cl.to_string(index=False))
    cl.to_csv(RESULTS_DIR / "competence_by_level.csv", index=False)

    print("\n── Split Pattern Analysis ──")
    sp = split_pattern_analysis(df)
    print(sp.to_string(index=False))
    sp.to_csv(RESULTS_DIR / "split_pattern.csv", index=False)

    print(f"\nAll analysis tables saved to {RESULTS_DIR}/")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    global _SPLITTER
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage",    default="all",
                        choices=["load", "split", "simulate", "score", "analyse", "all"])
    parser.add_argument("--workers",  type=int, default=8)
    parser.add_argument("--splitter", default="cidi", choices=["cidi", "sft"],
                        help="Split generator: 'cidi' (GPT-4/Groq, default) or 'sft' (local model, free)")
    args = parser.parse_args()
    _SPLITTER = args.splitter

    t0 = time.time()

    problems = None

    if args.stage in ("load", "all"):
        problems = stage_load()

    if args.stage in ("split", "all"):
        if problems is None:
            problems = _load_json(DATA_DIR / "math_sample.json")
        stage_split(problems, workers=args.workers)

    if args.stage in ("simulate", "all"):
        if problems is None:
            problems = _load_json(DATA_DIR / "math_sample.json")
        stage_simulate(problems, workers=args.workers)

    if args.stage in ("score", "all"):
        stage_score(workers=min(args.workers, 4))  # scoring is API-heavy

    if args.stage in ("analyse", "all"):
        stage_analyse()

    print(f"\nTotal time: {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
