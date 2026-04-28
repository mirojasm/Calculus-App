"""
Pipeline verification script — tests each component without API calls where possible.

Usage:
  python3 -m research.experiments.verify_pipeline           # all checks
  python3 -m research.experiments.verify_pipeline --no-api  # skip LLM calls
  python3 -m research.experiments.verify_pipeline --full    # include smoke test with LLM
"""
from __future__ import annotations
import argparse, json, sys, traceback
from pathlib import Path

PASS  = "✓"
FAIL  = "✗"
SKIP  = "–"
results: list[tuple[str, str, str]] = []   # (label, status, note)


def check(label: str, fn, skip: bool = False):
    if skip:
        results.append((label, SKIP, "skipped"))
        return True
    try:
        note = fn()
        results.append((label, PASS, note or ""))
        return True
    except Exception as e:
        results.append((label, FAIL, str(e)))
        if "--verbose" in sys.argv:
            traceback.print_exc()
        return False


# ── 1. Module 2: DAG closure ──────────────────────────────────────────────────

def test_dag_closure():
    from research.splitting.cidi.module2_feasibility import close_under_prerequisites
    # C2 requires: A1, A2 (via B2→A2), B1 (via C1→B1), B2 (via C2→C1→B2), C1
    result = set(close_under_prerequisites(["C2"]))
    expected = {"A1", "A2", "B1", "B2", "C1", "C2"}
    assert result == expected, f"Expected {expected}, got {result}"
    return f"close(['C2']) = {sorted(result)}"


def test_upset_property():
    from research.splitting.cidi.module2_feasibility import (
        close_under_prerequisites, PREREQ_DAG
    )
    # Every closed set must include all prerequisites
    for cell in PREREQ_DAG:
        closed = set(close_under_prerequisites([cell]))
        for c in list(closed):
            for prereq in PREREQ_DAG[c]:
                assert prereq in closed, f"{prereq} missing when closing {cell}"
    return "All cells: DAG closure is a valid upset"


def test_feasibility_check():
    from research.splitting.cidi.module2_feasibility import check_structural_feasibility
    anatomy_rich  = {"sub_problems": [{"id":"sp1"},{"id":"sp2"}], "information_bottlenecks": ["key info"]}
    anatomy_empty = {"sub_problems": [], "information_bottlenecks": []}
    ok, _  = check_structural_feasibility("C2", anatomy_rich)
    nok, _ = check_structural_feasibility("D3", anatomy_empty)
    assert ok  is True,  "C2 should be feasible with 2 sub-problems"
    assert nok is False, "D3 should be infeasible with 0 sub-problems"
    return "C2 feasible=True, D3 feasible=False"


# ── 2. Module 3: constraint table ────────────────────────────────────────────

def test_constraint_table():
    from research.splitting.cidi.module3_constraints import get_constraint, CELL_ASYMMETRY
    assert len(CELL_ASYMMETRY) == 12, f"Expected 12 cells, got {len(CELL_ASYMMETRY)}"
    c2 = get_constraint("C2")
    assert "unbreakable" in c2["prompt_instruction"].lower() or \
           "mathematically" in c2["prompt_instruction"].lower(), \
           "C2 constraint should mention mathematical dependency"
    return f"12 cells defined; C2 constraint ok"


def test_build_constraints_summary():
    from research.splitting.cidi.module3_constraints import build_constraints_summary
    anatomy = {
        "natural_split_axes": ["algebraic vs geometric"],
        "information_bottlenecks": ["hidden connection"],
        "sub_problems": [{"id":"sp1"},{"id":"sp2"}],
        "reasoning_type": ["algebraic", "geometric"],
    }
    cs = build_constraints_summary(["A1","B1","C2"], anatomy)
    assert "A1" in cs["cell_constraints"]
    assert "B1" in cs["cell_constraints"]
    assert "C2" in cs["cell_constraints"]
    assert cs["dominant_pattern"] in ("SPLIT-A","SPLIT-B","SPLIT-C","SPLIT-D","SPLIT-E","SPLIT-F","SPLIT-G")
    return f"Constraints built for 3 cells; pattern={cs['dominant_pattern']}"


# ── 3. Module 5: discriminator training ──────────────────────────────────────

def test_discriminator_training():
    from research.splitting.cidi.module5_validation import CPPDiscriminatorChain
    from research.splitting.cidi.train_discriminators import load_training_data

    texts, vectors = load_training_data(Path("outputs/splits"), Path("outputs/scores"))
    assert len(texts) >= 10, f"Need ≥10 training examples, got {len(texts)}"

    # Quick training on small subset
    chain = CPPDiscriminatorChain()
    auc = chain.fit(texts[:20], vectors[:20])
    assert len(auc) == 12
    auc_values = [v for v in auc.values() if v == v]  # exclude NaN
    return f"Trained on 20 examples; mean AUC (non-degenerate) = {sum(auc_values)/len(auc_values):.3f}"


def test_discriminator_predict():
    from research.splitting.cidi.module5_validation import CPPDiscriminatorChain
    from research.splitting.cidi.train_discriminators import load_training_data
    from research.splitting.cidi.module2_feasibility import CELL_ORDER

    texts, vectors = load_training_data(Path("outputs/splits"), Path("outputs/scores"))
    chain = CPPDiscriminatorChain()
    chain.fit(texts, vectors)

    # Predict on first test example
    mock_split = {"shared_context": texts[0], "packets": [], "agent_roles": []}
    pred = chain.predict(mock_split)

    assert "predicted_vector" in pred
    assert len(pred["predicted_vector"]) == 12
    assert 0.0 <= pred["predicted_cdi"] <= 1.0
    h = chain.hamming_to_target(pred["predicted_vector"], vectors[0])
    return f"Prediction ok; Hamming on first example = {h}/12"


def test_full_discriminator_pipeline():
    """Train on corpus and verify saves/loads correctly."""
    from research.splitting.cidi.module5_validation import CPPDiscriminatorChain
    from research.splitting.cidi.train_discriminators import load_training_data
    import tempfile, os

    texts, vectors = load_training_data(Path("outputs/splits"), Path("outputs/scores"))
    chain = CPPDiscriminatorChain()
    chain.fit(texts, vectors)

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        tmppath = Path(f.name)
    try:
        chain.save(tmppath)
        loaded = CPPDiscriminatorChain.load(tmppath)
        mock = {"shared_context": texts[0], "packets": [], "agent_roles": []}
        pred1 = chain.predict(mock)
        pred2 = loaded.predict(mock)
        assert pred1["predicted_vector"] == pred2["predicted_vector"], "Save/load inconsistent"
    finally:
        os.unlink(tmppath)
    return f"Save/load consistent; trained on {len(texts)} examples"


# ── 4. cpp_comparison: corpus loading ────────────────────────────────────────

def test_corpus_loading():
    from research.experiments.cpp_comparison import (
        _load_existing_splits, _load_existing_conversations, _load_existing_scores
    )
    splits = _load_existing_splits()
    convs  = _load_existing_conversations()
    scores = _load_existing_scores()
    assert len(splits) > 0, "No splits found in outputs/splits/"
    assert len(convs)  > 0, "No conversations found in outputs/conversations/"
    return f"splits={len(splits)}, conversations={len(convs)}, scores={len(scores)}"


def test_problem_selection():
    from research.experiments.cpp_comparison import select_pilot_problems
    problems = select_pilot_problems(4)
    assert len(problems) > 0, "No problems selected"
    for p in problems:
        assert "problem_id" in p
        assert "subject" in p or "problem" in p
    return f"Selected {len(problems)} problems: {[p['problem_id'] for p in problems]}"


def test_c1_loads():
    from research.experiments.cpp_comparison import run_c1, _load_existing_splits
    splits = _load_existing_splits()
    if not splits:
        return "No splits available — skip"
    pid = next(iter(splits))
    result = run_c1(pid, "")
    if "error" in result:
        return f"C1 missing conversation for {pid} (ok — not all splits have conversations)"
    assert result["condition"] == "C1"
    assert "cpp_vector" in result
    return f"C1 ok for {pid}: CDI={result['cdi']:.3f}"


# ── 5. CPP annotator ─────────────────────────────────────────────────────────

def test_annotator_imports():
    from research.scoring.cpp_annotator import annotate, CELL_LABELS, CPP_PROFILES, classify_cdi
    assert len(CELL_LABELS) == 12
    assert classify_cdi(0.0) == "CPP-Ø"
    assert classify_cdi(0.67) == "CPP-DEEP"
    assert classify_cdi(1.0) == "CPP-FULL"
    return "Annotator imports ok; classify_cdi correct"


# ── 6. openai_utils: Groq routing ────────────────────────────────────────────

def test_openai_utils_imports():
    from research.openai_utils import chat, chat_groq
    assert callable(chat)
    assert callable(chat_groq)
    return "openai_utils imports ok; chat_groq available"


# ── 7. Constitutional pipeline imports ───────────────────────────────────────

def test_constitutional_imports():
    from research.splitting.constitutional import (
        constitutional_split, ConstitutionalResult, MAX_ITER, APPROVAL_THRESHOLD
    )
    assert MAX_ITER == 3
    assert APPROVAL_THRESHOLD == 0.80
    return f"Constitutional imports ok; MAX_ITER={MAX_ITER}, threshold={APPROVAL_THRESHOLD}"


# ── 8. Monitor imports ───────────────────────────────────────────────────────

def test_monitor_imports():
    from research.simulation.monitor import (
        detect_phase, evaluate_phase, PHASE_INDICATORS, MONITOR_PHASES
    )
    assert set(MONITOR_PHASES) == {"A", "B"}
    # Quick phase detection
    history = [{"content": "let me calculate by substituting"}]
    phase = detect_phase(history)
    assert phase in ("A","B","C","D")
    return f"Monitor imports ok; detect_phase on test input → {phase}"


# ── 9. Simulator imports ─────────────────────────────────────────────────────

def test_simulator_imports():
    from research.simulation.simulator import simulate, simulate_with_monitor
    assert callable(simulate)
    assert callable(simulate_with_monitor)
    return "Simulator imports ok"


# ── 10. CIDI pipeline imports ────────────────────────────────────────────────

def test_cidi_pipeline_imports():
    from research.splitting.cidi.pipeline import split_cidi, CIDISplitResult, CPP_DEEP_TARGET
    assert len(CPP_DEEP_TARGET) == 8
    assert callable(split_cidi)
    return f"CIDI pipeline imports ok; CPP_DEEP_TARGET={CPP_DEEP_TARGET}"


# ── 11. LLM smoke test (requires API) ────────────────────────────────────────

def test_semantic_analysis_api():
    """Requires OpenAI/Groq API. Skipped with --no-api."""
    from research.splitting.cidi.module1_semantic import analyze
    problem = "Find the value of x if 2x + 3 = 7."
    anatomy = analyze(problem)
    assert isinstance(anatomy, dict)
    assert "reasoning_type" in anatomy
    return f"M1 ok; entities={len(anatomy.get('entities',[]))}, reasoning={anatomy.get('reasoning_type')}"


def test_cidi_smoke_test():
    """Full CIDI pipeline smoke test on a simple problem. Requires API."""
    from research.splitting.cidi.pipeline import split_cidi, CPP_DEEP_TARGET
    problem = "Find integers x and y such that x + y = 10 and x - y = 2."
    result = split_cidi(
        "smoke_test_001", problem,
        target_cpp=["A1","B1","C2"],
        n=2,
        max_retries=0,
        skip_validation=True,
    )
    assert result.split is not None
    assert len(result.split.packets) == 2
    return (f"CIDI smoke test ok: targeting_error={result.targeting_error:.2f}, "
            f"approved={result.approved}, pattern={result.split.pattern}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-api",  action="store_true", help="Skip LLM API calls")
    parser.add_argument("--full",    action="store_true", help="Include full smoke test")
    parser.add_argument("--verbose", action="store_true", help="Show tracebacks on failure")
    args = parser.parse_args()

    print("=" * 65)
    print("  CollabMath Pipeline Verification")
    print("=" * 65)

    # Deterministic checks (no API)
    check("M2 DAG closure (C2→prerequisites)",     test_dag_closure)
    check("M2 Upset property for all cells",        test_upset_property)
    check("M2 Structural feasibility checks",       test_feasibility_check)
    check("M3 Constraint table: 12 cells defined",  test_constraint_table)
    check("M3 build_constraints_summary",           test_build_constraints_summary)
    check("M5 Discriminator training (20 examples)",test_discriminator_training)
    check("M5 Discriminator predict",               test_discriminator_predict)
    check("M5 Save/load consistency",               test_full_discriminator_pipeline)
    check("Corpus loading (splits/convs/scores)",   test_corpus_loading)
    check("Problem selection (4 pilot problems)",   test_problem_selection)
    check("C1 baseline load",                       test_c1_loads)
    check("CPP annotator imports + classify_cdi",   test_annotator_imports)
    check("openai_utils: chat + chat_groq",         test_openai_utils_imports)
    check("Constitutional pipeline imports",        test_constitutional_imports)
    check("Monitor imports + detect_phase",         test_monitor_imports)
    check("Simulator imports",                      test_simulator_imports)
    check("CIDI pipeline imports + CPP_DEEP_TARGET",test_cidi_pipeline_imports)

    # API-required checks
    api_skip = args.no_api
    check("M1 Semantic analysis (API)",             test_semantic_analysis_api,
          skip=api_skip)
    check("CIDI smoke test (API, skip_validation)", test_cidi_smoke_test,
          skip=api_skip or not args.full)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    n_pass = sum(1 for _, s, _ in results if s == PASS)
    n_fail = sum(1 for _, s, _ in results if s == FAIL)
    n_skip = sum(1 for _, s, _ in results if s == SKIP)

    for label, status, note in results:
        note_str = f"  [{note}]" if note else ""
        print(f"  {status} {label}{note_str}")

    print("=" * 65)
    print(f"  PASS: {n_pass}  FAIL: {n_fail}  SKIP: {n_skip}")
    print("=" * 65)

    if n_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
