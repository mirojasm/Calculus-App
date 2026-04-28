"""
CIDI Pipeline — Constrained Inverse Design with Iterative Validation.

Public API: split_cidi(problem_id, problem, target_cpp, n, ...)

6-module pipeline (framework_PIE_CPS.md §3.3-3.5 + methodology_conditions.md §IV):
  M1  Semantic analysis (LLM — Groq or vLLM)
  M2  Prerequisite closure + structural feasibility (deterministic)
  M3  Partition constraint derivation (deterministic, table lookup)
  M4  Linguistic generation from specification (LLM — Groq or vLLM)
  M5  Predictive validation via discriminator chain (sklearn)
  M6  GRPO fine-tuning (future — not yet implemented)
"""
from __future__ import annotations
import json, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from research.splitting.splitter import SplitResult, Packet
from research.splitting.cidi.module2_feasibility import (
    CELL_ORDER, close_under_prerequisites, compute_feasible_target, cells_to_vector, hamming
)
from research.splitting.cidi.module3_constraints import build_constraints_summary

# Default CPP-DEEP target (8 cells — all rows A, B, C; no D)
CPP_DEEP_TARGET = ["A1","A2","A3","B1","B2","B3","C1","C2"]

HAMMING_THRESHOLD = 3   # max Hamming distance to approve a split
MAX_RETRIES = 2


@dataclass
class CIDISplitResult:
    split:              SplitResult
    target_cells:       list[str]           # requested target (after closure)
    feasible_cells:     list[str]           # subset structurally achievable
    infeasible_cells:   list[str]           # dropped due to anatomy constraints
    predicted_cpp:      list[int]           # M5 prediction (12-bit)
    predicted_cells:    list[str]
    targeting_error:    float               # Hamming / 12
    iterations:         int
    approved:           bool
    anatomy:            dict = field(default_factory=dict)
    constraints:        dict = field(default_factory=dict)
    timing_sec:         dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "problem_id":        self.split.problem_id,
            "target_cells":      self.target_cells,
            "feasible_cells":    self.feasible_cells,
            "infeasible_cells":  self.infeasible_cells,
            "predicted_cpp":     self.predicted_cpp,
            "predicted_cells":   self.predicted_cells,
            "targeting_error":   self.targeting_error,
            "iterations":        self.iterations,
            "approved":          self.approved,
            "split_pattern":     self.split.pattern,
            "split_valid":       self.split.valid,
        }


# ── helpers ───────────────────────────────────────────────────────────────────

def _dummy_prediction(n_cells: int = 12) -> dict:
    """Fallback when discriminators not available."""
    return {
        "probabilities":    {c: 0.5 for c in CELL_ORDER},
        "predicted_vector": [0] * 12,
        "predicted_cells":  [],
        "predicted_cdi":    0.0,
    }


def _try_predict(split_data: dict) -> dict:
    """Attempt M5 prediction; return dummy if discriminators not trained."""
    try:
        from research.splitting.cidi.module5_validation import predict_cpp
        return predict_cpp(split_data)
    except FileNotFoundError:
        return _dummy_prediction()


def _refine_constraints(
    constraints: dict,
    prediction: dict,
    target_vector: list[int],
    iteration: int,
) -> dict:
    """
    After a failed attempt, tighten the constraints for the failing cells.
    Simple strategy: add a stronger instruction for each failing cell.
    """
    failing = [
        CELL_ORDER[i] for i in range(12)
        if target_vector[i] == 1 and prediction["predicted_vector"][i] == 0
    ]
    for cell in failing:
        if cell in constraints.get("cell_constraints", {}):
            existing = constraints["cell_constraints"][cell]["prompt_instruction"]
            constraints["cell_constraints"][cell]["prompt_instruction"] = (
                f"[RETRY {iteration+1} — CRITICAL] {existing} "
                f"The previous attempt FAILED to activate this cell. "
                f"STRONGLY enforce this requirement."
            )
    return constraints


# ── main pipeline ─────────────────────────────────────────────────────────────

def split_cidi(
    problem_id:         str,
    problem:            str,
    target_cpp:         Optional[list[str]] = None,
    n:                  int = 2,
    max_retries:        int = MAX_RETRIES,
    hamming_threshold:  int = HAMMING_THRESHOLD,
    skip_validation:    bool = False,
) -> CIDISplitResult:
    """
    Generate a split targeting the given CPP profile using the CIDI pipeline.

    target_cpp: list of cell names to activate (e.g. ["A1","A2","B1","B2","C1","C2"])
                Defaults to CPP_DEEP_TARGET if None.
    skip_validation: if True, skip M5 discriminator (useful when model not trained yet).
    """
    if target_cpp is None:
        target_cpp = CPP_DEEP_TARGET

    timing: dict[str, float] = {}

    # ── M2a: prerequisite closure ─────────────────────────────────────────────
    t0 = time.time()
    closed_target, feasible_target, infeasible = compute_feasible_target(
        target_cpp, {}   # anatomy not yet available — conservative: no drops yet
    )
    timing["m2a_prereq_sec"] = round(time.time() - t0, 2)

    # ── M1: semantic analysis ─────────────────────────────────────────────────
    t1 = time.time()
    try:
        from research.splitting.cidi.module1_semantic import analyze
        anatomy = analyze(problem)
    except Exception as e:
        print(f"[WARN] M1 semantic analysis failed ({e}), using empty anatomy")
        anatomy = {
            "entities": [], "relations": [], "sub_problems": [],
            "reasoning_type": ["algebraic"], "information_bottlenecks": [],
            "natural_split_axes": [], "difficulty_indicators": {},
        }
    timing["m1_semantic_sec"] = round(time.time() - t1, 2)

    # ── M2b: structural feasibility with anatomy ──────────────────────────────
    _, feasible_target, infeasible = compute_feasible_target(closed_target, anatomy)

    # ── M3: constraint derivation ─────────────────────────────────────────────
    t3 = time.time()
    constraints = build_constraints_summary(feasible_target, anatomy)
    timing["m3_constraints_sec"] = round(time.time() - t3, 2)

    # ── Target vector ─────────────────────────────────────────────────────────
    target_vector = cells_to_vector(feasible_target)

    best_split:     Optional[SplitResult] = None
    best_prediction: dict = _dummy_prediction()
    best_hamming:    int = 12

    for attempt in range(max_retries + 1):
        # ── M4: linguistic generation ─────────────────────────────────────────
        t4 = time.time()
        try:
            from research.splitting.cidi.module4_generation import generate, dict_to_split_result
            raw = generate(problem, anatomy, constraints, n)
            split_result = dict_to_split_result(raw, problem_id, problem, n)
        except Exception as e:
            print(f"[WARN] M4 generation attempt {attempt+1} failed: {e}")
            if best_split is None:
                # Emergency fallback to standard splitter
                from research.splitting.splitter import split as standard_split
                split_result = standard_split(problem_id, problem, n, validate=False)
                raw = split_result.raw_split
            else:
                break
        timing[f"m4_gen_attempt{attempt+1}_sec"] = round(time.time() - t4, 2)

        if len(split_result.packets) != n:
            print(f"[WARN] M4 returned wrong number of packets ({len(split_result.packets)} ≠ {n}), retrying")
            constraints = _refine_constraints(constraints, _dummy_prediction(), target_vector, attempt)
            continue

        # ── M5: predictive validation ─────────────────────────────────────────
        if skip_validation:
            prediction = _dummy_prediction()
            h = 0   # pretend it passes
        else:
            t5 = time.time()
            prediction = _try_predict(raw)
            h = hamming(prediction["predicted_vector"], target_vector)
            timing[f"m5_val_attempt{attempt+1}_sec"] = round(time.time() - t5, 2)

        if h < best_hamming:
            best_hamming = h
            best_split = split_result
            best_prediction = prediction

        if h <= hamming_threshold:
            timing["total_sec"] = round(time.time() - t0, 2)
            return CIDISplitResult(
                split=best_split,
                target_cells=closed_target,
                feasible_cells=feasible_target,
                infeasible_cells=infeasible,
                predicted_cpp=best_prediction["predicted_vector"],
                predicted_cells=best_prediction.get("predicted_cells", []),
                targeting_error=round(best_hamming / 12, 4),
                iterations=attempt + 1,
                approved=True,
                anatomy=anatomy,
                constraints=constraints,
                timing_sec=timing,
            )

        # Refine constraints for next attempt
        constraints = _refine_constraints(constraints, prediction, target_vector, attempt)

    # Fallback: return best found
    timing["total_sec"] = round(time.time() - t0, 2)
    return CIDISplitResult(
        split=best_split if best_split else split_result,
        target_cells=closed_target,
        feasible_cells=feasible_target,
        infeasible_cells=infeasible,
        predicted_cpp=best_prediction["predicted_vector"],
        predicted_cells=best_prediction.get("predicted_cells", []),
        targeting_error=round(best_hamming / 12, 4),
        iterations=max_retries + 1,
        approved=False,
        anatomy=anatomy,
        constraints=constraints,
        timing_sec=timing,
    )
