"""
CIDI Module 2 — Feasibility: DAG prerequisite closure + structural checks.

Pure Python, no LLM calls. Deterministic.
"""
from __future__ import annotations

CELL_ORDER = ["A1","A2","A3","B1","B2","B3","C1","C2","C3","D1","D2","D3"]

# DAG of cognitive prerequisites (from framework_PIE_CPS.md §3.3)
PREREQ_DAG: dict[str, list[str]] = {
    "A1": [],
    "A2": ["A1"],
    "A3": ["A1", "A2"],
    "B1": ["A1"],
    "B2": ["A2", "B1"],
    "B3": ["A3", "B2"],
    "C1": ["B1", "B2"],
    "C2": ["C1", "B2"],
    "C3": ["B3"],
    "D1": ["C1", "A1"],
    "D2": ["C2", "D1"],
    "D3": ["D2", "B3"],
}

# Minimum structural requirements in the problem anatomy for each cell
# Returns True if the anatomy satisfies the requirement
_STRUCTURAL_REQS: dict[str, any] = {
    # C2 needs at least 2 sequential sub-problems with I/O dependency
    "C2": lambda a: (
        len(a.get("sub_problems", [])) >= 2
        or len(a.get("information_bottlenecks", [])) >= 1
    ),
    # D1 needs some ambiguity or hidden connection in the problem
    "D1": lambda a: any(
        kw in " ".join(a.get("information_bottlenecks", [])).lower()
        for kw in ["conexión", "connection", "hidden", "oculta", "ambig", "share", "shared"]
    ),
    # D3 needs enough sub-problems to allow role re-distribution
    "D3": lambda a: len(a.get("sub_problems", [])) >= 3,
}


def close_under_prerequisites(target_cells: list[str]) -> list[str]:
    """
    Complete target into the minimal upset of the DAG that contains it.
    Returns sorted list (CELL_ORDER order).
    """
    closed: set[str] = set(target_cells)
    changed = True
    while changed:
        changed = False
        for cell in list(closed):
            for prereq in PREREQ_DAG.get(cell, []):
                if prereq not in closed:
                    closed.add(prereq)
                    changed = True
    return [c for c in CELL_ORDER if c in closed]


def check_structural_feasibility(
    cell: str, anatomy: dict
) -> tuple[bool, str]:
    """
    Verify that the problem anatomy supports activating `cell`.
    Returns (is_feasible, reason).
    """
    req = _STRUCTURAL_REQS.get(cell)
    if req is None:
        return True, "no structural constraint for this cell"
    feasible = req(anatomy)
    if feasible:
        return True, "ok"
    return False, f"Cell {cell} requires richer problem structure (see STRUCTURAL_REQS)"


def compute_feasible_target(
    target_cells: list[str], anatomy: dict
) -> tuple[list[str], list[str], list[str]]:
    """
    Given a target CPP and problem anatomy:
    1. Close under prerequisites
    2. Check structural feasibility per cell
    Returns:
        closed_target   — target after prerequisite closure
        feasible_target — cells that are both in closed_target and structurally feasible
        infeasible      — cells dropped due to structural infeasibility
    """
    closed = close_under_prerequisites(target_cells)
    feasible, infeasible = [], []
    for cell in closed:
        ok, _ = check_structural_feasibility(cell, anatomy)
        (feasible if ok else infeasible).append(cell)
    return closed, feasible, infeasible


def cells_to_vector(cells: list[str]) -> list[int]:
    """Convert list of active cell names to 12-bit binary vector (CELL_ORDER)."""
    active = set(cells)
    return [1 if c in active else 0 for c in CELL_ORDER]


def vector_to_cells(vector: list[int]) -> list[str]:
    """Convert 12-bit binary vector to list of active cell names."""
    return [c for c, v in zip(CELL_ORDER, vector) if v == 1]


def hamming(v1: list[int], v2: list[int]) -> int:
    return sum(abs(a - b) for a, b in zip(v1, v2))
