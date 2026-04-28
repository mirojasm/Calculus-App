"""
CIDI Module 3 — Partition Constraints: cell → information asymmetry type.

Pure Python, no LLM calls. Implements the table from framework_PIE_CPS.md §3.5.
"""
from __future__ import annotations

# Cell → asymmetry type mapping (framework_PIE_CPS.md §3.5)
CELL_ASYMMETRY: dict[str, dict] = {
    "A1": {
        "description": "Los agentes no conocen las capacidades del otro a priori",
        "design_rule": (
            "Cada paquete contiene información que revela las capacidades del agente "
            "de forma que el otro NO puede inferirlas sin preguntar. "
            "NO describir explícitamente qué puede hacer el partner."
        ),
        "test_fail_if": "Un agente puede inferir qué sabe el otro sin ningún intercambio",
        "prompt_instruction": (
            "Make each agent's expertise only discoverable through dialogue. "
            "Neither agent's packet should describe what the other agent knows."
        ),
    },
    "A2": {
        "description": "Scripts de colaboración distintos o ausentes",
        "design_rule": (
            "NO especificar protocolo de colaboración en ningún paquete. "
            "Los agentes deben negociar cómo trabajar juntos desde cero."
        ),
        "test_fail_if": "El protocolo de colaboración está pre-especificado",
        "prompt_instruction": (
            "Do not include any collaboration protocol or turn-taking instructions "
            "in either packet or the shared context."
        ),
    },
    "A3": {
        "description": "Roles implicados por la información pero no nombrados explícitamente",
        "design_rule": (
            "Los paquetes implican roles complementarios sin asignar nombres de roles. "
            "Los roles emergen del descubrimiento mutuo, no de la asignación a priori."
        ),
        "test_fail_if": "Los roles están nombrados explícitamente en los paquetes",
        "prompt_instruction": (
            "Do not name the roles explicitly. The role names in agent_roles should be "
            "minimal placeholders. The role should emerge from the information structure."
        ),
    },
    "B1": {
        "description": "Representaciones distintas del mismo objeto matemático",
        "design_rule": (
            "Un paquete da una representación del objeto central (algebraica, geométrica, "
            "tabular, gráfica, etc.). El otro paquete da una representación distinta del mismo "
            "objeto. Ninguno puede mapear la representación del otro sin diálogo."
        ),
        "test_fail_if": "Ambos agentes tienen la misma representación del objeto central",
        "prompt_instruction": (
            "Give Agent 1 one mathematical representation of the core object "
            "(e.g., algebraic formula) and Agent 2 a different representation "
            "(e.g., geometric interpretation or data table). "
            "Neither representation alone is sufficient."
        ),
    },
    "B2": {
        "description": "Lista de sub-tareas distribuida entre paquetes",
        "design_rule": (
            "Cada paquete identifica algunas sub-tareas necesarias pero NO todas. "
            "La lista completa de pasos requeridos solo emerge combinando ambos paquetes."
        ),
        "test_fail_if": "Un agente puede listar todos los pasos necesarios solo con su paquete",
        "prompt_instruction": (
            "Distribute the necessary solution steps across both packets. "
            "Agent 1 knows steps related to their domain; Agent 2 knows different steps. "
            "Together they cover all required steps."
        ),
    },
    "B3": {
        "description": "Ambigüedad genuina sobre distribución del trabajo en la ejecución",
        "design_rule": (
            "El split NO debe especificar quién calcula qué. "
            "Múltiples distribuciones de trabajo son válidas dado los paquetes; "
            "los agentes deben negociarla durante la fase B."
        ),
        "test_fail_if": "Es obvio quién ejecuta qué sin negociación",
        "prompt_instruction": (
            "Do not specify which agent performs which calculation. "
            "The work distribution should be genuinely ambiguous and require negotiation."
        ),
    },
    "C1": {
        "description": "Output del paso i de A es requerido por B para validar antes de continuar",
        "design_rule": (
            "Diseñar una cadena de dependencia explícita: antes de que B ejecute el paso k+1, "
            "necesita el resultado intermedio de A del paso k — y debe confirmarlo. "
            "La comunicación intermedia es matemáticamente necesaria."
        ),
        "test_fail_if": "Un agente puede ejecutar todos sus pasos sin comunicación intermedia",
        "prompt_instruction": (
            "Design the information so that Agent B needs Agent A's intermediate result "
            "to proceed to the next step. Communication is not optional."
        ),
    },
    "C2": {
        "description": "Output(A, paso_k) = Input(B, paso_{k+1}) — dependencia matemática directa",
        "design_rule": (
            "Crear una cadena matemática inquebrante: el resultado numérico/algebraico de A "
            "es el input exacto que B necesita para su cálculo. "
            "B literalmente no puede calcular sin el resultado de A."
        ),
        "test_fail_if": "B puede calcular su parte sin el resultado específico de A",
        "prompt_instruction": (
            "Create a direct mathematical dependency: Agent A computes a value that "
            "Agent B literally cannot compute without. The chain is mathematically unbreakable."
        ),
    },
    "C3": {
        "description": "Mecanismo de verificación de contribución equitativa",
        "design_rule": (
            "El shared_context debe indicar que ambos agentes contribuyen activamente "
            "en cada fase. Puede incluir un criterio de éxito que requiere contribución de ambos."
        ),
        "test_fail_if": "Un agente puede dominar toda la ejecución sin que el otro contribuya",
        "prompt_instruction": (
            "The shared context should make explicit that success requires active "
            "contribution from both agents at each phase."
        ),
    },
    "D1": {
        "description": "Ambigüedad semántica deliberada que emerge durante la ejecución",
        "design_rule": (
            "Introducir un término, símbolo, o condición que cada paquete "
            "interpreta ligeramente diferente. La divergencia se descubre durante C "
            "y requiere renegociación."
        ),
        "test_fail_if": "Las interpretaciones de ambos paquetes son idénticas sobre todos los objetos",
        "prompt_instruction": (
            "Include a mathematical concept or term that each agent's packet treats "
            "slightly differently. The divergence should become apparent during execution "
            "and require the agents to reconcile their understanding."
        ),
    },
    "D2": {
        "description": "Criterios de evaluación distribuidos entre agentes",
        "design_rule": (
            "Un paquete tiene el criterio de corrección matemática ('es correcto'). "
            "El otro tiene el criterio de suficiencia/completitud ('es suficiente y completo'). "
            "Ninguno puede evaluar el éxito completo sin el criterio del otro."
        ),
        "test_fail_if": "Un agente puede decidir solo si la solución es correcta Y suficiente",
        "prompt_instruction": (
            "Give Agent 1 the mathematical correctness criterion and Agent 2 the "
            "completeness/sufficiency criterion. Neither can fully evaluate success alone."
        ),
    },
    "D3": {
        "description": "Sorpresa estructural mid-problem que requiere re-distribución de roles",
        "design_rule": (
            "El problema contiene una cláusula o condición que se activa durante la ejecución "
            "y requiere que los agentes renegocien los roles o la estrategia. "
            "El plan inicial es insuficiente para completar la solución."
        ),
        "test_fail_if": "El plan inicial de los agentes puede completarse sin ninguna adaptación",
        "prompt_instruction": (
            "Include a structural twist in the problem that becomes relevant mid-solution "
            "and requires the agents to adapt their plan or role distribution."
        ),
    },
}


def get_constraint(cell: str) -> dict:
    """Return the asymmetry constraint spec for a PISA cell."""
    return CELL_ASYMMETRY.get(cell, {
        "description": f"No constraint defined for cell {cell}",
        "design_rule": "",
        "test_fail_if": "",
        "prompt_instruction": "",
    })


def build_constraints_summary(
    target_cells: list[str],
    anatomy: dict,
) -> dict:
    """
    Build the full constraint specification for a given target CPP.
    Returns a dict suitable for passing to Module 4 (generation).
    """
    cell_constraints = {}
    for cell in target_cells:
        constraint = get_constraint(cell)
        cell_constraints[cell] = {
            "design_rule": constraint["design_rule"],
            "test_fail_if": constraint["test_fail_if"],
            "prompt_instruction": constraint["prompt_instruction"],
        }

    # Derive entity assignment hints from anatomy + cell rules
    entity_hints = _infer_entity_hints(target_cells, anatomy)

    return {
        "target_cells": target_cells,
        "cell_constraints": cell_constraints,
        "entity_hints": entity_hints,
        "dominant_pattern": _infer_pattern(target_cells, anatomy),
    }


def _infer_entity_hints(target_cells: list[str], anatomy: dict) -> list[str]:
    """Heuristic hints about how to distribute entities between I_A and I_B."""
    hints = []
    axes = anatomy.get("natural_split_axes", [])
    bottlenecks = anatomy.get("information_bottlenecks", [])

    if "B1" in target_cells and axes:
        hints.append(f"Assign entities from one natural axis to Agent 1, the other to Agent 2: {axes[0]}")
    if "C2" in target_cells and bottlenecks:
        hints.append(f"Create explicit mathematical I/O chain around: {bottlenecks[0] if bottlenecks else 'key computation'}")
    if "D1" in target_cells:
        hints.append("Introduce a subtly ambiguous shared term that each agent interprets slightly differently")
    if "D2" in target_cells:
        hints.append("Give correctness criterion to Agent 1, completeness criterion to Agent 2")

    return hints


def _infer_pattern(target_cells: list[str], anatomy: dict) -> str:
    """Suggest the best split pattern (SPLIT-A..G) for the target CPP."""
    reasoning = anatomy.get("reasoning_type", [])
    cells_set = set(target_cells)

    if "C2" in cells_set and "B2" in cells_set and len(anatomy.get("sub_problems", [])) >= 2:
        return "SPLIT-D"  # multi-step chain best activates C2
    if "B1" in cells_set and "geometric" in reasoning:
        return "SPLIT-A"  # composite figure for geometry
    if "B1" in cells_set and len(reasoning) >= 2:
        return "SPLIT-B"  # dual representation
    if "D2" in cells_set:
        return "SPLIT-E"  # objective × constraints
    if "probabilistic" in reasoning or "combinatorial" in reasoning:
        return "SPLIT-F"
    return "SPLIT-C"  # default
