"""
CIDI Module 4 — Linguistic generation of the split from a formal specification.

The LLM translates the algorithmic specification (entity assignments + cell
constraints) into natural-language packets. It is a translator, not a designer.
"""
from __future__ import annotations
import json, os, textwrap
from research.splitting.splitter import SplitResult, Packet

_GENERATION_SYSTEM = textwrap.dedent("""
Eres un traductor experto de especificaciones formales de splits jigsaw a lenguaje natural.

Se te proporciona:
1. Un problema matemático
2. Una anatomía estructural del problema (entidades, relaciones, sub-problemas)
3. Restricciones precisas derivadas algorítmicamente del perfil CPP objetivo
4. Instrucciones específicas de diseño por celda

Tu tarea: traducir fielmente esta especificación a un split jigsaw concreto en JSON.

REGLAS ABSOLUTAS:
- NO añadas restricciones que no estén en la especificación
- NO elimines restricciones presentes en la especificación
- NO asignes roles con nombres elaborados — usa nombres mínimos
- NO des protocolo de colaboración pre-especificado (a menos que C2/C3 lo requieran)
- SÍ asegúrate de que cada packet contiene suficiente información matemática para que
  el agente pueda contribuir matemáticamente — no solo describir su rol

Responde SOLO con JSON válido:
{
  "pattern": "SPLIT-A|B|C|D|E|F|G",
  "shared_context": "...",
  "agent_roles": [
    {"agent_id": 1, "role_name": "...", "role_description": "..."},
    {"agent_id": 2, "role_name": "...", "role_description": "..."}
  ],
  "packets": [
    {"agent_id": 1, "information": "..."},
    {"agent_id": 2, "information": "..."}
  ],
  "split_rationale": "...",
  "cidi_metadata": {
    "target_cells": [...],
    "design_rules_applied": ["..."]
  }
}
""").strip()


def generate(
    problem: str,
    anatomy: dict,
    constraints_summary: dict,
    n: int = 2,
) -> dict:
    """
    Generate a split JSON from the formal specification.
    Returns raw dict (not yet a SplitResult).
    """
    from research.openai_utils import chat, chat_groq
    from research.config import CFG

    # Build the specification message
    cell_instructions = "\n".join(
        f"  - Celda {cell}: {c['prompt_instruction']}"
        for cell, c in constraints_summary.get("cell_constraints", {}).items()
    )
    entity_hints = "\n".join(
        f"  - {hint}"
        for hint in constraints_summary.get("entity_hints", [])
    )
    pattern_hint = constraints_summary.get("dominant_pattern", "SPLIT-C")

    user_message = textwrap.dedent(f"""
    PROBLEMA:
    {problem}

    ANATOMÍA ESTRUCTURAL:
    - Sub-problemas: {json.dumps([sp.get('description','') for sp in anatomy.get('sub_problems',[])], ensure_ascii=False)}
    - Tipo de razonamiento: {anatomy.get('reasoning_type', ['algebraic'])}
    - Cuellos de botella informativos: {anatomy.get('information_bottlenecks', [])}
    - Ejes naturales de partición: {anatomy.get('natural_split_axes', [])}

    PATRÓN SUGERIDO: {pattern_hint}

    INSTRUCCIONES DE DISEÑO POR CELDA CPP OBJETIVO:
    {cell_instructions}

    HINTS DE ASIGNACIÓN DE ENTIDADES:
    {entity_hints}

    Genera el split para n={n} agentes.
    """).strip()

    # Prefer Groq for speed/cost if available, else use standard router
    use_groq = bool(os.environ.get("GROQ_API_KEY"))
    _fn = chat_groq if use_groq else chat
    model = (
        os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
        if use_groq
        else CFG.model_splitter
    )

    raw = _fn(
        messages=[
            {"role": "system", "content": _GENERATION_SYSTEM},
            {"role": "user",   "content": user_message},
        ],
        model=model,
        temperature=0.2,
        json_mode=True,
        max_tokens=4000,
    )
    return json.loads(raw)


def dict_to_split_result(
    raw: dict,
    problem_id: str,
    problem: str,
    n: int,
) -> SplitResult:
    """Convert the raw generated dict to a SplitResult."""
    roles = {r["agent_id"]: r for r in raw.get("agent_roles", [])}
    packets = [
        Packet(
            agent_id=p["agent_id"],
            information=p["information"],
            role_name=roles.get(p["agent_id"], {}).get("role_name", f"Agent {p['agent_id']}"),
            role_description=roles.get(p["agent_id"], {}).get("role_description", ""),
        )
        for p in raw.get("packets", [])
    ]
    return SplitResult(
        problem_id=problem_id,
        problem=problem,
        n=n,
        pattern=raw.get("pattern", "SPLIT-C"),
        shared_context=raw.get("shared_context", ""),
        packets=packets,
        valid=len(packets) == n,
        raw_split=raw,
    )
