"""
CIDI Module 4 — Linguistic generation of the split from a formal specification.

The LLM translates the algorithmic specification (entity assignments + cell
constraints) into natural-language packets. It is a translator, not a designer.
"""
from __future__ import annotations
import json, os, textwrap
from research.splitting.splitter import SplitResult, Packet

_GENERATION_SYSTEM = textwrap.dedent("""
Eres un diseñador experto de actividades jigsaw colaborativas para matemáticas (CPS).
Tu objetivo es crear splits de TAREA EN CADENA donde la colaboración emerge de la
estructura de la tarea, no de restricciones de conocimiento.

PRINCIPIO CENTRAL — DISEÑO POR ROL DE TAREA:
Asigna a cada agente una TAREA COMPUTACIONAL que produce un resultado que el otro necesita.
- NO escribas "no sabes X" ni "tu partner tiene X" — eso pre-computa la exploración y
  elimina Phase A del framework PISA CPS (Exploring & Understanding).
- SÍ escribe "tu tarea es computar X a partir de Y" — eso crea cadena funcional sin
  restringir el conocimiento del agente.
- La exploración de POR QUÉ el otro necesita tu output, y QUÉ necesitas tú del otro,
  debe emerger de la conversación (Phase A genuina).

ESTRUCTURA DE TAREA EN CADENA (Split-D):
1. Identifica la cadena de derivación natural del problema:
   paso_1 → resultado_intermedio → paso_2 → respuesta_final
2. Agente 1 recibe el input del paso 1 y la tarea de producir resultado_intermedio
3. Agente 2 recibe la tarea de aplicar paso_2, que requiere resultado_intermedio de A1
4. Ninguno puede completar la cadena sin el output del otro

ESTRUCTURA OBLIGATORIA DE CADA PAQUETE:
"Input: [información/datos disponibles para este agente]
Task: [qué debe derivar o computar]
Share: [qué resultado debe comunicar a su partner]
Needs from partner: [qué resultado necesita recibir para completar la cadena]"

REGLAS ABSOLUTAS:
- shared_context = SOLO la pregunta/objetivo final (sin datos del problema).
  Ejemplo correcto: "Find the integer m + n."
  Ejemplo incorrecto: "If sec θ + tan θ = 22/7, find m + n where csc θ + cot θ = m/n."
  (el segundo revela los datos — ambos agentes los verían y el split sería inútil)
- NUNCA incluyas en un paquete una fórmula o identidad que no sea parte del problema
  original o conocimiento matemático estándar verificable.
- La cadena debe requerir mínimo 2 intercambios secuenciales (no resuelve en 1 turno).
- NO des la solución completa a ningún agente.

VALIDACIÓN INTERNA (verifica antes de responder):
- ¿Puede el Agente 1 completar toda la cadena solo? Si SÍ, rediseña.
- ¿Puede el Agente 2 completar toda la cadena solo? Si SÍ, rediseña.
- ¿Combinar ambos produce la respuesta final? Si NO, rediseña.
- ¿Cada paquete describe una tarea, no una restricción de conocimiento? Si NO, rediseña.

Responde SOLO con JSON válido:
{
  "pattern": "SPLIT-A|B|C|D|E|F|G",
  "shared_context": "<SOLO la pregunta/objetivo — sin datos del problema>",
  "agent_roles": [
    {"agent_id": 1, "role_name": "...", "role_description": "..."},
    {"agent_id": 2, "role_name": "...", "role_description": "..."}
  ],
  "packets": [
    {"agent_id": 1, "information": "Input: ...\nTask: ...\nShare: ...\nNeeds from partner: ..."},
    {"agent_id": 2, "information": "Input: ...\nTask: ...\nShare: ...\nNeeds from partner: ..."}
  ],
  "split_rationale": "<una oración: qué cadena de derivación se usó y por qué crea interdependencia funcional>",
  "chain_verification": {
    "agent1_can_complete_alone": false,
    "agent2_can_complete_alone": false,
    "combined_produces_answer": true,
    "min_exchanges_required": 2
  },
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
