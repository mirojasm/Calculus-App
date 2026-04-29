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
Tu objetivo es crear splits de INFORMACIÓN PURA donde la colaboración emerge de la
estructura informacional del problema, no de instrucciones sobre qué computar.

PRINCIPIO CENTRAL — FRAMING MÍNIMO:
Cada agente recibe SOLO la información (datos, expresiones, condiciones) que le corresponde.
NO se le dice qué computar, qué compartir, ni qué pedirle al partner.
La colaboración debe emerger de la conversación: los agentes deben descubrir por sí mismos
qué tiene el otro y qué necesitan intercambiar. Eso es Phase A genuina (PISA CPS).

PROHIBIDO en cada paquete:
- "Task: ...", "Share: ...", "Needs from partner: ..." — prescribir la colaboración la elimina.
- "No sabes X", "tu partner tiene Y" — pre-computa el descubrimiento.
- Fórmulas o identidades que el problema no menciona — es ayuda, no información.
- La solución o cualquier paso intermedio de la solución.

OBLIGATORIO:
- El paquete de cada agente = solo los datos, expresiones o condiciones que ese agente posee.
- shared_context = SOLO la pregunta/objetivo final, sin ningún dato del problema.
  Correcto: "Find the integer m + n."
  Incorrecto: "If sec θ + tan θ = 22/7, find m + n..." (revela datos a ambos agentes)

VALIDACIÓN DE INTERDEPENDENCIA (verifica antes de responder):
- ¿Puede el Agente 1 responder al shared_context solo con su información? Si SÍ, rediseña.
- ¿Puede el Agente 2 responder al shared_context solo con su información? Si SÍ, rediseña.
- ¿Combinando la información de ambos se puede responder al shared_context? Si NO, rediseña.

La prueba clave de interdependencia: cada agente, viendo solo su packet y el shared_context,
debería concluir "necesito saber qué tiene mi partner" antes de poder avanzar.

Responde SOLO con JSON válido:
{
  "pattern": "SPLIT-A|B|C|D|E|F|G",
  "shared_context": "<SOLO la pregunta/objetivo — sin datos del problema>",
  "agent_roles": [
    {"agent_id": 1, "role_name": "...", "role_description": "..."},
    {"agent_id": 2, "role_name": "...", "role_description": "..."}
  ],
  "packets": [
    {"agent_id": 1, "information": "<solo los datos/expresiones/condiciones de A1>"},
    {"agent_id": 2, "information": "<solo los datos/expresiones/condiciones de A2>"}
  ],
  "split_rationale": "<una oración: qué información se separó y por qué ninguno puede responder sin el otro>",
  "interdependence_check": {
    "agent1_can_answer_alone": false,
    "agent2_can_answer_alone": false,
    "combined_can_answer": true
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

    EJES DE PARTICIÓN INFORMACIONAL:
    {entity_hints}

    CPP OBJETIVO (para guiar qué información activará cada celda):
    {cell_instructions}

    Genera el split de INFORMACIÓN PURA para n={n} agentes.
    Recuerda: cada paquete = solo los datos que ese agente posee. Sin tareas, sin instrucciones.
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
