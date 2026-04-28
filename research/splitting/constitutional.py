"""
Constitutional AI pipeline for jigsaw split generation (Condition C3).

3-stage iterative pipeline:
  Stage 1 — Generate initial split (standard splitter + Szewkis constraints)
  Stage 2 — Critic evaluates 24 checks (4 PISA phases × 6 Szewkis conditions)
  Stage 3 — Reviser improves split based on critique

Iterates up to MAX_ITER times; stops early when SQS >= APPROVAL_THRESHOLD.
Returns the best split found across all iterations.
"""
import json, textwrap
from dataclasses import dataclass, field
from typing import Optional

from research.config import CFG
from research.openai_utils import chat
from research.splitting.splitter import SplitResult, Packet, split as standard_split


MAX_ITER           = 3
APPROVAL_THRESHOLD = 0.80   # ≥ 80% of 24 checks to approve


# ── data classes ──────────────────────────────────────────────────────────────

@dataclass
class ConstitutionalResult:
    split:          SplitResult
    final_sqs:      float
    iterations:     int
    critique_history: list = field(default_factory=list)
    improvements_history: list = field(default_factory=list)
    approved:       bool = False


# ── prompts ───────────────────────────────────────────────────────────────────

_CRITIC_SYSTEM = textwrap.dedent("""
Eres un evaluador experto en Collaborative Problem Solving (CPS).
Se te proporciona un split jigsaw para {n} agentes y un problema matemático.

Evalúa si el split satisface las 6 condiciones de Szewkis en cada fase de resolución
PISA (A: Explorar, B: Formular, C: Ejecutar, D: Monitorear).

Las 6 condiciones de Szewkis:
S1 OBJETIVO COMÚN: ambos agentes trabajan hacia la misma meta explícita.
S2 INTERDEPENDENCIA POSITIVA: ningún agente puede resolver solo aunque reciba todo el info del otro.
S3 RESPONSABILIDAD INDIVIDUAL: cada agente tiene contribución única y necesaria.
S4 RECOMPENSA GRUPAL: el éxito se define como logro conjunto, no individual.
S5 CONCIENCIA GRUPAL: cada agente mantiene modelo mental del estado del otro.
S6 COORDINACIÓN Y COMUNICACIÓN: deben coordinar activamente — no trabajar en paralelo.

Para CADA combinación (fase PISA, condición Szewkis), razona imaginando qué ocurriría
si los agentes ejecutan fielmente este split. Responde si la condición se satisface.

Responde exclusivamente con JSON válido:
{{
  "evaluation": {{
    "A": {{"S1": {{"satisfied": bool, "critique": "..."}}, "S2": {{...}}, "S3": {{...}}, "S4": {{...}}, "S5": {{...}}, "S6": {{...}}}},
    "B": {{"S1": {{...}}, "S2": {{...}}, "S3": {{...}}, "S4": {{...}}, "S5": {{...}}, "S6": {{...}}}},
    "C": {{"S1": {{...}}, "S2": {{...}}, "S3": {{...}}, "S4": {{...}}, "S5": {{...}}, "S6": {{...}}}},
    "D": {{"S1": {{...}}, "S2": {{...}}, "S3": {{...}}, "S4": {{...}}, "S5": {{...}}, "S6": {{...}}}}
  }},
  "overall_sqs": 0.0,
  "critical_failures": ["descripción de los fallos más importantes"]
}}
Calcula overall_sqs como la fracción de los 24 checks que tienen satisfied=true.
""")

_REVISER_SYSTEM = textwrap.dedent("""
Eres un diseñador experto de actividades CPS matemáticas.

Se te proporciona:
1. Un split jigsaw con deficiencias identificadas
2. Una matriz de crítica con los fallos específicos por fase y condición

Tu tarea: mejorar el split resolviendo TODOS los fallos identificados.
No puedes reducir la calidad de los aspectos ya satisfechos.
Mantén el mismo problema y el mismo número de agentes.

Criterio de éxito: el split mejorado debe pasar los 24 checks (6 Szewkis × 4 fases).

Responde exclusivamente con JSON válido en el mismo formato que el split recibido.
Añade además el campo: "improvements_made": ["lista de cambios realizados"]
""")


# ── helpers ───────────────────────────────────────────────────────────────────

def _call(system: str, user: str, model: str = None, max_tokens: int = 4000) -> str:
    m = model or CFG.model_splitter
    return chat(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        model=m,
        temperature=0.2,
        json_mode=True,
        max_tokens=max_tokens,
    )


def _split_to_json(split: SplitResult) -> str:
    return json.dumps({
        "pattern":        split.pattern,
        "shared_context": split.shared_context,
        "agent_roles": [
            {"agent_id": p.agent_id, "role_name": p.role_name,
             "role_description": p.role_description}
            for p in split.packets
        ],
        "packets": [
            {"agent_id": p.agent_id, "information": p.information}
            for p in split.packets
        ],
        "split_rationale": split.raw_split.get("split_rationale", ""),
    }, ensure_ascii=False, indent=2)


def _critique(split: SplitResult, problem: str) -> dict:
    system = _CRITIC_SYSTEM.format(n=split.n)
    user   = f"Problema:\n{problem}\n\nSplit:\n{_split_to_json(split)}"
    raw    = _call(system, user, max_tokens=3000)
    return json.loads(raw)


def _revise(split: SplitResult, critique: dict, problem: str) -> SplitResult:
    user = (
        f"Problema:\n{problem}\n\n"
        f"Split actual:\n{_split_to_json(split)}\n\n"
        f"Críticas:\n{json.dumps(critique, ensure_ascii=False, indent=2)}"
    )
    raw      = _call(_REVISER_SYSTEM, user, max_tokens=4000)
    raw_dict = json.loads(raw)

    # Rebuild SplitResult from revised JSON
    roles = {r["agent_id"]: r for r in raw_dict.get("agent_roles", [])}
    packets = [
        Packet(
            agent_id=p["agent_id"],
            information=p["information"],
            role_name=roles.get(p["agent_id"], {}).get("role_name", f"Agent {p['agent_id']}"),
            role_description=roles.get(p["agent_id"], {}).get("role_description", ""),
        )
        for p in raw_dict.get("packets", [])
    ]
    return SplitResult(
        problem_id=split.problem_id,
        problem=split.problem,
        n=split.n,
        pattern=raw_dict.get("pattern", split.pattern),
        shared_context=raw_dict.get("shared_context", split.shared_context),
        packets=packets if len(packets) == split.n else split.packets,
        valid=True,
        raw_split=raw_dict,
    )


# ── public API ─────────────────────────────────────────────────────────────────

def constitutional_split(
    problem_id: str,
    problem:    str,
    n:          int = 2,
) -> ConstitutionalResult:
    """
    Generate a high-quality split via iterative constitutional critique.
    Returns ConstitutionalResult with the best split found.
    """
    # Stage 1: generate initial split (no extra validation — critic handles quality)
    current = standard_split(problem_id, problem, n, validate=True)

    best_split  = current
    best_sqs    = 0.0
    critique_history    = []
    improvements_history = []

    for i in range(MAX_ITER):
        # Stage 2: evaluate
        critique = _critique(current, problem)
        sqs      = float(critique.get("overall_sqs", 0.0))
        critique_history.append({"iteration": i, "sqs": sqs, "critique": critique})

        if sqs > best_sqs:
            best_sqs   = sqs
            best_split = current

        if sqs >= APPROVAL_THRESHOLD:
            return ConstitutionalResult(
                split=best_split,
                final_sqs=best_sqs,
                iterations=i + 1,
                critique_history=critique_history,
                improvements_history=improvements_history,
                approved=True,
            )

        # Stage 3: revise (only if more iterations remain)
        if i < MAX_ITER - 1:
            revised = _revise(current, critique, problem)
            improvements_history.append(
                revised.raw_split.get("improvements_made", [])
            )
            current = revised

    # Fallback: return best split found across all iterations
    return ConstitutionalResult(
        split=best_split,
        final_sqs=best_sqs,
        iterations=MAX_ITER,
        critique_history=critique_history,
        improvements_history=improvements_history,
        approved=best_sqs >= APPROVAL_THRESHOLD,
    )
