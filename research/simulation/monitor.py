"""
Szewkis Monitor for dynamic CPS facilitation (Condition C4).

After each PISA phase transition, evaluates whether the 6 Szewkis conditions
were maintained and optionally injects a corrective intervention.
"""
import json, textwrap
from dataclasses import dataclass, field
from typing import Optional

from research.config import CFG
from research.openai_utils import chat


# ── phase detection ───────────────────────────────────────────────────────────

PHASE_INDICATORS = {
    "A": ["tengo", "sé", "mi información", "me dieron", "mi parte es",
          "i have", "i know", "my information", "my part is", "i was given",
          "let me share", "what i see", "my role is"],
    "B": ["entonces el plan es", "podemos hacer", "yo haré", "tú haces",
          "propongo que", "el problema requiere",
          "so the plan", "we can do", "i will", "you do", "i propose",
          "the problem requires", "let's plan", "we need to"],
    "C": ["calculando", "si sustituyo", "obtengo", "el resultado es",
          "aplicando", "resolviendo", "deriving", "substituting",
          "computing", "calculating", "i get", "the result is", "applying",
          "solving", "let me compute", "if we plug"],
    "D": ["verificando", "comprobemos", "la respuesta es", "¿estás de acuerdo?",
          "revisando", "confirmo", "verifying", "let's check", "the answer is",
          "do you agree", "reviewing", "i confirm", "final answer",
          "does this match", "let's verify"],
}

MONITOR_PHASES = ("A", "B")   # only monitor after exploration and formulation


def detect_phase(conversation_history: list) -> str:
    """Return dominant PISA phase based on keyword scoring of last 3 turns."""
    recent = " ".join(t.get("content", "") for t in conversation_history[-3:]).lower()
    scores = {
        phase: sum(1 for kw in kws if kw in recent)
        for phase, kws in PHASE_INDICATORS.items()
    }
    return max(scores, key=scores.get) if any(scores.values()) else "A"


# ── monitor prompt ─────────────────────────────────────────────────────────────

_MONITOR_SYSTEM = textwrap.dedent("""
Eres un facilitador experto en Collaborative Problem Solving (CPS).
Observas una conversación entre {n} agentes LLM resolviendo un problema matemático.

Se acaba de completar la fase {current_phase} de la resolución.

Evalúa si las 6 condiciones de Szewkis se mantuvieron durante esta fase.
Razona exclusivamente sobre los turnos mostrados — no supongas nada fuera de ellos.

Condiciones a evaluar:
1. Objetivo común: ¿trabajaron ambos hacia la misma meta explícita?
2. Interdependencia positiva: ¿dependió cada uno del otro para avanzar?
3. Responsabilidad individual: ¿contribuyó activamente cada agente?
4. Recompensa grupal: ¿se orientaron al éxito del grupo, no individual?
5. Conciencia grupal: ¿se mantuvieron actualizados sobre lo que hacía el otro?
6. Coordinación: ¿coordinaron activamente o trabajaron en paralelo?

Si TODAS las condiciones se satisfacen responde:
{{"intervene": false, "sqs_phase": <float 0-1>, "failing_conditions": []}}

Si alguna condición falla responde:
{{
  "intervene": true,
  "sqs_phase": <float 0-1>,
  "failing_conditions": [<lista de números 1-6 que fallan>],
  "intervention": "<mensaje directo a los agentes, en primera persona plural, máximo 3 frases. Empieza con: 'Antes de continuar,'>"
}}
""")


# ── data class ─────────────────────────────────────────────────────────────────

@dataclass
class MonitorResult:
    phase:              str
    intervene:          bool
    sqs_phase:          float
    failing_conditions: list = field(default_factory=list)
    intervention:       Optional[str] = None


# ── core ───────────────────────────────────────────────────────────────────────

def _call(system: str, user: str) -> dict:
    raw = chat(
        messages=[{"role": "system", "content": system},
                  {"role": "user",   "content": user}],
        model=CFG.model_scorer,
        temperature=0.0,
        json_mode=True,
        max_tokens=600,
    )
    return json.loads(raw)


def evaluate_phase(
    phase: str,
    phase_turns: list,
    n: int,
    problem: str,
) -> MonitorResult:
    """
    Evaluate Szewkis conditions for a completed PISA phase.

    phase_turns: list of dicts with keys {"agent_id", "content"}
    """
    system = _MONITOR_SYSTEM.format(current_phase=phase, n=n)
    turn_text = "\n".join(
        f"[Agent {t.get('agent_id', '?')}]: {t.get('content', '')}"
        for t in phase_turns
    )
    user = f"Problema: {problem}\n\nConversación de la fase {phase}:\n{turn_text}"

    result = _call(system, user)
    return MonitorResult(
        phase=phase,
        intervene=result.get("intervene", False),
        sqs_phase=float(result.get("sqs_phase", 0.0)),
        failing_conditions=result.get("failing_conditions", []),
        intervention=result.get("intervention"),
    )
