"""
CPP (Collaborative Phase Profile) annotator.

Given a conversation, produces a 12-bit binary vector over the PISA 2015 CPS
4×3 matrix and computes CDI (CPS Depth Index).

Cells: A1 A2 A3 B1 B2 B3 C1 C2 C3 D1 D2 D3
CDI = sum(vector) / 12
"""
import json, textwrap
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from research.config import CFG
from research.simulation.simulator import Conversation
from research.openai_utils import chat


CELL_LABELS = ["A1","A2","A3","B1","B2","B3","C1","C2","C3","D1","D2","D3"]

CPP_PROFILES = {
    "CPP-Ø":    (0.00, 0.08,  "Solo information transfer, no collaboration"),
    "CPP-IC":   (0.08, 0.25,  "Information coordination only (A1-A2 active)"),
    "CPP-CG":   (0.25, 0.42,  "Common goal established (adds B1-B2)"),
    "CPP-RP":   (0.42, 0.58,  "Role play with task division (adds C1-C2)"),
    "CPP-DEEP": (0.58, 0.83,  "Deep collaboration (adds D1-D2, A3/B3)"),
    "CPP-FULL": (0.83, 1.01,  "Full 12-cell activation"),
}


def classify_cdi(cdi: float) -> str:
    for label, (lo, hi, _) in CPP_PROFILES.items():
        if lo <= cdi < hi:
            return label
    return "CPP-FULL"


# ── prompt ─────────────────────────────────────────────────────────────────────

_ANNOTATOR_SYSTEM = textwrap.dedent("""
Eres un experto en la matriz PISA 2015 CPS (4 procesos × 3 competencias = 12 celdas).

Lee la conversación entre agentes LLM resolviendo un problema matemático.
Para CADA una de las 12 celdas determina si la colaboración fue NECESARIA en esa celda:
una celda está activa (1) si ningún agente pudo completar esa operación sin input del otro.
Una celda está inactiva (0) si un agente pudo hacer esa operación solo o no ocurrió en absoluto.

Celdas a evaluar (usa las definiciones PISA oficiales):
A1: ¿Necesitaron descubrir perspectivas/habilidades del otro para avanzar?
A2: ¿Establecieron juntos las normas de interacción (quién lidera, cómo verifican)?
A3: ¿Los roles emergieron de exploración conjunta en vez de estar pre-asignados?
B1: ¿Negociaron explícitamente cómo representar o enmarcar el problema?
B2: ¿Identificar las sub-tareas requirió contribución activa de ambos?
B3: ¿Negociaron la distribución del trabajo durante la ejecución?
C1: ¿Comunicaron las acciones a realizar antes de ejecutarlas y recibieron confirmación?
C2: ¿Hay pasos de ejecución matemática que requirieron el output del otro como input?
C3: ¿Siguieron reglas de participación o se promovieron mutuamente activamente?
D1: ¿Monitorearon y repararon el entendimiento compartido cuando había divergencia?
D2: ¿Evaluaron conjuntamente el éxito de las acciones tomadas?
D3: ¿Adaptaron roles u organización en respuesta a lo que ocurrió en la conversación?

Responde exclusivamente con JSON válido:
{{
  "cpp_vector": [<A1>, <A2>, <A3>, <B1>, <B2>, <B3>, <C1>, <C2>, <C3>, <D1>, <D2>, <D3>],
  "cdi": <float 0-1>,
  "cdi_label": "<CPP-Ø|CPP-IC|CPP-CG|CPP-RP|CPP-DEEP|CPP-FULL>",
  "rationale": {{
    "A1": "...", "A2": "...", "A3": "...",
    "B1": "...", "B2": "...", "B3": "...",
    "C1": "...", "C2": "...", "C3": "...",
    "D1": "...", "D2": "...", "D3": "..."
  }}
}}
""")


# ── data class ─────────────────────────────────────────────────────────────────

@dataclass
class CPPAnnotation:
    problem_id:  str
    condition:   str
    cpp_vector:  List[int]          # 12-bit binary vector
    cdi:         float              # CDI = sum / 12
    cdi_label:   str
    rationale:   Dict[str, str] = field(default_factory=dict)


# ── core ───────────────────────────────────────────────────────────────────────

def annotate(conv: Conversation) -> CPPAnnotation:
    """Annotate a conversation with its 12-bit CPP vector and CDI."""
    transcript = "\n".join(
        f"[Agent {t.agent_id}]: {t.content}" for t in conv.turns
    )
    messages = [
        {"role": "system", "content": _ANNOTATOR_SYSTEM},
        {"role": "user",   "content": f"Conversación:\n{transcript}"},
    ]
    raw = chat(
        messages=messages,
        model=CFG.model_scorer,
        temperature=0.0,
        json_mode=True,
        max_tokens=2000,
    )
    data = json.loads(raw)

    vec = [int(v) for v in data.get("cpp_vector", [0] * 12)]
    cdi = sum(vec) / 12
    return CPPAnnotation(
        problem_id=conv.problem_id,
        condition=conv.condition,
        cpp_vector=vec,
        cdi=round(cdi, 4),
        cdi_label=classify_cdi(cdi),
        rationale=data.get("rationale", {}),
    )
