"""
CPP (Collaborative Phase Profile) annotator.

Given a conversation, produces:
  - cpp_vector: 12-bit binary vector (derived from quality_vector)
  - cdi:        CDI = sum(cpp_vector) / 12
  - quality_scores: quality 0-3 per cell (0=absent, 1=superficial, 2=functional, 3=emergent)
  - cqi:        CQI = Σ(q_i) / (3*12)  — quality-weighted CDI
  - phaq:       PhAQ = Σ q_i for {A1,A2,A3} / 9  — Phase A quality

Cells: A1 A2 A3 B1 B2 B3 C1 C2 C3 D1 D2 D3

Quality scale (mirrors ATC21S):
  0 = absent (cell did not fire)
  1 = superficial — exchange was scripted/mechanical, agents went through the motions
  2 = functional  — genuine exchange with real information need, authentic coordination
  3 = emergent    — new understanding neither agent had alone (synthesis, discovery, repair)
"""
import json, textwrap
from dataclasses import dataclass, field
from typing import List, Dict

from research.config import CFG
from research.simulation.simulator import Conversation
from research.openai_utils import chat


CELL_LABELS = ["A1","A2","A3","B1","B2","B3","C1","C2","C3","D1","D2","D3"]
PHASE_A_CELLS = ["A1", "A2", "A3"]

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
Para CADA una de las 12 celdas asigna una calidad de colaboración 0-3:

ESCALA DE CALIDAD:
0 = ausente — la celda no ocurrió en absoluto o fue puramente individual.
1 = superficial — la celda ocurrió pero de forma mecánica/guionada: los agentes
    ejecutaron el movimiento sin necesidad real del otro (p. ej. intercambio
    protocolar, cumplimiento de instrucciones, repetición sin síntesis).
2 = funcional — intercambio genuino: hubo una necesidad informacional real,
    los agentes coordinaron de forma auténtica y el output de uno fue input
    necesario del otro.
3 = emergente — nuevo entendimiento que ningún agente tenía por separado:
    síntesis, descubrimiento conjunto, corrección de error mutuo, o conclusión
    que no habría sido posible sin la colaboración.

Definiciones PISA por celda:
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
  "quality_vector": [<A1>, <A2>, <A3>, <B1>, <B2>, <B3>, <C1>, <C2>, <C3>, <D1>, <D2>, <D3>],
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
    problem_id:     str
    condition:      str
    cpp_vector:     List[int]         # 12-bit binary (1 if quality > 0)
    cdi:            float             # CDI = sum(cpp_vector) / 12
    cdi_label:      str
    quality_scores: Dict[str, int]    = field(default_factory=dict)  # cell → 0-3
    cqi:            float             = 0.0   # Σ q_i / (3*12) ∈ [0,1]
    phaq:           float             = 0.0   # Phase A quality: Σ q_A / 9 ∈ [0,1]
    rationale:      Dict[str, str]    = field(default_factory=dict)


def _compute_cqi(quality_scores: Dict[str, int]) -> float:
    return sum(quality_scores.get(c, 0) for c in CELL_LABELS) / (3 * 12)


def _compute_phaq(quality_scores: Dict[str, int]) -> float:
    return sum(quality_scores.get(c, 0) for c in PHASE_A_CELLS) / (3 * len(PHASE_A_CELLS))


# ── core ───────────────────────────────────────────────────────────────────────

def annotate(conv: Conversation) -> CPPAnnotation:
    """Annotate a conversation with CPP quality vector, CDI, CQI, and PhAQ."""
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

    raw_vec = [int(v) for v in data.get("quality_vector", [0] * 12)]
    quality_scores = {c: raw_vec[i] for i, c in enumerate(CELL_LABELS)}
    cpp_vector = [1 if q > 0 else 0 for q in raw_vec]
    cdi = sum(cpp_vector) / 12

    return CPPAnnotation(
        problem_id=conv.problem_id,
        condition=conv.condition,
        cpp_vector=cpp_vector,
        cdi=round(cdi, 4),
        cdi_label=classify_cdi(cdi),
        quality_scores=quality_scores,
        cqi=round(_compute_cqi(quality_scores), 4),
        phaq=round(_compute_phaq(quality_scores), 4),
        rationale=data.get("rationale", {}),
    )
