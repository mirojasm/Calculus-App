"""
ATC21S (Assessment and Teaching of 21st Century Skills) CPS scorer — v2.

Dimensions (7 total, 3 subscales):

  SOCIAL (PC, C, Co)
    PC — Participation & Contribution  (active engagement)
    C  — Communication                 (information sharing quality)
    Co — Collaboration                 (joint coordination)

  COGNITIVE / REGULATORY (CR, SR)
    CR — Co-Regulation                 (scaffolding partner's reasoning)
    SR — Shared Regulation             (joint metacognitive monitoring)

  EPISTEMIC / KNOWLEDGE (KB, TD)  ← NEW in v2
    KB — Knowledge Building            (constructing NEW shared understanding)
    TD — Transactive Discussion        (explicitly building on partner's math reasoning)

Two scoring modes:
  1. score_conversation()      — message-level MoE (4 calls/message)
  2. annotate_conversation()   — conversation-level (1 call), quality 0-3 per dim

Quality scale:
  0 = absent        — dimension not present
  1 = superficial   — mechanical/scripted, motions without genuine coordination
  2 = functional    — genuine coordination, real information need
  3 = emergent      — new group capability neither agent had alone

Backward compatibility: old result files scored only 5 dims (PC,C,Co,CR,SR).
Missing KB/TD values default to 0 when loading old files.

References:
  Griffin, McGaw & Care (2012) Assessment and Teaching of 21st Century Skills.
  Stegmann et al. (2012) Transactive discussion in CSCL.
  ATC21S CPS rubric: social/cognitive/epistemic subscales.
"""
import json, textwrap, math
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from collections import Counter

from research.config import CFG
from research.simulation.simulator import Conversation, Turn
from research.openai_utils import chat


# ── framework definitions ─────────────────────────────────────────────────────

ATC21S_DIMENSIONS = {
    # Social
    "PC": "Participation & Contribution — actively engages, contributes to the joint task",
    "C":  "Communication — shares information clearly and appropriately with partners",
    "Co": "Collaboration — coordinates actions, responds to partner contributions",
    # Cognitive / Regulatory
    "CR": "Co-Regulation — supports or scaffolds partner's reasoning or task management",
    "SR": "Shared Regulation — jointly monitors and regulates the group's collaborative process",
    # Epistemic / Knowledge (v2)
    "KB": "Knowledge Building — constructs NEW shared mathematical understanding neither agent had alone",
    "TD": "Transactive Discussion — explicitly references & builds on partner's mathematical reasoning",
}

ATC21S_DIMS          = ["PC", "C", "Co", "CR", "SR", "KB", "TD"]
ATC21S_SOCIAL_DIMS   = ["PC", "C", "Co"]
ATC21S_COGNITIVE_DIMS = ["CR", "SR"]
ATC21S_EPISTEMIC_DIMS = ["KB", "TD"]   # v2


# ── prompts ──────────────────────────────────────────────────────────────────

_SOCIAL_EXPERT = textwrap.dedent("""
You are an expert in the ATC21S (Assessment and Teaching of 21st Century Skills) framework.

Assess the MESSAGE below on three SOCIAL dimensions. For each, assign:
  0 = absent  1 = weak evidence  2 = clear evidence  3 = strong, elaborated evidence

Dimensions:
  PC — Participation & Contribution: Does the agent actively take part and add value?
  C  — Communication: Does the agent share information clearly and usefully?
  Co — Collaboration: Does the agent coordinate or respond to their partner's moves?

Consider the prior conversation context.

Return JSON:
{"PC": 0-3, "PC_rationale": "...", "C": 0-3, "C_rationale": "...", "Co": 0-3, "Co_rationale": "..."}
""")

_COGNITIVE_EXPERT = textwrap.dedent("""
You are an expert in the ATC21S (Assessment and Teaching of 21st Century Skills) framework.

Assess the MESSAGE below on two COGNITIVE/REGULATORY dimensions:
  CR — Co-Regulation: Does the agent help regulate or scaffold their partner's thinking?
       (e.g. asking guiding questions, pointing out errors in partner's reasoning)
  SR — Shared Regulation: Does the agent contribute to jointly monitoring progress,
       evaluating strategies, or adapting the group approach?

Assign 0–3 for each (0=absent, 1=weak, 2=clear, 3=strong).

Return JSON:
{"CR": 0-3, "CR_rationale": "...", "SR": 0-3, "SR_rationale": "..."}
""")

_EPISTEMIC_EXPERT = textwrap.dedent("""
You are an expert in collaborative learning and the ATC21S framework.

Assess the MESSAGE below on two EPISTEMIC/KNOWLEDGE dimensions:

  KB — Knowledge Building: Does this message contribute to constructing NEW mathematical
       understanding that neither agent had independently? Look for synthesis, joint
       inference, or conclusions that emerge from combining both agents' information.
       (0=absent, 1=restates known info, 2=extends shared understanding, 3=genuine
       joint construction — new insight neither could reach alone)

  TD — Transactive Discussion: Does this message explicitly reference AND build upon
       the partner's mathematical reasoning? Mere acknowledgment doesn't count —
       the agent must engage with the partner's math, extend it, correct it, or use it
       as a premise for new reasoning.
       (0=absent, 1=mentions partner's idea, 2=uses partner's idea as input,
       3=substantially extends or transforms partner's mathematical argument)

Consider the prior conversation context carefully.

Return JSON:
{"KB": 0-3, "KB_rationale": "...", "TD": 0-3, "TD_rationale": "..."}
""")

_JUDGE = textwrap.dedent("""
You are a senior ATC21S CPS judge. Review the expert scores for the message below.
You may adjust scores that seem inconsistent with the context.

Return the final profile as JSON:
{
  "PC": 0-3, "C": 0-3, "Co": 0-3,
  "CR": 0-3, "SR": 0-3,
  "KB": 0-3, "TD": 0-3,
  "dominant_dimension": "PC"|"C"|"Co"|"CR"|"SR"|"KB"|"TD",
  "overall_quality": 1-3
}
""")


# ── data classes ─────────────────────────────────────────────────────────────

@dataclass
class ATC21SScore:
    turn_index: int
    agent_id:   int
    content:    str
    PC: int = 0
    C:  int = 0
    Co: int = 0
    CR: int = 0
    SR: int = 0
    KB: int = 0    # v2
    TD: int = 0    # v2
    dominant_dimension: Optional[str] = None
    overall_quality:    Optional[int] = None

@dataclass
class ATC21SConversationScores:
    problem_id:      str
    condition:       str
    message_scores:  List[ATC21SScore] = field(default_factory=list)
    dim_means:       Dict[str, float]  = field(default_factory=dict)
    dim_presence:    Dict[str, float]  = field(default_factory=dict)
    social_index:    float = 0.0   # {PC,C,Co}   / (3×3) × 100
    cognitive_index: float = 0.0   # {CR,SR}      / (3×2) × 100
    epistemic_index: float = 0.0   # {KB,TD}      / (3×2) × 100  — v2
    global_atc_index: float = 0.0  # all 7 dims  / (3×7) × 100


# ── core scoring ──────────────────────────────────────────────────────────────

def _call(system: str, user: str) -> dict:
    return json.loads(chat(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        model=CFG.model_scorer,
        temperature=CFG.temperature_score,
        json_mode=True,
    ))


def _build_context(turns: List[Turn], idx: int) -> str:
    window = turns[max(0, idx - 6): idx]
    return "\n".join(f"[Agent {t.agent_id}]: {t.content}" for t in window) or "(start)"


def score_message(turn: Turn, context: str) -> dict:
    user_base = f"PRIOR CONTEXT:\n{context}\n\nMESSAGE TO SCORE:\n{turn.content}"

    social    = _call(_SOCIAL_EXPERT,    user_base)
    cogn      = _call(_COGNITIVE_EXPERT, user_base)
    epistemic = _call(_EPISTEMIC_EXPERT, user_base)

    judge_user = (
        f"{user_base}\n\n"
        f"Social: PC={social.get('PC')}, C={social.get('C')}, Co={social.get('Co')}\n"
        f"Cognitive: CR={cogn.get('CR')}, SR={cogn.get('SR')}\n"
        f"Epistemic: KB={epistemic.get('KB')}, TD={epistemic.get('TD')}"
    )
    return _call(_JUDGE, judge_user)


# ── aggregation ───────────────────────────────────────────────────────────────

def _aggregate(scores: List[ATC21SScore]) -> dict:
    n = len(scores)
    if n == 0:
        return {d: 0 for d in ATC21S_DIMS} | {
            "social_index": 0, "cognitive_index": 0,
            "epistemic_index": 0, "global_atc_index": 0,
            "dim_presence": {d: 0 for d in ATC21S_DIMS},
        }

    dim_sums  = {d: sum(getattr(s, d, 0) for s in scores) for d in ATC21S_DIMS}
    dim_means = {d: dim_sums[d] / n for d in ATC21S_DIMS}
    dim_pres  = {d: sum(1 for s in scores if getattr(s, d, 0) > 0) / n * 100 for d in ATC21S_DIMS}

    social_norm    = sum(dim_means[d] for d in ATC21S_SOCIAL_DIMS)    / (3 * 3) * 100
    cognitive_norm = sum(dim_means[d] for d in ATC21S_COGNITIVE_DIMS) / (3 * 2) * 100
    epistemic_norm = sum(dim_means[d] for d in ATC21S_EPISTEMIC_DIMS) / (3 * 2) * 100
    global_norm    = sum(dim_means[d] for d in ATC21S_DIMS)           / (3 * 7) * 100

    return {
        "dim_means": dim_means, "dim_presence": dim_pres,
        "social_index": social_norm, "cognitive_index": cognitive_norm,
        "epistemic_index": epistemic_norm, "global_atc_index": global_norm,
    }


# ── public API (message-level) ────────────────────────────────────────────────

def score_conversation(conv: Conversation) -> ATC21SConversationScores:
    scored: List[ATC21SScore] = []
    for idx, turn in enumerate(conv.turns):
        context = _build_context(conv.turns, idx)
        v = score_message(turn, context)
        scored.append(ATC21SScore(
            turn_index=idx, agent_id=turn.agent_id, content=turn.content,
            PC=int(v.get("PC", 0)), C=int(v.get("C", 0)),  Co=int(v.get("Co", 0)),
            CR=int(v.get("CR", 0)), SR=int(v.get("SR", 0)),
            KB=int(v.get("KB", 0)), TD=int(v.get("TD", 0)),
            dominant_dimension=v.get("dominant_dimension"),
            overall_quality=v.get("overall_quality"),
        ))

    agg = _aggregate(scored)
    result = ATC21SConversationScores(problem_id=conv.problem_id, condition=conv.condition,
                                      message_scores=scored)
    result.dim_means       = agg["dim_means"]
    result.dim_presence    = agg["dim_presence"]
    result.social_index    = agg["social_index"]
    result.cognitive_index = agg["cognitive_index"]
    result.epistemic_index = agg["epistemic_index"]
    result.global_atc_index = agg["global_atc_index"]
    return result


# ── conversation-level annotator (1 LLM call, CQI-analogous) ─────────────────

_CONV_ANNOTATOR_SYSTEM = textwrap.dedent("""
Eres un experto en el framework ATC21S (Assessment and Teaching of 21st Century Skills).

Lee la conversación COMPLETA entre agentes LLM resolviendo un problema matemático.
Para CADA una de las 7 dimensiones ATC21S asigna una calidad 0-3:

ESCALA:
0 = ausente      — la dimensión no ocurrió en la conversación.
1 = superficial  — ocurrió de forma mecánica/guionada, sin necesidad real.
2 = funcional    — coordinación genuina; necesidad real de la dimensión.
3 = emergente    — nueva capacidad grupal que ningún agente tenía por separado.

DIMENSIONES SOCIALES:
PC — Participation & Contribution: ¿Ambos participaron activamente y contribuyeron valor real?
C  — Communication: ¿Compartieron información de forma clara, oportuna y útil?
Co — Collaboration: ¿Coordinaron acciones y respondieron a las contribuciones del otro?

DIMENSIONES COGNITIVAS / REGULATORIAS:
CR — Co-Regulation: ¿Un agente apoyó o scaffoldeó el razonamiento o gestión del otro?
SR — Shared Regulation: ¿Monitorearon y regularon conjuntamente el proceso colaborativo?

DIMENSIONES EPISTÉMICAS (v2):
KB — Knowledge Building: ¿Construyeron comprensión matemática NUEVA que ninguno tenía
     por separado? Busca síntesis, inferencias conjuntas, conclusiones que emergen de
     combinar las informaciones de ambos agentes.
TD — Transactive Discussion: ¿Los agentes explícitamente referenciaron Y construyeron sobre
     el razonamiento matemático del otro, yendo más allá de solo compartir datos?
     Cuenta: extender el argumento del otro, corregir su razonamiento, usarlo como premisa.
     No cuenta: solo mencionar o agradecer.

Responde exclusivamente con JSON válido:
{{
  "dim_scores": {{
    "PC": 0|1|2|3, "C": 0|1|2|3, "Co": 0|1|2|3,
    "CR": 0|1|2|3, "SR": 0|1|2|3,
    "KB": 0|1|2|3, "TD": 0|1|2|3
  }},
  "rationale": {{
    "PC": "...", "C": "...", "Co": "...",
    "CR": "...", "SR": "...",
    "KB": "...", "TD": "..."
  }}
}}
""")


@dataclass
class ATC21SAnnotation:
    problem_id:    str
    condition:     str
    dim_scores:    Dict[str, int] = field(default_factory=dict)   # dim → 0-3
    atc_cqi:       float = 0.0   # Σ all 7 dims / (3×7) ∈ [0,1]
    social_qi:     float = 0.0   # {PC,C,Co} / 9
    cogn_qi:       float = 0.0   # {CR,SR} / 6
    epistemic_qi:  float = 0.0   # {KB,TD} / 6  — v2
    rationale:     Dict[str, str] = field(default_factory=dict)


def _compute_atc_cqi(dim_scores: Dict[str, int]) -> float:
    present = [d for d in ATC21S_DIMS if d in dim_scores]
    if not present:
        return 0.0
    return sum(dim_scores.get(d, 0) for d in ATC21S_DIMS) / (3 * len(ATC21S_DIMS))


def _compute_social_qi(dim_scores: Dict[str, int]) -> float:
    return sum(dim_scores.get(d, 0) for d in ATC21S_SOCIAL_DIMS) / (3 * len(ATC21S_SOCIAL_DIMS))


def _compute_cogn_qi(dim_scores: Dict[str, int]) -> float:
    return sum(dim_scores.get(d, 0) for d in ATC21S_COGNITIVE_DIMS) / (3 * len(ATC21S_COGNITIVE_DIMS))


def _compute_epistemic_qi(dim_scores: Dict[str, int]) -> float:
    return sum(dim_scores.get(d, 0) for d in ATC21S_EPISTEMIC_DIMS) / (3 * len(ATC21S_EPISTEMIC_DIMS))


def annotate_conversation(conv: Conversation) -> ATC21SAnnotation:
    """Score a full conversation on all 7 ATC21S dimensions (1 LLM call)."""
    transcript = "\n".join(f"[Agent {t.agent_id}]: {t.content}" for t in conv.turns)
    raw = chat(
        messages=[
            {"role": "system", "content": _CONV_ANNOTATOR_SYSTEM},
            {"role": "user",   "content": f"Conversación:\n{transcript}"},
        ],
        model=CFG.model_scorer,
        temperature=0.0,
        json_mode=True,
        max_tokens=1200,
    )
    data = json.loads(raw)
    dim_scores = {d: int(data.get("dim_scores", {}).get(d, 0)) for d in ATC21S_DIMS}

    return ATC21SAnnotation(
        problem_id=conv.problem_id,
        condition=conv.condition,
        dim_scores=dim_scores,
        atc_cqi=round(_compute_atc_cqi(dim_scores), 4),
        social_qi=round(_compute_social_qi(dim_scores), 4),
        cogn_qi=round(_compute_cogn_qi(dim_scores), 4),
        epistemic_qi=round(_compute_epistemic_qi(dim_scores), 4),
        rationale=data.get("rationale", {}),
    )


def annotate_from_dict(conv_dict: dict) -> ATC21SAnnotation:
    """Annotate from a stored result dict (no Conversation object needed)."""
    from research.simulation.simulator import Turn
    turns_raw = conv_dict.get("conversation", {}).get("turns", [])
    turns = [Turn(agent_id=t["agent_id"], role="assistant", content=t["content"])
             for t in turns_raw]

    class _FakeConv:
        def __init__(self):
            self.problem_id = conv_dict.get("problem_id", "")
            self.condition  = conv_dict.get("condition", "")
            self.turns = turns

    return annotate_conversation(_FakeConv())
