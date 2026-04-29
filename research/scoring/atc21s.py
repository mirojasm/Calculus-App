"""
ATC21S (Assessment and Teaching of 21st Century Skills) CPS scorer.

Dimensions scored per student message:
  PC — Participation & Contribution  (social: active engagement)
  C  — Communication                 (social: information sharing quality)
  Co — Collaboration                 (social: joint coordination)
  CR — Co-Regulation                 (cognitive: supporting partner's process)
  SR — Shared Regulation             (cognitive: joint metacognitive regulation)

Two scoring modes:
  1. score_conversation()      — message-level MoE (3 calls/message), used in n=30 paper
  2. annotate_conversation()   — conversation-level (1 call), quality 0-3 per dim, CQI-analogous

Quality scale (both modes):
  0 = absent        — dimension not present
  1 = superficial   — mechanical/scripted, motions without genuine coordination
  2 = functional    — genuine coordination, real information need
  3 = emergent      — new group capability neither agent had alone
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
    "PC": "Participation & Contribution — actively engages, contributes to the joint task",
    "C":  "Communication — shares information clearly and appropriately with partners",
    "Co": "Collaboration — coordinates actions, responds to partner contributions",
    "CR": "Co-Regulation — supports or scaffolds partner's reasoning or task management",
    "SR": "Shared Regulation — jointly monitors and regulates the group's collaborative process",
}


# ── prompts ──────────────────────────────────────────────────────────────────

_SOCIAL_EXPERT = textwrap.dedent("""
You are an expert in the ATC21S (Assessment and Teaching of 21st Century Skills) framework.

Assess the MESSAGE below on three SOCIAL dimensions. For each, assign:
  0 = absent
  1 = weak evidence
  2 = clear evidence
  3 = strong, elaborated evidence

Dimensions:
  PC — Participation & Contribution: Does the agent actively take part and add value?
  C  — Communication: Does the agent share information clearly and usefully?
  Co — Collaboration: Does the agent coordinate or respond to their partner's moves?

Consider the prior conversation context.

Return JSON:
{
  "PC": 0–3, "PC_rationale": "...",
  "C":  0–3, "C_rationale":  "...",
  "Co": 0–3, "Co_rationale": "..."
}
""")

_COGNITIVE_EXPERT = textwrap.dedent("""
You are an expert in the ATC21S (Assessment and Teaching of 21st Century Skills) framework.

Assess the MESSAGE below on two COGNITIVE/REGULATORY dimensions:
  CR — Co-Regulation: Does the agent help regulate or scaffold their partner's thinking?
       (e.g. asking guiding questions, pointing out errors in partner's reasoning)
  SR — Shared Regulation: Does the agent contribute to jointly monitoring progress,
       evaluating strategies, or adapting the group approach?

Assign 0–3 for each (0=absent, 1=weak, 2=clear, 3=strong).
Consider the prior conversation context.

Return JSON:
{
  "CR": 0–3, "CR_rationale": "...",
  "SR": 0–3, "SR_rationale": "..."
}
""")

_JUDGE = textwrap.dedent("""
You are a senior ATC21S CPS judge. Review the social and cognitive expert scores
for the message below. You may adjust scores that seem inconsistent with the context.

Return the final profile as JSON:
{
  "PC": 0–3,
  "C":  0–3,
  "Co": 0–3,
  "CR": 0–3,
  "SR": 0–3,
  "dominant_dimension": "PC" | "C" | "Co" | "CR" | "SR",
  "overall_quality": 1–3
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
    dominant_dimension: Optional[str] = None
    overall_quality:    Optional[int] = None

@dataclass
class ATC21SConversationScores:
    problem_id:   str
    condition:    str
    message_scores: List[ATC21SScore] = field(default_factory=list)
    dim_means:    Dict[str, float] = field(default_factory=dict)   # mean score per dim
    dim_presence: Dict[str, float] = field(default_factory=dict)   # % turns with score>0
    social_index:    float = 0.0   # mean(PC, C, Co) normalised 0–100
    cognitive_index: float = 0.0   # mean(CR, SR) normalised 0–100
    global_atc_index: float = 0.0  # mean of all 5 dims normalised 0–100


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

    social  = _call(_SOCIAL_EXPERT,   user_base)
    cogn    = _call(_COGNITIVE_EXPERT, user_base)

    judge_user = (
        f"{user_base}\n\n"
        f"Social expert scores: PC={social.get('PC')}, C={social.get('C')}, Co={social.get('Co')}\n"
        f"Cognitive expert scores: CR={cogn.get('CR')}, SR={cogn.get('SR')}"
    )
    verdict = _call(_JUDGE, judge_user)
    return verdict


# ── aggregation ───────────────────────────────────────────────────────────────

_DIMS = ["PC", "C", "Co", "CR", "SR"]

def _aggregate(scores: List[ATC21SScore]) -> dict:
    n = len(scores)
    if n == 0:
        return {d: 0 for d in _DIMS} | {
            "social_index": 0, "cognitive_index": 0, "global_atc_index": 0,
            "dim_presence": {d: 0 for d in _DIMS},
        }

    dim_sums = {d: sum(getattr(s, d) for s in scores) for d in _DIMS}
    dim_means = {d: dim_sums[d] / n for d in _DIMS}
    dim_presence = {d: sum(1 for s in scores if getattr(s, d) > 0) / n * 100 for d in _DIMS}

    # Normalise 0–100 (max raw score per message = 3)
    social_norm   = sum(dim_means[d] for d in ["PC", "C", "Co"]) / (3 * 3) * 100
    cognitive_norm = sum(dim_means[d] for d in ["CR", "SR"]) / (3 * 2) * 100
    global_norm   = sum(dim_means[d] for d in _DIMS) / (3 * 5) * 100

    return {
        "dim_means":      dim_means,
        "dim_presence":   dim_presence,
        "social_index":   social_norm,
        "cognitive_index": cognitive_norm,
        "global_atc_index": global_norm,
    }


# ── public API ────────────────────────────────────────────────────────────────

def score_conversation(conv: Conversation) -> ATC21SConversationScores:
    scored: List[ATC21SScore] = []

    for idx, turn in enumerate(conv.turns):
        context = _build_context(conv.turns, idx)
        v = score_message(turn, context)
        s = ATC21SScore(
            turn_index=idx,
            agent_id=turn.agent_id,
            content=turn.content,
            PC=int(v.get("PC", 0)),
            C=int(v.get("C",  0)),
            Co=int(v.get("Co", 0)),
            CR=int(v.get("CR", 0)),
            SR=int(v.get("SR", 0)),
            dominant_dimension=v.get("dominant_dimension"),
            overall_quality=v.get("overall_quality"),
        )
        scored.append(s)

    agg = _aggregate(scored)
    result = ATC21SConversationScores(
        problem_id=conv.problem_id,
        condition=conv.condition,
        message_scores=scored,
    )
    result.dim_means      = agg["dim_means"]
    result.dim_presence   = agg["dim_presence"]
    result.social_index   = agg["social_index"]
    result.cognitive_index = agg["cognitive_index"]
    result.global_atc_index = agg["global_atc_index"]
    return result


# ── conversation-level annotator (CQI-analogous) ──────────────────────────────

_CONV_ANNOTATOR_SYSTEM = textwrap.dedent("""
Eres un experto en el framework ATC21S (Assessment and Teaching of 21st Century Skills).

Lee la conversación COMPLETA entre agentes LLM resolviendo un problema matemático.
Para CADA una de las 5 dimensiones ATC21S asigna una calidad 0-3 para la conversación completa:

ESCALA DE CALIDAD:
0 = ausente      — la dimensión no ocurrió en la conversación.
1 = superficial  — la dimensión ocurrió pero de forma mecánica/guionada, sin coordinación real.
2 = funcional    — coordinación genuina; hubo necesidad real de la dimensión y se ejerció.
3 = emergente    — nueva capacidad grupal que ningún agente tenía por separado: síntesis,
                   corrección mutua, autorregulación conjunta genuina.

Dimensiones a evaluar:
PC — Participation & Contribution: ¿Los agentes participaron activamente y contribuyeron valor real al trabajo conjunto?
C  — Communication: ¿Los agentes compartieron información de forma clara, oportuna y útil para el otro?
Co — Collaboration: ¿Los agentes coordinaron acciones y respondieron a las contribuciones del otro?
CR — Co-Regulation: ¿Un agente apoyó o scaffoldeó el razonamiento o gestión del otro?
SR — Shared Regulation: ¿Los agentes monitorearon y regularon conjuntamente el proceso colaborativo?

Responde exclusivamente con JSON válido:
{{
  "dim_scores": {{
    "PC": 0|1|2|3,
    "C":  0|1|2|3,
    "Co": 0|1|2|3,
    "CR": 0|1|2|3,
    "SR": 0|1|2|3
  }},
  "rationale": {{
    "PC": "...", "C": "...", "Co": "...", "CR": "...", "SR": "..."
  }}
}}
""")

ATC21S_DIMS = ["PC", "C", "Co", "CR", "SR"]
ATC21S_SOCIAL_DIMS   = ["PC", "C", "Co"]
ATC21S_COGNITIVE_DIMS = ["CR", "SR"]


@dataclass
class ATC21SAnnotation:
    problem_id:  str
    condition:   str
    dim_scores:  Dict[str, int]    = field(default_factory=dict)   # dim → 0-3
    atc_cqi:     float             = 0.0   # Σ q_dim / (3*5) ∈ [0,1]
    social_qi:   float             = 0.0   # Σ q for {PC,C,Co} / 9
    cogn_qi:     float             = 0.0   # Σ q for {CR,SR} / 6
    rationale:   Dict[str, str]    = field(default_factory=dict)


def _compute_atc_cqi(dim_scores: Dict[str, int]) -> float:
    return sum(dim_scores.get(d, 0) for d in ATC21S_DIMS) / (3 * len(ATC21S_DIMS))


def _compute_social_qi(dim_scores: Dict[str, int]) -> float:
    return sum(dim_scores.get(d, 0) for d in ATC21S_SOCIAL_DIMS) / (3 * len(ATC21S_SOCIAL_DIMS))


def _compute_cogn_qi(dim_scores: Dict[str, int]) -> float:
    return sum(dim_scores.get(d, 0) for d in ATC21S_COGNITIVE_DIMS) / (3 * len(ATC21S_COGNITIVE_DIMS))


def annotate_conversation(conv: Conversation) -> ATC21SAnnotation:
    """Annotate a conversation with ATC21S quality scores (0-3 per dim, 1 LLM call)."""
    transcript = "\n".join(
        f"[Agent {t.agent_id}]: {t.content}" for t in conv.turns
    )
    messages = [
        {"role": "system", "content": _CONV_ANNOTATOR_SYSTEM},
        {"role": "user",   "content": f"Conversación:\n{transcript}"},
    ]
    raw = chat(
        messages=messages,
        model=CFG.model_scorer,
        temperature=0.0,
        json_mode=True,
        max_tokens=1000,
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
        rationale=data.get("rationale", {}),
    )
