"""
PISA 2015 CPS scorer — mirrors analisis_multiagente_cps.js exactly.

For the research pipeline we call the validated JS scorer via a bridge
(see bridge.py) for real experiments on Sapelo.

This module provides:
  1. score_conversation_python()  — pure Python fallback (routes to Responses API for GPT-5.x)
  2. The aggregation formulas used in both paths (H_Int, process/competence shares)

The bridge.py module is the preferred path for experiments; this is kept
for unit tests and local development without Node.

Aggregation formula (from journal paper + JS implementation):
  S_ct     = N_ct / TC * 100                    (share %)
  Q*_ct    = avg_quality / 3 * 100              (normalised quality)
  H_Int_ct = 0.5 * S_ct + 0.5 * Q*_ct          (integrated index)
  Exception C2: weight_freq=0.4, weight_qual=0.6, freq normalised as count/10*100
  Global CPS index = mean(H_Int) over all 12 codes
"""
import json, math, textwrap
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from collections import Counter

from research.config import CFG
from research.simulation.simulator import Conversation, Turn
from research.openai_utils import chat

VALID_CODES = ["A1","A2","A3","B1","B2","B3","C1","C2","C3","D1","D2","D3"]

# Exact codebook from analisis_multiagente_cps.js
CPS_CODEBOOK = textwrap.dedent("""
Unidad de análisis: cada mensaje del agente evaluado recibe exactamente un código.
Los mensajes de los otros agentes solo se usan como contexto.

Competencias (1/2/3):
1 = conocimiento compartido / comunicación de significado.
2 = acción matemática / ejecución del trabajo cognitivo.
3 = organización, coordinación, compromisos, delegación, continuidad o cierre del trabajo.

Procesos cognitivos (A/B/C/D):
A = explorar y comprender lo que se ve o lo que pide la tarea.
B = representar y formular relaciones, variables o expresiones.
C = planificar y ejecutar cálculos o derivaciones.
D = monitorear, corregir, validar o reflexionar sobre el resultado/progreso.

Reglas de oro:
- Si el mensaje tiene números, fórmulas o derivaciones claras hechas por el agente para avanzar → C2.
- Si el mensaje describe lo que ve o informa rasgos observados → A1.
- Si formula una relación matemática, equivalencia, o traduce lo observado a lenguaje matemático → B1 o B2.
- Si valida, corrige o evalúa si el resultado tiene sentido → D1 o D2.
- Si delega, propone seguir, pregunta cómo continuar o cambia de tema → competencia 3.
""").strip()

QUALITY_GUIDE = textwrap.dedent("""
Escala de calidad (1-3):
1 = evidencia débil o mínima.
2 = evidencia suficiente y clara.
3 = evidencia fuerte, explícita, elaborada o reflexiva.
""").strip()


# ── data classes ──────────────────────────────────────────────────────────────

@dataclass
class PISAMessageScore:
    turn_index:       int
    agent_id:         int
    content:          str
    social_competence: Optional[str] = None   # 1/2/3
    cognitive_process: Optional[str] = None   # A/B/C/D
    final_code:       Optional[str] = None    # A1–D3
    quality_score:    Optional[int] = None    # 1–3
    needs_review:     bool = False
    reviewed:         bool = False

@dataclass
class PISAConversationScores:
    problem_id:       str
    condition:        str
    scored_agent_id:  int   # which agent was scored (usually the focal "student" agent)
    message_scores:   List[PISAMessageScore] = field(default_factory=list)
    # aggregated
    code_counts:      Dict[str, int]   = field(default_factory=dict)
    code_h_int:       Dict[str, float] = field(default_factory=dict)
    process_share:    Dict[str, float] = field(default_factory=dict)
    competence_share: Dict[str, float] = field(default_factory=dict)
    global_cps_index: float = 0.0
    richness_entropy: float = 0.0


# ── aggregation (matches JS calculateIntegratedMetrics exactly) ───────────────

def _entropy(counts: Dict[str, int]) -> float:
    total = sum(counts.values())
    if not total:
        return 0.0
    return -sum((c/total)*math.log2(c/total) for c in counts.values() if c > 0)


def aggregate_pisa_scores(
    message_scores: List[PISAMessageScore],
) -> dict:
    total = len(message_scores)
    code_counts: Dict[str, int] = Counter(
        s.final_code for s in message_scores if s.final_code in VALID_CODES
    )
    quality_by_code: Dict[str, List[int]] = {c: [] for c in VALID_CODES}
    for s in message_scores:
        if s.final_code in VALID_CODES and s.quality_score:
            quality_by_code[s.final_code].append(s.quality_score)

    h_int: Dict[str, float] = {}
    for code in VALID_CODES:
        cnt   = code_counts.get(code, 0)
        quals = quality_by_code[code]
        avg_q = sum(quals)/len(quals) if quals else 0
        Q_star = avg_q / 3 * 100

        if code == "C2":
            H_Freq_Norm = cnt / 10 * 100
            h_int[code] = 0.4 * H_Freq_Norm + 0.6 * Q_star
        else:
            share = cnt / total * 100 if total else 0
            h_int[code] = 0.5 * share + 0.5 * Q_star

    global_idx = sum(h_int.values()) / 12

    process_share: Dict[str, float] = {}
    for proc in "ABCD":
        cnt = sum(code_counts.get(f"{proc}{c}", 0) for c in "123")
        process_share[proc] = cnt / total * 100 if total else 0

    competence_share: Dict[str, float] = {}
    for comp in "123":
        cnt = sum(code_counts.get(f"{p}{comp}", 0) for p in "ABCD")
        competence_share[comp] = cnt / total * 100 if total else 0

    return {
        "code_counts":       dict(code_counts),
        "code_h_int":        h_int,
        "process_share":     process_share,
        "competence_share":  competence_share,
        "global_cps_index":  global_idx,
        "richness_entropy":  _entropy(dict(code_counts)),
    }


# ── Python fallback scorer (gpt-4o chat completions) ─────────────────────────
# Used for local dev and unit tests. For real experiments use bridge.py.

_SOCIAL_SYS = textwrap.dedent("""
Eres el Agente Social CPS. Decide la competencia dominante (1, 2 o 3).
1 = conocimiento compartido. 2 = acción matemática. 3 = organización/coordinación.
Responde JSON: {"competence":"1"|"2"|"3","rationale":"..."}
""")

_COGNITIVE_SYS = textwrap.dedent("""
Eres el Agente Cognitivo CPS. Decide el proceso cognitivo (A, B, C o D).
A=explorar. B=representar/formular. C=planificar/ejecutar. D=monitorear/validar.
Responde JSON: {"process":"A"|"B"|"C"|"D","rationale":"..."}
""")

_JUDGE_SYS = textwrap.dedent(f"""
Eres el Juez Sintetizador CPS.
Códigos válidos: {', '.join(VALID_CODES)}.
Combina las salidas del agente social y cognitivo en el código final.
Responde JSON:
{{"final_code":"C2","quality_score":1|2|3,"quality_note":"...","needs_review":true|false,
  "review_target":"none"|"social"|"cognitive"|"both","rationale":"..."}}
""")


def _call(system: str, user: str) -> dict:
    return json.loads(chat(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        model=CFG.model_scorer,
        temperature=0.0,
        json_mode=True,
    ))


def _ctx(turns: List[Turn], idx: int) -> str:
    window = turns[max(0, idx-5): idx]
    return "\n".join(f"[Agent {t.agent_id}]: {t.content}" for t in window) or "(start)"


def _score_one(turn: Turn, context: str, prev_codes: list) -> PISAMessageScore:
    base = (
        f"{CPS_CODEBOOK}\n\n"
        f"CONTEXTO PREVIO:\n{context}\n\n"
        f"DECISIONES PREVIAS: {json.dumps(prev_codes[-4:])}\n\n"
        f"MENSAJE A CODIFICAR:\n{turn.content}"
    )
    social  = _call(_SOCIAL_SYS,   base)
    cogn    = _call(_COGNITIVE_SYS, base)
    judge_u = f"{base}\n\nAgente Social: {json.dumps(social)}\nAgente Cognitivo: {json.dumps(cogn)}"
    verdict = _call(_JUDGE_SYS, judge_u)

    reviewed = False
    if verdict.get("needs_review") and verdict.get("review_target","none") != "none":
        # One review pass
        rt = verdict["review_target"]
        if rt in ("social","both"):
            social = _call(_SOCIAL_SYS, f"REVISIÓN.\n{base}\nOtro agente: {json.dumps(cogn)}")
        if rt in ("cognitive","both"):
            cogn = _call(_COGNITIVE_SYS, f"REVISIÓN.\n{base}\nOtro agente: {json.dumps(social)}")
        judge_u2 = f"{base}\n\nAgente Social (revisado): {json.dumps(social)}\nAgente Cognitivo (revisado): {json.dumps(cogn)}"
        verdict = _call(_JUDGE_SYS, judge_u2)
        reviewed = True

    return PISAMessageScore(
        turn_index=turn.agent_id,
        agent_id=turn.agent_id,
        content=turn.content,
        social_competence=social.get("competence"),
        cognitive_process=cogn.get("process"),
        final_code=verdict.get("final_code"),
        quality_score=verdict.get("quality_score"),
        needs_review=verdict.get("needs_review", False),
        reviewed=reviewed,
    )


def score_conversation_python(
    conv: Conversation,
    scored_agent_id: int = 1,
) -> PISAConversationScores:
    """
    Python fallback. Scores only turns by scored_agent_id.
    For N=2 jigsaw, score Agent 1 (the focal 'student' side).
    """
    agent_turns = [t for t in conv.turns if t.agent_id == scored_agent_id]
    all_turns   = conv.turns

    scored: List[PISAMessageScore] = []
    prev_codes = []
    for turn in agent_turns:
        idx = all_turns.index(turn)
        ctx = _ctx(all_turns, idx)
        s   = _score_one(turn, ctx, prev_codes)
        s.turn_index = idx
        scored.append(s)
        if s.final_code:
            prev_codes.append({"turn": idx, "code": s.final_code, "quality": s.quality_score})

    agg = aggregate_pisa_scores(scored)
    result = PISAConversationScores(
        problem_id=conv.problem_id,
        condition=conv.condition,
        scored_agent_id=scored_agent_id,
        message_scores=scored,
    )
    for k, v in agg.items():
        setattr(result, k, v)
    return result
