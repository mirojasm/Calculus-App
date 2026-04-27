"""
Bridge between the Python simulation pipeline and the validated JS scoring scripts.

Strategy:
  1. Convert Conversation objects → .txt files in the format analisis_multiagente_cps.js expects
  2. Run the JS scorer via subprocess
  3. Parse the CSV output back into PISAConversationScores / ATC21SConversationScores

This lets us reuse the production-validated scoring code (analisis_multiagente_cps.js)
without rewriting it, while keeping the rest of the pipeline in Python.

The .txt format expected by the JS scorer:
  🧑 Usuario: <student message>
  🤖 Chatbot: <chatbot response>
  🕒 DD/MM/YYYY HH:MM:SS | 🧩 <segment>

For multi-agent (N=2): Agent 1 → Usuario, Agent 2 → Chatbot
For multi-agent (N>2): We score each agent separately by rotating who is 'Usuario'
"""
import csv, json, os, subprocess, textwrap, tempfile
from pathlib import Path
from datetime import datetime
from typing import List, Optional

from research.simulation.simulator import Conversation, Turn
from research.scoring.pisa import PISAConversationScores, PISAMessageScore, aggregate_pisa_scores, VALID_CODES

REPO_ROOT = Path(__file__).parent.parent.parent
JS_SCORER = REPO_ROOT / "analisis_multiagente_cps.js"


# ── conversation → .txt ───────────────────────────────────────────────────────

def _pair_turns(turns: List[Turn], student_agent: int) -> list:
    """
    Pair turns into (user_turn, chatbot_turn) exchanges.
    student_agent's turns become 🧑 Usuario.
    All other agents' turns become 🤖 Chatbot.
    """
    pairs = []
    i = 0
    while i < len(turns):
        t = turns[i]
        if t.agent_id == student_agent:
            user_text = t.content
            bot_text  = ""
            if i + 1 < len(turns) and turns[i+1].agent_id != student_agent:
                bot_text = turns[i+1].content
                i += 1
            pairs.append((user_text, bot_text))
        i += 1
    return pairs


def conversation_to_txt(conv: Conversation, student_agent: int = 1) -> str:
    """
    Convert a Conversation to the .txt format expected by the JS scorer.
    """
    pairs  = _pair_turns(conv.turns, student_agent)
    lines  = []
    now    = datetime.now()
    for idx, (user, bot) in enumerate(pairs, start=1):
        lines.append(f"🧑 Usuario: {user}")
        if bot:
            lines.append(f"🤖 Chatbot: {bot}")
        ts = now.strftime("%d/%m/%Y %H:%M:%S")
        lines.append(f"🕒 {ts} | 🧩 {idx}")
        lines.append("")
    return "\n".join(lines)


# ── call JS scorer ────────────────────────────────────────────────────────────

def _run_js_scorer(
    txt_dir: str,
    output_dir: str,
    model: str = "gpt-4o",
    reasoning_effort: str = "low",
    human_codes_csv: str = "",
) -> int:
    cmd = [
        "node", str(JS_SCORER),
        "--input-dir",  txt_dir,
        "--output-dir", output_dir,
        "--model",      model,
        "--reasoning-effort", reasoning_effort,
        "--limit",      "9999",
    ]
    if human_codes_csv:
        cmd += ["--human-message-codes", human_codes_csv]

    env = os.environ.copy()
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"JS scorer failed:\n{result.stderr}")
    return result.returncode


# ── parse CSV output ──────────────────────────────────────────────────────────

def _parse_coded_csv(csv_path: str) -> List[dict]:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def _csv_rows_to_pisa(
    rows: List[dict],
    problem_id: str,
    condition: str,
    scored_agent_id: int,
) -> PISAConversationScores:
    message_scores = []
    for idx, row in enumerate(rows):
        code = row.get("final_code","")
        q    = row.get("quality_score","")
        message_scores.append(PISAMessageScore(
            turn_index=idx,
            agent_id=scored_agent_id,
            content=row.get("user_text",""),
            social_competence=row.get("social_competence",""),
            cognitive_process=row.get("cognitive_process",""),
            final_code=code if code in VALID_CODES else None,
            quality_score=int(q) if q and q.isdigit() else None,
            needs_review=row.get("judge_needs_review","").lower() == "true",
            reviewed=row.get("judge_second_pass_used","").lower() == "true",
        ))

    agg = aggregate_pisa_scores(message_scores)
    result = PISAConversationScores(
        problem_id=problem_id,
        condition=condition,
        scored_agent_id=scored_agent_id,
        message_scores=message_scores,
    )
    for k, v in agg.items():
        setattr(result, k, v)
    return result


# ── public API ────────────────────────────────────────────────────────────────

def score_conversations_via_js(
    conversations: List[Conversation],
    output_root: str,
    student_agent: int = 1,
    model: str = "gpt-4o",
    human_codes_csv: str = "",
) -> List[PISAConversationScores]:
    """
    Score a list of conversations using the validated JS multi-agent scorer.

    Writes .txt files to a temp dir, runs analisis_multiagente_cps.js,
    and parses the CSV output into PISAConversationScores objects.
    """
    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)

    txt_dir    = str(output_root_path / "txt_logs")
    scores_dir = str(output_root_path / "js_scores")
    Path(txt_dir).mkdir(parents=True, exist_ok=True)

    # Write .txt files — filename must match ^\d+\.txt$ for the JS parser
    id_map = {}
    for i, conv in enumerate(conversations, start=1):
        txt = conversation_to_txt(conv, student_agent=student_agent)
        fname = f"{i}.txt"
        (Path(txt_dir) / fname).write_text(txt, encoding="utf-8")
        id_map[f"{i}"] = conv

    _run_js_scorer(txt_dir, scores_dir, model=model, human_codes_csv=human_codes_csv)

    results: List[PISAConversationScores] = []
    coded_dir = Path(scores_dir) / "coded_logs"
    for fname, conv in zip(sorted(id_map.keys(), key=int), id_map.values()):
        csv_path = coded_dir / f"{fname}_coded_messages.csv"
        if not csv_path.exists():
            continue
        rows = _parse_coded_csv(str(csv_path))
        pisa = _csv_rows_to_pisa(rows, conv.problem_id, conv.condition, student_agent)
        results.append(pisa)

    return results
