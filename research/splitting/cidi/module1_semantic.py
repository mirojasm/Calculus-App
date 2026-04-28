"""
CIDI Module 1 — Semantic Analysis of the problem.

Extracts the mathematical anatomy: entities, relations, sub-problems,
reasoning type, information bottlenecks, and natural split axes.

Tries Groq (Llama 3.3 70B) first if GROQ_API_KEY is set, falls back
to the standard openai_utils.chat() router.
"""
from __future__ import annotations
import json, os, textwrap
from research.openai_utils import chat

_SEMANTIC_SYSTEM = textwrap.dedent("""
Eres un experto en análisis estructural de problemas matemáticos para diseño
de actividades de aprendizaje colaborativo.

Analiza el problema matemático y extrae su anatomía estructural.

Responde SOLO con JSON válido:
{
  "entities": [
    {"name": "...", "type": "variable|constant|function|set|expression|other",
     "description": "qué representa en el problema"}
  ],
  "relations": [
    {"between": ["entidad1", "entidad2"], "type": "equal|depends|bounds|defines|other",
     "description": "cómo se relacionan"}
  ],
  "sub_problems": [
    {"id": "sp1", "description": "sub-tarea o sub-cálculo necesario",
     "requires": ["entidades o relaciones necesarias"]}
  ],
  "reasoning_type": ["algebraic|geometric|probabilistic|combinatorial|analytical|number_theory|other"],
  "information_bottlenecks": [
    "descripción de cada punto donde se NECESITA información no derivable localmente"
  ],
  "natural_split_axes": [
    "descripción de cómo podría partirse naturalmente la información del problema"
  ],
  "difficulty_indicators": {
    "n_variables": 0,
    "n_steps_estimated": 0,
    "has_hidden_connection": false
  }
}
""").strip()


def analyze(problem: str) -> dict:
    """
    Extract the structural anatomy of a math problem.
    Returns a dict with entities, relations, sub_problems, etc.
    Uses Groq if GROQ_API_KEY is set, otherwise standard router.
    """
    from research.openai_utils import chat as _chat, chat_groq
    _fn = chat_groq if os.environ.get("GROQ_API_KEY") else _chat

    messages = [
        {"role": "system", "content": _SEMANTIC_SYSTEM},
        {"role": "user",   "content": f"Problema:\n{problem}"},
    ]

    from research.config import CFG
    model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile") \
            if os.environ.get("GROQ_API_KEY") else CFG.model_splitter

    raw = _fn(
        messages=messages,
        model=model,
        temperature=0.1,
        json_mode=True,
        max_tokens=2000,
    )

    try:
        anatomy = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback anatomy if parsing fails
        anatomy = {
            "entities": [],
            "relations": [],
            "sub_problems": [],
            "reasoning_type": ["algebraic"],
            "information_bottlenecks": [],
            "natural_split_axes": [],
            "difficulty_indicators": {
                "n_variables": 1,
                "n_steps_estimated": 2,
                "has_hidden_connection": False,
            },
        }

    return anatomy
