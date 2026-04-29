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
  },
  "expected_answer_format": {
    "type": "integer|decimal|fraction|algebraic_expression|equation|set|multiple|other",
    "specification": "Your final answer should be [descripción exacta del formato y valor esperado, inferida del enunciado]",
    "partial_credit_indicators": [
      "valor o forma intermedia que indica progreso correcto aunque no sea la respuesta final"
    ]
  }
}

Para expected_answer_format.type usa:
- integer: cuando se pide un número entero (e.g. find m+n, find the value of k)
- decimal: cuando se espera un decimal (e.g. round to 2 decimal places)
- fraction: cuando la respuesta es una fracción (e.g. express as p/q in lowest terms)
- algebraic_expression: cuando la respuesta es una expresión con variables
- equation: cuando la respuesta es una ecuación o función
- set: cuando la respuesta es un conjunto o lista de valores
- multiple: cuando el problema pide varias cantidades separadas
- other: cualquier otro caso

Para specification: escribe la frase concisa que irá en el goal-anchor de la actividad
colaborativa, especificando el formato exacto. Ejemplos:
- "Your final answer must be a single integer."
- "Your final answer must be a reduced fraction p/q."
- "Your final answer must be an algebraic expression in terms of x."
- "Your final answer must be two values: the width and height."

Para partial_credit_indicators: lista valores intermedios que evidencian razonamiento
correcto aunque no sean la respuesta final (e.g. si la respuesta es m+n=44 donde
csc+cot=29/15, entonces "29/15" es un indicador parcial válido).
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
            "expected_answer_format": {
                "type": "other",
                "specification": "",
                "partial_credit_indicators": [],
            },
        }

    return anatomy


def extract_answer_format(problem: str) -> dict:
    """
    Lightweight call to M1 that returns only the expected_answer_format field.
    Used by conditions C3/C4 that don't run the full CIDI pipeline.
    """
    anatomy = analyze(problem)
    return anatomy.get("expected_answer_format", {
        "type": "other",
        "specification": "",
        "partial_credit_indicators": [],
    })
