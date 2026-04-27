import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# Load .env from the research/ directory if present
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

@dataclass
class Config:
    # --- OpenAI ---
    openai_api_key: str = field(default_factory=lambda: os.environ["OPENAI_API_KEY"])
    model_splitter:   str = "gpt-4.1"         # output-heavy (JSON); $2/8 per M wins
    model_validator:  str = "gpt-4o-mini"    # binary; avoids gpt-5.4-mini safety filter blocks
    model_simulator:  str = "gpt-5.4-mini"   # 400K ctx great for long convs; structured outputs
    model_scorer:     str = "gpt-5.4-mini"   # input-heavy scoring; JS scorer uses gpt-5.2 separately

    # --- MATH dataset ---
    dataset_id:     str = "EleutherAI/hendrycks_math"
    dataset_split:  str = "train"
    # Config names match HuggingFace subsets for EleutherAI/hendrycks_math
    subjects: List[str] = field(default_factory=lambda: [
        "algebra", "geometry", "precalculus",
        "counting_and_probability", "number_theory", "prealgebra"
    ])
    levels: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    problems_per_cell: int = field(
        default_factory=lambda: int(os.environ.get("COLLABMATH_PROBLEMS_PER_CELL", 5))
    )
    # 5 locally (150 problems), 20 on Sapelo2 (600 problems)
    # Override: export COLLABMATH_PROBLEMS_PER_CELL=20

    # --- Experiment conditions ---
    n_values: List[int] = field(default_factory=lambda: [1, 2, 3, 4])
    # condition names: "solo", "unrestricted_pair", "jigsaw_2", "jigsaw_3", "jigsaw_4"

    # --- Conversation ---
    max_turns:        int   = 20
    temperature_sim:  float = 0.7
    temperature_score: float = 0.0

    # --- Paths ---
    data_dir:    str = "outputs/data"
    results_dir: str = "outputs/results"
    scores_dir:  str = "outputs/scores"

CFG = Config()
