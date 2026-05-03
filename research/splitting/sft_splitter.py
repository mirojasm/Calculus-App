"""
SFT-based split generator — drop-in replacement for the CIDI pipeline.

Uses the fine-tuned Mistral-7B SFT adapter (outputs/training/split_adapter/final_adapter)
to generate jigsaw splits directly from a problem, with no API calls.

Cost comparison:
  CIDI pipeline  : ~$0.02–0.05 per split (GPT-4 / Groq, M1 + M4)
  SFT splitter   : ~$0.00 (local GPU inference, ~10s on A100)

Quality:
  Structural validity : 96.6% (n=333 eval, 1 sample) / 97.7% (3 samples)
  CDI ≥ 0.5 rate      : 53.6% (n=28 test problems, 3 samples)

Usage (Sapelo):
  export SFT_BASE_MODEL=/scratch/$USER/llm/models/hf/Mistral-7B-Instruct-v0.3
  export SFT_ADAPTER_PATH=outputs/training/split_adapter/final_adapter
  python -m research.run_experiment split --splitter sft

The model is loaded lazily on first call and cached for the process lifetime.
"""
from __future__ import annotations
import json, os, re, textwrap
from pathlib import Path
from typing import Optional

from research.splitting.splitter import SplitResult, Packet

# ── configuration ─────────────────────────────────────────────────────────────

DEFAULT_BASE    = "/scratch/mir85108/llm/models/hf/Mistral-7B-Instruct-v0.3"
DEFAULT_ADAPTER = "outputs/training/split_adapter/final_adapter"

REQUIRED_KEYS  = {"pattern", "shared_context", "packets", "interdependence_check"}
VALID_PATTERNS = {"SPLIT-A","SPLIT-B","SPLIT-C","SPLIT-D","SPLIT-E","SPLIT-F","SPLIT-G"}

_SPLIT_SYSTEM = textwrap.dedent("""\
You are an expert collaborative learning designer for mathematics education.

Given a math problem, create a 2-agent jigsaw split that satisfies these conditions:
1. COMMON GOAL — both agents share an explicit objective (shared_context).
2. POSITIVE INTERDEPENDENCE — neither agent can solve the problem alone with only
   their packet; together they can.
3. INFORMATION PURITY — packets contain only raw mathematical data/conditions,
   never task instructions, never "you need your partner for X".
4. BALANCE — packets have roughly equal informational weight.

Choose the split PATTERN that creates the strongest interdependence:
  SPLIT-C: Complementary conditions (equations/constraints split across agents)
  SPLIT-D: Multi-step chain (agent A computes intermediate; agent B needs it)
  SPLIT-B: Dual representation (same object, different representations)
  SPLIT-A: Composite figure (geometry: each agent sees one component)
  SPLIT-F: Sample space × counting principle (probability/combinatorics)
  SPLIT-G: Hypothesis × key lemma (proofs/number theory)
  SPLIT-E: Objective × constraints (optimization)

Return ONLY valid JSON:
{
  "pattern": "<SPLIT-A|B|C|D|E|F|G>",
  "shared_context": "<goal statement — NO mathematical data>",
  "packets": [
    {"agent_id": 1, "information": "<raw data/condition for agent 1>"},
    {"agent_id": 2, "information": "<raw data/condition for agent 2>"}
  ],
  "interdependence_check": {
    "agent1_can_answer_alone": false,
    "agent2_can_answer_alone": false,
    "combined_can_answer": true
  }
}
""").strip()

# ── singleton model state ─────────────────────────────────────────────────────

_model     = None
_tokenizer = None


def _load() -> None:
    global _model, _tokenizer
    if _model is not None:
        return

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    base_model   = os.environ.get("SFT_BASE_MODEL",    DEFAULT_BASE)
    adapter_path = os.environ.get("SFT_ADAPTER_PATH",  DEFAULT_ADAPTER)

    print(f"[SFT-splitter] Loading base model: {base_model}")
    _tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token
    _tokenizer.padding_side = "left"

    _model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.float16, trust_remote_code=True
    ).cuda()
    _model = PeftModel.from_pretrained(_model, str(adapter_path))
    _model.eval()
    print(f"[SFT-splitter] Adapter loaded: {adapter_path}")


# ── JSON parsing + validation ─────────────────────────────────────────────────

def _try_json(s: str) -> Optional[dict]:
    try:
        return json.loads(s)
    except Exception:
        return None


def _parse_robust(text: str) -> Optional[dict]:
    if d := _try_json(text.strip()):
        return d
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m and (d := _try_json(m.group(1))):
        return d
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m and (d := _try_json(m.group(0))):
        return d
    return None


def _validate(d: dict) -> bool:
    pkts = d.get("packets", [])
    ic   = d.get("interdependence_check") or {}
    return (
        REQUIRED_KEYS.issubset(d.keys())
        and d.get("pattern") in VALID_PATTERNS
        and isinstance(pkts, list) and len(pkts) == 2
        and all(isinstance(p, dict) and p.get("information") for p in pkts)
        and isinstance(ic, dict)
        and ic.get("agent1_can_answer_alone") is False
        and ic.get("agent2_can_answer_alone") is False
        and ic.get("combined_can_answer") is True
    )


# ── generation ────────────────────────────────────────────────────────────────

def _generate_raw(problem: str, n_samples: int = 3) -> Optional[dict]:
    import torch
    _load()

    messages = [
        {"role": "system", "content": _SPLIT_SYSTEM},
        {"role": "user",   "content": f"PROBLEM:\n{problem}"},
    ]
    prompt = _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = _tokenizer(prompt, return_tensors="pt", truncation=True,
                        max_length=1024).to("cuda")

    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=(n_samples > 1),
            temperature=0.7 if n_samples > 1 else 1.0,
            num_return_sequences=n_samples,
            pad_token_id=_tokenizer.eos_token_id,
        )

    prompt_len = inputs["input_ids"].shape[1]
    texts = [
        _tokenizer.decode(out[prompt_len:], skip_special_tokens=True)
        for out in outputs
    ]

    # Return first fully-valid sample; fall back to first parseable
    for text in texts:
        d = _parse_robust(text)
        if d and _validate(d):
            return d
    for text in texts:
        d = _parse_robust(text)
        if d:
            return d
    return None


# ── public API ────────────────────────────────────────────────────────────────

def generate_split(
    problem_id: str,
    problem:    str,
    n_samples:  int = 3,
) -> SplitResult:
    """
    Generate a jigsaw split using the local SFT model.

    Returns a SplitResult compatible with the rest of the pipeline.
    If generation fails, returns an invalid SplitResult with empty packets.
    """
    raw = _generate_raw(problem, n_samples)

    if raw is None:
        return SplitResult(
            problem_id=problem_id,
            problem=problem,
            n=2,
            pattern="SPLIT-C",
            shared_context="",
            packets=[],
            valid=False,
            raw_split={},
        )

    packets = [
        Packet(agent_id=p["agent_id"], information=p["information"])
        for p in raw.get("packets", [])
    ]
    return SplitResult(
        problem_id=problem_id,
        problem=problem,
        n=2,
        pattern=raw.get("pattern", "SPLIT-C"),
        shared_context=raw.get("shared_context", ""),
        packets=packets,
        valid=_validate(raw),
        raw_split=raw,
    )
