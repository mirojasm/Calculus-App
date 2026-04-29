"""
Route OpenAI calls to the correct API by model family.

  GPT-4.x        → Chat Completions API (client.chat.completions.create)
  GPT-5.x / o*   → Responses API       (client.responses.create)

Local vLLM override (set before running):
  LOCAL_MODEL_BASE_URL  — base URL of vLLM server  (e.g. http://localhost:8000/v1)
  LOCAL_MODEL_NAME      — model name passed to vLLM (default: Qwen/Qwen2.5-72B-Instruct-AWQ)
  KEEP_REMOTE_MODELS    — comma-separated model names that always stay on OpenAI
                          (default: gpt-4o-mini,gpt-4.1)
When LOCAL_MODEL_BASE_URL is set every model NOT in KEEP_REMOTE_MODELS is routed to the
local vLLM server using the Chat Completions protocol (vLLM does not expose the
Responses API).

Groq routing (cheap alternative to OpenAI for semantic/generation tasks):
  GROQ_API_KEY — if set, chat_groq() is available and used by CIDI M1/M4
  GROQ_MODEL   — model to use on Groq (default: llama-3.3-70b-versatile)
"""
import os
from typing import List
from openai import OpenAI
from research.config import CFG

# ── Remote (OpenAI) client ────────────────────────────────────────────────────
_client = OpenAI(api_key=CFG.openai_api_key)
_RESPONSES_PREFIXES = ("gpt-5", "o1", "o3", "o4")

# ── Local vLLM routing ────────────────────────────────────────────────────────
_LOCAL_BASE  = os.environ.get("LOCAL_MODEL_BASE_URL")
_LOCAL_MODEL = os.environ.get("LOCAL_MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct-AWQ")
_KEEP_REMOTE: set[str] = set(
    os.environ.get("KEEP_REMOTE_MODELS", "gpt-4o-mini,gpt-4.1").split(",")
)

_local_client: OpenAI | None = (
    OpenAI(base_url=_LOCAL_BASE, api_key="local") if _LOCAL_BASE else None
)

# ── Groq routing (OpenAI-compatible, cheap/fast for Llama) ───────────────────
_GROQ_KEY   = os.environ.get("GROQ_API_KEY")
_GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
_GROQ_BASE  = "https://api.groq.com/openai/v1"

_groq_client: OpenAI | None = (
    OpenAI(base_url=_GROQ_BASE, api_key=_GROQ_KEY) if _GROQ_KEY else None
)


def _route(model: str) -> tuple[OpenAI, str, bool]:
    """Return (client, effective_model_name, use_responses_api)."""
    if _local_client is not None and model not in _KEEP_REMOTE:
        return _local_client, _LOCAL_MODEL, False   # vLLM: Chat Completions only
    return _client, model, any(model.startswith(p) for p in _RESPONSES_PREFIXES)


def chat(
    messages: List[dict],
    model: str,
    temperature: float = 0.0,
    json_mode: bool = False,
    max_tokens: int = 2000,
) -> str:
    """
    Unified chat call. Accepts messages in Chat Completions format (role/content dicts).
    Routes to Responses API for GPT-5/o-series, or to a local vLLM server when
    LOCAL_MODEL_BASE_URL is set. Returns the assistant text content.
    """
    client, effective_model, use_responses = _route(model)

    if use_responses:
        # Responses API requires the word "json" somewhere in the input messages
        # when json_mode is enabled (unlike Chat Completions API).
        if json_mode:
            full_text = " ".join(
                m.get("content", "") if isinstance(m.get("content"), str) else ""
                for m in messages
            )
            if "json" not in full_text.lower():
                messages = list(messages)
                last = messages[-1]
                messages[-1] = {**last, "content": last.get("content", "") + "\n\nResponde en JSON."}
        kwargs: dict = dict(
            model=effective_model,
            input=messages,
            max_output_tokens=max_tokens,
        )
        if json_mode:
            kwargs["text"] = {"format": {"type": "json_object"}}
        resp = client.responses.create(**kwargs)
        return resp.output_text.strip()
    else:
        kwargs = dict(
            model=effective_model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=messages,
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content.strip()


def chat_groq(
    messages: List[dict],
    model: str = None,
    temperature: float = 0.1,
    json_mode: bool = False,
    max_tokens: int = 2000,
) -> str:
    """
    Route to Groq API (Llama 3.3 70B by default). Falls back to standard chat()
    if GROQ_API_KEY is not set.

    Costs roughly 100× less than GPT-4.1 for comparable quality on
    semantic analysis and structured generation tasks.
    """
    if _groq_client is None:
        return chat(messages, model or CFG.model_splitter, temperature, json_mode, max_tokens)

    effective_model = model if model and not model.startswith("gpt") else _GROQ_MODEL
    kwargs = dict(
        model=effective_model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=messages,
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = _groq_client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content.strip()
