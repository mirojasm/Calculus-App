"""
Evaluate the SFT split generator adapter on the held-out test set.

Metrics:
  - valid_json_rate      : % outputs parseable as JSON
  - struct_valid_rate    : % outputs with all required CIDI keys + 2 packets
  - interdep_correct_rate: % outputs with correct interdependence_check booleans
  - fully_valid_rate     : % outputs passing all checks
  - pattern_distribution : which SPLIT-X patterns the model generates

For each problem, generates --n-samples splits and keeps the best valid one
(sample-then-filter pipeline). If none are valid, keeps the first attempt.

Output
------
  outputs/eval/split_eval.jsonl    — per-problem results
  outputs/eval/split_eval_summary.json — aggregate metrics + 5 sample outputs

Usage
-----
  python -m research.experiments.eval_split_generator
  python -m research.experiments.eval_split_generator --n-samples 3 --show 5
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path

TEST_JSONL   = Path("outputs/training/split_dpo_test.jsonl")
ADAPTER_PATH = Path("outputs/training/split_adapter/final_adapter")
OUT_DIR      = Path("outputs/eval")
OUT_RESULTS  = OUT_DIR / "split_eval.jsonl"
OUT_SUMMARY  = OUT_DIR / "split_eval_summary.json"

DEFAULT_BASE = "/scratch/mir85108/llm/models/hf/Mistral-7B-Instruct-v0.3"

REQUIRED_KEYS  = {"pattern", "shared_context", "packets", "interdependence_check"}
VALID_PATTERNS = {"SPLIT-A", "SPLIT-B", "SPLIT-C", "SPLIT-D", "SPLIT-E", "SPLIT-F", "SPLIT-G"}


# ── JSON parsing ──────────────────────────────────────────────────────────────

def _parse_json(text: str) -> dict | None:
    """Parse JSON from model output, handling markdown code blocks."""
    for candidate in [
        text.strip(),
        re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL),
        re.search(r"\{.*\}", text, re.DOTALL),
    ]:
        if candidate is None:
            continue
        s = candidate if isinstance(candidate, str) else candidate.group(1 if hasattr(candidate, 'group') and candidate.lastindex else 0)
        try:
            return json.loads(s)
        except Exception:
            pass
    return None


def _parse_json_robust(text: str) -> dict | None:
    if d := _try_json(text.strip()):
        return d
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m and (d := _try_json(m.group(1))):
        return d
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m and (d := _try_json(m.group(0))):
        return d
    return None


def _try_json(s: str) -> dict | None:
    try:
        return json.loads(s)
    except Exception:
        return None


# ── structural validation ─────────────────────────────────────────────────────

def _validate(d: dict) -> dict:
    pkts = d.get("packets", [])
    ic   = d.get("interdependence_check") or {}
    return {
        "has_required_keys":     REQUIRED_KEYS.issubset(d.keys()),
        "valid_pattern":         d.get("pattern") in VALID_PATTERNS,
        "has_two_packets":       isinstance(pkts, list) and len(pkts) == 2,
        "packets_have_info":     all(isinstance(p, dict) and p.get("information") for p in pkts),
        "interdep_correct":      (
            isinstance(ic, dict)
            and ic.get("agent1_can_answer_alone") is False
            and ic.get("agent2_can_answer_alone") is False
            and ic.get("combined_can_answer")     is True
        ),
    }


# ── generation ────────────────────────────────────────────────────────────────

def _load_model(base_model: str, adapter_path: Path):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    print(f"[INFO] Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.float16, trust_remote_code=True
    ).cuda()
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()
    print(f"[INFO] Adapter loaded from: {adapter_path}")
    return model, tokenizer


def _generate_split(model, tokenizer, system: str, problem: str,
                    n_samples: int, max_new_tokens: int) -> list[str]:
    import torch
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": f"PROBLEM:\n{problem}"},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=1024).to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(n_samples > 1),
            temperature=0.7 if n_samples > 1 else 1.0,
            num_return_sequences=n_samples,
            pad_token_id=tokenizer.eos_token_id,
        )
    prompt_len = inputs["input_ids"].shape[1]
    return [
        tokenizer.decode(out[prompt_len:], skip_special_tokens=True)
        for out in outputs
    ]


# ── main evaluation ───────────────────────────────────────────────────────────

def evaluate(
    base_model:     str  = DEFAULT_BASE,
    adapter_path:   Path = ADAPTER_PATH,
    n_samples:      int  = 1,
    max_new_tokens: int  = 512,
    show:           int  = 5,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load test problems (extract system + user from "chosen" messages)
    test_records = []
    with open(TEST_JSONL, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            msgs   = d["chosen"]
            system = next(m["content"] for m in msgs if m["role"] == "system")
            user   = next(m["content"] for m in msgs if m["role"] == "user")
            gold   = next(m["content"] for m in msgs if m["role"] == "assistant")
            problem = user.removeprefix("PROBLEM:\n").strip()
            test_records.append({"problem": problem, "system": system, "gold": gold})

    print(f"[INFO] Test problems  : {len(test_records)}")
    print(f"[INFO] Samples/problem: {n_samples}")
    print(f"[INFO] Max new tokens : {max_new_tokens}")

    model, tokenizer = _load_model(base_model, adapter_path)

    results = []
    for i, rec in enumerate(test_records):
        samples  = _generate_split(model, tokenizer,
                                   rec["system"], rec["problem"],
                                   n_samples, max_new_tokens)

        # Pick best valid sample; fall back to first
        chosen_text   = samples[0]
        chosen_parsed = None
        chosen_checks = {}
        for s in samples:
            parsed = _parse_json_robust(s)
            if parsed is None:
                continue
            checks = _validate(parsed)
            if all(checks.values()):
                chosen_text   = s
                chosen_parsed = parsed
                chosen_checks = checks
                break
        if chosen_parsed is None and (parsed := _parse_json_robust(samples[0])):
            chosen_parsed = parsed
            chosen_checks = _validate(parsed)

        entry = {
            "problem_idx":  i,
            "problem":      rec["problem"],
            "generated":    chosen_text,
            "parsed":       chosen_parsed,
            "valid_json":   chosen_parsed is not None,
            "checks":       chosen_checks,
            "fully_valid":  all(chosen_checks.values()) if chosen_checks else False,
            "n_samples":    n_samples,
        }
        results.append(entry)

        status = "✓" if entry["fully_valid"] else ("~" if entry["valid_json"] else "✗")
        print(f"  [{i+1:>3}/{len(test_records)}] {status}  {rec['problem'][:60]}")

    # Write per-problem results
    with open(OUT_RESULTS, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Aggregate metrics
    n = len(results)
    valid_json   = [r for r in results if r["valid_json"]]
    fully_valid  = [r for r in results if r["fully_valid"]]
    struct_valid = [r for r in valid_json
                    if r["checks"].get("has_required_keys") and r["checks"].get("has_two_packets")]
    interdep_ok  = [r for r in valid_json if r["checks"].get("interdep_correct")]

    pattern_dist: dict[str, int] = {}
    for r in valid_json:
        if r["parsed"]:
            p = r["parsed"].get("pattern", "UNKNOWN")
            pattern_dist[p] = pattern_dist.get(p, 0) + 1

    summary = {
        "n_test":               n,
        "n_samples_per_problem": n_samples,
        "valid_json_rate":      round(len(valid_json)   / n, 3),
        "struct_valid_rate":    round(len(struct_valid) / n, 3),
        "interdep_correct_rate":round(len(interdep_ok)  / n, 3),
        "fully_valid_rate":     round(len(fully_valid)  / n, 3),
        "pattern_distribution": dict(sorted(pattern_dist.items())),
        "sample_outputs": [
            {
                "problem":   r["problem"][:120],
                "generated": r["generated"][:800],
                "valid_json": r["valid_json"],
                "fully_valid": r["fully_valid"],
            }
            for r in results[:show]
        ],
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    # Print summary
    print(f"""
=== SPLIT GENERATOR EVALUATION ===
Test problems      : {n}
Samples/problem    : {n_samples}

valid_json_rate    : {summary['valid_json_rate']:.1%}  ({len(valid_json)}/{n})
struct_valid_rate  : {summary['struct_valid_rate']:.1%}  ({len(struct_valid)}/{n})
interdep_correct   : {summary['interdep_correct_rate']:.1%}  ({len(interdep_ok)}/{n})
fully_valid_rate   : {summary['fully_valid_rate']:.1%}  ({len(fully_valid)}/{n})

Pattern distribution:
{json.dumps(pattern_dist, indent=2)}

Results → {OUT_RESULTS}
Summary → {OUT_SUMMARY}
""")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model",     default=DEFAULT_BASE)
    parser.add_argument("--adapter-path",   default=str(ADAPTER_PATH))
    parser.add_argument("--n-samples",      type=int, default=1,
                        help="samples generated per problem (1=greedy, 3=sample-then-filter)")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--show",           type=int, default=5,
                        help="number of sample outputs to include in summary")
    args = parser.parse_args()
    evaluate(
        base_model=args.base_model,
        adapter_path=Path(args.adapter_path),
        n_samples=args.n_samples,
        max_new_tokens=args.max_new_tokens,
        show=args.show,
    )


if __name__ == "__main__":
    main()
