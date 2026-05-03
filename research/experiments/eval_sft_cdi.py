"""
CDI evaluation of SFT-generated splits vs reference CIDI splits.

For 20 held-out test problems:
  1. Generate a split with the SFT adapter (Mistral-7B fine-tuned)
  2. Run it through the C7 student-simulation pipeline
  3. Compute CDI and compare with the reference CIDI split CDI

This tests whether the SFT split generator produces splits that are
epistemically effective (CDI >= 0.5), not just structurally valid.

Output
------
  outputs/eval/sft_cdi_results.jsonl   — per-problem CDI comparison
  outputs/eval/sft_cdi_summary.json    — aggregate stats

Usage
-----
  python -m research.experiments.eval_sft_cdi
  python -m research.experiments.eval_sft_cdi --n-problems 20 --n-samples 3
"""
from __future__ import annotations
import argparse, json, random, time
from pathlib import Path

PAIRS_FULL   = Path("outputs/training/split_dpo_pairs_full.jsonl")
ADAPTER_PATH = Path("outputs/training/split_adapter/final_adapter")
OUT_DIR      = Path("outputs/eval")

DEFAULT_BASE = "/scratch/mir85108/llm/models/hf/Mistral-7B-Instruct-v0.3"
TRAIN_FRAC   = 0.8
SEED         = 42

REQUIRED_KEYS  = {"pattern", "shared_context", "packets", "interdependence_check"}
VALID_PATTERNS = {"SPLIT-A", "SPLIT-B", "SPLIT-C", "SPLIT-D", "SPLIT-E", "SPLIT-F", "SPLIT-G"}

# System prompt (same as training)
import textwrap
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


# ── helpers ───────────────────────────────────────────────────────────────────

def _try_json(s: str) -> dict | None:
    try:
        return json.loads(s)
    except Exception:
        return None


def _parse_json_robust(text: str) -> dict | None:
    import re
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


def _build_split_result(problem_id: str, problem: str, split_json: dict):
    from research.splitting.splitter import SplitResult, Packet
    packets = [
        Packet(agent_id=p["agent_id"], information=p["information"])
        for p in split_json["packets"]
    ]
    return SplitResult(
        problem_id=problem_id,
        problem=problem,
        n=2,
        pattern=split_json.get("pattern", ""),
        shared_context=split_json.get("shared_context", ""),
        packets=packets,
        valid=True,
        raw_split=split_json,
    )


# ── SFT generation ────────────────────────────────────────────────────────────

def _load_model(base_model: str, adapter_path: Path, sft_adapter_path: Path | None = None):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    print(f"[INFO] Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=__import__("torch").float16, trust_remote_code=True
    ).cuda()

    if sft_adapter_path is not None:
        print(f"[INFO] Merging SFT adapter: {sft_adapter_path}")
        model = PeftModel.from_pretrained(model, str(sft_adapter_path))
        model = model.merge_and_unload()
        print(f"[INFO] SFT adapter merged.")

    model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()
    print(f"[INFO] Adapter loaded from: {adapter_path}")
    return model, tokenizer


def _generate_split(model, tokenizer, problem: str, n_samples: int) -> dict | None:
    import torch
    messages = [
        {"role": "system", "content": _SPLIT_SYSTEM},
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
            max_new_tokens=512,
            do_sample=(n_samples > 1),
            temperature=0.7 if n_samples > 1 else 1.0,
            num_return_sequences=n_samples,
            pad_token_id=tokenizer.eos_token_id,
        )
    prompt_len = inputs["input_ids"].shape[1]
    for out in outputs:
        text   = tokenizer.decode(out[prompt_len:], skip_special_tokens=True)
        parsed = _parse_json_robust(text)
        if parsed and _validate(parsed):
            return parsed
    # fallback: return first parseable even if not fully valid
    for out in outputs:
        text   = tokenizer.decode(out[prompt_len:], skip_special_tokens=True)
        parsed = _parse_json_robust(text)
        if parsed:
            return parsed
    return None


# ── C7 pipeline ───────────────────────────────────────────────────────────────

def _run_c7_with_split(split_result) -> dict:
    from research.simulation.simulator import simulate
    from research.scoring.cpp_annotator import annotate

    conv = simulate(split_result, "student_jigsaw_2", student_sim=True)
    cpp  = annotate(conv)
    return {"cdi": cpp.cdi, "cqi": cpp.cqi, "phaq": cpp.phaq}


# ── main ──────────────────────────────────────────────────────────────────────

def evaluate(
    base_model:       str       = DEFAULT_BASE,
    adapter_path:     Path      = ADAPTER_PATH,
    sft_adapter_path: Path|None = None,
    n_problems:       int       = 20,
    n_samples:        int       = 3,
    output_prefix:    str       = "sft_cdi",
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_RESULTS = OUT_DIR / f"{output_prefix}_results.jsonl"
    OUT_SUMMARY = OUT_DIR / f"{output_prefix}_summary.json"

    # Load all pairs and pick unique test-set problem IDs
    all_pairs: list[dict] = []
    with open(PAIRS_FULL, encoding="utf-8") as f:
        for line in f:
            all_pairs.append(json.loads(line))

    # Reproduce train/test split (same logic as prepare_split_dpo_data)
    pids = sorted({p["problem_id"] for p in all_pairs})
    rng  = random.Random(SEED)
    rng.shuffle(pids)
    cut  = int(len(pids) * TRAIN_FRAC)
    test_pids = set(pids[cut:])

    # Best reference CDI per test problem
    ref_cdi: dict[str, float] = {}
    problems: dict[str, str]  = {}
    for p in all_pairs:
        pid = p["problem_id"]
        if pid not in test_pids:
            continue
        if p["cdi_chosen"] > ref_cdi.get(pid, -1):
            ref_cdi[pid]  = p["cdi_chosen"]
            problems[pid] = p["problem"]

    eval_pids = sorted(ref_cdi.keys())[:n_problems]
    print(f"[INFO] Test problems available: {len(ref_cdi)}")
    print(f"[INFO] Evaluating              : {len(eval_pids)}")
    print(f"[INFO] SFT samples/problem     : {n_samples}")

    model, tokenizer = _load_model(base_model, adapter_path, sft_adapter_path)

    results = []
    for i, pid in enumerate(eval_pids):
        problem   = problems[pid]
        ref       = ref_cdi[pid]

        print(f"\n[{i+1:>2}/{len(eval_pids)}] {pid}  ref_CDI={ref:.3f}")
        print(f"  Problem: {problem[:80]}")

        # Generate split with SFT model
        t0         = time.time()
        split_json = _generate_split(model, tokenizer, problem, n_samples)
        gen_sec    = round(time.time() - t0, 1)

        if split_json is None:
            print(f"  [WARN] No valid split generated — skipping CDI evaluation")
            results.append({
                "problem_id": pid, "problem": problem,
                "ref_cdi": ref, "sft_cdi": None,
                "sft_cqi": None, "sft_phaq": None,
                "split_valid": False, "delta_cdi": None,
            })
            continue

        valid = _validate(split_json)
        print(f"  Split: pattern={split_json.get('pattern')}  valid={valid}  gen={gen_sec}s")

        # Run C7 pipeline
        split_result = _build_split_result(pid, problem, split_json)
        try:
            t0   = time.time()
            cpp  = _run_c7_with_split(split_result)
            c7_sec = round(time.time() - t0, 1)
            sft_cdi = cpp["cdi"]
            print(f"  CDI: SFT={sft_cdi:.3f}  ref={ref:.3f}  Δ={sft_cdi-ref:+.3f}  ({c7_sec}s)")
            results.append({
                "problem_id":  pid,
                "problem":     problem,
                "ref_cdi":     ref,
                "sft_cdi":     sft_cdi,
                "sft_cqi":     cpp["cqi"],
                "sft_phaq":    cpp["phaq"],
                "split_valid": valid,
                "split_json":  split_json,
                "delta_cdi":   round(sft_cdi - ref, 3),
            })
        except Exception as e:
            print(f"  [ERROR] C7 pipeline failed: {e}")
            results.append({
                "problem_id": pid, "problem": problem,
                "ref_cdi": ref, "sft_cdi": None,
                "split_valid": valid, "split_json": split_json,
                "error": str(e),
            })

    # Write results
    with open(OUT_RESULTS, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Summary
    ok = [r for r in results if r.get("sft_cdi") is not None]
    above_thresh = [r for r in ok if r["sft_cdi"] >= 0.5]
    ref_above    = [r for r in ok if r["ref_cdi"] >= 0.5]

    summary = {
        "n_evaluated":      len(eval_pids),
        "n_c7_ok":          len(ok),
        "n_sft_cdi_ge_05":  len(above_thresh),
        "sft_cdi_ge_05_rate": round(len(above_thresh) / len(ok), 3) if ok else None,
        "ref_cdi_ge_05_rate": round(len(ref_above) / len(ok), 3) if ok else None,
        "sft_cdi_mean":     round(sum(r["sft_cdi"] for r in ok) / len(ok), 3) if ok else None,
        "ref_cdi_mean":     round(sum(r["ref_cdi"] for r in ok) / len(ok), 3) if ok else None,
        "delta_cdi_mean":   round(sum(r["delta_cdi"] for r in ok) / len(ok), 3) if ok else None,
        "sft_cqi_mean":     round(sum(r["sft_cqi"] for r in ok) / len(ok), 3) if ok else None,
        "sft_phaq_mean":    round(sum(r["sft_phaq"] for r in ok) / len(ok), 3) if ok else None,
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"""
=== SFT CDI EVALUATION ===
Problems evaluated : {len(eval_pids)}
C7 pipeline ok     : {len(ok)}

SFT CDI mean       : {summary['sft_cdi_mean']}
Ref CDI mean       : {summary['ref_cdi_mean']}
Delta CDI mean     : {summary['delta_cdi_mean']}

SFT CDI >= 0.5     : {summary['sft_cdi_ge_05_rate']:.1%} ({len(above_thresh)}/{len(ok)})
Ref CDI >= 0.5     : {summary['ref_cdi_ge_05_rate']:.1%} ({len(ref_above)}/{len(ok)})

SFT CQI mean       : {summary['sft_cqi_mean']}
SFT PhAQ mean      : {summary['sft_phaq_mean']}

Results → {OUT_RESULTS}
Summary → {OUT_SUMMARY}
""")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model",      default=DEFAULT_BASE)
    parser.add_argument("--adapter-path",    default=str(ADAPTER_PATH))
    parser.add_argument("--sft-adapter-path", default=None,
                        help="SFT adapter to merge before loading the main adapter "
                             "(required for SFT→DPO adapters)")
    parser.add_argument("--n-problems",      type=int, default=20)
    parser.add_argument("--n-samples",       type=int, default=3,
                        help="samples per problem for sample-then-filter")
    parser.add_argument("--output-prefix",   default="sft_cdi",
                        help="prefix for output files (e.g. 'dpo_cdi')")
    args = parser.parse_args()
    evaluate(
        base_model=args.base_model,
        adapter_path=Path(args.adapter_path),
        sft_adapter_path=Path(args.sft_adapter_path) if args.sft_adapter_path else None,
        n_problems=args.n_problems,
        n_samples=args.n_samples,
        output_prefix=args.output_prefix,
    )


if __name__ == "__main__":
    main()
