"""
Agent Epistemic Contribution (AEC) — Shapley-based decomposition of collaborative outcome.

Measures how much each agent's private information contributed to the joint solution.
Exposes whether information asymmetry is the actual mechanism driving collaboration.

Value function  v(S) ∈ [0, 1]
  v({})    = 0  (trivially)
  v({A})   = correctness of agent A solving alone with ONLY their packet
  v({B})   = correctness of agent B solving alone with ONLY their packet
  v({A,B}) = correctness from stored C7 (L3 student-sim) conversation

Shapley values (N=2, symmetric):
  AEC_A = 0.5 * v({A}) + 0.5 * (v({A,B}) - v({B}))
  AEC_B = 0.5 * v({B}) + 0.5 * (v({A,B}) - v({A}))

Derived metrics:
  EN  (Epistemic Necessity)   = v({A,B}) > max(v({A}), v({B}))   [bool]
  EB  (Epistemic Balance)     = 1 - |AEC_A - AEC_B|              [0-1, higher = more balanced]
  CS  (Collaborative Surplus) = v({A,B}) - max(v({A}), v({B}))   [can be negative]

Output
------
  outputs/aec/aec_results.jsonl   — per-problem results with cached solo answers
  outputs/aec/aec_summary.json    — aggregate statistics

Usage
-----
  python -m research.experiments.compute_aec
  python -m research.experiments.compute_aec --workers 4 --dry-run
  python -m research.experiments.compute_aec --force-rerun
"""
from __future__ import annotations
import argparse, json, re, textwrap, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from research.config import CFG
from research.openai_utils import chat

PILOT_DIR   = Path("outputs/pilot")
AEC_DIR     = Path("outputs/aec")
OUT_RESULTS = AEC_DIR / "aec_results.jsonl"
OUT_SUMMARY = AEC_DIR / "aec_summary.json"

COMBINED_FILTERED = PILOT_DIR / "phase1_combined_filtered.json"

# Use the cheapest model — binary correctness check, no long generation needed
AEC_MODEL = "gpt-4o-mini"


# ── correctness scale ─────────────────────────────────────────────────────────

def _corr_value(correctness: str | None) -> float:
    return {"correct": 1.0, "partial": 0.5}.get(correctness or "", 0.0)


# ── packet parsing ────────────────────────────────────────────────────────────

def _parse_packets(raw_packets: list) -> dict[int, str]:
    """Return {agent_id: information_string} from raw packet list."""
    by_agent: dict[int, str] = {}
    for p in raw_packets:
        if isinstance(p, dict):
            by_agent[p["agent_id"]] = str(p.get("information", ""))
        elif isinstance(p, str):
            aid_m  = re.search(r"agent_id=(\d+)", p)
            info_m = re.search(r"information=(.+?)(?:,\s*role_name|$)", p, re.DOTALL)
            if aid_m:
                info = info_m.group(1).strip().rstrip(")") if info_m else ""
                by_agent[int(aid_m.group(1))] = info
    return by_agent


# ── limited-solo simulation ───────────────────────────────────────────────────

_SOLO_SYSTEM = textwrap.dedent("""\
    You are solving a math problem. You have been given PARTIAL information only.
    Work carefully with what you have. If the information you have is insufficient
    to determine a unique answer, say so explicitly.
    End your response with exactly one of:
      FINAL ANSWER: <your answer>
      FINAL ANSWER: CANNOT DETERMINE
""").strip()


def simulate_limited_solo(shared_context: str, agent_info: str) -> str:
    """Run one turn with only the agent's partial information. Returns raw response."""
    user_msg = (
        f"TASK: {shared_context}\n\n"
        f"YOUR INFORMATION: {agent_info}\n\n"
        "Solve what you can using only the information above."
    )
    return chat(
        messages=[
            {"role": "system",  "content": _SOLO_SYSTEM},
            {"role": "user",    "content": user_msg},
        ],
        model=AEC_MODEL,
        temperature=0.0,
        max_tokens=400,
    )


def _extract_answer(text: str) -> str | None:
    m = re.search(r"FINAL ANSWER:\s*(.+)", text, re.IGNORECASE)
    if not m:
        return None
    ans = m.group(1).strip()
    if re.search(r"cannot\s+determine|insufficient", ans, re.IGNORECASE):
        return None
    return ans


# ── answer checking ───────────────────────────────────────────────────────────

def _check_answer(predicted: str | None, known: str | None) -> str:
    """
    Returns 'correct', 'incorrect', 'incomplete', or 'unknown'.
    Requires ALL significant numbers from the known answer to appear in predicted.
    This prevents partial-match false positives (e.g. "0" in "(0,5)" ≠ "(6,0)").
    """
    if not known:
        return "unknown"
    if not predicted or not predicted.strip():
        return "incomplete"
    # exact string match first (strips whitespace, case-insensitive)
    if known.strip().lower() in predicted.strip().lower():
        return "correct"
    nums_predicted = set(re.findall(r"-?\d+(?:\.\d+)?", predicted))
    if not nums_predicted:
        return "incomplete"
    known_nums = re.findall(r"-?\d+(?:\.\d+)?", str(known))
    if not known_nums:
        return "incorrect"
    # all significant numbers from known must appear in predicted
    if all(kn in nums_predicted for kn in known_nums):
        return "correct"
    return "incorrect"


# ── data loading ──────────────────────────────────────────────────────────────

def _load_filtered_pids() -> set[str]:
    if COMBINED_FILTERED.exists():
        return {e["problem_id"] for e in json.loads(COMBINED_FILTERED.read_text())}
    return set()


def _load_best_c7(filtered_pids: set[str]) -> dict[str, dict]:
    """
    For each problem, keep the C7 result with the highest CDI.
    Must have: valid 2-agent split with packets, known_answer, and no error field.
    Restricts to N=2 only: Shapley for N>2 requires coalition simulations not yet available.
    """
    best: dict[str, tuple[float, dict]] = {}
    for f in sorted(PILOT_DIR.glob("math_*_C7_*.json")):
        parts = f.stem.split("_")
        if len(parts) < 5:
            continue
        pid = f"{parts[0]}_{parts[1]}"
        if filtered_pids and pid not in filtered_pids:
            continue
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        if "error" in d:
            continue
        split = d.get("split", {})
        if not isinstance(split, dict) or not split.get("packets"):
            continue
        if len(split["packets"]) != 2:
            continue   # N>2 requires full coalition simulations
        if not d.get("known_answer"):
            continue
        cdi = d.get("cdi", 0.0) or 0.0
        prev_cdi, _ = best.get(pid, (-1.0, {}))
        if cdi > prev_cdi:
            best[pid] = (cdi, d)
    return {pid: d for pid, (_, d) in best.items()}


# ── caching ───────────────────────────────────────────────────────────────────

def _load_cache() -> dict[str, dict]:
    """Load previously computed solo results keyed by problem_id."""
    cache: dict[str, dict] = {}
    if OUT_RESULTS.exists():
        with open(OUT_RESULTS, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                cache[r["problem_id"]] = r
    return cache


def _append_result(result: dict) -> None:
    with open(OUT_RESULTS, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


# ── per-problem AEC computation ───────────────────────────────────────────────

def compute_aec_for_problem(pid: str, d: dict) -> dict:
    """
    Run limited-solo simulations for both agents, compute Shapley AEC.
    Returns a result dict ready to cache and analyse.
    """
    split       = d["split"]
    shared      = split.get("shared_context", "")
    raw_packets = split.get("packets", [])
    known       = str(d.get("known_answer", "")).strip()
    packets     = _parse_packets(raw_packets)

    if len(packets) < 2:
        return {"problem_id": pid, "error": f"need 2 packets, got {len(packets)}"}

    agent_ids = sorted(packets.keys())
    a_id, b_id = agent_ids[0], agent_ids[1]

    # ── solo A (only packet A) ────────────────────────────────────────────────
    t0 = time.time()
    resp_a = simulate_limited_solo(shared, packets[a_id])
    ans_a  = _extract_answer(resp_a)
    corr_a = _check_answer(ans_a, known)
    v_a    = _corr_value(corr_a)

    # ── solo B (only packet B) ────────────────────────────────────────────────
    resp_b = simulate_limited_solo(shared, packets[b_id])
    ans_b  = _extract_answer(resp_b)
    corr_b = _check_answer(ans_b, known)
    v_b    = _corr_value(corr_b)

    # ── pair v({A,B}) from stored C7 result ───────────────────────────────────
    v_ab   = _corr_value(d.get("correctness"))
    elapsed = round(time.time() - t0, 1)

    # ── Shapley AEC ───────────────────────────────────────────────────────────
    aec_a = round(0.5 * v_a + 0.5 * (v_ab - v_b), 4)
    aec_b = round(0.5 * v_b + 0.5 * (v_ab - v_a), 4)

    en  = v_ab > max(v_a, v_b)                       # Epistemic Necessity
    eb  = round(1.0 - abs(aec_a - aec_b), 4)         # Epistemic Balance
    cs  = round(v_ab - max(v_a, v_b), 4)             # Collaborative Surplus

    return {
        "problem_id":     pid,
        "known_answer":   known,
        # solo results
        "v_a":            v_a,
        "v_b":            v_b,
        "correctness_a":  corr_a,
        "correctness_b":  corr_b,
        "answer_a":       ans_a,
        "answer_b":       ans_b,
        "solo_response_a": resp_a,
        "solo_response_b": resp_b,
        # pair result
        "v_ab":           v_ab,
        "correctness_ab": d.get("correctness"),
        "cdi_c7":         round(d.get("cdi", 0.0), 4),
        # AEC metrics
        "aec_a":          aec_a,
        "aec_b":          aec_b,
        "en":             en,
        "eb":             eb,
        "cs":             cs,
        "elapsed_sec":    elapsed,
    }


# ── summary statistics ────────────────────────────────────────────────────────

def _summarize(results: list[dict]) -> dict:
    ok = [r for r in results if "error" not in r]
    if not ok:
        return {"n": 0}

    def _mean(vals):
        return round(sum(vals) / len(vals), 4)

    v_a_vals  = [r["v_a"]  for r in ok]
    v_b_vals  = [r["v_b"]  for r in ok]
    v_ab_vals = [r["v_ab"] for r in ok]
    aec_a_vals = [r["aec_a"] for r in ok]
    aec_b_vals = [r["aec_b"] for r in ok]
    en_vals   = [r["en"]   for r in ok]
    eb_vals   = [r["eb"]   for r in ok]
    cs_vals   = [r["cs"]   for r in ok]

    return {
        "n":               len(ok),
        "errors":          len(results) - len(ok),
        "v_a_mean":        _mean(v_a_vals),
        "v_b_mean":        _mean(v_b_vals),
        "v_ab_mean":       _mean(v_ab_vals),
        "aec_a_mean":      _mean(aec_a_vals),
        "aec_b_mean":      _mean(aec_b_vals),
        "en_rate":         round(sum(en_vals) / len(en_vals), 4),
        "eb_mean":         _mean(eb_vals),
        "cs_mean":         _mean(cs_vals),
        "cs_positive_rate": round(sum(1 for c in cs_vals if c > 0) / len(cs_vals), 4),
        # Epistemic Necessity breakdown by packet-solvability
        "both_solo_zero_rate": round(
            sum(1 for r in ok if r["v_a"] == 0 and r["v_b"] == 0) / len(ok), 4
        ),
        "one_solo_nonzero_rate": round(
            sum(1 for r in ok if (r["v_a"] > 0) != (r["v_b"] > 0)) / len(ok), 4
        ),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def run(workers: int = 4, force_rerun: bool = False, dry_run: bool = False) -> list[dict]:
    AEC_DIR.mkdir(parents=True, exist_ok=True)

    filtered_pids = _load_filtered_pids()
    print(f"[INFO] Phase 1 filtered problems: {len(filtered_pids)}")

    best_c7 = _load_best_c7(filtered_pids)
    print(f"[INFO] C7 problems with valid split + known_answer: {len(best_c7)}")

    cache = {} if force_rerun else _load_cache()
    print(f"[INFO] Cached results: {len(cache)}")

    todo = {pid: d for pid, d in best_c7.items() if pid not in cache}
    print(f"[INFO] Problems to compute: {len(todo)}")

    if dry_run:
        print("[DRY-RUN] First 3 problems would be processed:")
        for pid in list(todo)[:3]:
            split = todo[pid].get("split", {})
            pkts  = _parse_packets(split.get("packets", []))
            print(f"  {pid}  known={todo[pid].get('known_answer')}  "
                  f"shared='{split.get('shared_context','')[:50]}'  "
                  f"n_packets={len(pkts)}")
        return []

    results: list[dict] = list(cache.values())
    n_done = len(results)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(compute_aec_for_problem, pid, d): pid
                for pid, d in todo.items()}
        for fut in as_completed(futs):
            pid = futs[fut]
            try:
                r = fut.result()
            except Exception as exc:
                r = {"problem_id": pid, "error": str(exc)}
                print(f"  [ERR] {pid}: {exc}")
            _append_result(r)
            results.append(r)
            n_done += 1
            status = "EN" if r.get("en") else "  "
            print(f"  [{n_done:>3}] {pid}  "
                  f"v_a={r.get('v_a','?'):.2f}  "
                  f"v_b={r.get('v_b','?'):.2f}  "
                  f"v_ab={r.get('v_ab','?'):.2f}  "
                  f"AEC_A={r.get('aec_a','?'):.3f}  "
                  f"AEC_B={r.get('aec_b','?'):.3f}  "
                  f"CS={r.get('cs','?'):.3f}  {status}")

    summary = _summarize(results)
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2))
    print(f"\n[OK] Results → {OUT_RESULTS}  ({len(results)} problems)")
    print(f"[OK] Summary → {OUT_SUMMARY}")
    print(f"\n  v_a  mean: {summary.get('v_a_mean','?')}")
    print(f"  v_b  mean: {summary.get('v_b_mean','?')}")
    print(f"  v_ab mean: {summary.get('v_ab_mean','?')}")
    print(f"  EN  rate:  {summary.get('en_rate','?')}")
    print(f"  EB  mean:  {summary.get('eb_mean','?')}")
    print(f"  CS  mean:  {summary.get('cs_mean','?')}")
    print(f"  CS positive: {summary.get('cs_positive_rate','?')}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers",     type=int,  default=4)
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--dry-run",     action="store_true")
    args = parser.parse_args()
    run(workers=args.workers, force_rerun=args.force_rerun, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
