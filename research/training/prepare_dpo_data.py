"""
Build DPO preference pairs for collaborative-agent fine-tuning.

Strategy
--------
For each of the 143 problems that have both C2 (L1 natural) and C7 (L3 student-sim)
conversations, we create one DPO pair **per agent** (2 agents → 2 pairs per problem):

  chosen  : agent's first turn from the highest-CDI C7 conversation
  rejected: agent's first turn from the lowest-CDI  C2 conversation

The system prompt is the L3 student-sim prompt (reconstructed from the split data),
which is the same for both chosen and rejected.  We train the model to generate
the L3-style opening given that framing.

Output (TRL DPO format)
-----------------------
  outputs/training/dpo_train.jsonl  (80 % of problems)
  outputs/training/dpo_test.jsonl   (20 % holdout)
  outputs/training/dpo_stats.json

Usage
-----
  python -m research.training.prepare_dpo_data
  python -m research.training.prepare_dpo_data --pilot-dir outputs/pilot --seed 42
"""
from __future__ import annotations
import argparse, json, re, textwrap
from collections import defaultdict
from pathlib import Path
import random

PILOT_DIR   = Path("outputs/pilot")
TRAIN_DIR   = Path("outputs/training")
OUT_TRAIN   = TRAIN_DIR / "dpo_train.jsonl"
OUT_TEST    = TRAIN_DIR / "dpo_test.jsonl"
OUT_STATS   = TRAIN_DIR / "dpo_stats.json"

CONDITIONS_CHOSEN   = ["C7"]
CONDITIONS_REJECTED = ["C2"]
TRAIN_FRAC = 0.8


# ── prompt reconstruction ─────────────────────────────────────────────────────

def _parse_packet(pkt_str: str) -> dict:
    """Extract agent_id and information from the Packet(...) string representation."""
    aid_m = re.search(r"agent_id=(\d+)", pkt_str)
    info_m = re.search(r"information=(.+?)(?:,\s*role_name|$)", pkt_str, re.DOTALL)
    aid = int(aid_m.group(1)) if aid_m else 0
    info = info_m.group(1).strip() if info_m else ""
    # Remove trailing parenthesis if present
    info = re.sub(r"\)$", "", info.strip())
    return {"agent_id": aid, "information": info}


def _build_l3_system(shared: str, agent_info: str) -> str:
    """Reconstruct the L3 student-sim system prompt used during simulation."""
    context = f"{shared}\n\n{agent_info}".strip() if shared else agent_info
    return textwrap.dedent(f"""
    You are a college student in a mathematics course working with 1 partner.

    How you reason and collaborate:
    - Work step by step, as a student learning — tentative, careful, willing to be wrong.
    - When you reach a step you are uncertain about, say so and ask your partner.
    - Do not jump to advanced techniques without verifying each step explicitly first.
    - You genuinely need your partner to make progress — ask questions, build on their ideas.
    - Acknowledge when you don't understand something your partner said before continuing.
    - Show your reasoning aloud, even when unsure. It's fine to say "I think..." or "I'm not sure if..."

    {context}
    """).strip()


def _build_goal_anchor(shared: str, fmt_spec: str = "") -> str:
    spec = fmt_spec.strip() if fmt_spec else "State your final answer clearly when both partners agree."
    return f"COLLABORATIVE TASK:\n\n{shared}\n\n{spec}"


# ── data loading ──────────────────────────────────────────────────────────────

def _load_best(pilot_dir: Path, conditions: list[str], prefer_high: bool) -> dict[str, dict]:
    """
    For each (problem_id, condition) group, keep the conversation with the
    highest CDI (prefer_high=True) or lowest CDI (prefer_high=False).
    Returns {problem_id: result_dict}.
    """
    best: dict[str, tuple[float, dict]] = {}
    for cond in conditions:
        for f in sorted(pilot_dir.glob(f"math_*_{cond}_*.json")):
            parts = f.stem.split("_")
            if len(parts) < 5:
                continue
            pid = f"{parts[0]}_{parts[1]}"
            try:
                d = json.loads(f.read_text())
            except Exception:
                continue
            if "error" in d or "conversation" not in d:
                continue
            cdi = d.get("cdi", None)
            if cdi is None:
                continue
            prev_cdi, _ = best.get(pid, (None, {}))
            if prev_cdi is None:
                best[pid] = (cdi, d)
            elif prefer_high and cdi > prev_cdi:
                best[pid] = (cdi, d)
            elif not prefer_high and cdi < prev_cdi:
                best[pid] = (cdi, d)
    return {pid: d for pid, (_, d) in best.items()}


# ── pair building ──────────────────────────────────────────────────────────────

def _extract_turns_by_agent(turns: list[dict]) -> dict[int, list[str]]:
    """Return {agent_id: [turn0_content, turn1_content, ...]} preserving order."""
    by_agent: dict[int, list[str]] = defaultdict(list)
    for t in turns:
        aid = t.get("agent_id", 0)
        by_agent[aid].append(t.get("content", ""))
    return dict(by_agent)


def _build_pairs(chosen_map: dict[str, dict],
                 rejected_map: dict[str, dict]) -> list[dict]:
    """
    For each problem present in both maps, build DPO pairs for each agent.
    We use the first turn only — this is the moment where L3 and L1 diverge most.
    """
    pairs = []
    common_pids = sorted(set(chosen_map) & set(rejected_map))

    for pid in common_pids:
        ch  = chosen_map[pid]
        rej = rejected_map[pid]

        split  = ch.get("split", {})
        shared = split.get("shared_context", "") if isinstance(split, dict) else ""
        raw_pkts = split.get("packets", []) if isinstance(split, dict) else []

        # goal anchor (same for both agents)
        fmt_spec = ""
        if isinstance(split, dict):
            af = split.get("answer_format", {}) or {}
            if isinstance(af, dict):
                fmt_spec = af.get("specification", "")
        goal = _build_goal_anchor(shared, fmt_spec)

        # turns by agent
        ch_turns  = _extract_turns_by_agent(ch["conversation"]["turns"])
        rej_turns = _extract_turns_by_agent(rej["conversation"]["turns"])

        # agent IDs present in chosen conversation
        agent_ids = sorted(ch_turns.keys())
        if not agent_ids:
            continue

        # parse packets for system prompt reconstruction
        pkt_by_agent: dict[int, str] = {}
        for pkt in raw_pkts:
            if isinstance(pkt, str):
                p = _parse_packet(pkt)
                pkt_by_agent[p["agent_id"]] = p["information"]
            elif isinstance(pkt, dict):
                pkt_by_agent[pkt.get("agent_id", 0)] = str(pkt.get("information", ""))

        for aid in agent_ids:
            ch_first  = ch_turns.get(aid, [None])[0]
            rej_first = rej_turns.get(aid, [None])[0]
            if not ch_first or not rej_first:
                continue
            # skip degenerate pairs where responses are identical
            if ch_first.strip() == rej_first.strip():
                continue

            agent_info = pkt_by_agent.get(aid, "")
            system = _build_l3_system(shared, agent_info)

            # TRL DPO messages format
            pairs.append({
                "problem_id": pid,
                "agent_id":   aid,
                "chosen_cdi": ch.get("cdi", 0),
                "rejected_cdi": rej.get("cdi", 0),
                "chosen": [
                    {"role": "system",    "content": system},
                    {"role": "user",      "content": goal},
                    {"role": "assistant", "content": ch_first},
                ],
                "rejected": [
                    {"role": "system",    "content": system},
                    {"role": "user",      "content": goal},
                    {"role": "assistant", "content": rej_first},
                ],
            })

    return pairs


# ── train / test split ────────────────────────────────────────────────────────

def _split_by_problem(pairs: list[dict],
                      train_frac: float,
                      seed: int) -> tuple[list[dict], list[dict]]:
    pids = sorted({p["problem_id"] for p in pairs})
    rng = random.Random(seed)
    rng.shuffle(pids)
    cut = int(len(pids) * train_frac)
    train_pids = set(pids[:cut])
    train = [p for p in pairs if p["problem_id"] in train_pids]
    test  = [p for p in pairs if p["problem_id"] not in train_pids]
    return train, test


# ── main ──────────────────────────────────────────────────────────────────────

def prepare(pilot_dir: Path = PILOT_DIR,
            train_frac: float = TRAIN_FRAC,
            seed: int = 42) -> tuple[list[dict], list[dict]]:
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading chosen conversations (C7, highest CDI per problem)...")
    chosen_map  = _load_best(pilot_dir, CONDITIONS_CHOSEN,   prefer_high=True)
    print(f"       {len(chosen_map)} problems")

    print("[INFO] Loading rejected conversations (C2, lowest CDI per problem)...")
    rejected_map = _load_best(pilot_dir, CONDITIONS_REJECTED, prefer_high=False)
    print(f"       {len(rejected_map)} problems")

    print("[INFO] Building DPO pairs...")
    pairs = _build_pairs(chosen_map, rejected_map)
    print(f"       {len(pairs)} raw pairs")

    # Verify CDI ordering for all pairs
    n_valid = sum(1 for p in pairs if p["chosen_cdi"] > p["rejected_cdi"])
    print(f"       {n_valid}/{len(pairs)} pairs satisfy chosen_cdi > rejected_cdi")

    train_pairs, test_pairs = _split_by_problem(pairs, train_frac, seed)
    print(f"       Train: {len(train_pairs)} pairs ({len({p['problem_id'] for p in train_pairs})} problems)")
    print(f"       Test:  {len(test_pairs)} pairs ({len({p['problem_id'] for p in test_pairs})} problems)")

    # Write JSONL (strip metadata fields not needed for TRL)
    def _write_jsonl(records: list[dict], path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                out = {"chosen": r["chosen"], "rejected": r["rejected"]}
                f.write(json.dumps(out, ensure_ascii=False) + "\n")

    _write_jsonl(train_pairs, OUT_TRAIN)
    _write_jsonl(test_pairs,  OUT_TEST)

    # Write full records (with metadata) for analysis
    with open(OUT_STATS.parent / "dpo_pairs_full.jsonl", "w", encoding="utf-8") as f:
        for r in pairs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats = {
        "total_pairs":       len(pairs),
        "train_pairs":       len(train_pairs),
        "test_pairs":        len(test_pairs),
        "train_problems":    len({p["problem_id"] for p in train_pairs}),
        "test_problems":     len({p["problem_id"] for p in test_pairs}),
        "valid_cdi_order":   n_valid,
        "chosen_cdi_mean":   round(sum(p["chosen_cdi"] for p in pairs) / len(pairs), 3) if pairs else 0,
        "rejected_cdi_mean": round(sum(p["rejected_cdi"] for p in pairs) / len(pairs), 3) if pairs else 0,
        "seed": seed,
    }
    OUT_STATS.write_text(json.dumps(stats, indent=2))
    print(f"\n[OK] {OUT_TRAIN}  ({len(train_pairs)} examples)")
    print(f"[OK] {OUT_TEST}   ({len(test_pairs)} examples)")
    print(f"[OK] {OUT_STATS}")
    print(f"     chosen CDI mean:   {stats['chosen_cdi_mean']}")
    print(f"     rejected CDI mean: {stats['rejected_cdi_mean']}")
    return train_pairs, test_pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-dir",   default=str(PILOT_DIR))
    parser.add_argument("--train-frac",  type=float, default=TRAIN_FRAC)
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()
    prepare(
        pilot_dir=Path(args.pilot_dir),
        train_frac=args.train_frac,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
