"""
Export a stratified validation set for human inter-rater reliability study.

Sampling design: 3 conditions × 4 quadrants × 3 conversations = 36 targets
  Conditions: C2 (L1 Natural), C6 (L2 Peer-aware), C7 (L3 Student-sim)
  Quadrants:  COUPLING, PROD_FAIL, COLLAPSE, TRIVIAL

Output: outputs/validation_set.json  — embedded into the HTML annotation tool.

Usage:
  python3 -m research.experiments.export_validation_set
  python3 -m research.experiments.export_validation_set --per-cell 3 --out outputs/validation_set.json
  python3 -m research.experiments.export_validation_set --translate --embed outputs/validation.html
"""
from __future__ import annotations
import argparse, base64, json, random, re
from collections import defaultdict
from pathlib import Path

PILOT_DIR  = Path("outputs/pilot")
CONV_DIR   = Path("outputs/conversations")
DEFAULT_OUT = Path("outputs/validation_set.json")
COMBINED_FILTERED = PILOT_DIR / "phase1_combined_filtered.json"

CONDITIONS = ["C2", "C6", "C7"]
QUADRANTS  = ["COUPLING", "PROD_FAIL", "COLLAPSE", "TRIVIAL"]
# Map rarer quadrants into primary ones for sampling
QUAD_MAP   = {"PARTIAL_COUP": "PROD_FAIL", "?": "COLLAPSE"}

CELL_ORDER = ["A1","A2","A3","B1","B2","B3","C1","C2","C3","D1","D2","D3"]

CELL_INFO = {
    "A1": {"phase":"A","label":"Descubrimiento de perspectivas",
           "desc":"¿Los agentes necesitaron descubrir las perspectivas/habilidades del otro para avanzar?"},
    "A2": {"phase":"A","label":"Normas de interacción",
           "desc":"¿Establecieron juntos las normas de interacción (quién lidera, cómo verifican)?"},
    "A3": {"phase":"A","label":"Roles emergentes",
           "desc":"¿Los roles emergieron de exploración conjunta en vez de estar pre-asignados?"},
    "B1": {"phase":"B","label":"Representación del problema",
           "desc":"¿Negociaron explícitamente cómo representar o enmarcar el problema?"},
    "B2": {"phase":"B","label":"Descomposición de tareas",
           "desc":"¿Identificar las sub-tareas requirió contribución activa de ambos?"},
    "B3": {"phase":"B","label":"Distribución del trabajo",
           "desc":"¿Negociaron la distribución del trabajo durante la ejecución?"},
    "C1": {"phase":"C","label":"Planificación comunicativa",
           "desc":"¿Comunicaron las acciones antes de ejecutarlas y recibieron confirmación?"},
    "C2": {"phase":"C","label":"Ejecución encadenada",
           "desc":"¿Hay pasos matemáticos que requirieron el output del otro como input necesario?"},
    "C3": {"phase":"C","label":"Participación activa",
           "desc":"¿Siguieron reglas de participación o se promovieron mutuamente activamente?"},
    "D1": {"phase":"D","label":"Monitoreo y reparación",
           "desc":"¿Monitorearon y repararon el entendimiento compartido cuando había divergencia?"},
    "D2": {"phase":"D","label":"Evaluación conjunta",
           "desc":"¿Evaluaron conjuntamente el éxito de las acciones tomadas?"},
    "D3": {"phase":"D","label":"Adaptación de roles",
           "desc":"¿Adaptaron roles u organización en respuesta a lo que ocurrió en la conversación?"},
}

ATC_INFO = {
    "PC": "Participación y contribución — ambos contribuyen activamente",
    "C":  "Comunicación — intercambio claro y útil de información",
    "Co": "Coordinación — sincronización efectiva de acciones",
    "CR": "Regulación colaborativa — apoyo mutuo en el razonamiento",
    "SR": "Regulación compartida — monitoreo conjunto del proceso grupal",
}

COND_LABELS = {"C2": "L1 Natural", "C6": "L2 Peer-aware", "C7": "L3 Student-sim"}


def _load_filtered_pids() -> set[str]:
    if COMBINED_FILTERED.exists():
        return {e["problem_id"] for e in json.loads(COMBINED_FILTERED.read_text())}
    return set()


def _load_phase2_pool(conditions: list[str], filtered_pids: set[str]) -> dict[tuple, list[dict]]:
    """Load result files grouped by (condition, quadrant)."""
    pool: dict[tuple, list[dict]] = defaultdict(list)
    for cond in conditions:
        # Most recent file per (pid, cond)
        best: dict[str, tuple[str, dict]] = {}
        for p in PILOT_DIR.glob(f"math_*_{cond}_*.json"):
            parts = p.stem.split("_")
            if len(parts) < 5:
                continue
            pid = f"{parts[0]}_{parts[1]}"
            ts  = f"{parts[3]}_{parts[4]}"
            if filtered_pids and pid not in filtered_pids:
                continue
            try:
                d = json.loads(p.read_text())
            except Exception:
                continue
            if "error" in d or "conversation" not in d:
                continue
            prev_ts, _ = best.get(pid, ("", {}))
            if ts > prev_ts:
                best[pid] = (ts, d)

        for pid, (_, d) in best.items():
            quad = QUAD_MAP.get(d.get("quadrant","?"), d.get("quadrant","?"))
            if quad in QUADRANTS:
                pool[(cond, quad)].append(d)
    return dict(pool)


def _sample_set(pool: dict, per_cell: int, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    sampled = []
    for cond in CONDITIONS:
        for quad in QUADRANTS:
            candidates = pool.get((cond, quad), [])
            rng.shuffle(candidates)
            take = candidates[:per_cell]
            sampled.extend(take)
    return sampled


def _build_record(d: dict, idx: int) -> dict:
    """Build a clean record for the HTML tool."""
    conv = d.get("conversation", {})
    turns = conv.get("turns", [])
    pid   = d["problem_id"]
    cond  = d["condition"]
    quad  = d.get("quadrant", "?")

    # Get problem text and split from the conversation or pilot file
    problem_text = conv.get("problem", "") or d.get("split", {}).get("problem", "")
    shared_ctx   = d.get("split", {}).get("shared_context", "")
    raw_packets  = d.get("split", {}).get("raw_split", {}).get("packets", [])
    # Fallback: parse from split field
    if not raw_packets:
        pkts = d.get("split", {}).get("packets", [])
        if pkts and isinstance(pkts[0], dict):
            raw_packets = pkts

    return {
        "idx":         idx,
        "problem_id":  pid,
        "condition":   cond,
        "cond_label":  COND_LABELS.get(cond, cond),
        "quadrant":    quad,
        "subject":     conv.get("subject", ""),
        "level":       conv.get("level", 0),
        "correctness": d.get("correctness", "unknown"),
        "known_answer":d.get("known_answer", ""),
        "final_answer":d.get("final_answer", ""),
        # Problem content
        "problem":     problem_text,
        "shared_context": shared_ctx,
        "packets":     [{"agent_id": p.get("agent_id"), "info": p.get("information","")}
                        for p in raw_packets],
        # Conversation
        "turns":       [{"agent": t.get("agent_id",0), "text": t.get("content","")}
                        for t in turns],
        "total_turns": conv.get("total_turns", len(turns)),
        # LLM annotations (shown after human rates, for reference)
        "llm_quality_scores": d.get("quality_scores", {}),
        "llm_cpp_vector":     d.get("cpp_vector", [0]*12),
        "llm_cdi":            d.get("cdi", 0),
        "llm_cqi":            d.get("cqi", 0),
        "llm_phaq":           d.get("phaq", 0),
        "llm_atc":            d.get("atc_dim_scores", {}),
        "llm_atc_cqi":        d.get("atc_cqi", 0),
        "llm_rationale":      d.get("cpp_rationale", {}),
        "llm_atc_rationale":  d.get("atc_rationale", {}),
    }


def export_validation_set(per_cell: int = 3, seed: int = 42, out: Path = DEFAULT_OUT) -> list[dict]:
    filtered_pids = _load_filtered_pids()
    print(f"[INFO] Filtered problem pool: {len(filtered_pids)} genuine epistemic problems")

    pool = _load_phase2_pool(CONDITIONS, filtered_pids)
    print(f"[INFO] Pool by (condition × quadrant):")
    for (cond, quad), items in sorted(pool.items()):
        print(f"  {cond} × {quad:12s}: {len(items):3d} available")

    sampled = _sample_set(pool, per_cell=per_cell, seed=seed)
    records = [_build_record(d, i) for i, d in enumerate(sampled)]

    payload = {
        "meta": {
            "n_conversations":  len(records),
            "per_cell":         per_cell,
            "conditions":       CONDITIONS,
            "quadrants":        QUADRANTS,
            "seed":             seed,
            "cell_info":        CELL_INFO,
            "atc_info":         ATC_INFO,
            "cell_order":       CELL_ORDER,
            "scale":            "0=ausente 1=superficial 2=funcional 3=emergente",
        },
        "conversations": records,
    }

    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\n[OK] {len(records)} conversations → {out}")
    # Summary
    from collections import Counter
    cond_counts = Counter(r["condition"] for r in records)
    quad_counts = Counter(r["quadrant"] for r in records)
    print(f"  By condition: {dict(cond_counts)}")
    print(f"  By quadrant:  {dict(quad_counts)}")
    return records


# ── Translation ───────────────────────────────────────────────────────────────

def _translate_batch(texts: list[str], client) -> list[str]:
    """Translate up to 25 texts to Spanish, preserving LaTeX math."""
    if not texts:
        return []
    numbered = "\n---\n".join(f"[{i}] {t}" for i, t in enumerate(texts))
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content":
             "Eres un traductor experto en matemáticas. Traduce al español. "
             "REGLA CRÍTICA: preserva EXACTAMENTE todas las expresiones LaTeX "
             "(todo lo que está entre $, $$, \\[...\\], \\(...\\), \\begin...\\end). "
             "Solo traduce el texto en lenguaje natural. "
             "Responde ÚNICAMENTE con JSON: {\"t\": [\"trad0\", \"trad1\", ...]}"},
            {"role": "user", "content": numbered},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )
    result = json.loads(resp.choices[0].message.content)
    translated = result.get("t", texts)
    # Safety: if lengths differ, fall back to originals for mismatched items
    if len(translated) != len(texts):
        print(f"  [WARN] batch size mismatch ({len(translated)} vs {len(texts)}), using originals")
        return texts
    return translated


def _translate_record(record: dict, client) -> dict:
    """Translate all natural-language fields in a record to Spanish."""
    import copy
    rec = copy.deepcopy(record)

    # --- Batch 1: problem, shared_context, packet infos ---
    b1_texts, b1_keys = [], []
    for key in ("problem", "shared_context"):
        if rec.get(key):
            b1_keys.append(("top", key)); b1_texts.append(rec[key])
    for i, pkt in enumerate(rec.get("packets", [])):
        if pkt.get("info"):
            b1_keys.append(("pkt", i)); b1_texts.append(pkt["info"])
    if b1_texts:
        translated = _translate_batch(b1_texts, client)
        for (kind, ref), txt in zip(b1_keys, translated):
            if kind == "top":
                rec[ref] = txt
            else:
                rec["packets"][ref]["info"] = txt

    # --- Batch 2+: conversation turns (25 per call) ---
    turns = rec.get("turns", [])
    BATCH = 25
    for start in range(0, len(turns), BATCH):
        chunk = turns[start:start + BATCH]
        translated = _translate_batch([t.get("text", "") for t in chunk], client)
        for j, txt in enumerate(translated):
            turns[start + j]["text"] = txt

    # --- Batch 3: LLM rationale (per-cell) ---
    rat = rec.get("llm_rationale", {})
    if rat:
        keys = list(rat.keys())
        translated = _translate_batch([rat[k] for k in keys], client)
        rec["llm_rationale"] = dict(zip(keys, translated))

    rat_atc = rec.get("llm_atc_rationale", {})
    if rat_atc:
        keys = list(rat_atc.keys())
        translated = _translate_batch([rat_atc[k] for k in keys], client)
        rec["llm_atc_rationale"] = dict(zip(keys, translated))

    return rec


def translate_records(records: list[dict]) -> list[dict]:
    try:
        from openai import OpenAI
    except ImportError:
        print("[ERROR] openai package not found. Run: pip install openai")
        return records
    client = OpenAI()
    translated = []
    for i, rec in enumerate(records):
        print(f"  [{i+1}/{len(records)}] {rec['problem_id']} × {rec['condition']} ...", end=" ", flush=True)
        try:
            translated.append(_translate_record(rec, client))
            print("✓")
        except Exception as e:
            print(f"ERROR: {e} — keeping original")
            translated.append(rec)
    return translated


# ── HTML embedding ─────────────────────────────────────────────────────────────

def embed_in_html(payload: dict, html_path: Path) -> bool:
    """Replace the base64 dataset blob in the existing HTML with new payload."""
    html = html_path.read_text(encoding="utf-8")
    new_b64 = base64.b64encode(
        json.dumps(payload, ensure_ascii=False).encode("utf-8")
    ).decode("ascii")
    new_html, n = re.subn(
        r'(JSON\.parse\(atob\(")[A-Za-z0-9+/=]+("\)\))',
        r'\g<1>' + new_b64 + r'\g<2>',
        html,
    )
    if n == 0:
        print("[WARN] Could not find base64 dataset in HTML — not updated.")
        return False
    html_path.write_text(new_html, encoding="utf-8")
    print(f"[OK] HTML updated: {html_path} ({len(new_html)//1024}KB)")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-cell",  type=int, default=3)
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--out",       default=str(DEFAULT_OUT))
    parser.add_argument("--translate", action="store_true",
                        help="Translate problems/conversations to Spanish via OpenAI")
    parser.add_argument("--embed",     default=None, metavar="HTML",
                        help="Path to validation.html — re-embed dataset after export")
    args = parser.parse_args()

    records = export_validation_set(per_cell=args.per_cell, seed=args.seed, out=Path(args.out))

    if args.translate:
        print(f"\n[TRANSLATE] Translating {len(records)} conversations to Spanish...")
        records = translate_records(records)
        # Rebuild payload with translated records
        payload = json.loads(Path(args.out).read_text(encoding="utf-8"))
        payload["conversations"] = records
        payload["meta"]["lang"] = "es"
        Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[OK] Translated dataset saved → {args.out}")

    if args.embed:
        payload = json.loads(Path(args.out).read_text(encoding="utf-8"))
        embed_in_html(payload, Path(args.embed))


if __name__ == "__main__":
    main()
