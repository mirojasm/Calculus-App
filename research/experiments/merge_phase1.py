"""
Merge Phase 1 results from Corpus 1 and Corpus 2.

Usage:
  python3 -m research.experiments.merge_phase1
  python3 -m research.experiments.merge_phase1 --threshold 0.5 --out outputs/pilot/phase1_combined.json

What it does:
  1. Loads Corpus 1 filtered set (outputs/pilot/phase1_filtered_problems.json)
  2. Finds the most recent Corpus 2 Phase 1 consolidated results file
  3. Filters C7 entries with CDI >= threshold
  4. Merges both sets (deduplication by problem_id)
  5. Enriches with subject/level/answer from corpus JSON files
  6. Saves combined filtered set + prints statistics
"""
from __future__ import annotations
import argparse, json
from collections import Counter
from pathlib import Path

PILOT_DIR = Path("outputs/pilot")
C1_FILTERED = PILOT_DIR / "phase1_filtered_problems.json"
C2_PROBLEMS = PILOT_DIR / "corpus2_problems.json"
DEFAULT_OUT  = PILOT_DIR / "phase1_combined_filtered.json"
CDI_THRESHOLD = 0.5


def _load_corpus_metadata(*files: Path) -> dict[str, dict]:
    """Build problem_id → metadata dict from corpus JSON files."""
    meta: dict[str, dict] = {}
    for f in files:
        if not f.exists():
            continue
        for item in json.loads(f.read_text()):
            pid = item.get("problem_id") or item.get("id", "")
            if pid:
                meta[pid] = {
                    "subject": item.get("subject", ""),
                    "level":   item.get("level", 0),
                    "answer":  item.get("answer", ""),
                }
    return meta


def _load_corpus1_splits_metadata() -> dict[str, dict]:
    """Extract subject/level from corpus jigsaw_2 conversation files (Corpus 1)."""
    conv_dir = Path("outputs/conversations")
    meta: dict[str, dict] = {}
    for p in conv_dir.glob("*_jigsaw_2.json"):
        try:
            d = json.loads(p.read_text())
            pid = d.get("problem_id", "")
            if pid:
                meta[pid] = {
                    "subject": d.get("subject", ""),
                    "level":   d.get("level", 0),
                    "answer":  d.get("ground_truth", ""),
                }
        except Exception:
            pass
    return meta


def _load_phaq_from_results(results_file: Path, max_pid_index: int = 149) -> dict[str, float]:
    """Load phaq_c7 values from a full pilot results JSON for problems with index ≤ max_pid_index."""
    phaq: dict[str, float] = {}
    try:
        data = json.loads(results_file.read_text())
        for e in data:
            if e.get("condition") != "C7":
                continue
            pid = e.get("problem_id", "")
            try:
                idx = int(pid.split("_")[1])
            except (IndexError, ValueError):
                continue
            if idx <= max_pid_index:
                phaq[pid] = round(e.get("phaq", 0.0), 4)
    except Exception:
        pass
    return phaq


def _find_corpus1_full_results() -> Path | None:
    """Return the Corpus 1 Phase 1 full results file (contains ≥50 C7 entries for IDs <150)."""
    for p in sorted(PILOT_DIR.glob("pilot_results_*.json"), reverse=True):
        try:
            d = json.loads(p.read_text())
            c7_corpus1 = sum(
                1 for e in d
                if e.get("condition") == "C7"
                and int(e.get("problem_id", "math_99999").split("_")[1]) < 150
            )
            if c7_corpus1 >= 50:
                return p
        except Exception:
            continue
    return None


def _find_corpus2_results() -> Path | None:
    """Return the most recent corpus2 Phase 1 consolidated results file."""
    # Corpus 2 result file has C7 entries with problem_ids >= math_00150.
    # The consolidated file is named pilot_results_YYYYMMDD_HHMMSS.json.
    # We identify it by checking which file contains math_001** entries.
    candidates = []
    for p in sorted(PILOT_DIR.glob("pilot_results_*.json"), reverse=True):
        try:
            d = json.loads(p.read_text())
            # Check if any entry is a Corpus 2 problem (ID >= math_00150)
            if any(int(e.get("problem_id", "math_00000").split("_")[1]) >= 150
                   for e in d if "problem_id" in e):
                candidates.append(p)
        except Exception:
            continue
    return candidates[0] if candidates else None


def merge_phase1(threshold: float = CDI_THRESHOLD, out_path: Path = DEFAULT_OUT) -> list[dict]:
    # ── Corpus 1 filtered set + PhAQ enrichment ───────────────────────────────
    if not C1_FILTERED.exists():
        raise FileNotFoundError(f"Corpus 1 filtered set not found: {C1_FILTERED}")
    c1_filtered: list[dict] = json.loads(C1_FILTERED.read_text())
    print(f"Corpus 1: {len(c1_filtered)} problems (CDI≥{threshold})")

    # Enrich with phaq_c7 from full results (not in the compact filtered file)
    c1_full_results = _find_corpus1_full_results()
    phaq_map: dict[str, float] = {}
    if c1_full_results:
        phaq_map = _load_phaq_from_results(c1_full_results, max_pid_index=149)
        print(f"  PhAQ enrichment from: {c1_full_results.name} ({len(phaq_map)} entries)")
    for entry in c1_filtered:
        pid = entry["problem_id"]
        if "phaq_c7" not in entry:
            entry["phaq_c7"] = phaq_map.get(pid, 0.0)

    # ── Corpus 2 Phase 1 results ───────────────────────────────────────────────
    c2_results_file = _find_corpus2_results()
    if not c2_results_file:
        raise FileNotFoundError(
            "No Corpus 2 Phase 1 results file found in outputs/pilot/. "
            "Run: python3 -m research.experiments.cpp_comparison "
            "--corpus-file outputs/pilot/corpus2_problems.json --conditions C7 --workers 6"
        )
    print(f"Corpus 2 results: {c2_results_file.name}")
    c2_all: list[dict] = json.loads(c2_results_file.read_text())

    c2_c7 = [e for e in c2_all if e.get("condition") == "C7" and "error" not in e]
    c2_filtered = [
        {
            "problem_id": e["problem_id"],
            "cdi_c7":     round(e.get("cdi", 0), 4),
            "cqi_c7":     round(e.get("cqi", 0), 4),
            "phaq_c7":    round(e.get("phaq", 0), 4),
            "atc_cqi_c7": round(e.get("atc_cqi", 0), 4),
            "quadrant":   e.get("quadrant", ""),
            "profile":    e.get("cdi_label", ""),
            "corpus":     2,
        }
        for e in c2_c7
        if e.get("cdi", 0) >= threshold
    ]
    print(f"Corpus 2: {len(c2_c7)} C7 results → {len(c2_filtered)} pass CDI≥{threshold}")

    # ── Enrich Corpus 1 with corpus/subject/level fields ──────────────────────
    c1_meta = _load_corpus1_splits_metadata()
    c2_meta = _load_corpus_metadata(C2_PROBLEMS)

    def _enrich(entry: dict, meta: dict, corpus: int) -> dict:
        pid = entry["problem_id"]
        m = meta.get(pid, {})
        return {**entry,
                "corpus":  corpus,
                "subject": m.get("subject", ""),
                "level":   m.get("level", 0),
                "answer":  m.get("answer", "")}

    c1_enriched = [_enrich(e, c1_meta, 1) for e in c1_filtered]

    # Corpus 2 entries already have corpus=2, enrich with subject/level
    c2_enriched = [_enrich(e, c2_meta, 2) for e in c2_filtered]

    # ── Merge (dedup by problem_id — no overlap expected) ─────────────────────
    seen: set[str] = set()
    combined: list[dict] = []
    for entry in c1_enriched + c2_enriched:
        pid = entry["problem_id"]
        if pid not in seen:
            seen.add(pid)
            combined.append(entry)

    combined.sort(key=lambda x: x["problem_id"])

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path.write_text(json.dumps(combined, indent=2, ensure_ascii=False))
    print(f"\nSaved {len(combined)} combined problems → {out_path}")

    # ── Statistics ────────────────────────────────────────────────────────────
    _print_stats(combined, threshold)
    return combined


def _print_stats(combined: list[dict], threshold: float) -> None:
    print("\n" + "="*70)
    print(f"COMBINED PHASE 1 STATS  (CDI≥{threshold})")
    print("="*70)

    c1 = [x for x in combined if x.get("corpus") == 1]
    c2 = [x for x in combined if x.get("corpus") == 2]
    print(f"  Corpus 1: {len(c1):3d}  |  Corpus 2: {len(c2):3d}  |  Total: {len(combined)}")

    cdis = [x["cdi_c7"] for x in combined]
    print(f"\n  CDI(C7) distribution:")
    from collections import Counter
    dist = Counter(round(v, 3) for v in cdis)
    for cdi_val in sorted(dist):
        bar = "#" * dist[cdi_val]
        print(f"    {cdi_val:.3f}: {dist[cdi_val]:3d}  {bar}")

    print(f"\n  By subject:")
    subj = Counter(x.get("subject", "?") for x in combined)
    for s, cnt in sorted(subj.items()):
        print(f"    {s:20s}: {cnt}")

    print(f"\n  By level:")
    lvl = Counter(x.get("level", 0) for x in combined)
    for l, cnt in sorted(lvl.items()):
        print(f"    Level {l}: {cnt}")

    print(f"\n  Quadrant distribution:")
    quad = Counter(x.get("quadrant", "?") for x in combined)
    for q, cnt in sorted(quad.items()):
        print(f"    {q:12s}: {cnt}")

    # Phase A presence
    phaq_nonzero = sum(1 for x in combined if x.get("phaq_c7", 0) > 0)
    print(f"\n  PhAQ>0 (Phase A emerged): {phaq_nonzero}/{len(combined)} ({100*phaq_nonzero/len(combined):.1f}%)")

    # Subject × level coverage
    print(f"\n  Subject × Level coverage (problems per cell):")
    subjects = sorted(set(x.get("subject", "") for x in combined if x.get("subject")))
    levels   = sorted(set(x.get("level", 0) for x in combined if x.get("level")))
    header = f"  {'':20s}" + "".join(f" L{l}" for l in levels)
    print(header)
    for s in subjects:
        row = f"  {s:20s}"
        for l in levels:
            cnt = sum(1 for x in combined if x.get("subject") == s and x.get("level") == l)
            row += f" {cnt:2d} "
        print(row)
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description="Merge Corpus 1 + Corpus 2 Phase 1 filtered results")
    parser.add_argument("--threshold", type=float, default=CDI_THRESHOLD,
                        help=f"CDI threshold for filtering (default: {CDI_THRESHOLD})")
    parser.add_argument("--out", default=str(DEFAULT_OUT),
                        help=f"Output file path (default: {DEFAULT_OUT})")
    args = parser.parse_args()
    merge_phase1(threshold=args.threshold, out_path=Path(args.out))


if __name__ == "__main__":
    main()
