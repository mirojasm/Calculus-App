"""
Build the anonymized OSF release ZIP for CIDI-140.

Usage (from repo root):
  python3 docs/artifacts/build_osf_release.py

Output: docs/artifacts/cidi_140_release.zip
"""
from __future__ import annotations
import json
import glob
import zipfile
import os
from pathlib import Path

ROOT   = Path(__file__).parent.parent.parent
OUT    = ROOT / "docs" / "artifacts" / "cidi_140_release.zip"
PILOT  = ROOT / "outputs" / "pilot"
ABL    = ROOT / "outputs" / "ablations"
AEC    = ROOT / "outputs" / "aec"
ART    = ROOT / "docs" / "artifacts"
SCRIPTS = ROOT / "research"

# ── helpers ───────────────────────────────────────────────────────────────────

def stripped_corpus(probs: list) -> list:
    """Return corpus records without original problem text (MATH license)."""
    keep_keys = {"problem_id", "subject", "level", "answer", "cdi_c7", "corpus"}
    stripped = []
    for p in probs:
        rec = {k: v for k, v in p.items() if k in keep_keys}
        stripped.append(rec)
    return stripped


def pilot_ids(probs: list) -> set:
    return {p["problem_id"] for p in probs}


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    probs = json.loads((PILOT / "phase2_problems.json").read_text())
    ids   = pilot_ids(probs)
    print(f"Phase 2 corpus: {len(ids)} problems")

    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6) as zf:

        # ── top-level docs ─────────────────────────────────────────────────
        for name in ("benchmark_card.md", "REPRODUCE.md", "croissant.json"):
            src = ART / name
            if src.exists():
                zf.write(src, f"cidi_140_release/{name}")
                print(f"  + {name}")
            else:
                print(f"  ! MISSING: {name}")

        # ── corpus (no problem text) ───────────────────────────────────────
        anon = stripped_corpus(probs)
        zf.writestr(
            "cidi_140_release/corpus/phase2_problems.json",
            json.dumps(anon, indent=2, ensure_ascii=False)
        )
        print(f"  + corpus/phase2_problems.json  ({len(anon)} records, problem text stripped)")

        # ── pilot transcripts: C2 / C6 / C7 ──────────────────────────────
        for cond in ("C2", "C6", "C7"):
            files = sorted(
                f for f in glob.glob(str(PILOT / f"*_{cond}_*.json"))
                if any(Path(f).name.startswith(pid + "_" + cond) for pid in ids)
            )
            for fpath in files:
                arcname = f"cidi_140_release/transcripts/{cond}/{Path(fpath).name}"
                zf.write(fpath, arcname)
            print(f"  + transcripts/{cond}/  ({len(files)} files)")

        # ── ablation transcripts ───────────────────────────────────────────
        for cond in ("CFULL", "CEXP"):
            files = sorted(glob.glob(str(ABL / cond / "*.json")))
            for fpath in files:
                arcname = f"cidi_140_release/ablations/{cond}/{Path(fpath).name}"
                zf.write(fpath, arcname)
            print(f"  + ablations/{cond}/  ({len(files)} files)")

        # ── ablation summary ───────────────────────────────────────────────
        abl_sum = ABL / "ablations_summary.json"
        if abl_sum.exists():
            zf.write(abl_sum, "cidi_140_release/ablations/ablations_summary.json")
            print("  + ablations/ablations_summary.json")

        # ── AEC results ────────────────────────────────────────────────────
        for name in ("aec_results.jsonl", "aec_summary.json"):
            src = AEC / name
            if src.exists():
                zf.write(src, f"cidi_140_release/aec/{name}")
                print(f"  + aec/{name}")

        # ── scoring scripts ────────────────────────────────────────────────
        for rel in (
            "scoring/cpp_annotator.py",
            "scoring/atc21s.py",
            "scoring/pisa.py",
            "experiments/run_ablations.py",
            "experiments/compute_aec.py",
            "config.py",
        ):
            src = SCRIPTS / rel
            if src.exists():
                arcname = f"cidi_140_release/scripts/{Path(rel).name}"
                zf.write(src, arcname)
                print(f"  + scripts/{Path(rel).name}")

    size_mb = OUT.stat().st_size / 1_048_576
    print(f"\nZIP written: {OUT}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
