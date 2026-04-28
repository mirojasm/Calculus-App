"""
Train the CPP discriminator chain on the existing corpus.

Derives CPP binary vectors from existing PISA scores (code_counts > 0 → cell active).
Matches split files (n=2) with jigsaw_2 score files.

Usage:
  python3 -m research.splitting.cidi.train_discriminators
  python3 -m research.splitting.cidi.train_discriminators --splits-dir outputs/splits \\
      --scores-dir outputs/scores --output outputs/models/cpp_discriminator_chain.pkl
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

from research.splitting.cidi.module2_feasibility import CELL_ORDER
from research.splitting.cidi.module5_validation import CPPDiscriminatorChain, _split_to_text


def load_training_data(
    splits_dir: Path,
    scores_dir: Path,
) -> tuple[list[str], list[list[int]]]:
    """
    Load matched (split_text, cpp_vector) pairs from existing corpus.

    Split text = shared_context + packets (from n=2 split files).
    CPP vector = binary activation derived from PISA code_counts in jigsaw_2 scores.
    """
    split_texts: list[str] = []
    cpp_vectors:  list[list[int]] = []
    skipped = 0

    for split_path in sorted(splits_dir.glob("*_n2.json")):
        problem_id = split_path.stem.replace("_n2", "")
        score_path = scores_dir / f"{problem_id}_jigsaw_2_scores.json"

        if not score_path.exists():
            skipped += 1
            continue

        try:
            split_data = json.loads(split_path.read_text())
            score_data = json.loads(score_path.read_text())
        except (json.JSONDecodeError, OSError):
            skipped += 1
            continue

        # Derive CPP vector from PISA code_counts
        code_counts = score_data.get("pisa", {}).get("code_counts", {})
        if not code_counts:
            # Try nested structure
            code_counts = score_data.get("code_counts", {})
        cpp_vector = [1 if code_counts.get(cell, 0) > 0 else 0 for cell in CELL_ORDER]

        # Split text from the raw split data
        text = _split_to_text(split_data)
        if not text.strip():
            skipped += 1
            continue

        split_texts.append(text)
        cpp_vectors.append(cpp_vector)

    return split_texts, cpp_vectors


def main():
    parser = argparse.ArgumentParser(description="Train CPP discriminator chain")
    parser.add_argument("--splits-dir", default="outputs/splits", type=Path)
    parser.add_argument("--scores-dir", default="outputs/scores", type=Path)
    parser.add_argument("--output", default="outputs/models/cpp_discriminator_chain.pkl", type=Path)
    parser.add_argument("--holdout", default=0.2, type=float, help="Fraction for validation")
    args = parser.parse_args()

    print(f"[INFO] Loading data from {args.splits_dir} + {args.scores_dir} ...")
    texts, vectors = load_training_data(args.splits_dir, args.scores_dir)
    print(f"[INFO] Loaded {len(texts)} examples")

    if len(texts) < 10:
        print("[ERROR] Not enough training examples (need ≥ 10)")
        return

    # Train/validation split
    n_val = max(1, int(len(texts) * args.holdout))
    n_train = len(texts) - n_val
    train_texts,  val_texts  = texts[:n_train],  texts[n_train:]
    train_vectors, val_vectors = vectors[:n_train], vectors[n_train:]
    print(f"[INFO] Train: {n_train}, Val: {n_val}")

    # Train
    chain = CPPDiscriminatorChain()
    print("[INFO] Training chain ...")
    train_auc = chain.fit(train_texts, train_vectors)

    print("\n[RESULTS] Training AUC by cell:")
    for cell in CELL_ORDER:
        auc = train_auc.get(cell, float("nan"))
        bar = "█" * int(auc * 20) if not (auc != auc) else "  (degenerate)"
        print(f"  {cell}: {auc:.3f}  {bar}")

    # Validation
    if val_texts:
        print("\n[RESULTS] Validation predictions:")
        n_correct = 0
        total_hamming = 0
        for text, true_vec in zip(val_texts, val_vectors):
            split_mock = {"shared_context": text, "packets": [], "agent_roles": []}
            pred = chain.predict(split_mock)
            h = chain.hamming_to_target(pred["predicted_vector"], true_vec)
            total_hamming += h
            n_correct += int(h == 0)
        print(f"  Exact match (Hamming=0): {n_correct}/{n_val}")
        print(f"  Mean Hamming distance:    {total_hamming/n_val:.2f} / 12")

    # Save
    chain.save(args.output)
    print(f"\n[OK] Discriminator chain saved → {args.output}")


if __name__ == "__main__":
    main()
