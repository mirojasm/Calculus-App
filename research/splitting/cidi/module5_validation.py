"""
CIDI Module 5 — Predictive Validation: chain of CPP discriminators.

Trains 12 logistic-regression classifiers in DAG order, each conditioned
on the predictions of its prerequisites. Uses TF-IDF text features + previous
predictions as input.

At inference: predicts the CPP binary vector for a given split (before simulation).
"""
from __future__ import annotations
import json, pickle
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from research.splitting.cidi.module2_feasibility import CELL_ORDER, PREREQ_DAG

MODEL_PATH = Path("outputs/models/cpp_discriminator_chain.pkl")
HAMMING_THRESHOLD = 3   # max acceptable cells off-target


class _TrivialClassifier:
    """Always predicts a fixed majority class. Module-level so pickle works."""
    def __init__(self, majority: int):
        self.majority = majority

    def predict_proba(self, X):
        n = X.shape[0]
        p = np.zeros((n, 2))
        p[:, self.majority] = 1.0
        return p


def _split_to_text(split_data: dict) -> str:
    """Serialize split to a single text for TF-IDF encoding."""
    parts = [split_data.get("shared_context", "")]
    for pkt in split_data.get("packets", []):
        parts.append(pkt.get("information", ""))
    for role in split_data.get("agent_roles", []):
        parts.append(role.get("role_description", ""))
    parts.append(split_data.get("split_rationale", ""))
    return " ".join(parts)


class CPPDiscriminatorChain:
    """
    12 logistic-regression classifiers in CELL_ORDER, each conditioned on
    the predictions of all prerequisite cells (respecting DAG structure).
    Input = TF-IDF(split_text) ⊕ predictions of prerequisite cells.
    """

    def __init__(self):
        self.vectorizer  = TfidfVectorizer(
            max_features=3000,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.classifiers: dict[str, LogisticRegression] = {}
        self.trained = False
        self._tfidf_matrix = None   # cached from fit for training

    # ── training ──────────────────────────────────────────────────────────────

    def fit(
        self,
        split_texts: list[str],
        cpp_vectors: list[list[int]],
    ) -> dict[str, float]:
        """
        Train all 12 classifiers.
        Returns AUC per cell (computed on training data — for diagnostic only).
        """
        X_tfidf = self.vectorizer.fit_transform(split_texts).toarray()
        N = len(split_texts)
        # Previous predictions accumulated column by column
        prev_preds = np.zeros((N, 0))
        auc_scores: dict[str, float] = {}

        for cell in CELL_ORDER:
            X = np.hstack([X_tfidf, prev_preds])
            y = np.array([v[CELL_ORDER.index(cell)] for v in cpp_vectors])

            # Handle degenerate case (all same class)
            if len(np.unique(y)) < 2:
                majority = int(y.mean() >= 0.5)
                self.classifiers[cell] = _TrivialClassifier(majority)
                pred_probs = np.full(N, float(majority))
                auc_scores[cell] = float("nan")
            else:
                clf = LogisticRegression(
                    max_iter=500,
                    class_weight="balanced",
                    C=0.5,
                    solver="lbfgs",
                )
                clf.fit(X, y)
                self.classifiers[cell] = clf
                pred_probs = clf.predict_proba(X)[:, 1]
                try:
                    auc_scores[cell] = roc_auc_score(y, pred_probs)
                except Exception:
                    auc_scores[cell] = float("nan")

            # Add this cell's predictions as feature for subsequent cells
            prev_preds = np.hstack([prev_preds, pred_probs.reshape(-1, 1)])

        self.trained = True
        return auc_scores

    # ── inference ─────────────────────────────────────────────────────────────

    def predict(self, split_data: dict) -> dict:
        """
        Predict CPP vector for a single split (before simulation).
        split_data can be a raw dict (from generation) or a JSON-serialized split.

        Returns:
          probabilities    — dict {cell: P(cell=1)}
          predicted_vector — 12-bit binary list
          predicted_cdi    — CDI = sum / 12
          predicted_cells  — list of predicted active cells
        """
        if not self.trained:
            raise RuntimeError("Discriminator chain not trained. Call fit() or load() first.")

        text = _split_to_text(split_data)
        X_tfidf = self.vectorizer.transform([text]).toarray()
        prev_preds = np.zeros((1, 0))
        probs: dict[str, float] = {}

        for cell in CELL_ORDER:
            X = np.hstack([X_tfidf, prev_preds])
            prob = float(self.classifiers[cell].predict_proba(X)[0, 1])
            probs[cell] = prob
            prev_preds = np.hstack([prev_preds, [[prob]]])

        predicted_vector = [1 if probs[c] >= 0.5 else 0 for c in CELL_ORDER]
        from research.splitting.cidi.module2_feasibility import vector_to_cells, hamming
        predicted_cells = vector_to_cells(predicted_vector)

        return {
            "probabilities":     probs,
            "predicted_vector":  predicted_vector,
            "predicted_cells":   predicted_cells,
            "predicted_cdi":     sum(predicted_vector) / 12,
        }

    def hamming_to_target(
        self, predicted_vector: list[int], target_vector: list[int]
    ) -> int:
        from research.splitting.cidi.module2_feasibility import hamming
        return hamming(predicted_vector, target_vector)

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path = MODEL_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "CPPDiscriminatorChain":
        with open(path, "rb") as f:
            return pickle.load(f)


# ── module-level singleton ────────────────────────────────────────────────────

_chain: CPPDiscriminatorChain | None = None


def get_chain() -> CPPDiscriminatorChain:
    """Return the loaded discriminator chain, auto-loading from disk if available."""
    global _chain
    if _chain is None:
        if MODEL_PATH.exists():
            _chain = CPPDiscriminatorChain.load()
        else:
            raise FileNotFoundError(
                f"Discriminator chain not found at {MODEL_PATH}. "
                "Run: python3 -m research.splitting.cidi.train_discriminators"
            )
    return _chain


def predict_cpp(split_data: dict) -> dict:
    """Convenience function: predict CPP for a split dict."""
    return get_chain().predict(split_data)
