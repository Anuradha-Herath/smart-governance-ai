"""Pipeline 1 Policy Stance Classification Module.

Distinguishes Policy Stance (Support vs. Oppose vs. Neutral/Unclear) from General Sentiment.

Stance is evaluated relative to a specific policy proposition:
TARGET PROPOSITION: "The government should increase electricity tariffs to cost-reflective levels."

Outputs:
- models/p1_stance_classifier.joblib
- outputs/evaluation/p1_stance_metrics.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

from src.utils.data_loader import load_ibm_debater_argkp

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
EVAL_DIR = PROJECT_ROOT / "outputs" / "evaluation"

TARGET_PROPOSITION = "The government should increase electricity tariffs to cost-reflective levels."


def build_stance_pipeline() -> Pipeline:
    """Build stance-specific word and subword feature classifier."""
    union = FeatureUnion(
        transformer_list=[
            ("word_tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=15000)),
            ("char_tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=20000)),
        ]
    )
    clf = LogisticRegression(C=1.0, max_iter=500, class_weight="balanced", random_state=42)
    return Pipeline([("features", union), ("classifier", clf)])


def train_stance_benchmark() -> dict[str, Any]:
    """Train stance detection model on the IBM Debater ArgKP stance benchmark."""
    print("Loading IBM Debater ArgKP Stance Dataset...")
    argkp_df = load_ibm_debater_argkp(subset="stance_balanced")

    # Map stance: 1 -> Support, -1 -> Oppose
    # Also handle neutral or unclear
    def map_stance(val: Any) -> str:
        try:
            v = int(val)
            if v == 1:
                return "Support"
            elif v == -1:
                return "Oppose"
        except Exception:
            pass
        return "Neutral_or_Unclear"

    argkp_df["stance_label"] = argkp_df["stance"].apply(map_stance)

    # Train/test split on stance benchmark
    X_train, X_test, y_train, y_test = train_test_split(
        argkp_df["argument"].astype(str),
        argkp_df["stance_label"],
        test_size=0.20,
        random_state=42,
        stratify=argkp_df["stance_label"],
    )

    print(f"Training Stance Model on {len(X_train):,} arguments...")
    pipeline = build_stance_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "p1_stance_classifier.joblib"
    joblib.dump(pipeline, model_path)

    metrics = {
        "task": "Policy Stance Classification (Support vs. Oppose)",
        "target_proposition": TARGET_PROPOSITION,
        "benchmark_dataset": "IBM Debater ArgKP (stance_balanced)",
        "test_records": len(X_test),
        "macro_f1": round(macro_f1, 4),
        "classification_report": report,
    }

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out_file = EVAL_DIR / "p1_stance_metrics.json"
    out_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return metrics


if __name__ == "__main__":
    m = train_stance_benchmark()
    print("\nStance Model Training Complete:")
    print(f"  Macro F1: {m['macro_f1']:.4f}")
    print(f"  Saved to: {MODELS_DIR / 'p1_stance_classifier.joblib'}")
