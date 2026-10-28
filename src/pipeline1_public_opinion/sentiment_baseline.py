"""Pipeline 1 Baseline Sentiment Classification Model.

Architecture:
- Subword Character & Word N-Gram Feature Union (optimized for Sinhala script & Singlish)
- Class-Balanced Multinomial Logistic Regression

Outputs:
- models/p1_sentiment_baseline.joblib
- outputs/evaluation/p1_sentiment_baseline_metrics.json
- outputs/figures/p1_confusion_matrix_baseline.png
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import FeatureUnion, Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
EVAL_DIR = PROJECT_ROOT / "outputs" / "evaluation"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"


def build_sentiment_pipeline() -> Pipeline:
    """Build hybrid word + character subword TF-IDF Logistic Regression pipeline."""
    union = FeatureUnion(
        transformer_list=[
            (
                "word_tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=3,
                    max_features=25000,
                    sublinear_tf=True,
                ),
            ),
            (
                "char_tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=5,
                    max_features=35000,
                    sublinear_tf=True,
                ),
            ),
        ]
    )

    clf = LogisticRegression(
        C=1.5,
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    )

    return Pipeline([("features", union), ("classifier", clf)])


def evaluate_split(
    model: Pipeline,
    df: pd.DataFrame,
    target_names: list[str],
) -> dict[str, Any]:
    """Compute overall and per-language evaluation metrics."""
    X = df["text"].astype(str).tolist()
    y_true = df["sentiment_label"].tolist()
    y_pred = model.predict(X)

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted"))
    macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=target_names).tolist()

    # Per-language performance breakdown
    per_language: dict[str, Any] = {}
    for lang in df["language"].unique():
        lang_mask = (df["language"] == lang).values
        if lang_mask.sum() > 0:
            sub_true = np.array(y_true)[lang_mask]
            sub_pred = np.array(y_pred)[lang_mask]
            per_language[lang] = {
                "count": int(lang_mask.sum()),
                "accuracy": round(float(accuracy_score(sub_true, sub_pred)), 4),
                "macro_f1": round(float(f1_score(sub_true, sub_pred, average="macro")), 4),
                "weighted_f1": round(float(f1_score(sub_true, sub_pred, average="weighted")), 4),
            }

    return {
        "accuracy": round(acc, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class_report": report,
        "per_language_metrics": per_language,
        "confusion_matrix": cm,
        "target_names": target_names,
    }


def train_and_evaluate_baseline() -> dict[str, Any]:
    """Train baseline sentiment model and save metrics, model artifacts, and confusion matrix."""
    train_path = PROCESSED_DIR / "p1_train.parquet"
    val_path = PROCESSED_DIR / "p1_val.parquet"
    test_path = PROCESSED_DIR / "p1_test.parquet"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Processed splits not found. Run split.py first.")

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)

    classes = ["Negative", "Neutral", "Positive"]

    print("Fitting Baseline Sentiment Pipeline (TF-IDF Word+Char + Logistic Regression)...")
    pipeline = build_sentiment_pipeline()
    pipeline.fit(train_df["text"].astype(str), train_df["sentiment_label"])

    print("Evaluating on Validation Set...")
    val_metrics = evaluate_split(pipeline, val_df, target_names=classes)

    print("Evaluating on Test Set...")
    test_metrics = evaluate_split(pipeline, test_df, target_names=classes)

    # Save model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "p1_sentiment_baseline.joblib"
    joblib.dump(pipeline, model_path)
    print(f"Model saved to: {model_path}")

    # Save metrics
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    all_metrics = {
        "model_name": "TF-IDF (Word + Char Subwords) + Class-Balanced Logistic Regression",
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    eval_file = EVAL_DIR / "p1_sentiment_baseline_metrics.json"
    eval_file.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    print(f"Metrics saved to: {eval_file}")

    # Generate Confusion Matrix Chart
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    cm = np.array(test_metrics["confusion_matrix"])
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
    )
    plt.title(f"Pipeline 1 Test Confusion Matrix (Macro F1: {test_metrics['macro_f1']:.3f})", fontsize=11, fontweight="bold")
    plt.xlabel("Predicted Sentiment", fontsize=10)
    plt.ylabel("True Sentiment", fontsize=10)
    plt.tight_layout()
    cm_path = FIGURES_DIR / "p1_confusion_matrix_baseline.png"
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"Confusion matrix plot saved to: {cm_path}")

    return all_metrics


if __name__ == "__main__":
    res = train_and_evaluate_baseline()
    t = res["test_metrics"]
    print("\n==========================================")
    print(f"Test Accuracy:  {t['accuracy'] * 100:.2f}%")
    print(f"Test Macro F1:  {t['macro_f1']:.4f}")
    print(f"Test Weighted:  {t['weighted_f1']:.4f}")
    print("Per-Language Macro F1:")
    for lang, metrics in t["per_language_metrics"].items():
        print(f"  - {lang:15s}: F1 = {metrics['macro_f1']:.4f} (N={metrics['count']:,})")
    print("==========================================")
