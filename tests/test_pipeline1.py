"""Automated tests for Pipeline 1 models, splits, and inference services."""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import pytest

from src.pipeline1_public_opinion.predict import (
    analyze_comment,
    analyze_comment_batch,
    predict_sentiment,
    predict_stance,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


def test_splits_integrity_and_zero_leakage() -> None:
    train_file = PROCESSED_DIR / "p1_train.parquet"
    val_file = PROCESSED_DIR / "p1_val.parquet"
    test_file = PROCESSED_DIR / "p1_test.parquet"

    assert train_file.exists()
    assert val_file.exists()
    assert test_file.exists()

    df_train = pd.read_parquet(train_file)
    df_val = pd.read_parquet(val_file)
    df_test = pd.read_parquet(test_file)

    assert len(df_train) > 15000
    assert len(df_val) > 3000
    assert len(df_test) > 3000

    # Ensure zero data leakage
    s_train = set(df_train["comment_id"])
    s_val = set(df_val["comment_id"])
    s_test = set(df_test["comment_id"])

    assert len(s_train.intersection(s_val)) == 0
    assert len(s_train.intersection(s_test)) == 0
    assert len(s_val.intersection(s_test)) == 0


def test_sentiment_model_predictions() -> None:
    res = predict_sentiment("Great policy, electricity supply is reliable now!")
    assert res["sentiment"] in ["Positive", "Negative", "Neutral"]
    assert 0.0 <= res["confidence"] <= 1.0
    probs = res["probabilities"]
    assert len(probs) == 3
    assert abs(sum(probs.values()) - 1.0) < 1e-3


def test_stance_model_predictions() -> None:
    res = predict_stance("We strongly oppose the power tariff increases for poor households.")
    assert res["stance"] in ["Support", "Oppose", "Neutral_or_Unclear"]
    assert 0.0 <= res["confidence"] <= 1.0


def test_batch_aggregation_and_disclaimer() -> None:
    comments = [
        "CEB bill eka wadi karapu eka hondayi.",
        "Electricity costs are ruining small businesses.",
        "No comment on power tariffs.",
    ]
    batch = analyze_comment_batch(comments)
    assert batch["total_analyzed"] == 3
    assert "sampling_disclaimer" in batch
    assert "sentiment_distribution_pct" in batch
    assert "stance_distribution_pct" in batch

    # Check that sentiment percentages sum to ~100%
    sent_sum = sum(batch["sentiment_distribution_pct"].values())
    assert abs(sent_sum - 100.0) < 1.0
