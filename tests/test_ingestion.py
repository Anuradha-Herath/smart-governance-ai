"""Automated tests for Phase 3 ingestion and interim data validation."""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INTERIM_DIR = DATA_DIR / "interim"


def test_data_dictionary_schema() -> None:
    dd_path = DATA_DIR / "data_dictionary.json"
    assert dd_path.exists(), f"Data dictionary not found at {dd_path}"
    with dd_path.open("r", encoding="utf-8") as f:
        dd = json.load(f)
    assert "pipeline1_public_opinion" in dd
    assert "pipeline2_expert_documents" in dd
    assert "pipeline3_indicators" in dd


def test_pipeline1_interim() -> None:
    p1_file = INTERIM_DIR / "p1_public_opinion_interim.parquet"
    assert p1_file.exists(), f"Pipeline 1 interim file not found at {p1_file}"
    df = pd.read_parquet(p1_file)
    assert len(df) >= 20000
    expected_cols = ["comment_id", "text", "language", "sentiment_label", "source_dataset"]
    for c in expected_cols:
        assert c in df.columns, f"Missing column {c} in Pipeline 1 interim data"
    assert set(df["sentiment_label"].unique()).issubset({"Positive", "Negative", "Neutral"})


def test_pipeline2_interim() -> None:
    p2_file = INTERIM_DIR / "p2_expert_documents_interim.parquet"
    assert p2_file.exists(), f"Pipeline 2 interim file not found at {p2_file}"
    df = pd.read_parquet(p2_file)
    assert len(df) >= 50
    expected_cols = ["doc_id", "source_type", "date_str", "text_chunk", "matched_keywords"]
    for c in expected_cols:
        assert c in df.columns, f"Missing column {c} in Pipeline 2 interim data"


def test_pipeline3_interim() -> None:
    p3_file = INTERIM_DIR / "p3_indicators_interim.csv"
    assert p3_file.exists(), f"Pipeline 3 interim file not found at {p3_file}"
    df = pd.read_csv(p3_file)
    assert len(df) >= 50
    expected_cols = ["indicator_code", "year", "value", "policy_window", "pct_change_from_baseline"]
    for c in expected_cols:
        assert c in df.columns, f"Missing column {c} in Pipeline 3 interim data"
    assert "Implementation Year (2023)" in df["policy_window"].values
