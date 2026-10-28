"""Unit tests for dataset loading and policy filtering utilities."""

from __future__ import annotations

import pandas as pd
import pytest

from src.utils.data_loader import (
    filter_by_keywords,
    load_cabinet_decisions,
    load_dataset_register,
    load_macro_indicators,
    load_sinhala_sentiment,
)


def test_load_dataset_register() -> None:
    df = load_dataset_register()
    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 19
    assert "dataset_id" in df.columns
    assert "pipeline_fit" in df.columns


def test_load_sinhala_sentiment() -> None:
    df_train = load_sinhala_sentiment(split="train", format="parquet")
    assert isinstance(df_train, pd.DataFrame)
    assert len(df_train) > 1000
    assert "comment" in df_train.columns or "text" in df_train.columns or len(df_train.columns) >= 2


def test_load_cabinet_decisions() -> None:
    df_cab = load_cabinet_decisions()
    assert isinstance(df_cab, pd.DataFrame)
    assert len(df_cab) >= 10000
    assert "description" in df_cab.columns
    assert "date_str" in df_cab.columns


def test_load_macro_indicators() -> None:
    df_all = load_macro_indicators()
    assert isinstance(df_all, pd.DataFrame)
    assert len(df_all) >= 500
    assert "indicator_code" in df_all.columns
    assert "value" in df_all.columns

    # Test filtering by impact category
    df_econ = load_macro_indicators(impact_category="Economic")
    assert len(df_econ) > 0
    assert (df_econ["impact_category"] == "Economic").all()

    df_social = load_macro_indicators(impact_category="Social")
    assert len(df_social) > 0
    assert (df_social["impact_category"] == "Social").all()

    df_sector = load_macro_indicators(impact_category="Sector-specific")
    assert len(df_sector) > 0
    assert (df_sector["impact_category"] == "Sector-specific").all()


def test_filter_by_keywords() -> None:
    data = {
        "id": [1, 2, 3],
        "text": [
            "Discussion on electricity tariff increase and CEB reform",
            "Agricultural fertilizer subsidy and crop yields",
            "Fuel prices adjusted under the new formula",
        ],
    }
    df = pd.DataFrame(data)
    filtered = filter_by_keywords(df, text_column="text", keywords=["electricity", "CEB"])
    assert len(filtered) == 1
    assert filtered.iloc[0]["id"] == 1
