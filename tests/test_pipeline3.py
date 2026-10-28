"""Automated unit tests for Pipeline 3 Historical Retrieval & Indicator Trajectories."""

from __future__ import annotations

import pytest

from src.pipeline3_historical_analysis.confounders import (
    MANDATORY_CORRELATION_DISCLAIMER,
    get_confounders_for_window,
)
from src.pipeline3_historical_analysis.retrieval import search_historical_decisions
from src.pipeline3_historical_analysis.trajectory import IndicatorTrajectoryAnalyzer


def test_historical_decision_retrieval() -> None:
    """Test TF-IDF retrieval on historical Sri Lanka Cabinet decisions."""
    # Query for the primary electricity tariff revision
    results = search_historical_decisions(
        query="Proposal for revision of electricity tariff 2023 CEB",
        top_k=3,
        min_score=0.10,
    )
    assert len(results) > 0
    top = results[0]
    assert "doc_id" in top
    assert "date_str" in top
    assert "title" in top
    assert "similarity_score" in top
    assert top["similarity_score"] > 0.15
    assert "2023-01-09" in top["date_str"] or "electricity" in top["title"].lower()


def test_retrieval_empty_query() -> None:
    """Test retrieval robustness on blank query string."""
    results = search_historical_decisions(query="   ", top_k=5)
    assert results == []


def test_retrieval_date_filter() -> None:
    """Test date-bounded retrieval filtering."""
    results = search_historical_decisions(
        query="tariff",
        top_k=5,
        date_range=("2022-01-01", "2022-12-31"),
    )
    for r in results:
        assert r["date_str"].startswith("2022")


def test_indicator_trajectory_analysis() -> None:
    """Test symmetric window indicator trajectory calculation."""
    analyzer = IndicatorTrajectoryAnalyzer()
    res = analyzer.analyze_trajectory(decision_year=2023, window_years=2)

    assert res["decision_year"] == 2023
    assert res["window_range"] == [2021, 2025]
    assert res["indicators_analyzed_count"] >= 3
    assert "trajectories" in res

    # Check CPI inflation indicator
    cpi_code = "FP.CPI.TOTL.ZG"
    assert cpi_code in res["trajectories"]
    cpi_traj = res["trajectories"][cpi_code]
    assert cpi_traj["baseline_year"] == 2022
    assert cpi_traj["baseline_value"] is not None
    assert cpi_traj["event_value"] is not None
    assert "time_series" in cpi_traj
    assert len(cpi_traj["time_series"]) >= 3


def test_confounder_registry_and_disclaimer() -> None:
    """Verify that historical confounders and causal disclaimers are strictly attached."""
    confounders = get_confounders_for_window(start_year=2021, end_year=2023)
    assert len(confounders) >= 4

    # Ensure sovereign debt default and currency devaluation are recorded
    titles = [c["event_title"] for c in confounders]
    assert any("Debt" in t for t in titles)
    assert any("Currency" in t or "Devaluation" in t for t in titles)

    # Ensure mandatory disclaimer text is defined
    assert "CAUSAL INFERENCE DISCLAIMER" in MANDATORY_CORRELATION_DISCLAIMER
