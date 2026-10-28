"""Unit and Smoke Tests for Streamlit Dashboard Data Loaders (Phase 8)."""

from __future__ import annotations

import pytest

from app.main import (
    load_briefing_markdown,
    load_p1_sample_comments,
    load_p2_statements,
    load_p3_trajectories,
    load_synthesis_data,
)


def test_ui_synthesis_loader() -> None:
    """Verify synthesis data loader returns valid briefing dictionary."""
    data = load_synthesis_data()
    assert isinstance(data, dict)
    assert data.get("case_id") == "DEC_2023_ELEC_01"
    assert "pillar1_public_opinion" in data
    assert "pillar2_expert_evidence" in data
    assert "pillar3_historical_indicators" in data


def test_ui_markdown_loader() -> None:
    """Verify policy briefing markdown is loaded non-empty."""
    md = load_briefing_markdown()
    assert isinstance(md, str)
    assert len(md) > 100
    assert "DEC_2023_ELEC_01" in md


def test_ui_p1_comments_loader() -> None:
    """Verify public opinion comments table loader."""
    df = load_p1_sample_comments()
    assert not df.empty
    assert "comment_id" in df.columns
    assert "text" in df.columns
    assert "language" in df.columns


def test_ui_p2_statements_loader() -> None:
    """Verify expert statements list loader."""
    stmts = load_p2_statements()
    assert len(stmts) >= 1000
    first = stmts[0]
    assert "statement_id" in first
    assert "statement_type" in first
    assert "extracted_statement" in first


def test_ui_p3_trajectories_loader() -> None:
    """Verify historical trajectories dictionary loader."""
    trajs = load_p3_trajectories()
    assert isinstance(trajs, dict)
    assert "trajectories" in trajs
    assert "confounding_events" in trajs
