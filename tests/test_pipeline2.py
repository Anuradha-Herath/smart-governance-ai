"""Automated tests for Pipeline 2 grounded evidence extraction and verification."""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import pytest

from src.pipeline2_expert_opinion.extractor import (
    classify_sentence,
    extract_grounded_evidence,
    split_into_sentences,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_FILE = PROJECT_ROOT / "data" / "processed" / "p2_extracted_evidence.json"
MULTIDOMAIN_EXPERT_PQ = PROJECT_ROOT / "data" / "processed" / "p2_multidomain_expert_opinions.parquet"


def test_sentence_segmentation() -> None:
    sample_text = (
        "Cabinet considered the proposal on electricity tariffs. "
        "CEB incurred massive losses of Rs. 167 billion in 2022. "
        "The commission must study the proposal before February 2023."
    )
    sents = split_into_sentences(sample_text)
    assert len(sents) == 3
    for s in sents:
        assert len(s) >= 20


def test_statement_classification() -> None:
    # Test Evidence with numerical facts
    stmt_ev, conf_ev = classify_sentence("CEB accumulated operating losses exceeding Rs. 150 billion in 2022.")
    assert stmt_ev == "Evidence"
    assert conf_ev >= 0.60

    # Test Recommendation
    stmt_rec, conf_rec = classify_sentence("The government must implement cost-reflective tariffs immediately.")
    assert stmt_rec == "Recommendation"
    assert conf_rec >= 0.60

    # Test Risk
    stmt_risk, conf_risk = classify_sentence("Higher electricity tariffs impose a heavy burden on low-income consumers.")
    assert stmt_risk == "Risk"
    assert conf_risk >= 0.60


def test_grounded_extraction_integrity() -> None:
    sample_chunk = (
        "The Ceylon Electricity Board is facing a liquidity crisis. "
        "The General Treasury is no longer possible to provide required funds within existing fiscal space. "
        "Approval was granted to revise tariffs in order to ensure uninterrupted supply."
    )
    extractions = extract_grounded_evidence(
        text_chunk=sample_chunk,
        doc_id="TEST_DOC_001",
        source_type="CabinetDecision",
        date_str="2023-01-09",
        speaker_or_title="Test Tariff Policy",
    )
    assert len(extractions) > 0
    for stmt in extractions:
        assert "statement_type" in stmt
        assert stmt["statement_type"] in ["Claim", "Risk", "Benefit", "Evidence", "Recommendation"]
        assert "verbatim_quote" in stmt
        # Verification: If flagged grounded, verbatim quote must be present in source chunk
        if stmt["grounding_verified"]:
            assert stmt["verbatim_quote"] in sample_chunk


def test_p2_extracted_evidence_file() -> None:
    assert PROCESSED_FILE.exists(), f"Processed evidence file not found at {PROCESSED_FILE}"
    with PROCESSED_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)
    assert len(data) >= 1000
    first = data[0]
    assert "statement_id" in first
    assert "doc_id" in first
    assert "grounding_verified" in first
    assert "confidence" in first

    # Verify domain expert coverage across technical experts, think tanks, and regulators
    source_types = set(s.get("source_type") for s in data)
    assert "TechnicalExpert" in source_types
    assert "ThinkTankAnalysis" in source_types
    assert "RegulatoryAuthority" in source_types
    assert "IndustryStakeholder" in source_types


def test_multidomain_expert_opinion_dataset() -> None:
    assert MULTIDOMAIN_EXPERT_PQ.exists(), f"Multi-domain expert dataset missing at {MULTIDOMAIN_EXPERT_PQ}"
    df = pd.read_parquet(MULTIDOMAIN_EXPERT_PQ)
    assert len(df) >= 800
    assert set(df["domain_id"].unique()) == {1, 2, 3, 4, 5, 6}
    assert len(df["policy_case_id"].unique()) == 9

    # Verify zero Tamil script leakage per research scope
    assert not df["core_argument"].astype(str).str.contains(r"[\u0B80-\u0BFF]", regex=True).any()
    assert not df["text_full_passage"].astype(str).str.contains(r"[\u0B80-\u0BFF]", regex=True).any()

    # Verify 100% verbatim source grounding
    assert df["grounding_verified"].all()
    for _, row in df.iterrows():
        assert str(row["core_argument"]) in str(row["text_full_passage"])

    required_cols = {
        "expert_opinion_id", "domain_id", "domain_name", "sub_focus", "policy_case_id",
        "legal_anchor", "expert_name", "primary_affiliation", "stakeholder_category",
        "primary_discipline", "source_type", "source_url_or_doc_id", "publication_date",
        "temporal_window", "language", "text_full_passage", "core_argument", "char_span",
        "grounding_verified", "statement_type", "stance_label", "aspect_category",
        "expert_mechanism", "confidence", "annotation_method",
    }
    assert required_cols.issubset(set(df.columns))

    gold_file = PROJECT_ROOT.parent / "Datasets" / "23_sl_multidomain_expert_opinion" / "gold_expert_annotation_sample_300.csv"
    assert gold_file.exists(), "300-row Gold expert verification sample sheet missing"
