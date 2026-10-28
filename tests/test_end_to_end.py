"""End-to-End System Integration Test for the Smart Governance AI Framework."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from src.common.evaluation_suite import aggregate_master_metrics
from src.integration.synthesis import PolicyDecisionSynthesizer
from src.pipeline1_public_opinion.predict import analyze_comment
from src.pipeline2_expert_opinion.extractor import extract_grounded_evidence
from src.pipeline3_historical_analysis.retrieval import search_historical_decisions
from src.pipeline3_historical_analysis.trajectory import IndicatorTrajectoryAnalyzer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MASTER_EVAL_FILE = PROJECT_ROOT / "outputs" / "evaluation" / "master_evaluation_metrics.json"
LATEX_TABLES_FILE = PROJECT_ROOT / "outputs" / "evaluation" / "thesis_tables.tex"


def test_end_to_end_pipeline_workflow() -> None:
    """Test entire cross-pipeline workflow from raw user input to policy briefing."""
    # 1. Pipeline 1: Live Stance & Sentiment inference
    test_comment = "විදුලි බිල වැඩි කරපු එක සාමාන්‍ය මිනිස්සුන්ට දරාගන්න බෑ."
    p1_res = analyze_comment(test_comment, target_decision="DEC_2023_ELEC_01")
    assert p1_res["stance"] in ["Oppose", "Support", "Neutral_or_Unclear"]
    assert p1_res["sentiment"] in ["Negative", "Neutral", "Positive"]
    assert "sampling_disclaimer" in p1_res

    # 2. Pipeline 2: Grounded Evidence Extraction
    sample_expert_chunk = (
        "Ceylon Electricity Board incurred massive losses of Rs. 167 billion in 2022. "
        "The General Treasury is unable to provide subsidies within the current fiscal space. "
        "Approval was granted to revise tariffs to ensure uninterrupted electricity supply."
    )
    p2_res = extract_grounded_evidence(
        text_chunk=sample_expert_chunk,
        doc_id="E2E_TEST_DOC",
        source_type="CabinetDecision",
        date_str="2023-01-09",
        speaker_or_title="E2E Test Policy",
    )
    assert len(p2_res) > 0
    for stmt in p2_res:
        if stmt["grounding_verified"]:
            assert stmt["verbatim_quote"] in sample_expert_chunk

    # 3. Pipeline 3: Historical Retrieval & Trajectory
    p3_ret = search_historical_decisions("electricity tariff revision 2023 CEB", top_k=1)
    assert len(p3_ret) == 1
    assert "electricity" in p3_ret[0]["title"].lower() or "tariff" in p3_ret[0]["title"].lower()

    analyzer = IndicatorTrajectoryAnalyzer()
    p3_traj = analyzer.analyze_trajectory(decision_year=2023, window_years=2)
    assert p3_traj["decision_year"] == 2023
    assert len(p3_traj["confounding_events"]) > 0

    # 4. Multi-Pipeline Synthesis
    synth = PolicyDecisionSynthesizer()
    briefing = synth.build_policy_briefing()
    assert briefing["case_id"] == "DEC_2023_ELEC_01"
    assert len(briefing["cross_pipeline_tensions"]) >= 3
    assert len(briefing["evidence_based_recommendations"]) >= 3


def test_master_evaluation_artifacts_exist() -> None:
    """Ensure master metrics and LaTeX dissertation tables exist and are valid."""
    assert MASTER_EVAL_FILE.exists(), f"Missing {MASTER_EVAL_FILE}"
    assert LATEX_TABLES_FILE.exists(), f"Missing {LATEX_TABLES_FILE}"

    with MASTER_EVAL_FILE.open("r", encoding="utf-8") as f:
        master = json.load(f)

    assert "rq1_public_opinion" in master
    assert "rq2_expert_evidence" in master
    assert "rq3_historical_analysis" in master
    assert "rq4_decision_synthesis" in master

    tex = LATEX_TABLES_FILE.read_text(encoding="utf-8")
    assert "\\begin{table}" in tex
    assert "tab:pipeline1_metrics" in tex
    assert "tab:pipeline2_metrics" in tex
    assert "tab:pipeline3_retrieval" in tex
