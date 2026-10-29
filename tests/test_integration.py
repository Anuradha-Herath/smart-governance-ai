"""Automated Integration Tests for Smart Governance Decision Synthesis (Phase 7)."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from src.integration.synthesis import PolicyDecisionSynthesizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BRIEFING_JSON = PROJECT_ROOT / "data" / "processed" / "synthesis_briefing.json"
BRIEFING_MD = PROJECT_ROOT / "outputs" / "reports" / "policy_briefing_DEC_2023_ELEC_01.md"


@pytest.fixture
def synthesizer() -> PolicyDecisionSynthesizer:
    return PolicyDecisionSynthesizer()


def test_synthesizer_pillars_integrity(synthesizer: PolicyDecisionSynthesizer) -> None:
    """Verify that all three analytical pillars are populated in synthesis briefing."""
    briefing = synthesizer.build_policy_briefing()

    assert briefing["case_id"] == "DEC_2023_ELEC_01"
    assert "executive_summary" in briefing

    # Pillar 1 checks
    p1 = briefing["pillar1_public_opinion"]
    assert "stance_distribution_pct" in p1
    assert "ground_truth_sentiment_pct" in p1
    assert "stance_validation_warning" in p1
    assert "grievance_themes" in p1
    assert len(p1["grievance_themes"]) >= 3
    for g in p1["grievance_themes"]:
        assert g.get("comment_id"), "Every grievance quote must have a real comment_id"
        assert len(g.get("sample_quote", "")) > 10

    # Pillar 2 checks
    p2 = briefing["pillar2_expert_evidence"]
    assert p2["verbatim_grounding_rate"] == "99.88%"
    assert "top_citations" in p2
    assert len(p2["top_citations"]["empirical_evidence"]) >= 2
    for cat_items in p2["top_citations"].values():
        for cit in cat_items:
            assert cit.get("statement_id"), "Every top citation must trace to a real statement_id"
            assert cit.get("grounded") is True

    # Pillar 3 checks
    p3 = briefing["pillar3_historical_indicators"]
    assert "historical_precedents" in p3
    assert len(p3["historical_precedents"]) >= 1
    assert "indicator_trajectories" in p3
    assert len(p3["indicator_trajectories"]) >= 3


def test_tensions_mapping(synthesizer: PolicyDecisionSynthesizer) -> None:
    """Verify cross-pipeline tension points and resolutions."""
    tensions = synthesizer.synthesize_policy_tensions()
    assert len(tensions) >= 3
    for t in tensions:
        assert "tension_name" in t
        assert "public_perspective" in t
        assert "expert_perspective" in t
        assert "governance_resolution" in t


def test_evidence_grounded_recommendations(synthesizer: PolicyDecisionSynthesizer) -> None:
    """Ensure all policy recommendations are backed by empirical evidence."""
    briefing = synthesizer.build_policy_briefing()
    recs = briefing["evidence_based_recommendations"]
    assert len(recs) >= 3
    for r in recs:
        assert "action_title" in r
        assert "detail" in r
        assert "evidence_backing" in r
        assert len(r["evidence_backing"]) > 10


def test_mandatory_governance_disclaimers(synthesizer: PolicyDecisionSynthesizer) -> None:
    """Ensure both sampling and causal disclaimers are preserved in output."""
    briefing = synthesizer.build_policy_briefing()
    caveats = briefing["governance_caveats"]
    assert "sampling_limitation" in caveats
    assert "causal_inference_limitation" in caveats
    assert "Disclaimer" in caveats["sampling_limitation"]
    assert "CAUSAL INFERENCE DISCLAIMER" in caveats["causal_inference_limitation"]


def test_exported_artifacts_on_disk() -> None:
    """Verify saved JSON and Markdown policy briefing files."""
    assert BRIEFING_JSON.exists(), f"Missing {BRIEFING_JSON}"
    assert BRIEFING_MD.exists(), f"Missing {BRIEFING_MD}"

    with BRIEFING_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["case_id"] == "DEC_2023_ELEC_01"

    md_text = BRIEFING_MD.read_text(encoding="utf-8")
    assert "# Evidence-Based Policy Briefing" in md_text
    assert "Pillar 1: Public Opinion" in md_text
    assert "Pillar 2: Grounded Expert Evidence" in md_text
    assert "Pillar 3: Historical Precedents" in md_text
