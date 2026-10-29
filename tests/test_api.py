"""Unit tests for FastAPI endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient
import pytest

from src.api.main import app

client = TestClient(app)


def test_api_health() -> None:
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"


def test_api_policies() -> None:
    res = client.get("/api/policies")
    assert res.status_code == 200
    data = res.json()
    assert "policies" in data
    assert len(data["policies"]) >= 3
    assert data["default"] == "DEC_2023_ELEC_01"


def test_api_synthesis() -> None:
    res = client.get("/api/synthesis")
    assert res.status_code == 200
    data = res.json()
    assert "case_id" in data
    assert "cross_pipeline_tensions" in data


def test_api_comments() -> None:
    res = client.get("/api/comments?limit=5")
    assert res.status_code == 200
    data = res.json()
    assert "comments" in data
    assert "total" in data
    assert len(data["comments"]) <= 5


def test_api_expert_evidence() -> None:
    res = client.get("/api/expert-evidence?limit=5")
    assert res.status_code == 200
    data = res.json()
    assert "statements" in data
    assert "top_citations" in data
    assert "source_types" in data
    assert len(data["statements"]) <= 5

    # Test filtering by technical expert source
    res_expert = client.get("/api/expert-evidence?source_type=TechnicalExpert")
    assert res_expert.status_code == 200
    expert_data = res_expert.json()
    assert expert_data["total"] > 0
    assert any("Siyambalapitiya" in s["speaker_or_title"] or "Samarajiva" in s["speaker_or_title"] for s in expert_data["statements"])


def test_api_trajectories() -> None:
    res = client.get("/api/trajectories?year=2023")
    assert res.status_code == 200
    data = res.json()
    assert "trajectories" in data
    assert "confounding_events" in data


def test_api_historical_search() -> None:
    res = client.get("/api/historical/search?q=electricity tariff")
    assert res.status_code == 200
    data = res.json()
    assert "results" in data
    assert len(data["results"]) > 0


def test_api_predict_stance() -> None:
    payload = {"text": "විදුලි බිල වැඩි කරපු එක ජනතාවට දරාගන්න බෑ.", "target_decision": "DEC_2023_ELEC_01"}
    res = client.post("/api/predict/stance", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "stance" in data
    assert "sentiment" in data
