"""FastAPI backend for Smart Governance AI Decision Support Platform.

Provides unified REST endpoints serving:
1. Executive Synthesis & Policy Briefings
2. Decision-linked Multilingual Public Comments (P1)
3. Grounded Expert Statements & Verbatim Quotes (P2)
4. Cabinet Precedents & Macroeconomic Trajectories (P3)
5. Live Policy Stance & Sentiment Inference
"""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import sys
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline1_public_opinion.predict import analyze_comment
from src.pipeline3_historical_analysis.retrieval import search_historical_decisions
from src.pipeline3_historical_analysis.trajectory import IndicatorTrajectoryAnalyzer

app = FastAPI(
    title="Smart Governance AI API",
    description="Evidence-Based National Policy Decision Support Backend",
    version="1.0.0",
)

# Enable CORS for local Vite frontend and modern web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = PROJECT_ROOT / "data"
BRIEFING_JSON = DATA_DIR / "processed" / "synthesis_briefing.json"
BRIEFING_MD = PROJECT_ROOT / "outputs" / "reports" / "policy_briefing_DEC_2023_ELEC_01.md"
P1_DECISION_COMMENTS = DATA_DIR / "processed" / "p1_decision_linked_comments.parquet"
P1_INTERIM = DATA_DIR / "interim" / "p1_public_opinion_interim.parquet"
P2_PROCESSED = DATA_DIR / "processed" / "p2_extracted_evidence.json"
P3_PROCESSED = DATA_DIR / "processed" / "p3_historical_trajectories.json"

POLICIES: dict[str, dict[str, Any]] = {
    "DEC_2023_ELEC_01": {
        "id": "DEC_2023_ELEC_01",
        "label": "DEC_2023_ELEC_01 · Tariff Revision (Feb 2023)",
        "title": "Cost-Reflective Electricity Tariff Revision (February 2023)",
        "year": 2023,
        "date": "2023-02-15",
        "badge": "ACTIVE POLICY CASE STUDY",
        "domain": "Energy & Public Utilities",
        "desc": "Comprehensive evaluation of Sri Lanka's 66% retail power tariff restructuring approved by the Cabinet of Ministers on January 9, 2023 (Directive 2023-01-09-001-en) and enacted by PUCSL on February 15, 2023 to eliminate untargeted subsidies, halt the Ceylon Electricity Board (CEB) Rs. 167B fiscal bleed, and end daily 10–14 hour islandwide blackouts.",
        "meta": [
            {"key": "Enactment Date", "value": "February 15, 2023"},
            {"key": "Governing Authority", "value": "Cabinet of Ministers & PUCSL"},
            {"key": "Affected Consumer Base", "value": "4.8M Domestic, SME & Industrial Units"},
            {"key": "Stated Primary Mandate", "value": "CEB Fiscal Solvency & IMF EFF Targets"},
        ],
        "vline": "Tariff Revision (Feb 2023)",
    },
    "DEC_2022_ELEC_01": {
        "id": "DEC_2022_ELEC_01",
        "label": "DEC_2022_ELEC_01 · Tariff Hike (Aug 2022)",
        "title": "Initial Electricity Tariff Revision (August 2022)",
        "year": 2022,
        "date": "2022-08-10",
        "badge": "COMPARATIVE CASE · PRECEDING DECISION",
        "domain": "Energy & Public Utilities",
        "desc": "First electricity tariff increase after a 9-year freeze (weighted average of ~75%), introduced at the peak of the fuel and foreign-exchange crisis to recover skyrocketing generation costs.",
        "meta": [
            {"key": "Decision Date", "value": "August 10, 2022"},
            {"key": "Governing Authority", "value": "PUCSL"},
            {"key": "Relation to Primary Case", "value": "Preceded the Feb 2023 revision"},
            {"key": "Analysis Window", "value": "2020 – 2024 (t-2 to t+2)"},
        ],
        "vline": "Tariff Revision (Aug 2022)",
    },
    "DEC_2024_ELEC_01": {
        "id": "DEC_2024_ELEC_01",
        "label": "DEC_2024_ELEC_01 · Tariff Cut (Mar 2024)",
        "title": "Downward Electricity Tariff Revision (March 2024)",
        "year": 2024,
        "date": "2024-03-04",
        "badge": "COMPARATIVE CASE · SUBSEQUENT DECISION",
        "domain": "Energy & Public Utilities",
        "desc": "Subsequent downward tariff adjustment (~ -21.9% in March 2024, followed by a further cut in July) after heavy monsoon hydro generation and lower fuel costs.",
        "meta": [
            {"key": "Decision Date", "value": "March 4, 2024"},
            {"key": "Governing Authority", "value": "PUCSL"},
            {"key": "Relation to Primary Case", "value": "Followed the Feb 2023 revision"},
            {"key": "Analysis Window", "value": "2022 – 2026 (data available to 2025)"},
        ],
        "vline": "Tariff Cut (Mar 2024)",
    },
}


# -----------------------------------------------------------------------------
# Data Loading Helpers
# -----------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_synthesis_data() -> dict[str, Any]:
    if not BRIEFING_JSON.exists():
        from src.integration.synthesis import PolicyDecisionSynthesizer
        synth = PolicyDecisionSynthesizer()
        briefing, _, _ = synth.export_briefing()
        return briefing
    with BRIEFING_JSON.open("r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def get_briefing_markdown() -> str:
    if BRIEFING_MD.exists():
        return BRIEFING_MD.read_text(encoding="utf-8")
    return "Briefing markdown report not yet generated."


@lru_cache(maxsize=1)
def get_comments_df() -> pd.DataFrame:
    if P1_DECISION_COMMENTS.exists():
        df = pd.read_parquet(P1_DECISION_COMMENTS)
        return df.rename(columns={"sentiment_label": "sentiment", "stance_label": "stance"})
    if P1_INTERIM.exists():
        from src.integration.synthesis import filter_p1_utility_comments
        df = pd.read_parquet(P1_INTERIM)
        sub = filter_p1_utility_comments(df)
        return sub[["comment_id", "text", "language", "sentiment_label"]].rename(
            columns={"sentiment_label": "sentiment"}
        )
    return pd.DataFrame()


@lru_cache(maxsize=1)
def get_p2_statements() -> list[dict[str, Any]]:
    if P2_PROCESSED.exists():
        with P2_PROCESSED.open("r", encoding="utf-8") as f:
            return json.load(f)
    return []


# -----------------------------------------------------------------------------
# Request & Response Models
# -----------------------------------------------------------------------------
class PredictRequest(BaseModel):
    text: str
    target_decision: str = "DEC_2023_ELEC_01"


# -----------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------
@app.get("/api/health")
def health_check() -> dict[str, str]:
    return {"status": "healthy", "service": "Smart Governance AI Decision Support API"}


@app.get("/api/policies")
def get_policies() -> dict[str, Any]:
    return {"policies": list(POLICIES.values()), "default": "DEC_2023_ELEC_01"}


@app.get("/api/synthesis")
def get_synthesis() -> dict[str, Any]:
    return get_synthesis_data()


@app.get("/api/briefing/markdown")
def get_markdown_report() -> dict[str, str]:
    return {"markdown": get_briefing_markdown()}


@app.get("/api/comments")
def get_comments(
    decision_id: Optional[str] = None,
    language: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> dict[str, Any]:
    df = get_comments_df()
    if df.empty:
        return {"total": 0, "comments": [], "languages": []}

    all_languages = [str(x) for x in df["language"].dropna().unique().tolist()]

    if decision_id and "decision_id" in df.columns:
        df = df[df["decision_id"] == decision_id]

    if language:
        df = df[df["language"] == language]

    if search:
        df = df[df["text"].str.contains(search, case=False, na=False)]

    total = len(df)
    page_df = df.iloc[offset : offset + limit].fillna("")
    comments = page_df.to_dict(orient="records")

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "languages": all_languages,
        "comments": comments,
    }


AUTHORITY_RANK = {
    "TechnicalExpert": 1,
    "RegulatoryAuthority": 2,
    "ThinkTankAnalysis": 3,
    "IndustryStakeholder": 4,
    "CabinetDecision": 5,
    "ParliamentaryHansard": 6,
}


@app.get("/api/expert-evidence")
def get_evidence(
    statement_type: Optional[str] = None,
    source_type: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> dict[str, Any]:
    stmts = get_p2_statements()
    synthesis = get_synthesis_data()
    top_citations = synthesis.get("pillar2_expert_evidence", {}).get("top_citations", {})
    grounding_rate = synthesis.get("pillar2_expert_evidence", {}).get("verbatim_grounding_rate", "99.88%")

    filtered = stmts
    if statement_type and statement_type != "All":
        filtered = [s for s in filtered if s.get("statement_type") == statement_type]

    if source_type and source_type != "All":
        filtered = [s for s in filtered if s.get("source_type") == source_type]

    if search:
        search_lower = search.lower()
        filtered = [
            s for s in filtered
            if search_lower in s.get("extracted_statement", "").lower()
            or search_lower in s.get("speaker_or_title", "").lower()
            or search_lower in s.get("doc_id", "").lower()
            or search_lower in s.get("source_type", "").lower()
        ]

    # Prioritize substantive domain experts, regulators, think tanks, and cabinet directives over general debate
    filtered = sorted(
        filtered,
        key=lambda s: AUTHORITY_RANK.get(s.get("source_type", ""), 99),
    )

    total = len(filtered)
    page = filtered[offset : offset + limit]
    all_source_types = sorted(list(set(s.get("source_type", "") for s in stmts if s.get("source_type"))))

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "verbatim_grounding_rate": grounding_rate,
        "top_citations": top_citations,
        "source_types": all_source_types,
        "statements": page,
    }


@app.get("/api/trajectories")
def get_trajectories(
    year: int = 2023,
    window_years: int = 2,
) -> dict[str, Any]:
    analyzer = IndicatorTrajectoryAnalyzer()
    res = analyzer.analyze_trajectory(decision_year=year, window_years=window_years)
    return res


@app.get("/api/historical/search")
def search_cabinet_decisions(
    q: str = Query(..., min_length=2),
    top_k: int = Query(5, ge=1, le=20),
) -> dict[str, Any]:
    results = search_historical_decisions(query=q, top_k=top_k)
    return {"query": q, "count": len(results), "results": results}


@app.post("/api/predict/stance")
def predict_stance(payload: PredictRequest) -> dict[str, Any]:
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    result = analyze_comment(payload.text, target_decision=payload.target_decision)
    return result


# -----------------------------------------------------------------------------
# Static SPA Hosting (Serves production React build if dist exists)
# -----------------------------------------------------------------------------
from fastapi.staticfiles import StaticFiles

FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")

