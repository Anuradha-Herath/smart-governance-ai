"""Pipeline 3 Retrieval Evaluation and Historical Trajectory Synthesis Module.

Evaluates the historical decision retrieval engine on benchmark policy queries,
computes standard information retrieval metrics (Hit@1, Hit@3, Hit@5, MRR),
and generates the Phase 6 historical analysis report.

Output:
- outputs/evaluation/p3_retrieval_metrics.json
- outputs/reports/phase6_historical_analysis_report.md
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

from src.pipeline3_historical_analysis.retrieval import HistoricalDecisionRetriever
from src.pipeline3_historical_analysis.trajectory import IndicatorTrajectoryAnalyzer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = PROJECT_ROOT / "outputs" / "evaluation"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"

# Benchmark evaluation queries representing key policy search scenarios in Sri Lanka
BENCHMARK_QUERIES: list[dict[str, Any]] = [
    {
        "query_id": "Q1_ELEC_TARIFF_2023",
        "query": "Proposal for revision of electricity tariff Ceylon Electricity Board 2023",
        "target_date": "2023-01-09",
        "target_keywords": ["electricity tariff", "ceb", "2023"],
    },
    {
        "query_id": "Q2_FUEL_PRICE_FORMULA",
        "query": "Introduction of a fuel price mechanism selling price petroleum",
        "target_date": "2022-05-23",
        "target_keywords": ["fuel price", "petroleum"],
    },
    {
        "query_id": "Q3_RENEWABLE_TARIFF",
        "query": "Revision of the Renewable Energy Purchase Tariff standard power purchase",
        "target_date": "2024-07-01",
        "target_keywords": ["renewable energy", "tariff"],
    },
    {
        "query_id": "Q4_ELECTRICITY_ACT_AMEND",
        "query": "Introducing amendments to the Sri Lanka Electricity Act No 20 of 2009",
        "target_date": "2022-04-25",
        "target_keywords": ["electricity act", "amendments"],
    },
    {
        "query_id": "Q5_SOLAR_ROOFTOP",
        "query": "Revision of Solar Rooftop Tariff Ceylon Electricity Board",
        "target_date": "2022-10-25",
        "target_keywords": ["solar rooftop", "tariff"],
    },
]


def evaluate_retrieval_engine() -> dict[str, Any]:
    """Evaluate retrieval engine across benchmark policy queries."""
    retriever = HistoricalDecisionRetriever()
    retriever.build_index()

    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0
    reciprocal_ranks: list[float] = []
    latencies_ms: list[float] = []
    query_details: list[dict[str, Any]] = []

    for bq in BENCHMARK_QUERIES:
        t0 = time.perf_counter()
        results = retriever.search(bq["query"], top_k=5)
        latency = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(latency)

        # Check rank of target match by date or target keywords
        rank = None
        for idx, res in enumerate(results, start=1):
            date_match = (res["date_str"] == bq["target_date"])
            title_lower = res["title"].lower()
            kw_match = all(kw in title_lower for kw in bq["target_keywords"])
            if date_match or kw_match:
                rank = idx
                break

        if rank == 1:
            hits_at_1 += 1
            hits_at_3 += 1
            hits_at_5 += 1
            reciprocal_ranks.append(1.0)
        elif rank and rank <= 3:
            hits_at_3 += 1
            hits_at_5 += 1
            reciprocal_ranks.append(1.0 / rank)
        elif rank and rank <= 5:
            hits_at_5 += 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

        query_details.append({
            "query_id": bq["query_id"],
            "query": bq["query"],
            "target_date": bq["target_date"],
            "found_rank": rank,
            "top_match_title": results[0]["title"] if results else None,
            "top_match_score": results[0]["similarity_score"] if results else 0.0,
            "latency_ms": round(latency, 2),
        })

    total_q = len(BENCHMARK_QUERIES)
    metrics = {
        "benchmark_queries_count": total_q,
        "hit_at_1_rate": round(hits_at_1 / total_q, 4),
        "hit_at_3_rate": round(hits_at_3 / total_q, 4),
        "hit_at_5_rate": round(hits_at_5 / total_q, 4),
        "mean_reciprocal_rank_mrr": round(sum(reciprocal_ranks) / total_q, 4),
        "avg_query_latency_ms": round(sum(latencies_ms) / total_q, 2),
        "query_evaluations": query_details,
    }
    return metrics


def generate_phase6_report(
    retrieval_metrics: dict[str, Any],
    trajectory_data: dict[str, Any],
) -> str:
    """Generate Markdown evaluation and trajectory synthesis report."""
    md_lines = [
        "# Phase 6: Historical Decision Retrieval & Indicator Trajectory Report",
        "",
        "## 1. Overview & Research Objectives",
        "- **Target Policy Case**: `DEC_2023_ELEC_01` (Sri Lanka Cost-Reflective Electricity Tariff Revision, February 2023)",
        "- **Retrieval Corpus**: 11,175 Sri Lanka Cabinet Decisions (2010-2026)",
        "- **Indicator Baseline**: World Bank / WDI Sri Lanka Outcome Indicators (1980-2025)",
        "- **Window Alignment**: Symmetric $t-2$ to $t+2$ years (2021 to 2025)",
        "",
        "## 2. Historical Decision Retrieval Evaluation",
        f"- **Mean Reciprocal Rank (MRR)**: `{retrieval_metrics['mean_reciprocal_rank_mrr']:.4f}`",
        f"- **Hit@1 Rate**: `{retrieval_metrics['hit_at_1_rate'] * 100:.1f}%`",
        f"- **Hit@3 Rate**: `{retrieval_metrics['hit_at_3_rate'] * 100:.1f}%`",
        f"- **Hit@5 Rate**: `{retrieval_metrics['hit_at_5_rate'] * 100:.1f}%`",
        f"- **Average Query Latency**: `{retrieval_metrics['avg_query_latency_ms']:.2f} ms`",
        "",
        "| Query ID | Target Date | Found Rank | Top Matched Cabinet Decision | Score | Latency |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for q in retrieval_metrics["query_evaluations"]:
        md_lines.append(
            f"| `{q['query_id']}` | {q['target_date']} | **#{q['found_rank']}** | {q['top_match_title'][:45]}... | {q['top_match_score']} | {q['latency_ms']} ms |"
        )

    md_lines.extend([
        "",
        "## 3. Pre/Post Indicator Trajectories (DEC_2023_ELEC_01)",
        "",
        "| Indicator Code | Indicator Name | Baseline (2022) | Event (2023) | Post / Latest | % Delta | Qualitative Trend |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ])

    for code, t in trajectory_data["trajectories"].items():
        base_str = f"{t['baseline_value']:.2f}" if t['baseline_value'] is not None else "N/A"
        event_str = f"{t['event_value']:.2f}" if t['event_value'] is not None else "N/A"
        post_str = f"{t['post_value']:.2f}" if t['post_value'] is not None else "Pending"
        pct_str = f"{t['pct_change']:+.2f}%" if t['pct_change'] is not None else "N/A"
        md_lines.append(
            f"| `{code}` | {t['indicator_name'][:35]} | {base_str} | {event_str} | {post_str} | **{pct_str}** | {t['trend_direction']} |"
        )

    md_lines.extend([
        "",
        "## 4. Contextual Confounders (2021-2025)",
        f"A total of **{trajectory_data['confounders_count']} macro confounding events** were identified and attached to the analysis window:",
        "",
    ])

    for c in trajectory_data["confounding_events"]:
        md_lines.append(f"- **{c['year']} ({c['date_period']}) - {c['event_title']}** [{c['category']}]: {c['description']}")

    md_lines.extend([
        "",
        "## 5. Causal Inference Boundary & Ethical Governance Guidelines",
        "> [!IMPORTANT]",
        f"> {trajectory_data['causal_disclaimer']}",
        "",
        "---",
        "*Report auto-generated by Phase 6 Pipeline 3 Evaluation Module.*",
    ])

    return "\n".join(md_lines)


def run_pipeline3_evaluation() -> None:
    """Run full Phase 6 evaluation and write reports."""
    print("Evaluating Historical Decision Retrieval Engine on Benchmark Queries...")
    ret_metrics = evaluate_retrieval_engine()
    print(f"MRR: {ret_metrics['mean_reciprocal_rank_mrr']:.4f} | Hit@1: {ret_metrics['hit_at_1_rate']*100:.1f}% | Hit@3: {ret_metrics['hit_at_3_rate']*100:.1f}%")

    print("\nExecuting Trajectory Analysis on Case DEC_2023_ELEC_01...")
    analyzer = IndicatorTrajectoryAnalyzer()
    traj_data = analyzer.analyze_trajectory(decision_year=2023, window_years=2)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_file = EVAL_DIR / "p3_retrieval_metrics.json"
    metrics_file.write_text(json.dumps(ret_metrics, indent=2), encoding="utf-8")
    print(f"Saved retrieval metrics to: {metrics_file}")

    report_content = generate_phase6_report(ret_metrics, traj_data)
    report_file = REPORTS_DIR / "phase6_historical_analysis_report.md"
    report_file.write_text(report_content, encoding="utf-8")
    print(f"Saved Phase 6 Report to: {report_file}")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    run_pipeline3_evaluation()
