"""Master Exploratory Data Analysis & Ingestion Orchestrator for Phase 3.

Runs repeatable ingestion for all three pipelines, generates publication-ready
diagnostic figures, writes the formal data dictionary, and compiles the
Phase 3 Exploratory Analysis Report.
"""

from __future__ import annotations

import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.common.data_inspector import format_inspection_markdown
from src.pipeline1_public_opinion.ingest import ingest_pipeline1
from src.pipeline2_expert_opinion.ingest import ingest_pipeline2
from src.pipeline3_historical_analysis.ingest import ingest_pipeline3

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
DATA_DIR = PROJECT_ROOT / "data"


def generate_figures(p1_df: pd.DataFrame, p2_df: pd.DataFrame, p3_df: pd.DataFrame) -> dict[str, str]:
    """Generate and save publication-ready diagnostic charts."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig_paths = {}

    # Figure 1: Pipeline 1 Sentiment Distribution by Language
    plt.figure(figsize=(9, 5))
    sns.set_theme(style="whitegrid")
    ax1 = sns.countplot(
        data=p1_df,
        x="language",
        hue="sentiment_label",
        palette={"Positive": "#2ecc71", "Neutral": "#95a5a6", "Negative": "#e74c3c"},
    )
    plt.title("Pipeline 1: Sentiment Distribution Across Language Subsets (N=30,398)", fontsize=13, fontweight="bold")
    plt.xlabel("Language Variety", fontsize=11)
    plt.ylabel("Number of Comments", fontsize=11)
    plt.legend(title="Sentiment Polarity")
    for p in ax1.patches:
        h = p.get_height()
        if h > 0:
            ax1.annotate(f"{int(h):,}", (p.get_x() + p.get_width() / 2., h / 2),
                         ha="center", va="center", color="white", fontsize=9, fontweight="bold")
    plt.tight_layout()
    p1_fig_path = FIGURES_DIR / "p1_sentiment_distribution.png"
    plt.savefig(p1_fig_path, dpi=200)
    plt.close()
    fig_paths["p1_figure"] = str(p1_fig_path)

    # Figure 2: Pipeline 2 Expert Passages by Source
    plt.figure(figsize=(8, 4.5))
    ax2 = sns.countplot(
        data=p2_df,
        x="source_type",
        hue="source_type",
        palette="viridis",
        legend=False,
    )
    plt.title("Pipeline 2: Case Study Policy Matches (DEC_2023_ELEC_01, N=763)", fontsize=13, fontweight="bold")
    plt.xlabel("Official Source Type", fontsize=11)
    plt.ylabel("Policy-Relevant Text Passages", fontsize=11)
    for p in ax2.patches:
        h = p.get_height()
        if h > 0:
            ax2.annotate(f"{int(h):,}", (p.get_x() + p.get_width() / 2., h + 10),
                         ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.tight_layout()
    p2_fig_path = FIGURES_DIR / "p2_expert_doc_keyword_matches.png"
    plt.savefig(p2_fig_path, dpi=200)
    plt.close()
    fig_paths["p2_figure"] = str(p2_fig_path)

    # Figure 3: Pipeline 3 Outcome Indicators Pre/Post Window (2018-2025)
    plt.figure(figsize=(10, 5))
    cpi_data = p3_df[p3_df["indicator_code"] == "FP.CPI.TOTL.ZG"].copy()
    cpi_recent = cpi_data[(cpi_data["year"] >= 2018) & (cpi_data["year"] <= 2025)]
    if not cpi_recent.empty:
        plt.plot(cpi_recent["year"], cpi_recent["value"], marker="o", color="#d35400", linewidth=2.5, label="Headline Inflation (Annual %)")
        plt.axvline(x=2023, color="#c0392b", linestyle="--", linewidth=1.8, label="2023 Electricity Tariff Revision")
        plt.axvspan(2021, 2022, color="#f39c12", alpha=0.15, label="Pre-Reform Window")
        plt.axvspan(2023, 2025, color="#27ae60", alpha=0.15, label="Post-Reform Window")
        plt.title("Pipeline 3: Sri Lanka Inflation Trajectory Around February 2023 Tariff Revision", fontsize=13, fontweight="bold")
        plt.xlabel("Year", fontsize=11)
        plt.ylabel("Annual Inflation (%)", fontsize=11)
        plt.legend(loc="upper right")
        plt.tight_layout()
        p3_fig_path = FIGURES_DIR / "p3_pre_post_indicators.png"
        plt.savefig(p3_fig_path, dpi=200)
        plt.close()
        fig_paths["p3_figure"] = str(p3_fig_path)

    return fig_paths


def build_data_dictionary() -> dict[str, Any]:
    """Compile the formal data dictionary defining all interim datasets."""
    return {
        "pipeline1_public_opinion": {
            "file": "data/interim/p1_public_opinion_interim.parquet",
            "description": "Standardized multi-source public opinion comments.",
            "fields": {
                "comment_id": {"type": "string", "description": "Unique identifier (e.g. SI_NEWS_00001, SING_YT_00001)."},
                "text": {"type": "string", "description": "Preprocessed comment text with normalized whitespace."},
                "language": {"type": "string", "allowed_values": ["si", "singlish", "si_en_codemixed"], "description": "Linguistic classification."},
                "sentiment_label": {"type": "string", "allowed_values": ["Positive", "Negative", "Neutral"], "description": "Harmonized 3-class sentiment."},
                "source_dataset": {"type": "string", "description": "Originating research corpus."},
                "char_length": {"type": "integer", "description": "Character count of cleaned text."},
                "word_count": {"type": "integer", "description": "Token count based on whitespace splitting."},
            },
        },
        "pipeline2_expert_documents": {
            "file": "data/interim/p2_expert_documents_interim.parquet",
            "description": "Policy-relevant expert passages and debates filtered for target decision keywords.",
            "fields": {
                "doc_id": {"type": "string", "description": "Unique document or speech identifier."},
                "source_type": {"type": "string", "allowed_values": ["ParliamentaryHansard", "CabinetDecision", "CentralBankReport"], "description": "Document authority."},
                "date_str": {"type": "string", "description": "Publication or debate date."},
                "speaker_or_title": {"type": "string", "description": "Speaker name or document section title."},
                "text_chunk": {"type": "string", "description": "Verbatim text chunk containing matched arguments."},
                "char_length": {"type": "integer", "description": "Character count of chunk."},
                "word_count": {"type": "integer", "description": "Token count of chunk."},
                "matched_keywords": {"type": "string", "description": "Comma-separated policy keywords detected in chunk."},
                "keyword_match_count": {"type": "integer", "description": "Number of unique target keywords present."},
            },
        },
        "pipeline3_indicators": {
            "file": "data/interim/p3_indicators_interim.csv",
            "description": "Sri Lanka macroeconomic and energy indicators aligned around the 2023 reform window.",
            "fields": {
                "indicator_code": {"type": "string", "description": "World Bank / CBSL indicator code (e.g., FP.CPI.TOTL.ZG)."},
                "indicator_name": {"type": "string", "description": "Human-readable indicator description."},
                "impact_category": {"type": "string", "allowed_values": ["Economic", "Social", "Sector-specific"], "description": "Governance outcome domain."},
                "year": {"type": "integer", "description": "Observation year (1980-2025)."},
                "value": {"type": "float", "description": "Recorded indicator measurement."},
                "policy_window": {"type": "string", "allowed_values": ["Historical Pre-Crisis", "Pre-Implementation (2021-2022)", "Implementation Year (2023)", "Post-Implementation (2024-2025)"], "description": "Temporal categorization relative to February 2023."},
                "baseline_2022_value": {"type": "float", "description": "Indicator value in pre-reform base year 2022."},
                "delta_from_baseline": {"type": "float", "description": "Absolute delta (value - baseline_2022_value)."},
                "pct_change_from_baseline": {"type": "float", "description": "Percentage change relative to 2022."},
            },
        },
    }


def main() -> None:
    print("==================================================")
    print("Starting Phase 3: Exploratory Data Analysis & Ingestion")
    print("==================================================")

    # 1. Ingest all 3 pipelines
    print("\n--- Pipeline 1 Ingestion ---")
    p1_df, p1_diag = ingest_pipeline1(save_interim=True)

    print("\n--- Pipeline 2 Ingestion ---")
    p2_df, p2_diag = ingest_pipeline2(save_interim=True)

    print("\n--- Pipeline 3 Ingestion ---")
    p3_df, p3_diag = ingest_pipeline3(save_interim=True)

    # 2. Generate Figures
    print("\nGenerating Diagnostic Figures...")
    fig_paths = generate_figures(p1_df, p2_df, p3_df)
    for k, p in fig_paths.items():
        print(f"  Saved figure: {p}")

    # 3. Save Data Dictionary
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dd_path = DATA_DIR / "data_dictionary.json"
    data_dict = build_data_dictionary()
    dd_path.write_text(json.dumps(data_dict, indent=2), encoding="utf-8")
    print(f"\nSaved Data Dictionary: {dd_path}")

    # 4. Generate Comprehensive Markdown Report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "phase3_eda_report.md"

    report_lines = [
        "# Phase 3: Data Ingestion and Exploratory Data Analysis (EDA) Report",
        "",
        "**Target Case Study**: Sri Lanka Electricity Tariff Revisions (`DEC_2023_ELEC_01`)  ",
        "**Date**: October 2026",
        "",
        "---",
        "",
        "## 1. Executive Summary",
        f"- **Pipeline 1 Public Opinion Corpus**: Ingested and standardized **{len(p1_df):,}** comments across Sinhala (`si`), Singlish, and Sinhala-English code-mixed social media.",
        f"- **Pipeline 2 Expert Document Corpus**: Scanned official parliamentary Hansards and Cabinet decisions, identifying **{len(p2_df):,}** policy-relevant expert passages discussing electricity tariffs, CEB, and PUCSL.",
        f"- **Pipeline 3 Policy Outcome Indicators**: Normalized **{len(p3_df):,}** time-series indicator records across economic, energy, and social domains, aligned around the February 2023 reform baseline.",
        "",
        "---",
        "",
        "## 2. Pipeline 1: Public Opinion & Sentiment Diagnostics",
        format_inspection_markdown(p1_diag, title="Pipeline 1 Standardized Comments"),
        "",
        "### Sentiment Class Distribution",
        "| Sentiment | Count | Proportion |",
        "| :--- | ---: | ---: |",
    ]
    for sent, count in p1_diag["sentiment_distribution"].items():
        pct = (count / len(p1_df)) * 100
        report_lines.append(f"| **{sent}** | {count:,} | {pct:.2f}% |")

    report_lines.extend([
        "",
        "### Language Distribution",
        "| Language Variety | Count | Proportion |",
        "| :--- | ---: | ---: |",
    ])
    for lang, count in p1_diag["language_distribution"].items():
        pct = (count / len(p1_df)) * 100
        report_lines.append(f"| `{lang}` | {count:,} | {pct:.2f}% |")

    report_lines.extend([
        "",
        "---",
        "",
        "## 3. Pipeline 2: Expert Documents & Evidence Passages",
        format_inspection_markdown(p2_diag, title="Pipeline 2 Case Study Passages"),
        "",
        "### Source Distribution",
        "| Official Source | Policy-Relevant Passages |",
        "| :--- | ---: |",
    ])
    for src, count in p2_diag["source_distribution"].items():
        report_lines.append(f"| **{src}** | {count:,} |")

    report_lines.extend([
        "",
        "---",
        "",
        "## 4. Pipeline 3: Macroeconomic & Energy Outcome Indicators",
        format_inspection_markdown(p3_diag, title="Pipeline 3 Normalized Indicator Series"),
        "",
        "### Policy Windows Alignment",
        "| Policy Window | Records Count |",
        "| :--- | ---: |",
    ])
    for win, count in p3_diag["policy_windows_count"].items():
        report_lines.append(f"| **{win}** | {count:,} |")

    report_lines.extend([
        "",
        "---",
        "",
        "## 5. Methodological Safeguards & Data Leakage Assessment",
        "1. **Sampling Bias**: Online comments over-represent urban, internet-connected consumers and cannot be treated as a randomized poll of all Sri Lankan households.",
        "2. **Causal Confounding**: Pre/post changes in inflation or electricity consumption reflect multi-factor economic adjustments (e.g. currency floating, fuel pricing, sovereign debt restructuring) and are documented as descriptive associations rather than direct policy causes.",
        "3. **Data Leakage Prevention**: Split generation in Phase 4 will use strictly grouped splits by document/source to prevent identical comment variants from leaking across train and test folds.",
    ])

    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\nSaved Comprehensive EDA Report: {report_path}")
    print("\nPhase 3 EDA Complete!")


if __name__ == "__main__":
    main()
