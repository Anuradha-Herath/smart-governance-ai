"""Pipeline 2 Data Ingestion & Case Study Evidence Filtering Module.

Ingests expert and official policy document corpora:
1. Sri Lanka Parliamentary Hansard speeches (2017-2026)
2. Sri Lanka Cabinet Decisions (2010-2026)
3. Central Bank of Sri Lanka (CBSL) Annual Report text chunks

Filters documents matching DEC_2023_ELEC_01 case study keywords across
English, Sinhala, and Tamil.

Output: data/interim/p2_expert_documents_interim.parquet
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import pandas as pd
import yaml

from src.common.data_inspector import inspect_dataframe
from src.utils.data_loader import (
    load_cabinet_decisions,
    load_document_chunks,
    load_hansard_speeches,
)

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "case_study.yaml"
INTERIM_DIR = Path(__file__).resolve().parents[2] / "data" / "interim"


def load_case_keywords() -> list[str]:
    """Load target policy keywords from configs/case_study.yaml."""
    if not CONFIG_PATH.exists():
        return ["electricity", "tariff", "CEB", "PUCSL", "විදුලි ගාස්තු", "විදුලි බිල"]
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    kw_dict = cfg.get("keywords", {})
    all_kws: list[str] = []
    for lang_kws in kw_dict.values():
        all_kws.extend(lang_kws)
    return [k.strip() for k in all_kws if k.strip()]


def ingest_pipeline2(save_interim: bool = True) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Ingest, filter, and normalize policy-relevant expert texts for DEC_2023_ELEC_01."""
    keywords = load_case_keywords()
    records: list[dict[str, Any]] = []

    # 1. Parliamentary Hansards (Speeches)
    print("  Processing Parliamentary Hansard speeches...")
    hansard_df = load_hansard_speeches(split="train")
    for idx, row in hansard_df.iterrows():
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        # Search for keyword matches (case-insensitive for Latin letters)
        matched = [k for k in keywords if k.lower() in text.lower()]
        if matched:
            records.append({
                "doc_id": str(row.get("speech_id", f"HANS_{idx}")),
                "source_type": "ParliamentaryHansard",
                "date_str": str(row.get("date", "")),
                "speaker_or_title": str(row.get("speaker", "Unknown Speaker")),
                "text_chunk": text,
                "char_length": len(text),
                "word_count": len(text.split()),
                "matched_keywords": ", ".join(matched),
                "keyword_match_count": len(matched),
            })

    # 2. Cabinet Decisions
    print("  Processing Cabinet Decisions...")
    cab_df = load_cabinet_decisions()
    for idx, row in cab_df.iterrows():
        title = str(row.get("description", ""))
        body = str(row.get("decision_details_body", ""))
        full_text = f"{title}\n{body}".strip()
        if not full_text:
            continue
        matched = [k for k in keywords if k.lower() in full_text.lower()]
        if matched:
            records.append({
                "doc_id": str(row.get("doc_id", f"CAB_{idx}")),
                "source_type": "CabinetDecision",
                "date_str": str(row.get("date_str", "")),
                "speaker_or_title": title[:200],
                "text_chunk": full_text,
                "char_length": len(full_text),
                "word_count": len(full_text.split()),
                "matched_keywords": ", ".join(matched),
                "keyword_match_count": len(matched),
            })

    # 3. CBSL Annual Report Chunks (Sample top policy chunks to maintain speed)
    print("  Processing Central Bank Annual Report chunks...")
    try:
        cbsl_chunks = load_document_chunks("cbsl_annual_reports")
        for chunk in cbsl_chunks:
            text = str(chunk.get("text", "")).strip()
            if not text:
                continue
            matched = [k for k in keywords if k.lower() in text.lower()]
            if matched:
                doc_id = str(chunk.get("doc_id", chunk.get("chunk_id", "CBSL_CHUNK")))
                records.append({
                    "doc_id": doc_id,
                    "source_type": "CentralBankReport",
                    "date_str": str(chunk.get("date_str", "")),
                    "speaker_or_title": str(chunk.get("title", "CBSL Economic Review")),
                    "text_chunk": text,
                    "char_length": len(text),
                    "word_count": len(text.split()),
                    "matched_keywords": ", ".join(matched),
                    "keyword_match_count": len(matched),
                })
    except Exception as exc:
        print(f"    [Notice] CBSL chunks scan: {exc}")

    expert_df = pd.DataFrame(records)

    # Diagnostic inspection
    diagnostics = inspect_dataframe(expert_df, text_columns=["text_chunk"])
    diagnostics["source_distribution"] = expert_df["source_type"].value_counts().to_dict()
    diagnostics["total_policy_matches"] = len(expert_df)

    if save_interim:
        INTERIM_DIR.mkdir(parents=True, exist_ok=True)
        out_parquet = INTERIM_DIR / "p2_expert_documents_interim.parquet"
        expert_df.to_parquet(out_parquet, index=False)

        summary_file = INTERIM_DIR / "p2_expert_documents_summary.json"
        summary_file.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")

    return expert_df, diagnostics


if __name__ == "__main__":
    df, stats = ingest_pipeline2()
    print(f"\nPipeline 2 Ingestion Complete: {len(df):,} policy-matched expert passages.")
    print("Source distribution:", stats["source_distribution"])
