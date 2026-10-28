"""Pipeline 1 Data Ingestion & Normalization Module.

Combines, cleans, and standardizes multi-source public opinion datasets:
1. Sinhala News Comments (Azri-Muhsin / sinhala-nlp)
2. Singlish YouTube Comments (1516CS)
3. Sinhala-English Code-Mixed Comments (NLPC-UOM)

Output: data/interim/p1_public_opinion_interim.parquet
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import pandas as pd

from src.common.data_inspector import inspect_dataframe
from src.utils.data_loader import (
    load_codemixed_uom,
    load_singlish_sentiment,
    load_sinhala_sentiment,
)

INTERIM_DIR = Path(__file__).resolve().parents[2] / "data" / "interim"


def clean_text(text: Any) -> str:
    """Strip whitespace and normalize string representations."""
    if pd.isna(text):
        return ""
    s = str(text).strip()
    # Replace multiple internal whitespace/tabs/newlines with a single space
    return " ".join(s.split())


def ingest_pipeline1(save_interim: bool = True) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Ingest, clean, and standardize Pipeline 1 public opinion corpora."""
    records: list[dict[str, Any]] = []

    # 1. Sinhala News Comments (train + test splits)
    si_train = load_sinhala_sentiment(split="train")
    si_test = load_sinhala_sentiment(split="test")
    si_all = pd.concat([si_train, si_test], ignore_index=True)

    si_label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    for idx, row in si_all.iterrows():
        raw_text = clean_text(row.get("text", ""))
        if not raw_text:
            continue
        label_raw = row.get("label")
        sentiment = si_label_map.get(label_raw, "Neutral")
        records.append({
            "comment_id": f"SI_NEWS_{idx:05d}",
            "text": raw_text,
            "language": "si",
            "sentiment_label": sentiment,
            "source_dataset": "sinhala_news_comment_sentiment",
            "char_length": len(raw_text),
            "word_count": len(raw_text.split()),
        })

    # 2. Singlish YouTube Comments
    singlish_df = load_singlish_sentiment()
    sing_label_map = {"Pos": "Positive", "Neg": "Negative", "Neu": "Neutral"}
    for idx, row in singlish_df.iterrows():
        raw_text = clean_text(row.get("Comments", ""))
        if not raw_text:
            continue
        sentiment = sing_label_map.get(str(row.get("Sentiment")).strip(), "Neutral")
        records.append({
            "comment_id": f"SING_YT_{idx:05d}",
            "text": raw_text,
            "language": "singlish",
            "sentiment_label": sentiment,
            "source_dataset": "singlish_sentiment_youtube",
            "char_length": len(raw_text),
            "word_count": len(raw_text.split()),
        })

    # 3. UOM Sinhala-English Code-Mixed Comments
    uom_df = load_codemixed_uom(level="sentence")
    uom_label_map = {"Positive": "Positive", "Negative": "Negative", "Neutral": "Neutral", "Conflict": "Neutral"}
    for idx, row in uom_df.iterrows():
        raw_text = clean_text(row.get("Sentence", ""))
        if not raw_text:
            continue
        sentiment = uom_label_map.get(str(row.get("Sentiment")).strip(), "Neutral")
        records.append({
            "comment_id": f"UOM_CM_{idx:05d}",
            "text": raw_text,
            "language": "si_en_codemixed",
            "sentiment_label": sentiment,
            "source_dataset": "sinhala_english_codemixed_uom",
            "char_length": len(raw_text),
            "word_count": len(raw_text.split()),
        })

    unified_df = pd.DataFrame(records)

    # Diagnostic inspection
    diagnostics = inspect_dataframe(unified_df, text_columns=["text"])
    diagnostics["sentiment_distribution"] = unified_df["sentiment_label"].value_counts().to_dict()
    diagnostics["language_distribution"] = unified_df["language"].value_counts().to_dict()
    diagnostics["source_distribution"] = unified_df["source_dataset"].value_counts().to_dict()

    if save_interim:
        INTERIM_DIR.mkdir(parents=True, exist_ok=True)
        out_parquet = INTERIM_DIR / "p1_public_opinion_interim.parquet"
        unified_df.to_parquet(out_parquet, index=False)

        summary_file = INTERIM_DIR / "p1_public_opinion_summary.json"
        summary_file.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")

    return unified_df, diagnostics


if __name__ == "__main__":
    df, stats = ingest_pipeline1()
    print(f"Pipeline 1 Ingestion Complete: {len(df):,} comments.")
    print("Sentiment distribution:", stats["sentiment_distribution"])
    print("Language distribution:", stats["language_distribution"])
