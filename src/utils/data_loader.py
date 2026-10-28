"""Data loading utilities for accessing local research datasets across all pipelines.

Provides a unified interface to load tabular, text, and JSON records from the
central Datasets/ directory into Pandas DataFrames and dictionaries without duplicating
data files on disk.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd

# The Datasets/ directory sits adjacent to the Project folder
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = PROJECT_ROOT.parent / "Datasets"


def get_datasets_dir() -> Path:
    """Return the absolute path to the Datasets directory, ensuring it exists."""
    if not DATASETS_DIR.exists():
        raise FileNotFoundError(f"Datasets directory not found at: {DATASETS_DIR}")
    return DATASETS_DIR


def load_dataset_register() -> pd.DataFrame:
    """Load the master dataset register listing all 19 downloaded datasets."""
    reg_path = get_datasets_dir() / "dataset_register.csv"
    if not reg_path.exists():
        raise FileNotFoundError(f"Dataset register not found at {reg_path}")
    return pd.read_csv(reg_path)


# ------------------------------------------------------------------------------
# Pipeline 1 Loaders: Public Opinion & Stance
# ------------------------------------------------------------------------------

def load_sinhala_sentiment(split: str = "train", format: str = "parquet") -> pd.DataFrame:
    """Load the 9,048 Sinhala news comment sentiment dataset.

    Args:
        split: 'train' or 'test'.
        format: 'parquet' (from Azri-Muhsin) or 'tsv' (from sinhala-nlp).
    """
    ds_dir = get_datasets_dir()
    if format == "parquet":
        file_path = ds_dir / "sinhala_news_comment_sentiment" / f"{split}-00000-of-00001.parquet"
        if not file_path.exists():
            raise FileNotFoundError(f"Parquet split not found: {file_path}")
        return pd.read_parquet(file_path)
    elif format == "tsv":
        file_path = ds_dir / "sinhala_sentiment_analysis_tsv" / f"{split}.tsv"
        if not file_path.exists():
            raise FileNotFoundError(f"TSV split not found: {file_path}")
        return pd.read_csv(file_path, sep="\t")
    else:
        raise ValueError(f"Unsupported format '{format}'. Choose 'parquet' or 'tsv'.")


def load_singlish_sentiment() -> pd.DataFrame:
    """Load the 7,832 Singlish YouTube sentiment comments."""
    csv_path = get_datasets_dir() / "singlish_sentiment_youtube" / "Singlish_Dataset.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Singlish dataset not found at {csv_path}")
    return pd.read_csv(csv_path)


def load_codemixed_uom(level: str = "sentence") -> pd.DataFrame:
    """Load the NLPC-UOM Sinhala-English code-mixed and code-switched comments.

    Args:
        level: 'sentence' (sentence-level annotations) or 'word' (word-level).
    """
    fname = f"{level}-level-annotation.csv"
    csv_path = get_datasets_dir() / "sinhala_english_codemixed_uom" / fname
    if not csv_path.exists():
        raise FileNotFoundError(f"UOM dataset not found at {csv_path}")
    return pd.read_csv(csv_path)


def load_ibm_debater_argkp(subset: str = "stance_balanced") -> pd.DataFrame:
    """Load IBM Debater Argument Key Point and Stance benchmark.

    Args:
        subset: 'stance_balanced', 'balanced', or 'combined'.
    """
    fname = f"ArgKP_{subset}.csv"
    csv_path = get_datasets_dir() / "ibm_debater_argkp" / fname
    if not csv_path.exists():
        raise FileNotFoundError(f"ArgKP subset not found at {csv_path}")
    return pd.read_csv(csv_path)


# ------------------------------------------------------------------------------
# Pipeline 2 Loaders: Expert Opinion, Reports & Parliamentary Debates
# ------------------------------------------------------------------------------

def load_hansard_speeches(split: str = "train") -> pd.DataFrame:
    """Load Sri Lanka Parliamentary Hansard speeches (2017-2026)."""
    file_path = (
        get_datasets_dir()
        / "sl_parliamentary_hansard_17_26"
        / f"{split}-00000-of-00001.parquet"
    )
    if not file_path.exists():
        raise FileNotFoundError(f"Hansard split not found at {file_path}")
    return pd.read_parquet(file_path)


def load_document_chunks(dataset_id: str) -> list[dict[str, Any]]:
    """Load pre-extracted text chunks for any document corpus (e.g., 'cbsl_annual_reports',

    'lk_cabinet_decisions', 'lk_acts', 'lk_bills', 'lk_extraordinary_gazettes_2020s').
    """
    chunks_path = get_datasets_dir() / dataset_id / "chunks.json"
    if not chunks_path.exists():
        raise FileNotFoundError(f"chunks.json not found for dataset '{dataset_id}' at {chunks_path}")
    with chunks_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_document_metadata(dataset_id: str) -> pd.DataFrame:
    """Load document metadata table (docs_all.tsv) for a legal/governance corpus."""
    tsv_path = get_datasets_dir() / dataset_id / "docs_all.tsv"
    if not tsv_path.exists():
        raise FileNotFoundError(f"docs_all.tsv not found for dataset '{dataset_id}' at {tsv_path}")
    return pd.read_csv(tsv_path, sep="\t")


# ------------------------------------------------------------------------------
# Pipeline 3 Loaders: Historical Decisions & Macro Indicators
# ------------------------------------------------------------------------------

def load_cabinet_decisions() -> pd.DataFrame:
    """Load 11,175 Sri Lanka Cabinet Decisions (2010-2026)."""
    docs_json = get_datasets_dir() / "lk_cabinet_decisions" / "docs.json"
    if not docs_json.exists():
        raise FileNotFoundError(f"Cabinet decisions docs.json not found at {docs_json}")
    with docs_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)


def load_macro_indicators(impact_category: str | None = None) -> pd.DataFrame:
    """Load official Sri Lanka time-series indicators (1980-2025).

    Args:
        impact_category: Filter by 'Economic', 'Social', 'Sector-specific',
          or 'Governance' (WGI). If None, returns all indicators.
    """
    csv_path = get_datasets_dir() / "sri_lanka_macro_indicators" / "sri_lanka_outcome_indicators.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Macro indicators CSV not found at {csv_path}")
    df = pd.read_csv(csv_path)
    if impact_category:
        df = df[df["impact_category"].str.lower() == impact_category.lower()].copy()
    return df


# ------------------------------------------------------------------------------
# Generic Policy Filtering Utility
# ------------------------------------------------------------------------------

def filter_by_keywords(
    df: pd.DataFrame,
    text_column: str,
    keywords: list[str],
    case_sensitive: bool = False,
) -> pd.DataFrame:
    """Filter DataFrame rows where text_column contains any of the provided keywords.

    Handles NaN values and multi-language strings safely.
    """
    if text_column not in df.columns:
        raise KeyError(f"Column '{text_column}' not found in DataFrame. Available: {list(df.columns)}")

    valid_mask = df[text_column].notna()
    series = df.loc[valid_mask, text_column].astype(str)

    pattern = "|".join([k.strip() for k in keywords if k.strip()])
    match_mask = series.str.contains(pattern, case=case_sensitive, regex=True)

    return df.loc[valid_mask][match_mask].copy()
