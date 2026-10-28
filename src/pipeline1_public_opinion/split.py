"""Pipeline 1 Data Splitting Module.

Performs stratified, leak-free train/validation/test splitting across
combined language varieties (Sinhala, Singlish, Code-Mixed) and sentiment classes.

Output:
- data/processed/p1_train.parquet (70%)
- data/processed/p1_val.parquet   (15%)
- data/processed/p1_test.parquet  (15%)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INTERIM_FILE = PROJECT_ROOT / "data" / "interim" / "p1_public_opinion_interim.parquet"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def generate_pipeline1_splits(
    random_state: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> dict[str, Any]:
    """Generate reproducible, stratified train/validation/test splits."""
    if not INTERIM_FILE.exists():
        raise FileNotFoundError(f"Interim dataset not found at {INTERIM_FILE}. Run Phase 3 ingestion first.")

    df = pd.read_parquet(INTERIM_FILE)
    total_len = len(df)

    # Create composite stratification key to balance both language and sentiment classes
    strat_key = df["language"] + "_" + df["sentiment_label"]

    # First split: Train vs Temp (Val + Test)
    temp_ratio = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_ratio,
        random_state=random_state,
        stratify=strat_key,
    )

    # Second split: Val vs Test
    temp_strat_key = temp_df["language"] + "_" + temp_df["sentiment_label"]
    val_share_of_temp = val_ratio / temp_ratio
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_share_of_temp),
        random_state=random_state,
        stratify=temp_strat_key,
    )

    # Reset indices
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Verification: Zero comment_id overlap
    train_ids = set(train_df["comment_id"])
    val_ids = set(val_df["comment_id"])
    test_ids = set(test_df["comment_id"])

    assert len(train_ids.intersection(val_ids)) == 0, "Data leakage detected between Train and Val!"
    assert len(train_ids.intersection(test_ids)) == 0, "Data leakage detected between Train and Test!"
    assert len(val_ids.intersection(test_ids)) == 0, "Data leakage detected between Val and Test!"

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(PROCESSED_DIR / "p1_train.parquet", index=False)
    val_df.to_parquet(PROCESSED_DIR / "p1_val.parquet", index=False)
    test_df.to_parquet(PROCESSED_DIR / "p1_test.parquet", index=False)

    summary = {
        "total_records": total_len,
        "train_count": len(train_df),
        "val_count": len(val_df),
        "test_count": len(test_df),
        "train_pct": round(len(train_df) / total_len * 100, 2),
        "val_pct": round(len(val_df) / total_len * 100, 2),
        "test_pct": round(len(test_df) / total_len * 100, 2),
        "class_distributions": {
            "train": train_df["sentiment_label"].value_counts().to_dict(),
            "val": val_df["sentiment_label"].value_counts().to_dict(),
            "test": test_df["sentiment_label"].value_counts().to_dict(),
        },
        "language_distributions": {
            "train": train_df["language"].value_counts().to_dict(),
            "val": val_df["language"].value_counts().to_dict(),
            "test": test_df["language"].value_counts().to_dict(),
        },
    }

    summary_file = PROCESSED_DIR / "p1_split_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary


if __name__ == "__main__":
    stats = generate_pipeline1_splits()
    print("Pipeline 1 Splits Generated Successfully:")
    print(f"  Train: {stats['train_count']:,} ({stats['train_pct']}%)")
    print(f"  Val:   {stats['val_count']:,} ({stats['val_pct']}%)")
    print(f"  Test:  {stats['test_count']:,} ({stats['test_pct']}%)")
