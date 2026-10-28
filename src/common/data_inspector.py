"""Dataset inspection and exploratory quality assessment utilities.

Computes repeatable summary statistics, missing-value ratios, duplicates,
text length distributions, and categorical balances.
"""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd


def inspect_dataframe(df: pd.DataFrame, text_columns: list[str] | None = None) -> dict[str, Any]:
    """Generate a comprehensive diagnostic dictionary for a DataFrame."""
    total_rows = len(df)
    total_cols = len(df.columns)

    # Missing values
    null_counts = df.isnull().sum().to_dict()
    null_ratios = {k: round(v / total_rows, 4) if total_rows > 0 else 0.0 for k, v in null_counts.items()}

    # Duplicates
    dup_count = int(df.duplicated().sum())
    dup_ratio = round(dup_count / total_rows, 4) if total_rows > 0 else 0.0

    # Text length metrics
    text_stats: dict[str, Any] = {}
    if text_columns:
        for col in text_columns:
            if col in df.columns:
                lengths = df[col].dropna().astype(str).str.len()
                if len(lengths) > 0:
                    text_stats[col] = {
                        "count": int(len(lengths)),
                        "min": int(lengths.min()),
                        "max": int(lengths.max()),
                        "mean": round(float(lengths.mean()), 2),
                        "median": float(lengths.median()),
                        "p95": float(np.percentile(lengths, 95)),
                        "empty_or_whitespace_count": int((df[col].dropna().astype(str).str.strip() == "").sum()),
                    }

    # Column dtypes
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}

    return {
        "total_rows": total_rows,
        "total_columns": total_cols,
        "dtypes": dtypes,
        "null_counts": null_counts,
        "null_ratios": null_ratios,
        "duplicate_rows": dup_count,
        "duplicate_ratio": dup_ratio,
        "text_statistics": text_stats,
    }


def format_inspection_markdown(inspection: dict[str, Any], title: str = "Dataset Inspection") -> str:
    """Format diagnostic metrics into GitHub-flavored Markdown."""
    lines = [
        f"### {title}",
        f"- **Total Records**: {inspection['total_rows']:,}",
        f"- **Total Columns**: {inspection['total_columns']}",
        f"- **Duplicate Rows**: {inspection['duplicate_rows']:,} ({inspection['duplicate_ratio'] * 100:.2f}%)",
        "",
        "| Column | Type | Missing Count | Missing % |",
        "| :--- | :--- | ---: | ---: |",
    ]
    for col, dtype in inspection["dtypes"].items():
        null_c = inspection["null_counts"].get(col, 0)
        null_r = inspection["null_ratios"].get(col, 0.0) * 100
        lines.append(f"| `{col}` | `{dtype}` | {null_c:,} | {null_r:.2f}% |")

    if inspection.get("text_statistics"):
        lines.extend([
            "",
            "#### Text Length Metrics (Characters)",
            "| Column | Count | Min | Median | Mean | 95th % | Empty Count |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for col, stats in inspection["text_statistics"].items():
            lines.append(
                f"| `{col}` | {stats['count']:,} | {stats['min']} | {stats['median']:.1f} | "
                f"{stats['mean']:.1f} | {stats['p95']:.1f} | {stats['empty_or_whitespace_count']:,} |"
            )

    return "\n".join(lines)
