"""Pipeline 3 Data Ingestion & Pre/Post Policy Indicator Normalization Module.

Ingests official Sri Lanka macroeconomic, energy, and social indicators (1980-2025).
Aligns indicators around the DEC_2023_ELEC_01 implementation window (2021-2025)
and computes relative trajectory metrics against the 2022 pre-reform baseline.

Output: data/interim/p3_indicators_interim.csv
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import pandas as pd
import yaml

from src.common.data_inspector import inspect_dataframe
from src.utils.data_loader import load_macro_indicators

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "case_study.yaml"
INTERIM_DIR = Path(__file__).resolve().parents[2] / "data" / "interim"


def load_target_indicator_codes() -> list[str]:
    """Load target indicators defined in configs/case_study.yaml."""
    if not CONFIG_PATH.exists():
        return ["FP.CPI.TOTL.ZG", "FP.CPI.TOTL", "EG.USE.ELEC.KH.PC", "EG.ELC.ACCS.ZS"]
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("indicators", {}).get("target_codes", [])


def ingest_pipeline3(save_interim: bool = True) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Ingest, align, and normalize pre/post policy indicators for DEC_2023_ELEC_01."""
    target_codes = load_target_indicator_codes()
    raw_df = load_macro_indicators()

    # Filter for target codes, or retain all if codes list is empty
    if target_codes:
        df = raw_df[raw_df["indicator_code"].isin(target_codes)].copy()
    else:
        df = raw_df.copy()

    # Sort deterministically
    df = df.sort_values(by=["indicator_code", "year"]).reset_index(drop=True)

    # Classify policy timing windows relative to February 2023 implementation
    def classify_window(year: int) -> str:
        if year < 2021:
            return "Historical Pre-Crisis"
        elif year in (2021, 2022):
            return "Pre-Implementation (2021-2022)"
        elif year == 2023:
            return "Implementation Year (2023)"
        else:
            return "Post-Implementation (2024-2025)"

    df["policy_window"] = df["year"].apply(classify_window)

    # Compute baseline 2022 value for each indicator
    baseline_2022 = (
        df[df["year"] == 2022]
        .set_index("indicator_code")["value"]
        .to_dict()
    )

    df["baseline_2022_value"] = df["indicator_code"].map(baseline_2022)
    df["delta_from_baseline"] = df["value"] - df["baseline_2022_value"]
    df["pct_change_from_baseline"] = (
        (df["delta_from_baseline"] / df["baseline_2022_value"].abs()) * 100.0
    ).round(2)

    diagnostics = inspect_dataframe(df)
    diagnostics["indicators_covered"] = df["indicator_code"].nunique()
    diagnostics["years_range"] = [int(df["year"].min()), int(df["year"].max())]
    diagnostics["policy_windows_count"] = df["policy_window"].value_counts().to_dict()

    if save_interim:
        INTERIM_DIR.mkdir(parents=True, exist_ok=True)
        out_csv = INTERIM_DIR / "p3_indicators_interim.csv"
        df.to_csv(out_csv, index=False)

        summary_file = INTERIM_DIR / "p3_indicators_summary.json"
        summary_file.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")

    return df, diagnostics


if __name__ == "__main__":
    df, stats = ingest_pipeline3()
    print(f"\nPipeline 3 Ingestion Complete: {len(df):,} indicator records.")
    print("Indicators covered:", stats["indicators_covered"])
    print("Policy windows:", stats["policy_windows_count"])
