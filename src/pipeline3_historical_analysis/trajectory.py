"""Pipeline 3 Pre/Post Indicator Trajectory Analysis Module.

Analyzes macroeconomic and sector outcome indicators across symmetric time windows
(t - W to t + W years) around a policy intervention year (e.g. DEC_2023_ELEC_01 in 2023).

Features:
- Calculates baseline shifts, absolute change, percentage change, and trend direction.
- Strictly pairs trajectory metrics with verified historical confounders.
- Generates mandatory causal inference disclaimers to prevent erroneous attribution.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import pandas as pd
import yaml

from src.pipeline3_historical_analysis.confounders import (
    MANDATORY_CORRELATION_DISCLAIMER,
    get_confounders_for_window,
)
from src.utils.data_loader import load_macro_indicators

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "case_study.yaml"
INTERIM_CSV = PROJECT_ROOT / "data" / "interim" / "p3_indicators_interim.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def load_default_target_codes() -> list[str]:
    """Read target indicator codes from case study configuration."""
    if not CONFIG_PATH.exists():
        return ["FP.CPI.TOTL.ZG", "FP.CPI.TOTL", "EG.USE.ELEC.KH.PC", "EG.ELC.ACCS.ZS", "GC.DOD.TOTL.GD.ZS", "NE.CON.PRVT.ZS"]
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("indicators", {}).get("target_codes", [])


class IndicatorTrajectoryAnalyzer:
    """Computes pre/post policy indicator trajectories with contextual confounding events."""

    def __init__(self, indicators_df: pd.DataFrame | None = None) -> None:
        if indicators_df is not None:
            self.df = indicators_df.copy()
        elif INTERIM_CSV.exists():
            self.df = pd.read_csv(INTERIM_CSV)
        else:
            self.df = load_macro_indicators()

    def analyze_trajectory(
        self,
        decision_year: int = 2023,
        window_years: int = 2,
        indicator_codes: list[str] | None = None,
    ) -> dict[str, Any]:
        """Compute pre/post indicator trajectory for a policy decision year.

        Args:
            decision_year: The implementation or announcement year of the policy.
            window_years: Symmetric window size in years (e.g., 2 means t-2 to t+2).
            indicator_codes: Optional subset of indicator codes to analyze.

        Returns:
            Dictionary containing indicator trajectories, confounders, and causal disclaimer.
        """
        start_year = decision_year - window_years
        end_year = decision_year + window_years

        if indicator_codes is None:
            indicator_codes = load_default_target_codes()

        # Filter dataset to selected codes and window
        subset = self.df[
            (self.df["indicator_code"].isin(indicator_codes))
            & (self.df["year"] >= start_year)
            & (self.df["year"] <= end_year)
        ].copy()

        trajectories: dict[str, Any] = {}

        for code in indicator_codes:
            ind_rows = subset[subset["indicator_code"] == code].sort_values("year")
            if ind_rows.empty:
                # Look in master df if code exists outside interim subset
                master_rows = self.df[
                    (self.df["indicator_code"] == code)
                    & (self.df["year"] >= start_year)
                    & (self.df["year"] <= end_year)
                ].sort_values("year")
                if master_rows.empty:
                    continue
                ind_rows = master_rows

            ind_name = ind_rows["indicator_name"].iloc[0]
            impact_cat = ind_rows["impact_category"].iloc[0] if "impact_category" in ind_rows.columns else "Economic"

            # Build year-by-year points
            series_points: list[dict[str, Any]] = []
            values_by_year: dict[int, float] = {}

            for _, r in ind_rows.iterrows():
                yr = int(r["year"])
                val = float(r["value"])
                values_by_year[yr] = val

                if yr < decision_year:
                    phase = "Pre-Policy"
                elif yr == decision_year:
                    phase = "Implementation Year"
                else:
                    phase = "Post-Policy"

                series_points.append({
                    "year": yr,
                    "value": round(val, 2),
                    "phase": phase,
                })

            # Identify pre-policy baseline (prefer t-1, fallback to latest pre-year)
            pre_years = [y for y in values_by_year if y < decision_year]
            baseline_val = None
            if pre_years:
                baseline_val = values_by_year[max(pre_years)]

            # Identify event year value (t0)
            event_val = values_by_year.get(decision_year)

            # Identify latest post-policy value (prefer t+1 or t+2)
            post_years = [y for y in values_by_year if y > decision_year]
            post_val = None
            if post_years:
                post_val = values_by_year[max(post_years)]

            # Compute delta and percentage change
            # Reference comparison: post_val vs baseline_val, or event_val vs baseline_val
            target_val = post_val if post_val is not None else event_val
            abs_delta = None
            pct_delta = None
            trend_desc = "Insufficient Data"

            if baseline_val is not None and target_val is not None:
                abs_delta = round(target_val - baseline_val, 2)
                if abs(baseline_val) > 1e-6:
                    pct_delta = round((abs_delta / abs(baseline_val)) * 100.0, 2)

                # Classify qualitative trend direction
                if abs_delta > 1.0:
                    trend_desc = "Substantial Increase" if (pct_delta and pct_delta > 15) else "Moderate Increase"
                elif abs_delta < -1.0:
                    trend_desc = "Substantial Contraction" if (pct_delta and pct_delta < -15) else "Moderate Decline"
                else:
                    trend_desc = "Relatively Stable"

            trajectories[code] = {
                "indicator_code": code,
                "indicator_name": ind_name,
                "impact_category": impact_cat,
                "baseline_year": max(pre_years) if pre_years else None,
                "baseline_value": round(baseline_val, 2) if baseline_val is not None else None,
                "event_value": round(event_val, 2) if event_val is not None else None,
                "post_value": round(post_val, 2) if post_val is not None else None,
                "latest_observed_year": max(post_years) if post_years else (decision_year if event_val is not None else None),
                "absolute_change": abs_delta,
                "pct_change": pct_delta,
                "trend_direction": trend_desc,
                "time_series": series_points,
            }

        # Retrieve relevant confounders for the analyzed window
        confounders = get_confounders_for_window(start_year, end_year)

        return {
            "policy_case_id": "DEC_2023_ELEC_01",
            "decision_year": decision_year,
            "window_years": window_years,
            "window_range": [start_year, end_year],
            "indicators_analyzed_count": len(trajectories),
            "trajectories": trajectories,
            "confounders_count": len(confounders),
            "confounding_events": confounders,
            "causal_disclaimer": MANDATORY_CORRELATION_DISCLAIMER,
        }


def run_pipeline3_analysis(save_processed: bool = True) -> dict[str, Any]:
    """Execute pre/post trajectory analysis for target case DEC_2023_ELEC_01."""
    analyzer = IndicatorTrajectoryAnalyzer()
    results = analyzer.analyze_trajectory(decision_year=2023, window_years=2)

    if save_processed:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        out_file = PROCESSED_DIR / "p3_historical_trajectories.json"
        out_file.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Pipeline 3 Trajectory Analysis saved to: {out_file}")

    return results


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    res = run_pipeline3_analysis()
    print(f"\nAnalyzed {res['indicators_analyzed_count']} indicators over [{res['window_range'][0]}-{res['window_range'][1]}].")
    print(f"Attached {res['confounders_count']} historical confounders.")
    for code, t in res["trajectories"].items():
        print(f"- {code} ({t['indicator_name']}): {t['baseline_value']} -> {t['event_value']} (Trend: {t['trend_direction']}, Change: {t['pct_change']}%)")
