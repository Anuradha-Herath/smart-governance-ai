"""Macroeconomic and External Confounder Registry for Sri Lanka Policy Analysis.

Academic Policy Boundary:
Observed indicator trajectories (inflation, GDP growth, electricity consumption)
reflect overlapping macroeconomic shocks, fiscal measures, and external crises.
This module documents verified historical confounding events and generates
mandatory correlation vs. causation disclaimers for smart governance decision support.
"""

from __future__ import annotations

from typing import Any

# Verified macroeconomic, political, and external shocks for Sri Lanka (2018-2025)
HISTORICAL_CONFOUNDERS: list[dict[str, Any]] = [
    {
        "year": 2019,
        "date_period": "2019-04",
        "event_title": "Easter Sunday Terror Attacks",
        "category": "Security / Macroeconomic Shock",
        "description": "Devastating terror attacks severely disrupted the tourism sector, foreign exchange earnings, and investor confidence.",
        "impact_direction": "Negative on GDP growth, tourist receipts, and foreign reserves",
    },
    {
        "year": 2019,
        "date_period": "2019-11",
        "event_title": "Sweeping Fiscal Tax Cuts (Late 2019)",
        "category": "Fiscal Policy Shock",
        "description": "Reduction of VAT from 15% to 8%, abolishment of NBT (Nation Building Tax), and expansion of income tax thresholds eroded government revenue by ~25%.",
        "impact_direction": "Critical fiscal deficit expansion and loss of sovereign market access",
    },
    {
        "year": 2020,
        "date_period": "2020-03 to 2021-12",
        "event_title": "COVID-19 Global Pandemic & Lockdowns",
        "category": "Global Health & Economic Shock",
        "description": "Island-wide lockdowns, global supply chain paralysis, total collapse of tourist arrivals, and substantial fiscal relief expenditures.",
        "impact_direction": "Severe contraction in economic activity and loss of foreign remittances",
    },
    {
        "year": 2021,
        "date_period": "2021-04 to 2021-11",
        "event_title": "Chemical Fertilizer Import Ban",
        "category": "Agricultural & Trade Policy Shock",
        "description": "Overnight ban on chemical fertilizer imports led to severe domestic crop failure (paddy -30%, tea -15%), surging food inflation.",
        "impact_direction": "Dramatic spike in food inflation and agricultural output contraction",
    },
    {
        "year": 2022,
        "date_period": "2022-03",
        "event_title": "LKR Currency Float & Massive Devaluation",
        "category": "Monetary / Exchange Rate Shock",
        "description": "Central Bank floated the Sri Lankan Rupee; LKR depreciated from ~200/USD to over 360/USD within weeks, multiplying import costs.",
        "impact_direction": "Severe headline inflation spike (peaking over 70% in Sep 2022)",
    },
    {
        "year": 2022,
        "date_period": "2022-04",
        "event_title": "Sovereign Debt Standstill (External Debt Default)",
        "category": "Sovereign Debt Crisis",
        "description": "Ministry of Finance declared a preemptive moratorium on external debt servicing ($51 billion total external obligations).",
        "impact_direction": "Total cutoff from international financial markets and trade credit",
    },
    {
        "year": 2022,
        "date_period": "2022-04 to 2022-07",
        "event_title": "Acute Fuel, Cooking Gas & Power Shortages",
        "category": "Utility & Commodity Supply Shock",
        "description": "Depletion of foreign exchange reserves caused daily 10-13 hour power cuts and widespread fuel queues, paralyzing industrial and domestic transport.",
        "impact_direction": "Massive drop in electricity consumption and economic standstill",
    },
    {
        "year": 2022,
        "date_period": "2022-08",
        "event_title": "Electricity Tariff Revision (August 2022)",
        "category": "Utility Pricing Reform",
        "description": "First electricity tariff increase in 9 years (weighted average ~75% increase) to recover skyrocketing generation costs.",
        "impact_direction": "Initial step toward cost-recovery; increased household utility expenditure",
    },
    {
        "year": 2023,
        "date_period": "2023-01 to 2023-03",
        "event_title": "Major Tax Restructuring & Tight Monetary Policy",
        "category": "Fiscal & Monetary Policy",
        "description": "Implementation of progressive income tax revisions, increased corporate tax rates to 30%, and high Central Bank policy interest rates (~15-16%).",
        "impact_direction": "Contraction in private consumption expenditure and disposable income",
    },
    {
        "year": 2023,
        "date_period": "2023-03",
        "event_title": "IMF Extended Fund Facility (EFF) Approval",
        "category": "International Macroeconomic Program",
        "description": "IMF approved a 48-month, $3 billion EFF arrangement, unlocking multilateral financing from ADB and World Bank.",
        "impact_direction": "Stabilization of foreign reserves and exchange rate; strict fiscal targets",
    },
    {
        "year": 2024,
        "date_period": "2024-01",
        "event_title": "VAT Increase to 18% & Broadening of Base",
        "category": "Fiscal Policy Shock",
        "description": "Standard Value Added Tax increased from 15% to 18%, eliminating exemptions on fuel and electricity inputs.",
        "impact_direction": "One-off increase in consumer price index and living costs",
    },
    {
        "year": 2024,
        "date_period": "2024-03 to 2024-07",
        "event_title": "Downward Electricity Tariff Adjustments",
        "category": "Utility Pricing Reform",
        "description": "PUCSL approved consecutive tariff cuts (~21.9% in March, ~22.5% in July 2024) following heavy monsoon hydro generation and lower fuel costs.",
        "impact_direction": "Relief for domestic and industrial energy consumers",
    },
]

MANDATORY_CORRELATION_DISCLAIMER = (
    "CAUSAL INFERENCE DISCLAIMER: Observed indicator trajectories (e.g., inflation, "
    "GDP growth, energy consumption) correlate with policy execution but MUST NOT be "
    "interpreted as purely caused by the specific policy intervention. Macroeconomic outcomes "
    "in Sri Lanka during 2020-2025 were heavily confounded by sovereign debt restructuring, "
    "currency devaluation, global commodity price volatility, and multilateral IMF program conditions. "
    "All trajectory analyses presented here are descriptive and comparative, not causal assertions."
)


def get_confounders_for_window(
    start_year: int,
    end_year: int,
) -> list[dict[str, Any]]:
    """Retrieve historical confounding events falling within the specified time window."""
    return [
        c for c in HISTORICAL_CONFOUNDERS
        if start_year <= c["year"] <= end_year
    ]


def format_confounder_summary(start_year: int, end_year: int) -> dict[str, Any]:
    """Return a structured summary of confounders and mandatory disclaimer for a time window."""
    matched = get_confounders_for_window(start_year, end_year)
    return {
        "window": f"{start_year}-{end_year}",
        "confounders_count": len(matched),
        "disclaimer": MANDATORY_CORRELATION_DISCLAIMER,
        "confounding_events": matched,
    }
