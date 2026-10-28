"""Pipeline 3 Historical Policy Decision Retrieval Engine.

Searches 11,175 Sri Lanka Cabinet Decisions (2010-2026) to find past policy
interventions similar to a proposed reform or policy query.

Features:
- Multilingual indexing (English & Sinhala) using subword/word TF-IDF representations.
- Flexible filtering by sector keywords and date ranges.
- Relevance tiering (High, Moderate, Low) with continuous cosine similarity scores.
"""

from __future__ import annotations

import re
from typing import Any
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils.data_loader import load_cabinet_decisions


class HistoricalDecisionRetriever:
    """Retrieves historically similar Cabinet decisions using vectorized semantic matching."""

    def __init__(self, max_features: int = 35000) -> None:
        self.max_features = max_features
        self.df: pd.DataFrame | None = None
        self.vectorizer: TfidfVectorizer | None = None
        self.doc_matrix: Any = None
        self._is_indexed: bool = False

    def build_index(self) -> None:
        """Load cabinet decisions and fit the TF-IDF index across titles and body texts."""
        if self._is_indexed:
            return

        self.df = load_cabinet_decisions()
        # Combine title and body into unified search corpus
        titles = self.df["description"].fillna("").astype(str)
        bodies = self.df["decision_details_body"].fillna("").astype(str)
        corpus = (titles + " " + bodies).tolist()

        # Unicode token pattern supports Latin, Sinhala, and Tamil characters
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            token_pattern=r"(?u)\b\w+\b",
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )
        self.doc_matrix = self.vectorizer.fit_transform(corpus)
        self._is_indexed = True

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.05,
        sector_keywords: list[str] | None = None,
        date_range: tuple[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        """Search historical decisions for a policy query.

        Args:
            query: Policy description or keywords in English or Sinhala.
            top_k: Maximum number of ranked decisions to return.
            min_score: Minimum cosine similarity threshold.
            sector_keywords: Optional list of required keywords in title or body.
            date_range: Optional ('YYYY-MM-DD', 'YYYY-MM-DD') tuple to filter dates.

        Returns:
            Ranked list of structured decision dictionaries.
        """
        if not self._is_indexed:
            self.build_index()

        assert self.df is not None
        assert self.vectorizer is not None
        assert self.doc_matrix is not None

        clean_query = query.strip()
        if not clean_query:
            return []

        # Vectorize incoming query
        query_vec = self.vectorizer.transform([clean_query])
        # Compute cosine similarity
        raw_scores = (self.doc_matrix * query_vec.T).toarray().flatten()

        # Pre-filter by date or sector if specified
        valid_indices = np.arange(len(self.df))
        if date_range:
            start_d, end_d = date_range
            date_mask = (self.df["date_str"] >= start_d) & (self.df["date_str"] <= end_d)
            valid_indices = valid_indices[date_mask.to_numpy()]

        if sector_keywords:
            kw_pattern = "|".join([re.escape(k.strip()) for k in sector_keywords if k.strip()])
            if kw_pattern:
                combined_text = (
                    self.df["description"].fillna("") + " " + self.df["decision_details_body"].fillna("")
                ).astype(str)
                sector_mask = combined_text.str.contains(kw_pattern, case=False, regex=True)
                valid_indices = np.intersect1d(valid_indices, np.where(sector_mask)[0])

        if len(valid_indices) == 0:
            return []

        filtered_scores = raw_scores[valid_indices]
        # Rank by descending score
        top_subset_idx = filtered_scores.argsort()[::-1]

        results: list[dict[str, Any]] = []
        for sub_idx in top_subset_idx:
            orig_idx = valid_indices[sub_idx]
            score = float(raw_scores[orig_idx])
            if score < min_score:
                break

            row = self.df.iloc[orig_idx]
            body_text = str(row["decision_details_body"]) if pd.notna(row["decision_details_body"]) else ""
            date_str = str(row["date_str"]) if pd.notna(row["date_str"]) else "Unknown"
            year = int(date_str.split("-")[0]) if "-" in date_str and date_str[:4].isdigit() else None

            # Categorize relevance tier
            if score >= 0.35:
                tier = "High"
            elif score >= 0.15:
                tier = "Moderate"
            else:
                tier = "Low"

            results.append({
                "doc_id": str(row["doc_id"]),
                "date_str": date_str,
                "year": year,
                "title": str(row["description"]),
                "body_snippet": body_text[:280] + ("..." if len(body_text) > 280 else ""),
                "full_body": body_text,
                "similarity_score": round(score, 4),
                "relevance_tier": tier,
                "lang": str(row.get("lang", "en")),
            })

            if len(results) >= top_k:
                break

        return results


# Module-level singleton instance for shared fast access
_DEFAULT_RETRIEVER: HistoricalDecisionRetriever | None = None


def get_default_retriever() -> HistoricalDecisionRetriever:
    """Return an initialized singleton retriever instance."""
    global _DEFAULT_RETRIEVER
    if _DEFAULT_RETRIEVER is None:
        _DEFAULT_RETRIEVER = HistoricalDecisionRetriever()
        _DEFAULT_RETRIEVER.build_index()
    return _DEFAULT_RETRIEVER


def search_historical_decisions(
    query: str,
    top_k: int = 5,
    min_score: float = 0.05,
    sector_keywords: list[str] | None = None,
    date_range: tuple[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Functional convenience wrapper for retrieving historical decisions."""
    retriever = get_default_retriever()
    return retriever.search(
        query=query,
        top_k=top_k,
        min_score=min_score,
        sector_keywords=sector_keywords,
        date_range=date_range,
    )
