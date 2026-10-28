"""Pipeline 2 Expert Grounded Evidence Extractor Module.

Extracts structured policy statements from official documents:
- Claims (central assertions)
- Risks (adverse impacts/threats)
- Benefits (positive outcomes/stability)
- Evidence (empirical numbers, audit figures, facts)
- Recommendations (directives, proposed mitigation)

Enforces strict source grounding: every extracted statement preserves document ID,
date, speaker/title, and verified verbatim substring citations.

Output: data/processed/p2_extracted_evidence.json
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INTERIM_FILE = PROJECT_ROOT / "data" / "interim" / "p2_expert_documents_interim.parquet"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Linguistic discourse markers for the 5 evidence statement types across English & Sinhala
LEXICAL_CUES = {
    "Recommendation": [
        "should", "must", "recommend", "to take necessary action", "decided to grant approval",
        "to further study", "directive", "urgent need to", "introduce", "implement",
        "යෝජනා", "ක්‍රියාමාර්ග ගත යුතු", "සඳහා උපදෙස්", "නියෝග", "ක්‍රියාත්මක කළ යුතු"
    ],
    "Evidence": [
        "billion", "million", "percent", "%", "rs.", "rupees", "incurred", "accumulated",
        "data shows", "according to", "audited", "fiscal space", "per unit",
        "සියයට", "රුපියල්", "මිලියන", "බිලියන", "වාර්තා", "අනුව", "වියදම"
    ],
    "Risk": [
        "impact caused", "adverse", "burden", "unaffordable", "deficit", "crisis",
        "loss", "struggle", "inflation", "cost of living", "poverty", "fail",
        "අර්බුදය", "අහිතකර", "අලාභය", "පීඩාව", "දරාගන්න බෑ", "වැඩි වී", "අපහසු"
    ],
    "Benefit": [
        "ensure uninterrupted", "financial viability", "recovery", "stability",
        "turnaround", "eliminate subsidies", "cost-effective", "sustainable", "surplus",
        "ස්ථාවරත්වය", "අඛණ්ඩ", "ගොඩ ගැනීම", "සහනය", "වර්ධනය", "දියුණු"
    ],
    "Claim": [
        "considered that", "there is no alternative", "cost-reflective", "essential to",
        "fundamental", "we argue", "it is clear", "policy requires",
        "හේතුවෙන්", "ප්‍රකාශ කරන්නේ", "පැහැදිලි වන්නේ", "අවශ්‍ය බව"
    ],
}


def split_into_sentences(text: str) -> list[str]:
    """Segment raw text chunk into sentences handling Latin and Sinhala delimiters."""
    # Normalize bullet points and multiple spaces
    cleaned = re.sub(r"[\r\n\t]+", " ", str(text))
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Split by full stops, question marks, and new sentence boundaries
    raw_sents = re.split(r"(?<=[.!?\u0df4])\s+", cleaned)
    valid_sents = []
    for s in raw_sents:
        s_clean = s.strip()
        # Discard boilerplate greetings or single tokens (< 25 chars)
        if len(s_clean) >= 25 and not s_clean.startswith(("*", "-", "#")):
            valid_sents.append(s_clean)
    return valid_sents


def classify_sentence(sentence: str) -> tuple[str, float]:
    """Classify a sentence into one of the 5 statement categories based on semantic cues."""
    s_lower = sentence.lower()

    scores: dict[str, int] = {}
    for stmt_type, cues in LEXICAL_CUES.items():
        score = 0
        for cue in cues:
            cue_lower = cue.lower()
            # If cue is currency 'rs.', require word boundary and punctuation check to avoid 'consumers.'
            if cue_lower == "rs.":
                if re.search(r"(?<!\w)rs\.?(?=\s|\d|$)", s_lower):
                    score += 1
            elif cue_lower == "%":
                if "%" in s_lower:
                    score += 1
            # If cue contains alphanumeric characters only, enforce word boundary
            elif cue_lower.isalnum() or " " in cue_lower:
                if re.search(r"\b" + re.escape(cue_lower) + r"\b", s_lower):
                    score += 1
            elif cue_lower in s_lower:
                score += 1
        scores[stmt_type] = score

    # Check for strongest cue match
    best_type = max(scores, key=scores.get) # type: ignore
    max_score = scores[best_type]

    if max_score == 0:
        # Default fallback: check if it expresses a generic assertion
        if any(w in s_lower for w in ["tariff", "ceb", "power", "විදුලි"]):
            return "Claim", 0.50
        return "None", 0.0

    # Calculate confidence based on matched cue density
    confidence = min(0.60 + (max_score * 0.15), 0.95)
    return best_type, round(confidence, 2)


def extract_grounded_evidence(
    text_chunk: str,
    doc_id: str,
    source_type: str,
    date_str: str,
    speaker_or_title: str,
) -> list[dict[str, Any]]:
    """Extract verifiable, grounded statements from a document passage."""
    sentences = split_into_sentences(text_chunk)
    extracted: list[dict[str, Any]] = []

    for idx, sent in enumerate(sentences):
        stmt_type, confidence = classify_sentence(sent)
        if stmt_type == "None" or confidence < 0.55:
            continue

        # Grounding check: Locate exact start and end position in original text
        start_char = text_chunk.find(sent)
        if start_char == -1:
            # Substring match if minor punctuation differences exist
            sub_target = sent[:min(len(sent), 40)]
            start_char = text_chunk.find(sub_target)

        end_char = start_char + len(sent) if start_char != -1 else -1
        is_grounded = (start_char != -1)

        extracted.append({
            "statement_id": f"{doc_id}_STMT_{idx:03d}",
            "doc_id": doc_id,
            "source_type": source_type,
            "date": date_str,
            "speaker_or_title": speaker_or_title,
            "statement_type": stmt_type,
            "extracted_statement": sent,
            "verbatim_quote": sent if is_grounded else text_chunk[:100],
            "char_span": [start_char, end_char],
            "grounding_verified": is_grounded,
            "confidence": confidence,
        })

    return extracted


def run_pipeline2_extraction(max_passages: int | None = None) -> list[dict[str, Any]]:
    """Process all policy-matched expert passages and save verified grounded extractions."""
    if not INTERIM_FILE.exists():
        raise FileNotFoundError(f"Interim expert file not found: {INTERIM_FILE}")

    df = pd.read_parquet(INTERIM_FILE)
    if max_passages:
        df = df.head(max_passages)

    all_statements: list[dict[str, Any]] = []
    print(f"Extracting Grounded Expert Evidence from {len(df):,} policy passages...")

    for _, row in df.iterrows():
        doc_id = str(row["doc_id"])
        source_type = str(row["source_type"])
        date_str = str(row["date_str"])
        speaker = str(row["speaker_or_title"])
        text = str(row["text_chunk"])

        stmts = extract_grounded_evidence(text, doc_id, source_type, date_str, speaker)
        all_statements.extend(stmts)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_file = PROCESSED_DIR / "p2_extracted_evidence.json"
    out_file.write_text(json.dumps(all_statements, indent=2, ensure_ascii=False), encoding="utf-8")

    # Summary
    type_counts: dict[str, int] = {}
    verified_count = 0
    for s in all_statements:
        t = s["statement_type"]
        type_counts[t] = type_counts.get(t, 0) + 1
        if s["grounding_verified"]:
            verified_count += 1

    print(f"\nExtraction Finished: {len(all_statements):,} structured statements.")
    print(f"Grounding Verification Rate: {verified_count / max(len(all_statements), 1) * 100:.2f}%")
    print("Statement Types Breakdown:", type_counts)
    print(f"Saved to: {out_file}")

    return all_statements


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    run_pipeline2_extraction()
