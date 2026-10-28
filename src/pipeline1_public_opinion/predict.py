"""Pipeline 1 Inference & Prediction Service.

Provides unified prediction APIs for:
1. Multilingual Sentiment Classification (Positive / Negative / Neutral)
2. Policy Stance Classification (Support / Oppose / Neutral_or_Unclear)
3. Batch Aggregation with Confidence Thresholding & Sampling Disclaimers
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SENTIMENT_MODEL_PATH = PROJECT_ROOT / "models" / "p1_sentiment_baseline.joblib"
STANCE_MODEL_PATH = PROJECT_ROOT / "models" / "p1_stance_classifier.joblib"

SAMPLING_DISCLAIMER = (
    "Disclaimer: Online comments reflect self-selected social media participants and "
    "do not represent a randomized statistical sample of the entire population."
)

# Global model cache to avoid re-loading on each inference call
_SENTIMENT_PIPELINE = None
_STANCE_PIPELINE = None


def get_sentiment_model() -> Any:
    global _SENTIMENT_PIPELINE
    if _SENTIMENT_PIPELINE is None:
        if not SENTIMENT_MODEL_PATH.exists():
            raise FileNotFoundError(f"Sentiment model not found at {SENTIMENT_MODEL_PATH}. Train it first.")
        _SENTIMENT_PIPELINE = joblib.load(SENTIMENT_MODEL_PATH)
    return _SENTIMENT_PIPELINE


def get_stance_model() -> Any:
    global _STANCE_PIPELINE
    if _STANCE_PIPELINE is None:
        if not STANCE_MODEL_PATH.exists():
            raise FileNotFoundError(f"Stance model not found at {STANCE_MODEL_PATH}. Train it first.")
        _STANCE_PIPELINE = joblib.load(STANCE_MODEL_PATH)
    return _STANCE_PIPELINE


def predict_sentiment(text: str) -> dict[str, Any]:
    """Predict sentiment polarity and class probabilities for input text."""
    model = get_sentiment_model()
    text_clean = str(text).strip()
    if not text_clean:
        return {"sentiment": "Neutral", "confidence": 1.0, "probabilities": {"Neutral": 1.0}}

    probs = model.predict_proba([text_clean])[0]
    classes = list(model.classes_)
    best_idx = int(np.argmax(probs))
    best_label = classes[best_idx]
    confidence = round(float(probs[best_idx]), 4)

    prob_dict = {cls_name: round(float(p), 4) for cls_name, p in zip(classes, probs)}

    return {
        "sentiment": best_label,
        "confidence": confidence,
        "probabilities": prob_dict,
    }


def predict_stance(text: str, target: str = "Electricity Tariff Revision") -> dict[str, Any]:
    """Predict policy stance (Support vs. Oppose) toward target decision."""
    model = get_stance_model()
    text_clean = str(text).strip()
    if not text_clean:
        return {"stance": "Neutral_or_Unclear", "confidence": 1.0, "target": target}

    probs = model.predict_proba([text_clean])[0]
    classes = list(model.classes_)
    best_idx = int(np.argmax(probs))
    best_label = classes[best_idx]
    confidence = round(float(probs[best_idx]), 4)

    return {
        "stance": best_label,
        "confidence": confidence,
        "target": target,
        "probabilities": {cls_name: round(float(p), 4) for cls_name, p in zip(classes, probs)},
    }


def analyze_comment(text: str, target_decision: str = "DEC_2023_ELEC_01") -> dict[str, Any]:
    """Jointly analyze sentiment and stance for a single policy comment."""
    sent_res = predict_sentiment(text)
    stance_res = predict_stance(text, target=target_decision)

    return {
        "text": text,
        "target_decision": target_decision,
        "sentiment": sent_res["sentiment"],
        "sentiment_confidence": sent_res["confidence"],
        "sentiment_probabilities": sent_res["probabilities"],
        "stance": stance_res["stance"],
        "stance_confidence": stance_res["confidence"],
        "sampling_disclaimer": SAMPLING_DISCLAIMER,
    }


def analyze_comment_batch(texts: list[str], target_decision: str = "DEC_2023_ELEC_01") -> dict[str, Any]:
    """Analyze a batch of comments and aggregate sentiment and stance distributions."""
    results = [analyze_comment(t, target_decision=target_decision) for t in texts if str(t).strip()]
    total = len(results)
    if total == 0:
        return {"total_analyzed": 0, "sentiment_distribution": {}, "stance_distribution": {}}

    sentiment_counts: dict[str, int] = {}
    stance_counts: dict[str, int] = {}

    for r in results:
        sent = r["sentiment"]
        st = r["stance"]
        sentiment_counts[sent] = sentiment_counts.get(sent, 0) + 1
        stance_counts[st] = stance_counts.get(st, 0) + 1

    sentiment_dist = {k: round(v / total * 100, 2) for k, v in sentiment_counts.items()}
    stance_dist = {k: round(v / total * 100, 2) for k, v in stance_counts.items()}

    return {
        "total_analyzed": total,
        "sentiment_distribution_pct": sentiment_dist,
        "stance_distribution_pct": stance_dist,
        "sample_results": results[:5],
        "sampling_disclaimer": SAMPLING_DISCLAIMER,
    }


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    sample_comments = [
        "CEB bill eka wadi una eka sadaranai, nathnam power cuts enawa.",
        "විදුලි බිල 66% කින් වැඩි කරපු එක ජනතාවට දරාගන්න බෑ. මේක වහාම නවත්වන්න.",
        "This electricity tariff hike is essential to stop fiscal bleeding at CEB.",
        "PUCSL should publish the breakdown of generation costs before increasing prices.",
    ]

    print("--- Single Comment Analysis ---")
    single = analyze_comment(sample_comments[1])
    print(f"Text:      {single['text']}")
    print(f"Sentiment: {single['sentiment']} (conf: {single['sentiment_confidence']})")
    print(f"Stance:    {single['stance']} (conf: {single['stance_confidence']})")

    print("\n--- Batch Aggregation ---")
    batch = analyze_comment_batch(sample_comments)
    print("Sentiment %:", batch["sentiment_distribution_pct"])
    print("Stance %:   ", batch["stance_distribution_pct"])
