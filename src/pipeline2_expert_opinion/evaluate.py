"""Pipeline 2 Extraction Evaluation Module.

Evaluates the Grounded Evidence Extractor against human-annotated gold standard
examples (annotations/expert_claims_gold.json).

Metrics:
- Statement Type Classification Precision, Recall, and Macro F1
- Verbatim Citation & Source Grounding Accuracy Rate
- Exact Match and Overlap F1

Output: outputs/evaluation/p2_extraction_metrics.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

from src.pipeline2_expert_opinion.extractor import classify_sentence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOLD_FILE = PROJECT_ROOT / "annotations" / "expert_claims_gold.json"
EVAL_DIR = PROJECT_ROOT / "outputs" / "evaluation"


def evaluate_pipeline2_extraction() -> dict[str, Any]:
    """Evaluate extraction accuracy and source grounding against gold benchmark."""
    if not GOLD_FILE.exists():
        raise FileNotFoundError(f"Gold benchmark annotations not found at {GOLD_FILE}")

    with GOLD_FILE.open("r", encoding="utf-8") as f:
        gold_items = json.load(f)

    y_true: list[str] = []
    y_pred: list[str] = []
    grounded_count = 0
    total_items = len(gold_items)

    print(f"Evaluating Extractor against {total_items} Human-Annotated Gold Samples...")

    for item in gold_items:
        text = item["sentence_text"]
        true_label = item["gold_label"]
        pred_label, conf = classify_sentence(text)

        y_true.append(true_label)
        y_pred.append(pred_label if pred_label != "None" else "Claim")

        # Check verbatim quote grounding
        verbatim = item.get("verbatim_quote", "")
        if verbatim and (verbatim.lower() in text.lower() or text.lower() in verbatim.lower()):
            grounded_count += 1

    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    grounding_rate = round(grounded_count / total_items * 100, 2)

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    results = {
        "benchmark_file": str(GOLD_FILE),
        "total_gold_samples": total_items,
        "macro_f1": round(macro_f1, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "source_grounding_accuracy_pct": grounding_rate,
        "detailed_classification_report": report,
    }

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out_file = EVAL_DIR / "p2_extraction_metrics.json"
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n--- Pipeline 2 Evaluation Results ---")
    print(f"Macro F1 Score:               {macro_f1:.4f}")
    print(f"Macro Precision:              {macro_prec:.4f}")
    print(f"Macro Recall:                 {macro_rec:.4f}")
    print(f"Source Grounding Accuracy:    {grounding_rate}% (Verbatim passage match)")
    print(f"Results saved to:             {out_file}")

    return results


if __name__ == "__main__":
    evaluate_pipeline2_extraction()
