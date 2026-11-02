"""Evaluation and model acceptance policy."""

from typing import Any

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from sentiment_mlops.data.prepare_data import LABELS


def evaluate_predictions(actual: list[str], predicted: list[str]) -> dict[str, Any]:
    """Calculate labeled test-set metrics for both sentiment classes."""

    def per_class(metric: Any, label: str) -> float:
        return float(metric(actual, predicted, pos_label=label, zero_division=0))

    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "precision_positive": per_class(precision_score, "positive"),
        "recall_positive": per_class(recall_score, "positive"),
        "f1_positive": per_class(f1_score, "positive"),
        "f1_negative": per_class(f1_score, "negative"),
        "macro_f1": float(f1_score(actual, predicted, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(actual, predicted, labels=LABELS).tolist(),
        "classification_report": classification_report(actual, predicted, output_dict=True, zero_division=0),
    }


def candidate_is_accepted(metrics: dict[str, Any], minimum_f1: float) -> bool:
    """Return whether the candidate meets the explicit macro F1 threshold."""

    return float(metrics["macro_f1"]) >= minimum_f1
