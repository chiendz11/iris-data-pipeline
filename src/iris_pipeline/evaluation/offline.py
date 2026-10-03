from __future__ import annotations

from dataclasses import dataclass

from sklearn.metrics import accuracy_score, f1_score


@dataclass(frozen=True)
class OfflineMetrics:
    accuracy: float
    f1_macro: float


def evaluate_predictions(expected, predicted) -> OfflineMetrics:
    """Calculate deterministic offline classification metrics."""
    return OfflineMetrics(
        accuracy=float(accuracy_score(expected, predicted)),
        f1_macro=float(f1_score(expected, predicted, average="macro")),
    )


def passes_quality_gate(
    accuracy: float,
    *,
    min_accuracy: float,
    baseline_accuracy: float | None,
    min_improvement: float,
) -> bool:
    """Require the absolute threshold and, when present, a champion improvement."""
    return accuracy >= min_accuracy and (
        baseline_accuracy is None or accuracy >= baseline_accuracy + min_improvement
    )
