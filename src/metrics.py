from __future__ import annotations

import numpy as np


def weighted_precision_recall_f1(
    labels: np.ndarray,
    predictions: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float]:
    labels = labels.astype(bool)
    predictions = predictions.astype(bool)
    weights = weights.astype(np.float64)

    true_positive = weights[labels & predictions].sum()
    false_positive = weights[~labels & predictions].sum()
    false_negative = weights[labels & ~predictions].sum()

    precision = true_positive / max(true_positive + false_positive, 1e-8)
    recall = true_positive / max(true_positive + false_negative, 1e-8)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp_weight": float(true_positive),
        "fp_weight": float(false_positive),
        "fn_weight": float(false_negative),
    }


def find_best_threshold(
    probabilities: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> tuple[float, dict[str, float]]:
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 17)

    best_threshold = 0.5
    best_metrics = {"f1": -1.0}
    for threshold in thresholds:
        metrics = weighted_precision_recall_f1(
            labels,
            probabilities >= threshold,
            weights,
        )
        if metrics["f1"] > best_metrics["f1"]:
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics
