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
    iou = true_positive / max(true_positive + false_positive + false_negative, 1e-8)

    unweighted_tp = (labels & predictions).sum()
    unweighted_fp = (~labels & predictions).sum()
    unweighted_fn = (labels & ~predictions).sum()
    unweighted_iou = unweighted_tp / max(unweighted_tp + unweighted_fp + unweighted_fn, 1e-8)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
        "unweighted_iou": float(unweighted_iou),
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
        thresholds = np.linspace(0.05, 0.95, 37)

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


def find_best_threshold_per_region(
    probabilities: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    regions: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> dict[str, tuple[float, dict[str, float]]]:
    """Return best (threshold, metrics) for each region label."""
    result: dict[str, tuple[float, dict[str, float]]] = {}
    unique_regions = np.unique(regions)
    for region in unique_regions:
        region_mask = regions == region
        if not region_mask.any():
            continue
        threshold, metrics = find_best_threshold(
            probabilities[region_mask],
            labels[region_mask],
            weights[region_mask],
            thresholds=thresholds,
        )
        result[str(region)] = (threshold, metrics)
    return result


def weighted_average_precision(
    labels: np.ndarray,
    probabilities: np.ndarray,
    weights: np.ndarray,
) -> float:
    labels = labels.astype(bool)
    probabilities = probabilities.astype(np.float64)
    weights = weights.astype(np.float64)

    total_positive_weight = weights[labels].sum()
    if total_positive_weight <= 0:
        return 0.0

    order = np.argsort(-probabilities, kind="mergesort")
    sorted_labels = labels[order]
    sorted_weights = weights[order]

    cumulative_weight = np.cumsum(sorted_weights)
    cumulative_positive_weight = np.cumsum(sorted_weights * sorted_labels)
    precision = cumulative_positive_weight / np.clip(cumulative_weight, 1e-8, None)
    recall_delta = (sorted_weights * sorted_labels) / total_positive_weight
    return float(np.sum(precision * recall_delta))
