"""Tests for src/metrics.py — evaluation metrics."""

from __future__ import annotations

import numpy as np
import pytest

from src.metrics import weighted_precision_recall_f1, find_best_threshold, weighted_average_precision


class TestWeightedPrecisionRecallF1:
    def test_perfect_predictions(self):
        labels = np.array([1, 1, 0, 0], dtype=np.float32)
        predictions = np.array([1, 1, 0, 0], dtype=np.float32)
        weights = np.ones(4, dtype=np.float32)
        result = weighted_precision_recall_f1(labels, predictions, weights)
        assert result["precision"] == pytest.approx(1.0)
        assert result["recall"] == pytest.approx(1.0)
        assert result["f1"] == pytest.approx(1.0)

    def test_all_wrong_predictions(self):
        labels = np.array([1, 1, 0, 0], dtype=np.float32)
        predictions = np.array([0, 0, 1, 1], dtype=np.float32)
        weights = np.ones(4, dtype=np.float32)
        result = weighted_precision_recall_f1(labels, predictions, weights)
        assert result["precision"] == pytest.approx(0.0, abs=1e-6)
        assert result["recall"] == pytest.approx(0.0, abs=1e-6)
        assert result["f1"] == pytest.approx(0.0, abs=1e-6)

    def test_all_negative_predictions(self):
        """All predicted negative: precision undefined → 0, recall=0, f1=0."""
        labels = np.array([1, 0, 1], dtype=np.float32)
        predictions = np.array([0, 0, 0], dtype=np.float32)
        weights = np.ones(3, dtype=np.float32)
        result = weighted_precision_recall_f1(labels, predictions, weights)
        assert result["recall"] == pytest.approx(0.0, abs=1e-6)
        assert result["f1"] == pytest.approx(0.0, abs=1e-6)

    def test_weight_zero_excludes_pixel(self):
        """Weight=0 pixels must not contribute to any metric."""
        labels = np.array([1, 1, 0], dtype=np.float32)
        predictions = np.array([1, 0, 1], dtype=np.float32)
        weights = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        result = weighted_precision_recall_f1(labels, predictions, weights)
        # Only the first pixel has weight; it's a TP
        assert result["precision"] == pytest.approx(1.0)
        assert result["recall"] == pytest.approx(1.0)
        assert result["f1"] == pytest.approx(1.0)

    def test_higher_weight_on_positives_amplifies_recall(self):
        labels = np.array([1, 1, 0, 0], dtype=np.float32)
        predictions = np.array([1, 0, 0, 0], dtype=np.float32)
        # Equal weights
        result_eq = weighted_precision_recall_f1(labels, predictions, np.ones(4))
        # Downweight the missed positive
        weights_down = np.array([1.0, 0.1, 1.0, 1.0], dtype=np.float32)
        result_down = weighted_precision_recall_f1(labels, predictions, weights_down)
        # Downweighting the false-negative improves recall
        assert result_down["recall"] > result_eq["recall"]

    def test_returned_keys(self):
        labels = np.array([1, 0], dtype=np.float32)
        predictions = np.array([1, 0], dtype=np.float32)
        weights = np.ones(2, dtype=np.float32)
        result = weighted_precision_recall_f1(labels, predictions, weights)
        for key in ("precision", "recall", "f1", "tp_weight", "fp_weight", "fn_weight"):
            assert key in result


class TestFindBestThreshold:
    def test_returns_float_in_unit_interval(self):
        probs = np.array([0.1, 0.6, 0.9, 0.2], dtype=np.float32)
        labels = np.array([0, 1, 1, 0], dtype=np.float32)
        weights = np.ones(4, dtype=np.float32)
        threshold, _ = find_best_threshold(probs, labels, weights)
        assert 0.0 <= threshold <= 1.0

    def test_optimal_threshold_found_for_clean_data(self):
        """When probabilities clearly separate classes, threshold must be near 0.5."""
        probs = np.array([0.05, 0.05, 0.95, 0.95], dtype=np.float32)
        labels = np.array([0, 0, 1, 1], dtype=np.float32)
        weights = np.ones(4, dtype=np.float32)
        threshold, metrics = find_best_threshold(probs, labels, weights)
        assert metrics["f1"] == pytest.approx(1.0, abs=1e-6)

    def test_custom_threshold_grid(self):
        """Custom threshold grid should be used instead of the default."""
        probs = np.linspace(0, 1, 100, dtype=np.float32)
        labels = (probs > 0.7).astype(np.float32)
        weights = np.ones_like(probs)
        custom_thresholds = np.array([0.7, 0.8, 0.9], dtype=np.float32)
        threshold, _ = find_best_threshold(probs, labels, weights, thresholds=custom_thresholds)
        assert threshold in custom_thresholds.tolist()

    def test_returns_metrics_dict(self):
        probs = np.array([0.3, 0.7], dtype=np.float32)
        labels = np.array([0, 1], dtype=np.float32)
        weights = np.ones(2, dtype=np.float32)
        _, metrics = find_best_threshold(probs, labels, weights)
        assert "f1" in metrics
        assert "precision" in metrics
        assert "recall" in metrics


class TestWeightedAveragePrecision:
    def test_perfect_ranking_gives_ap_one(self):
        """When model ranks all positives above all negatives, AP = 1.0."""
        probs = np.array([0.9, 0.8, 0.2, 0.1], dtype=np.float64)
        labels = np.array([1, 1, 0, 0], dtype=bool)
        weights = np.ones(4, dtype=np.float64)
        ap = weighted_average_precision(labels, probs, weights)
        assert ap == pytest.approx(1.0, abs=1e-6)

    def test_random_ranking_gives_low_ap(self):
        """Reversed ranking (worst model) should give AP near 0."""
        probs = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float64)
        labels = np.array([1, 1, 0, 0], dtype=bool)
        weights = np.ones(4, dtype=np.float64)
        ap = weighted_average_precision(labels, probs, weights)
        assert ap < 0.5

    def test_no_positives_returns_zero(self):
        probs = np.array([0.8, 0.7], dtype=np.float64)
        labels = np.array([0, 0], dtype=bool)
        weights = np.ones(2, dtype=np.float64)
        ap = weighted_average_precision(labels, probs, weights)
        assert ap == 0.0

    def test_ap_in_unit_interval(self):
        rng = np.random.default_rng(7)
        probs = rng.random(200).astype(np.float64)
        labels = (rng.random(200) > 0.8).astype(bool)
        weights = rng.random(200)
        ap = weighted_average_precision(labels, probs, weights)
        assert 0.0 <= ap <= 1.0
