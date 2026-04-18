"""Tests for src/dataset.py — pixel sampling and dataset construction."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.dataset import (
    PixelSampleDataset,
    build_eval_matrix,
    sample_tile_training_data,
)
from src.feature_bank import TileFeatures, build_tile_features
from tests.conftest import TARGET_YEARS, H, W, _make_metadata


def _dummy_tile_features(
    n_features: int = 10,
    *,
    has_labels: bool = True,
) -> TileFeatures:
    """Build a minimal TileFeatures for dataset tests (avoids full feature bank)."""
    forest = np.ones((H, W), dtype=bool)
    labels = np.zeros((H, W), dtype=np.uint8) if has_labels else None
    weights = np.ones((H, W), dtype=np.float32) if has_labels else None
    if has_labels:
        # Place a few positives
        labels[2, 2] = 1
        labels[3, 5] = 1
    return TileFeatures(
        feature_cube=np.random.randn(n_features, H, W).astype(np.float32),
        feature_names=tuple(f"f{i}" for i in range(n_features)),
        forest_mask=forest,
        labels=labels,
        weights=weights,
        persistence_score=np.zeros((H, W), dtype=np.float32),
        persistence_mask=np.zeros((H, W), dtype=bool),
    )


class TestPixelSampleDataset:
    def test_length_matches_input(self):
        features = np.random.randn(100, 10).astype(np.float32)
        labels = np.zeros(100, dtype=np.float32)
        weights = np.ones(100, dtype=np.float32)
        ds = PixelSampleDataset(features, labels, weights)
        assert len(ds) == 100

    def test_getitem_returns_dict(self):
        features = np.random.randn(10, 5).astype(np.float32)
        labels = np.zeros(10, dtype=np.float32)
        weights = np.ones(10, dtype=np.float32)
        ds = PixelSampleDataset(features, labels, weights)
        item = ds[0]
        assert isinstance(item, dict)
        assert "features" in item
        assert "labels" in item
        assert "weights" in item

    def test_getitem_values_are_tensors(self):
        features = np.random.randn(10, 5).astype(np.float32)
        labels = np.zeros(10, dtype=np.float32)
        weights = np.ones(10, dtype=np.float32)
        ds = PixelSampleDataset(features, labels, weights)
        item = ds[0]
        assert isinstance(item["features"], torch.Tensor)
        assert isinstance(item["labels"], torch.Tensor)
        assert isinstance(item["weights"], torch.Tensor)

    def test_getitem_shapes(self):
        n_features = 15
        features = np.random.randn(50, n_features).astype(np.float32)
        labels = np.zeros(50, dtype=np.float32)
        weights = np.ones(50, dtype=np.float32)
        ds = PixelSampleDataset(features, labels, weights)
        item = ds[0]
        assert item["features"].shape == (n_features,)
        assert item["labels"].shape == ()
        assert item["weights"].shape == ()

    def test_sequence_included_when_provided(self):
        n, t, c = 20, 12, 5
        features = np.random.randn(n, 10).astype(np.float32)
        labels = np.zeros(n, dtype=np.float32)
        weights = np.ones(n, dtype=np.float32)
        sequences = np.random.randn(n, t, c).astype(np.float32)
        ds = PixelSampleDataset(features, labels, weights, sequences=sequences)
        item = ds[0]
        assert "sequence" in item
        assert item["sequence"].shape == (t, c)

    def test_no_sequence_key_when_none(self):
        features = np.random.randn(10, 5).astype(np.float32)
        labels = np.zeros(10, dtype=np.float32)
        weights = np.ones(10, dtype=np.float32)
        ds = PixelSampleDataset(features, labels, weights)
        item = ds[0]
        assert "sequence" not in item


class TestSampleTileTrainingData:
    def test_positive_pixels_included(self, rng):
        metadata = _make_metadata()
        tf = _dummy_tile_features()
        sampled = sample_tile_training_data(metadata, tf, rng, negative_ratio=3.0)
        assert sampled.positive_count > 0
        assert set(sampled.labels.astype(int).tolist()).issubset({0, 1})

    def test_negative_ratio_respected(self, rng):
        metadata = _make_metadata()
        tf = _dummy_tile_features()
        ratio = 3.0
        sampled = sample_tile_training_data(metadata, tf, rng, negative_ratio=ratio)
        n_pos = int((sampled.labels == 1).sum())
        n_neg = int((sampled.labels == 0).sum())
        if sampled.positive_count > 0:
            actual_ratio = n_neg / max(n_pos, 1)
            assert actual_ratio <= ratio + 0.5

    def test_only_weighted_pixels_sampled(self, rng):
        """All sampled pixels must have weight > 0 (forest + labelled)."""
        metadata = _make_metadata()
        tf = _dummy_tile_features()
        sampled = sample_tile_training_data(metadata, tf, rng, negative_ratio=1.0)
        assert (sampled.weights > 0).all()

    def test_output_shapes_consistent(self, rng):
        metadata = _make_metadata()
        tf = _dummy_tile_features()
        sampled = sample_tile_training_data(metadata, tf, rng, negative_ratio=2.0)
        n = sampled.features.shape[0]
        assert sampled.labels.shape == (n,)
        assert sampled.weights.shape == (n,)

    def test_max_samples_per_tile_limit(self, rng):
        metadata = _make_metadata()
        tf = _dummy_tile_features()
        max_samples = 10
        sampled = sample_tile_training_data(
            metadata, tf, rng, negative_ratio=10.0, max_samples_per_tile=max_samples,
        )
        assert sampled.features.shape[0] <= max_samples

    def test_tile_without_labels_raises(self, rng):
        metadata = _make_metadata()
        tf = _dummy_tile_features(has_labels=False)
        with pytest.raises(ValueError, match="labels"):
            sample_tile_training_data(metadata, tf, rng, negative_ratio=1.0)

    def test_full_tile_sampling_with_real_features(self, synthetic_tile, rng):
        """End-to-end: build features from TileData and sample training data."""
        metadata = _make_metadata()
        tf = build_tile_features(synthetic_tile, TARGET_YEARS, use_cache=False)
        if tf.labels is None:
            pytest.skip("Synthetic tile produced no training labels (persistence threshold not met)")
        sampled = sample_tile_training_data(metadata, tf, rng, negative_ratio=3.0)
        assert sampled.features.shape[1] == tf.feature_cube.shape[0]


class TestBuildEvalMatrix:
    def test_returns_four_arrays(self):
        tf = _dummy_tile_features()
        result = build_eval_matrix(tf)
        assert len(result) == 4

    def test_shapes_consistent(self):
        tf = _dummy_tile_features()
        features, labels, weights, eval_mask = build_eval_matrix(tf)
        n = features.shape[0]
        assert labels.shape == (n,)
        assert weights.shape == (n,)
        assert eval_mask.shape == (H, W)

    def test_only_weighted_pixels_in_eval(self):
        """Eval set must contain only pixels with weight > 0."""
        tf = _dummy_tile_features()
        _, _, weights, _ = build_eval_matrix(tf)
        assert (weights > 0).all()

    def test_raises_without_labels(self):
        tf = _dummy_tile_features(has_labels=False)
        with pytest.raises(ValueError):
            build_eval_matrix(tf)
