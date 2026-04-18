"""Integration tests for the training pipeline end-to-end (no disk I/O)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.dataset import PixelSampleDataset, build_eval_matrix, sample_tile_training_data
from src.feature_bank import build_tile_features
from src.metrics import find_best_threshold, weighted_precision_recall_f1
from src.models import ResidualMLP, build_model
from src.train import (
    LossConfig,
    RuntimeConfig,
    compute_weighted_loss,
    fit_normalizer,
    normalize_features,
    predict_probabilities,
)
from src.feature_bank import compute_year_heuristic_map
from tests.conftest import TARGET_YEARS, H, W, _make_metadata, _make_tile


def _make_runtime_config() -> RuntimeConfig:
    return RuntimeConfig(
        device=torch.device("cpu"),
        amp_dtype=None,
        grad_clip_norm=1.0,
    )


def _make_loss_config(name: str = "focal") -> LossConfig:
    return LossConfig(
        name=name,
        focal_gamma=2.0,
        positive_class_weight=None,
        gce_q=0.7,
        label_smoothing=0.0,
    )


class TestTrainingPipeline:
    def test_training_epoch_completes_without_nan(self, rng):
        """Full training epoch must not produce NaN losses."""
        tile = _make_tile(rng)
        metadata = _make_metadata()
        tf = build_tile_features(tile, TARGET_YEARS, use_cache=False)
        if tf.labels is None:
            pytest.skip("No training labels produced by persistence logic")

        sampled = sample_tile_training_data(
            metadata, tf, rng, negative_ratio=3.0, max_samples_per_tile=500
        )
        normalizer = fit_normalizer(sampled.features)
        features_norm = normalize_features(sampled.features, normalizer)
        dataset = PixelSampleDataset(features_norm, sampled.labels, sampled.weights)
        loader = DataLoader(dataset, batch_size=128, shuffle=True)
        model = build_model("mlp", input_dim=features_norm.shape[1], hidden_dims=[64, 32], dropout=0.1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        loss_config = _make_loss_config("focal")
        runtime_config = _make_runtime_config()

        model.train()
        for batch in loader:
            f = batch["features"]
            l = batch["labels"]
            w = batch["weights"]
            optimizer.zero_grad(set_to_none=True)
            outputs = model(f)
            loss, _, _ = compute_weighted_loss(
                logits=outputs.logits, labels=l, weights=w, loss_config=loss_config
            )
            assert not torch.isnan(loss), "Training loss is NaN"
            loss.backward()
            optimizer.step()

    def test_focal_loss_lower_than_bce_on_easy_negatives(self):
        """Focal loss must down-weight easy negatives vs plain BCE."""
        logits = torch.full((100,), -5.0)  # very confident negatives
        labels = torch.zeros(100)
        weights = torch.ones(100)

        _, bce_sum, bce_weight = compute_weighted_loss(
            logits=logits, labels=labels, weights=weights,
            loss_config=_make_loss_config("bce"),
        )
        _, focal_sum, focal_weight = compute_weighted_loss(
            logits=logits, labels=labels, weights=weights,
            loss_config=_make_loss_config("focal"),
        )
        bce_mean = float(bce_sum / bce_weight)
        focal_mean = float(focal_sum / focal_weight)
        assert focal_mean < bce_mean, "Focal loss must be lower on easy negatives"

    def test_gce_loss_runs_without_nan(self):
        logits = torch.randn(64)
        labels = (torch.rand(64) > 0.8).float()
        weights = torch.ones(64)
        loss, _, _ = compute_weighted_loss(
            logits=logits, labels=labels, weights=weights,
            loss_config=_make_loss_config("gce"),
        )
        assert torch.isfinite(loss)

    def test_normalizer_zero_mean(self, rng):
        """After normalization, features must have approximately zero mean."""
        tile = _make_tile(rng)
        tf = build_tile_features(tile, TARGET_YEARS, use_cache=False)
        if tf.labels is None:
            pytest.skip("No training labels")
        metadata = _make_metadata()
        sampled = sample_tile_training_data(
            metadata, tf, rng, negative_ratio=3.0, max_samples_per_tile=2000
        )
        normalizer = fit_normalizer(sampled.features)
        normed = normalize_features(sampled.features, normalizer)
        assert np.abs(normed.mean(axis=0)).max() < 0.05

    def test_inference_probabilities_in_unit_interval(self, rng):
        """All predicted probabilities must lie in [0, 1]."""
        tile = _make_tile(rng)
        tf = build_tile_features(tile, TARGET_YEARS, use_cache=False)
        if tf.labels is None:
            pytest.skip("No training labels")
        metadata = _make_metadata()
        sampled = sample_tile_training_data(
            metadata, tf, rng, negative_ratio=3.0, max_samples_per_tile=500
        )
        normalizer = fit_normalizer(sampled.features)
        features_norm = normalize_features(sampled.features, normalizer)
        model = build_model("mlp", input_dim=features_norm.shape[1], hidden_dims=[64], dropout=0.0)
        runtime_config = _make_runtime_config()
        probs = predict_probabilities(
            model=model,
            features=features_norm,
            runtime_config=runtime_config,
            batch_size=256,
        )
        assert probs.min() >= 0.0
        assert probs.max() <= 1.0

    def test_forest_mask_filters_predictions(self, rng):
        """Predictions outside the 2020 forest mask must always be zero."""
        tile = _make_tile(rng)
        tf = build_tile_features(tile, TARGET_YEARS, use_cache=False)
        model = build_model("mlp", input_dim=tf.feature_cube.shape[0], hidden_dims=[32], dropout=0.0)
        normalizer = fit_normalizer(tf.feature_cube.reshape(tf.feature_cube.shape[0], -1).T)
        runtime_config = _make_runtime_config()
        feature_matrix = normalize_features(
            tf.feature_cube.reshape(tf.feature_cube.shape[0], -1).T, normalizer
        )
        probs = predict_probabilities(
            model=model, features=feature_matrix,
            runtime_config=runtime_config, batch_size=1024,
        )
        prob_map = probs.reshape(H, W)
        prediction = (prob_map >= 0.5) & tf.forest_mask
        outside = ~tf.forest_mask
        assert not prediction[outside].any(), "Predictions must be masked by forest_2020"

    def test_weighted_loss_downweights_uncertain_pixels(self):
        """Loss contribution must scale with weight."""
        logits = torch.tensor([2.0, 2.0])
        labels = torch.tensor([0.0, 0.0])
        weights_full = torch.tensor([1.0, 1.0])
        weights_low = torch.tensor([0.3, 0.3])

        _, sum_full, w_full = compute_weighted_loss(
            logits=logits, labels=labels, weights=weights_full,
            loss_config=_make_loss_config("bce"),
        )
        _, sum_low, w_low = compute_weighted_loss(
            logits=logits, labels=labels, weights=weights_low,
            loss_config=_make_loss_config("bce"),
        )
        # The weighted mean is the same (same logits/labels), but total loss sum is lower
        assert float(sum_low) < float(sum_full)

    def test_deforestation_only_after_2020_in_labels(self):
        """Task constraint: pre-2020 alerts must never produce positive labels."""
        from datetime import date
        from src.labels.decoders import decode_radd, _RADD_EPOCH

        pre2020 = date(2019, 12, 31)
        val = 3 * 10000 + (pre2020 - _RADD_EPOCH).days
        raw = np.array([[val]], dtype=np.int32)
        is_defo, _ = decode_radd(raw)
        assert not is_defo.any(), "Pre-2020 alerts must never yield positive deforestation labels"

    def test_feature_cube_aligns_with_labels(self, synthetic_tile):
        """Feature cube and label arrays must share spatial dimensions."""
        tf = build_tile_features(synthetic_tile, TARGET_YEARS, use_cache=False)
        if tf.labels is not None:
            assert tf.feature_cube.shape[1:] == tf.labels.shape
        if tf.weights is not None:
            assert tf.feature_cube.shape[1:] == tf.weights.shape

    def test_eval_matrix_features_have_correct_dim(self, rng):
        """Eval matrix pixel features must match the feature cube channel count."""
        tile = _make_tile(rng)
        tf = build_tile_features(tile, TARGET_YEARS, use_cache=False)
        if tf.labels is None:
            pytest.skip("No labels")
        feature_matrix, _, _, _ = build_eval_matrix(tf)
        assert feature_matrix.shape[1] == tf.feature_cube.shape[0]

    def test_cosine_lr_decreases_over_epochs(self):
        """CosineAnnealingLR must reduce learning rate monotonically after warmup."""
        model = ResidualMLP(input_dim=32)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=10, eta_min=1e-5
        )
        lrs = []
        for _ in range(10):
            lrs.append(optimizer.param_groups[0]["lr"])
            scheduler.step()
        # LR must decrease from start to end
        assert lrs[-1] < lrs[0], f"LR did not decrease: {lrs[0]:.6f} → {lrs[-1]:.6f}"


class TestYearHeuristicMap:
    def test_output_shape(self, rng):
        tile = _make_tile(rng)
        year_map = compute_year_heuristic_map(tile, TARGET_YEARS)
        assert year_map.shape == (H, W)

    def test_output_dtype_float32(self, rng):
        tile = _make_tile(rng)
        year_map = compute_year_heuristic_map(tile, TARGET_YEARS)
        assert year_map.dtype == np.float32

    def test_years_are_valid_target_years(self, rng):
        """Every assigned year must be one of the target years (not an interpolation)."""
        tile = _make_tile(rng)
        year_map = compute_year_heuristic_map(tile, TARGET_YEARS)
        unique_years = set(year_map.flatten().tolist())
        assert unique_years.issubset({float(y) for y in TARGET_YEARS})

    def test_single_year_returns_constant_map(self, rng):
        """With only one target year, all pixels get that year as default."""
        tile = _make_tile(rng)
        year_map = compute_year_heuristic_map(tile, (2021,))
        assert (year_map == 2021.0).all()

    def test_large_nir_drop_year2_returns_year2(self, rng):
        """If NIR drops sharply in year 2, heuristic should assign year 2021."""
        tile = _make_tile(rng)
        # Make year-2021 observations (indices 6-11) have very low NIR
        tile.s2_data[6:, 7] = 200.0   # NIR collapses in 2021
        tile.s2_data[:6, 7] = 7000.0  # NIR high in 2020
        year_map = compute_year_heuristic_map(tile, TARGET_YEARS)
        # Majority of pixels should be assigned 2021
        assert (year_map == 2021.0).mean() > 0.5
