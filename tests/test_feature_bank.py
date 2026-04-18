"""Tests for src/feature_bank.py — feature construction from tile data."""

from __future__ import annotations

import numpy as np
import pytest

from src.feature_bank import (
    TileFeatures,
    build_tile_features,
    build_monthly_sequence,
    build_target_months,
    compute_persistence_targets,
    SEQUENCE_CHANNELS,
)
from tests.conftest import TARGET_YEARS, H, W, _make_tile


class TestBuildTileFeatures:
    def test_returns_tile_features_instance(self, synthetic_tile):
        result = build_tile_features(synthetic_tile, TARGET_YEARS, use_cache=False)
        assert isinstance(result, TileFeatures)

    def test_feature_cube_has_correct_spatial_shape(self, synthetic_tile):
        result = build_tile_features(synthetic_tile, TARGET_YEARS, use_cache=False)
        assert result.feature_cube.shape[1:] == (H, W)

    def test_feature_names_length_matches_feature_cube_depth(self, synthetic_tile):
        result = build_tile_features(synthetic_tile, TARGET_YEARS, use_cache=False)
        n_features = result.feature_cube.shape[0]
        assert len(result.feature_names) == n_features, (
            f"feature_cube depth={n_features} but len(feature_names)={len(result.feature_names)}"
        )

    def test_feature_names_are_unique(self, synthetic_tile):
        result = build_tile_features(synthetic_tile, TARGET_YEARS, use_cache=False)
        names = result.feature_names
        assert len(names) == len(set(names)), "Feature names must be unique"

    def test_no_nan_or_inf_in_output(self, synthetic_tile):
        """nan_to_num must be applied — no NaN or Inf in feature cube."""
        result = build_tile_features(synthetic_tile, TARGET_YEARS, use_cache=False)
        assert np.isfinite(result.feature_cube).all(), "Feature cube contains NaN or Inf"

    def test_feature_cube_dtype_is_float32(self, synthetic_tile):
        result = build_tile_features(synthetic_tile, TARGET_YEARS, use_cache=False)
        assert result.feature_cube.dtype == np.float32

    def test_delta_features_exist(self, synthetic_tile):
        """Delta features capture vegetation change — the core deforestation signal."""
        result = build_tile_features(synthetic_tile, TARGET_YEARS, use_cache=False)
        delta_names = [name for name in result.feature_names if "delta" in name]
        assert len(delta_names) > 0, "No delta features found"

    def test_ndvi_features_exist(self, synthetic_tile):
        result = build_tile_features(synthetic_tile, TARGET_YEARS, use_cache=False)
        ndvi_names = [name for name in result.feature_names if "ndvi" in name]
        assert len(ndvi_names) > 0, "No NDVI features found"

    def test_slope_features_exist(self, synthetic_tile):
        """Temporal slope features (trend over time) must be present."""
        result = build_tile_features(synthetic_tile, TARGET_YEARS, use_cache=False)
        slope_names = [name for name in result.feature_names if "slope" in name]
        assert len(slope_names) > 0, "No slope features found (required for trend detection)"

    def test_min_max_features_exist(self, synthetic_tile):
        """Min/max features capture extreme events (e.g., bare soil after logging)."""
        result = build_tile_features(synthetic_tile, TARGET_YEARS, use_cache=False)
        feature_names = result.feature_names
        assert any("_min" in name for name in feature_names), "No min features found"
        assert any("_max" in name for name in feature_names), "No max features found"

    def test_s1_features_exist(self, synthetic_tile):
        result = build_tile_features(synthetic_tile, TARGET_YEARS, use_cache=False)
        s1_names = [name for name in result.feature_names if "s1" in name]
        assert len(s1_names) > 0, "No Sentinel-1 features found"

    def test_aef_features_exist(self, synthetic_tile):
        result = build_tile_features(synthetic_tile, TARGET_YEARS, use_cache=False)
        aef_names = [name for name in result.feature_names if "aef" in name]
        assert len(aef_names) > 0, "No AEF embedding features found"

    def test_persistence_score_in_features(self, synthetic_tile):
        """Persistence score feature captures multi-temporal vegetation loss signal."""
        result = build_tile_features(synthetic_tile, TARGET_YEARS, use_cache=False)
        assert "persistence_score" in result.feature_names

    def test_forest_mask_preserved(self, synthetic_tile):
        result = build_tile_features(synthetic_tile, TARGET_YEARS, use_cache=False)
        assert result.forest_mask.shape == (H, W)
        assert result.forest_mask.dtype == bool

    def test_persistence_score_shape(self, synthetic_tile):
        result = build_tile_features(synthetic_tile, TARGET_YEARS, use_cache=False)
        assert result.persistence_score.shape == (H, W)
        assert result.persistence_mask.shape == (H, W)

    def test_single_year_does_not_crash(self, synthetic_tile):
        result = build_tile_features(synthetic_tile, (2020,), use_cache=False)
        assert result.feature_cube.ndim == 3

    def test_feature_count_scales_with_target_years(self, synthetic_tile):
        r2 = build_tile_features(synthetic_tile, (2020, 2021), use_cache=False)
        r1 = build_tile_features(synthetic_tile, (2020,), use_cache=False)
        assert r2.feature_cube.shape[0] > r1.feature_cube.shape[0]

    def test_s2_delta_captures_nir_drop(self, rng):
        """NIR band drop in year 2 should yield a negative s2_delta_b08 mean."""
        tile = _make_tile(rng)
        # Year-2 observations: drastically lower NIR → simulates forest clearing
        tile.s2_data[6:, 7] = 500.0
        result = build_tile_features(tile, TARGET_YEARS, use_cache=False)
        delta_idx = [i for i, name in enumerate(result.feature_names) if name == "s2_delta_b08"]
        assert len(delta_idx) == 1
        assert result.feature_cube[delta_idx[0]].mean() < 0, "Delta must be negative after NIR drop"


class TestBuildMonthlySequence:
    def test_sequence_shape(self, synthetic_tile):
        target_months = build_target_months(TARGET_YEARS)
        seq = build_monthly_sequence(synthetic_tile, target_months)
        T = len(target_months)
        C = len(SEQUENCE_CHANNELS)
        assert seq.sequence.shape == (T, C, H, W)
        assert seq.sequence_mask.shape == (T, H, W)

    def test_channel_names_match(self, synthetic_tile):
        target_months = build_target_months(TARGET_YEARS)
        seq = build_monthly_sequence(synthetic_tile, target_months)
        assert seq.channel_names == SEQUENCE_CHANNELS

    def test_empty_months_produces_zeros(self, synthetic_tile):
        seq = build_monthly_sequence(synthetic_tile, ())
        assert seq.sequence.shape[0] == 0
        assert seq.sequence_mask.shape[0] == 0

    def test_no_nan_in_sequence(self, synthetic_tile):
        target_months = build_target_months(TARGET_YEARS)
        seq = build_monthly_sequence(synthetic_tile, target_months)
        assert np.isfinite(seq.sequence).all(), "Sequence contains NaN or Inf"

    def test_sequence_mask_is_boolean(self, synthetic_tile):
        target_months = build_target_months(TARGET_YEARS)
        seq = build_monthly_sequence(synthetic_tile, target_months)
        assert seq.sequence_mask.dtype == bool


class TestComputePersistenceTargets:
    def _make_index_yearly(self, value: float, n_years: int = 2) -> np.ndarray:
        return np.full((n_years, H, W), value, dtype=np.float32)

    def test_returns_persistence_targets(self, synthetic_tile):
        ndvi = self._make_index_yearly(0.7)
        nbr = self._make_index_yearly(0.5)
        ndmi = self._make_index_yearly(0.4)
        result = compute_persistence_targets(
            forest_mask=synthetic_tile.forest_mask,
            raw_labels=synthetic_tile.label_consensus,
            raw_weights=synthetic_tile.label_weight,
            raw_votes=synthetic_tile.label_votes,
            ndvi_yearly=ndvi,
            nbr_yearly=nbr,
            ndmi_yearly=ndmi,
        )
        assert result.persistence_score.shape == (H, W)
        assert result.persistence_mask.shape == (H, W)
        assert result.feature_layers.shape[0] == 8  # 8 fixed feature layers

    def test_no_labels_returns_none_training_data(self, synthetic_tile):
        ndvi = self._make_index_yearly(0.7)
        nbr = self._make_index_yearly(0.5)
        ndmi = self._make_index_yearly(0.4)
        result = compute_persistence_targets(
            forest_mask=synthetic_tile.forest_mask,
            raw_labels=None,
            raw_weights=None,
            raw_votes=None,
            ndvi_yearly=ndvi,
            nbr_yearly=nbr,
            ndmi_yearly=ndmi,
        )
        assert result.training_labels is None
        assert result.training_weights is None

    def test_training_labels_subset_of_consensus(self, synthetic_tile):
        """Training labels must never be positive where raw consensus is negative."""
        ndvi = self._make_index_yearly(0.7)
        nbr = self._make_index_yearly(0.6)
        ndmi = self._make_index_yearly(0.4)
        result = compute_persistence_targets(
            forest_mask=synthetic_tile.forest_mask,
            raw_labels=synthetic_tile.label_consensus,
            raw_weights=synthetic_tile.label_weight,
            raw_votes=synthetic_tile.label_votes,
            ndvi_yearly=ndvi,
            nbr_yearly=nbr,
            ndmi_yearly=ndmi,
        )
        assert result.training_labels is not None
        # A pixel that was never flagged by any source cannot become a training positive
        false_positives = (result.training_labels == 1) & (synthetic_tile.label_consensus == 0)
        assert not false_positives.any(), "training_labels must be a subset of raw consensus"

    def test_out_of_forest_weight_zero(self, synthetic_tile):
        """Pixels outside forest_mask must have weight=0 after refinement."""
        ndvi = self._make_index_yearly(0.7)
        nbr = self._make_index_yearly(0.5)
        ndmi = self._make_index_yearly(0.4)
        result = compute_persistence_targets(
            forest_mask=synthetic_tile.forest_mask,
            raw_labels=synthetic_tile.label_consensus,
            raw_weights=synthetic_tile.label_weight,
            raw_votes=synthetic_tile.label_votes,
            ndvi_yearly=ndvi,
            nbr_yearly=nbr,
            ndmi_yearly=ndmi,
        )
        assert result.training_weights is not None
        outside = ~synthetic_tile.forest_mask
        if outside.any():
            assert (result.training_weights[outside] == 0.0).all()

    def test_large_ndvi_drop_raises_persistence_score(self, synthetic_tile):
        """A large NDVI drop (0.9 → 0.1) should produce high persistence score."""
        ndvi_high = np.full((1, H, W), 0.9, dtype=np.float32)
        ndvi_low = np.full((1, H, W), 0.1, dtype=np.float32)
        ndvi = np.concatenate([ndvi_high, ndvi_low], axis=0)

        nbr_high = np.full((1, H, W), 0.7, dtype=np.float32)
        nbr_low = np.full((1, H, W), 0.1, dtype=np.float32)
        nbr = np.concatenate([nbr_high, nbr_low], axis=0)

        ndmi = self._make_index_yearly(0.3)

        result = compute_persistence_targets(
            forest_mask=np.ones((H, W), dtype=bool),
            raw_labels=None, raw_weights=None, raw_votes=None,
            ndvi_yearly=ndvi, nbr_yearly=nbr, ndmi_yearly=ndmi,
        )
        # Pixels in forest with large drop should have high score
        forest_scores = result.persistence_score[np.ones((H, W), dtype=bool)]
        assert forest_scores.mean() > 0.5, "Large NDVI/NBR drop should yield high persistence score"
