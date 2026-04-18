"""Tests for src/postprocess.py — probability map binarisation and postprocessing."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest
from rasterio import Affine

from src.postprocess import (
    PostprocessConfig,
    hectares_to_pixels,
    pixel_area_square_meters,
    postprocess_probability_map,
)


# 10 m pixel Sentinel-2 UTM transform (100 m² per pixel)
_TRANSFORM_10M = Affine(10.0, 0.0, 300000.0, 0.0, -10.0, 5000000.0)


def _all_forest(h: int = 16, w: int = 16) -> np.ndarray:
    return np.ones((h, w), dtype=bool)


def _no_forest(h: int = 16, w: int = 16) -> np.ndarray:
    return np.zeros((h, w), dtype=bool)


def _config(
    threshold: float = 0.5,
    closing_radius: int = 0,
    opening_radius: int = 0,
    min_area_ha: float = 0.0,
    min_hole_area_ha: float = 0.0,
    persistence_threshold: float = 0.0,
) -> PostprocessConfig:
    return PostprocessConfig(
        threshold=threshold,
        closing_radius=closing_radius,
        opening_radius=opening_radius,
        min_area_ha=min_area_ha,
        min_hole_area_ha=min_hole_area_ha,
        persistence_threshold=persistence_threshold,
    )


@pytest.fixture(autouse=True)
def mock_skimage(monkeypatch):
    """Inject a mock skimage.morphology into sys.modules so tests run without skimage."""
    mock_morph = MagicMock()
    mock_morph.binary_closing.side_effect = lambda x, footprint=None: x
    mock_morph.binary_opening.side_effect = lambda x, footprint=None: x
    mock_morph.remove_small_objects.side_effect = lambda x, min_size=1: x
    mock_morph.remove_small_holes.side_effect = lambda x, area_threshold=1: x
    mock_morph.disk.return_value = None

    monkeypatch.setitem(sys.modules, "skimage", MagicMock())
    monkeypatch.setitem(sys.modules, "skimage.morphology", mock_morph)
    return mock_morph


# ---------------------------------------------------------------------------
# hectares_to_pixels / pixel_area_square_meters
# ---------------------------------------------------------------------------

class TestPixelAreaSquareMeters:
    def test_10m_pixel(self):
        assert pixel_area_square_meters(_TRANSFORM_10M) == pytest.approx(100.0)

    def test_symmetric_pixel(self):
        t = Affine(20.0, 0.0, 0.0, 0.0, -20.0, 0.0)
        assert pixel_area_square_meters(t) == pytest.approx(400.0)

    def test_non_square_pixel(self):
        t = Affine(10.0, 0.0, 0.0, 0.0, -20.0, 0.0)
        assert pixel_area_square_meters(t) == pytest.approx(200.0)


class TestHectaresToPixels:
    def test_zero_area_returns_zero(self):
        assert hectares_to_pixels(0.0, _TRANSFORM_10M) == 0

    def test_negative_area_returns_zero(self):
        assert hectares_to_pixels(-1.0, _TRANSFORM_10M) == 0

    def test_one_hectare_at_10m(self):
        # 1 ha = 10000 m², pixel = 100 m² → 100 pixels
        assert hectares_to_pixels(1.0, _TRANSFORM_10M) == 100

    def test_half_hectare_at_10m(self):
        assert hectares_to_pixels(0.5, _TRANSFORM_10M) == 50

    def test_fractional_rounds_up(self):
        # 0.001 ha = 10 m² / 100 m² per pixel = 0.1 pixels → ceil → 1
        assert hectares_to_pixels(0.001, _TRANSFORM_10M) >= 1

    def test_minimum_is_one(self):
        assert hectares_to_pixels(1e-6, _TRANSFORM_10M) >= 1


# ---------------------------------------------------------------------------
# postprocess_probability_map
# ---------------------------------------------------------------------------

class TestThresholdGating:
    def test_above_threshold_predicts_positive(self):
        probs = np.full((8, 8), 0.9, dtype=np.float32)
        result = postprocess_probability_map(probs, forest_mask=_all_forest(8, 8), transform=_TRANSFORM_10M, config=_config(threshold=0.5))
        assert result.all()

    def test_below_threshold_predicts_negative(self):
        probs = np.full((8, 8), 0.3, dtype=np.float32)
        result = postprocess_probability_map(probs, forest_mask=_all_forest(8, 8), transform=_TRANSFORM_10M, config=_config(threshold=0.5))
        assert not result.any()

    def test_exactly_at_threshold_is_positive(self):
        probs = np.full((8, 8), 0.5, dtype=np.float32)
        result = postprocess_probability_map(probs, forest_mask=_all_forest(8, 8), transform=_TRANSFORM_10M, config=_config(threshold=0.5))
        assert result.all()

    def test_mixed_probs_split_correctly(self):
        probs = np.zeros((4, 4), dtype=np.float32)
        probs[:2, :] = 0.8
        result = postprocess_probability_map(probs, forest_mask=_all_forest(4, 4), transform=_TRANSFORM_10M, config=_config(threshold=0.5))
        assert result[:2, :].all()
        assert not result[2:, :].any()


class TestForestMaskEnforcement:
    def test_out_of_forest_always_zero(self):
        """Model predictions outside the 2020 forest mask must be zeroed out."""
        probs = np.ones((8, 8), dtype=np.float32)
        result = postprocess_probability_map(probs, forest_mask=_no_forest(8, 8), transform=_TRANSFORM_10M, config=_config(threshold=0.0))
        assert not result.any()

    def test_partial_forest_mask(self):
        forest = np.zeros((8, 8), dtype=bool)
        forest[:4, :] = True
        probs = np.ones((8, 8), dtype=np.float32)
        result = postprocess_probability_map(probs, forest_mask=forest, transform=_TRANSFORM_10M, config=_config(threshold=0.5))
        assert not result[4:, :].any()

    def test_output_dtype_is_bool(self):
        probs = np.random.rand(8, 8).astype(np.float32)
        result = postprocess_probability_map(probs, forest_mask=_all_forest(8, 8), transform=_TRANSFORM_10M, config=_config())
        assert result.dtype == bool

    def test_output_shape_matches_input(self):
        probs = np.ones((16, 32), dtype=np.float32)
        forest = np.ones((16, 32), dtype=bool)
        result = postprocess_probability_map(probs, forest_mask=forest, transform=_TRANSFORM_10M, config=_config())
        assert result.shape == (16, 32)


class TestPersistenceFilter:
    def test_gate_suppresses_low_score_pixels(self):
        probs = np.ones((8, 8), dtype=np.float32)
        low_persistence = np.full((8, 8), 0.1, dtype=np.float32)
        result = postprocess_probability_map(
            probs, forest_mask=_all_forest(8, 8), transform=_TRANSFORM_10M,
            config=_config(threshold=0.5, persistence_threshold=0.5),
            persistence_score=low_persistence,
        )
        assert not result.any()

    def test_gate_keeps_high_score_pixels(self):
        probs = np.ones((8, 8), dtype=np.float32)
        high_persistence = np.full((8, 8), 0.9, dtype=np.float32)
        result = postprocess_probability_map(
            probs, forest_mask=_all_forest(8, 8), transform=_TRANSFORM_10M,
            config=_config(threshold=0.5, persistence_threshold=0.5),
            persistence_score=high_persistence,
        )
        assert result.all()

    def test_zero_threshold_disables_gate(self):
        probs = np.ones((8, 8), dtype=np.float32)
        zero_persistence = np.zeros((8, 8), dtype=np.float32)
        result = postprocess_probability_map(
            probs, forest_mask=_all_forest(8, 8), transform=_TRANSFORM_10M,
            config=_config(threshold=0.5, persistence_threshold=0.0),
            persistence_score=zero_persistence,
        )
        assert result.all()

    def test_none_persistence_score_disables_gate(self):
        probs = np.ones((8, 8), dtype=np.float32)
        result = postprocess_probability_map(
            probs, forest_mask=_all_forest(8, 8), transform=_TRANSFORM_10M,
            config=_config(threshold=0.5, persistence_threshold=0.9),
            persistence_score=None,
        )
        assert result.all()


class TestMorphologicalOpsInvoked:
    def test_closing_called_when_radius_nonzero(self, mock_skimage):
        probs = np.ones((8, 8), dtype=np.float32)
        postprocess_probability_map(probs, forest_mask=_all_forest(8, 8), transform=_TRANSFORM_10M, config=_config(closing_radius=2))
        mock_skimage.binary_closing.assert_called_once()

    def test_closing_not_called_when_radius_zero(self, mock_skimage):
        probs = np.ones((8, 8), dtype=np.float32)
        postprocess_probability_map(probs, forest_mask=_all_forest(8, 8), transform=_TRANSFORM_10M, config=_config(closing_radius=0))
        mock_skimage.binary_closing.assert_not_called()

    def test_opening_called_when_radius_nonzero(self, mock_skimage):
        probs = np.ones((8, 8), dtype=np.float32)
        postprocess_probability_map(probs, forest_mask=_all_forest(8, 8), transform=_TRANSFORM_10M, config=_config(opening_radius=1))
        mock_skimage.binary_opening.assert_called_once()

    def test_opening_not_called_when_radius_zero(self, mock_skimage):
        probs = np.ones((8, 8), dtype=np.float32)
        postprocess_probability_map(probs, forest_mask=_all_forest(8, 8), transform=_TRANSFORM_10M, config=_config(opening_radius=0))
        mock_skimage.binary_opening.assert_not_called()
