"""Tests for src/features.py — forest mask computation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.features import compute_forest_mask, linear_backscatter_to_db
from src.raster_utils import RasterGrid


H, W = 16, 16


def _make_grid() -> RasterGrid:
    from rasterio.crs import CRS
    from rasterio.transform import Affine
    return RasterGrid(
        width=W,
        height=H,
        crs=CRS.from_epsg(32618),
        transform=Affine(10, 0, 300000, 0, -10, 1000000),
    )


def _make_observation(year: int, ndvi_value: float, month: int = 6) -> MagicMock:
    """Return a mock RasterTimePoint that yields a known NDVI when read."""
    from src.index import RasterTimePoint
    obs = MagicMock(spec=RasterTimePoint)
    obs.year = year
    obs.month = month
    obs.path = Path(f"/fake/{year}_{month}.tif")
    return obs


def _patch_read_reprojected(red_value: float, nir_value: float):
    """Context manager: patches read_reprojected to return [red, nir] arrays."""
    def fake_read(path, grid, resampling, band_indexes, out_dtype):
        red = np.full((H, W), red_value, dtype=np.float32)
        nir = np.full((H, W), nir_value, dtype=np.float32)
        return np.stack([red, nir], axis=0)
    return patch("src.features.read_reprojected", side_effect=fake_read)


class TestComputeForestMask:
    def test_high_ndvi_classified_as_forest(self):
        """NDVI > 0.6 with sufficient observations → all True."""
        # NDVI = (nir - red) / (nir + red) = (8000 - 500) / (8000 + 500) ≈ 0.88
        observations = [_make_observation(2020, ndvi_value=0.88) for _ in range(3)]
        grid = _make_grid()
        with _patch_read_reprojected(red_value=500, nir_value=8000):
            mask = compute_forest_mask(observations, grid, year=2020, min_observations=3)
        assert mask.shape == (H, W)
        assert mask.all(), "High NDVI should produce all-forest mask"

    def test_low_ndvi_not_classified_as_forest(self):
        """NDVI < 0.6 → all False (no forest)."""
        # NDVI = (400 - 600) / (400 + 600) = -0.2
        observations = [_make_observation(2020, ndvi_value=-0.2) for _ in range(3)]
        grid = _make_grid()
        with _patch_read_reprojected(red_value=600, nir_value=400):
            mask = compute_forest_mask(observations, grid, year=2020, min_observations=3)
        assert not mask.any(), "Low NDVI should produce empty forest mask"

    def test_insufficient_observations_not_forest(self):
        """Fewer than min_observations valid observations → not forest."""
        observations = [_make_observation(2020, 0.88) for _ in range(2)]
        grid = _make_grid()
        with _patch_read_reprojected(red_value=500, nir_value=8000):
            mask = compute_forest_mask(
                observations, grid, year=2020, min_observations=3
            )
        assert not mask.any(), "Too few observations → should not be classified as forest"

    def test_fallback_to_nearest_year(self):
        """If requested year absent, falls back to nearest available year."""
        observations = [_make_observation(2021, 0.88) for _ in range(3)]
        grid = _make_grid()
        with _patch_read_reprojected(red_value=500, nir_value=8000):
            with pytest.warns(RuntimeWarning, match="falling back"):
                mask = compute_forest_mask(
                    observations, grid, year=2020, min_observations=3
                )
        assert mask.all()

    def test_no_observations_raises(self):
        grid = _make_grid()
        with pytest.raises(ValueError):
            compute_forest_mask([], grid, year=2020)

    def test_output_is_boolean(self):
        observations = [_make_observation(2020, 0.88) for _ in range(3)]
        grid = _make_grid()
        with _patch_read_reprojected(red_value=500, nir_value=8000):
            mask = compute_forest_mask(observations, grid, year=2020, min_observations=1)
        assert mask.dtype == bool

    def test_output_shape_matches_grid(self):
        observations = [_make_observation(2020, 0.88) for _ in range(3)]
        grid = _make_grid()
        with _patch_read_reprojected(red_value=500, nir_value=8000):
            mask = compute_forest_mask(observations, grid, year=2020, min_observations=1)
        assert mask.shape == (H, W)


class TestLinearBackscatterToDb:
    def test_known_conversion(self):
        """10 * log10(0.1) ≈ -10.0 dB."""
        values = np.array([0.1], dtype=np.float32)
        result = linear_backscatter_to_db(values)
        assert np.isclose(result[0], -10.0, atol=0.01)

    def test_zero_becomes_nan(self):
        """Zero linear backscatter → NaN (log10(0) undefined)."""
        values = np.array([0.0], dtype=np.float32)
        result = linear_backscatter_to_db(values)
        assert np.isnan(result[0])

    def test_clipped_to_range(self):
        """Result is clipped to [-30, 0] dB by default."""
        values = np.array([1e-10, 1.0, 1000.0], dtype=np.float32)
        result = linear_backscatter_to_db(values)
        assert result[0] == pytest.approx(-30.0)  # below min → clipped to -30
        assert result[1] == pytest.approx(0.0)    # 10*log10(1) = 0
        assert result[2] == pytest.approx(0.0)    # above max → clipped to 0

    def test_output_shape_preserved(self):
        values = np.ones((5, 6), dtype=np.float32) * 0.01
        result = linear_backscatter_to_db(values)
        assert result.shape == (5, 6)
