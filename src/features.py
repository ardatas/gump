from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
from rasterio.enums import Resampling

from .index import RasterTimePoint
from .raster_utils import RasterGrid, read_reprojected


def compute_forest_mask(
    s2_observations: list[RasterTimePoint],
    grid: RasterGrid,
    *,
    year: int = 2020,
    red_band_index: int = 4,
    nir_band_index: int = 8,
    ndvi_threshold: float = 0.6,
    min_observations: int = 3,
) -> np.ndarray:
    available_years = sorted({observation.year for observation in s2_observations})
    if not available_years:
        raise ValueError("No Sentinel-2 observations available for forest mask computation.")

    selected_year = year
    if selected_year not in available_years:
        selected_year = min(available_years, key=lambda candidate: (abs(candidate - year), candidate))
        warnings.warn(
            f"No Sentinel-2 observations for forest mask year {year}; "
            f"falling back to nearest available year {selected_year}.",
            RuntimeWarning,
            stacklevel=2,
        )

    ndvi_stack: list[np.ndarray] = []
    for observation in s2_observations:
        if observation.year != selected_year:
            continue
        bands = read_reprojected(
            observation.path,
            grid,
            resampling=Resampling.bilinear,
            band_indexes=[red_band_index, nir_band_index],
            out_dtype=np.float32,
        )
        red = bands[0]
        nir = bands[1]
        denom = nir + red
        valid = np.isfinite(red) & np.isfinite(nir) & (denom > 0)
        ndvi = np.full(grid.shape, np.nan, dtype=np.float32)
        ndvi[valid] = (nir[valid] - red[valid]) / denom[valid]
        ndvi_stack.append(ndvi)

    if not ndvi_stack:
        raise ValueError(
            f"No Sentinel-2 observations found for forest mask year {selected_year}."
        )

    stacked = np.stack(ndvi_stack, axis=0)
    valid_count = np.sum(np.isfinite(stacked), axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median_ndvi = np.nanmedian(stacked, axis=0)
    return (valid_count >= min_observations) & (median_ndvi >= ndvi_threshold)


def linear_backscatter_to_db(
    values: np.ndarray,
    *,
    clip_min: float = -30.0,
    clip_max: float = 0.0,
) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        db = np.where(values > 0, 10.0 * np.log10(values), np.nan)
    return np.clip(db, clip_min, clip_max)
