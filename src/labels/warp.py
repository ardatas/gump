"""Warp a label TIFF onto the target S2 UTM grid using nearest-neighbour resampling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling

from src.data.reproject import read_on_grid


def warp_label(
    src_path: Path,
    dst_transform: rasterio.Affine,
    dst_crs: rasterio.crs.CRS,
    dst_shape: tuple[int, int],
    nodata_fill: float = 0.0,
) -> np.ndarray:
    """Warp a single-band label raster onto the target grid.

    Returns (H, W) float32 with nearest-neighbour resampling.
    nodata_fill=0 so missing-coverage pixels are treated as no-alert.
    """
    arr = read_on_grid(
        src_path,
        dst_transform,
        dst_crs,
        dst_shape,
        resampling=Resampling.nearest,
        nodata_fill=nodata_fill,
    )
    return arr[0]  # drop band dim → (H, W)
