"""Warp helpers: read a raster on an arbitrary destination grid."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject


def open_reference_grid(path: Path) -> tuple[rasterio.Affine, rasterio.crs.CRS, int, int]:
    """Extract (transform, crs, height, width) from a raster to use as master grid."""
    with rasterio.open(path) as src:
        return src.transform, src.crs, src.height, src.width


def read_on_grid(
    src_path: Path,
    dst_transform: rasterio.Affine,
    dst_crs: rasterio.crs.CRS,
    dst_shape: tuple[int, int],
    resampling: Resampling = Resampling.bilinear,
    nodata_fill: float = np.nan,
) -> np.ndarray:
    """Warp all bands of ``src_path`` onto the destination grid.

    Returns ``(bands, H, W)`` float32. Pixels outside the source coverage are
    filled with ``nodata_fill``.
    """
    H, W = dst_shape
    with rasterio.open(src_path) as src:
        n_bands = src.count
        src_nodata = src.nodata
        dst = np.full((n_bands, H, W), nodata_fill, dtype=np.float32)
        reproject(
            source=rasterio.band(src, list(range(1, n_bands + 1))),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src_nodata,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_nodata=nodata_fill,
            resampling=resampling,
        )
    return dst
