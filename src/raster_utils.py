from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject


@dataclass(frozen=True)
class RasterGrid:
    width: int
    height: int
    crs: CRS
    transform: Affine

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width


def grid_from_raster(path: Path) -> RasterGrid:
    with rasterio.open(path) as src:
        return RasterGrid(
            width=src.width,
            height=src.height,
            crs=src.crs,
            transform=src.transform,
        )


def read_reprojected(
    path: Path,
    grid: RasterGrid,
    *,
    resampling: Resampling,
    band_indexes: Sequence[int] | None = None,
    out_dtype: np.dtype | type | None = None,
    fill_value: float | int | None = None,
) -> np.ndarray:
    with rasterio.open(path) as src:
        indexes = tuple(band_indexes or src.indexes)
        if out_dtype is None:
            out_dtype = np.dtype(src.dtypes[indexes[0] - 1])
        else:
            out_dtype = np.dtype(out_dtype)

        if fill_value is None:
            fill_value = np.nan if np.issubdtype(out_dtype, np.floating) else 0

        destination = np.full(
            (len(indexes), grid.height, grid.width),
            fill_value,
            dtype=out_dtype,
        )

        same_grid = (
            src.crs == grid.crs
            and src.transform == grid.transform
            and src.width == grid.width
            and src.height == grid.height
        )
        if same_grid:
            data = src.read(indexes, out_dtype=out_dtype)
            if len(indexes) == 1:
                destination[0] = data[0]
            else:
                destination[:] = data
            return destination

        src_nodatavals = src.nodatavals
        for offset, band_index in enumerate(indexes):
            src_nodata = src_nodatavals[band_index - 1]
            reproject(
                source=rasterio.band(src, band_index),
                destination=destination[offset],
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src_nodata,
                dst_transform=grid.transform,
                dst_crs=grid.crs,
                dst_nodata=fill_value,
                resampling=resampling,
            )
        return destination


def to_jsonable_transform(transform: Affine) -> list[float]:
    return [transform.a, transform.b, transform.c, transform.d, transform.e, transform.f]
