"""Hansen GFC treecover2020 forest mask: download, reproject, cache, and build.

Tiles are served as Cloud-Optimized GeoTIFFs via Google Cloud Storage and are
accessed using GDAL's ``/vsicurl/`` driver — no full download required.

Public API
----------
- :func:`get_hansen_tile_ids`   — map a WGS84 bbox to Hansen tile IDs
- :func:`download_hansen_tile`  — reproject one S2 tile's worth of treecover onto UTM
- :func:`build_forest_mask`     — threshold a cached treecover GeoTIFF → bool mask
- :func:`get_or_build_forest_mask` — one-call helper with caching + fallback
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from filelock import FileLock
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject as _rasterio_reproject, transform_bounds

LOGGER = logging.getLogger(__name__)

# Hansen GFC 2020 v1.8 treecover2020 layer on GCS
_HANSEN_BASE_URL = (
    "https://storage.googleapis.com/earthenginepartners-hansen"
    "/GFC-2020-v1.8/Hansen_GFC-2020-v1.8_treecover2020"
)
_WGS84: CRS = CRS.from_epsg(4326)
_NODATA: int = 255  # treecover2020 NoData sentinel


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def get_hansen_tile_ids(bbox_lonlat: tuple[float, float, float, float]) -> list[str]:
    """Return the Hansen 10°×10° tile IDs that overlap a WGS84 bounding box.

    Hansen tiles are identified by their NW corner.  The tile ``10N_080W``
    has its north edge at 10°N and west edge at 80°W, covering
    0°N–10°N and 80°W–70°W.  Formula::

        north_edge = ceil(lat / 10) * 10
        west_edge  = floor(lon / 10) * 10

    Args:
        bbox_lonlat: ``(west, south, east, north)`` in WGS84 decimal degrees.

    Returns:
        List of tile ID strings, e.g. ``["00N_080W"]``.  One Sentinel-2 tile
        (≈ 110 km²) almost always maps to exactly one Hansen tile (1° × 1°),
        but edge cases at 10° meridian/parallel boundaries return multiple IDs.
    """
    west, south, east, north = bbox_lonlat

    # North-edge of the southernmost required tile: ceil(south/10)*10
    lat_start = int(math.ceil(south / 10.0)) * 10
    lat_stop  = int(math.ceil(north / 10.0)) * 10
    lon_start = int(math.floor(west / 10.0)) * 10
    lon_stop  = int(math.floor(east / 10.0)) * 10

    tile_ids: list[str] = []
    lat = lat_start
    while lat <= lat_stop:
        lon = lon_start
        while lon <= lon_stop:
            lat_str = f"{abs(lat):02d}{'N' if lat >= 0 else 'S'}"
            lon_str = f"{abs(lon):03d}{'E' if lon >= 0 else 'W'}"
            tile_ids.append(f"{lat_str}_{lon_str}")
            lon += 10
        lat += 10
    return tile_ids


def download_hansen_tile(
    tile_id: str,
    target_utm_crs: CRS,
    target_transform: Affine,
    target_shape: tuple[int, int],
    out_path: Path,
) -> None:
    """Download and reproject Hansen GFC treecover2020 onto a target UTM grid.

    Streams the source via ``/vsicurl/`` using GDAL.  If the target grid spans
    more than one 10°×10° Hansen tile, all tiles are reprojected separately and
    the per-pixel maximum is composited.

    Args:
        tile_id: Sentinel-2 (or other) tile identifier — used only for logging
            and cache file naming; not the Hansen tile ID.
        target_utm_crs: CRS of the destination raster (UTM).
        target_transform: Affine transform of the destination grid.
        target_shape: ``(height, width)`` of the destination grid in pixels.
        out_path: Destination uint8 GeoTIFF path (created if absent).

    Raises:
        RuntimeError: If a required Hansen tile cannot be accessed.
    """
    H, W = target_shape

    # Compute WGS84 bbox of the UTM grid to find the Hansen tiles we need
    left   = target_transform.c
    top    = target_transform.f
    right  = left + W * target_transform.a
    bottom = top  + H * target_transform.e   # .e is negative for north-up
    bbox_wgs84: tuple[float, float, float, float] = transform_bounds(
        target_utm_crs, _WGS84, left, bottom, right, top
    )
    hansen_ids = get_hansen_tile_ids(bbox_wgs84)
    LOGGER.debug(
        "tile=%s  WGS84 bbox=%s  Hansen tiles=%s", tile_id, bbox_wgs84, hansen_ids
    )

    merged   = np.zeros((H, W), dtype=np.uint8)
    covered  = np.zeros((H, W), dtype=bool)

    for hid in hansen_ids:
        url = f"/vsicurl/{_HANSEN_BASE_URL}_{hid}.tif"
        try:
            with rasterio.open(url) as src:
                buf = np.full((H, W), np.nan, dtype=np.float32)
                _rasterio_reproject(
                    source=rasterio.band(src, 1),
                    destination=buf,
                    src_crs=src.crs,
                    src_transform=src.transform,
                    src_nodata=float(_NODATA),
                    dst_crs=target_utm_crs,
                    dst_transform=target_transform,
                    dst_nodata=np.nan,
                    resampling=Resampling.bilinear,
                )
        except Exception as exc:
            raise RuntimeError(
                f"Cannot access Hansen tile {hid!r} for S2 tile {tile_id!r}: {exc}"
            ) from exc

        valid = np.isfinite(buf)
        # Composite: take the per-pixel maximum (tiles don't overlap except at edges)
        merged[valid] = np.maximum(
            merged[valid],
            np.clip(buf[valid], 0, 100).astype(np.uint8),
        )
        covered |= valid

    if not covered.any():
        raise RuntimeError(
            f"No Hansen treecover data projected onto {tile_id!r}; "
            f"tried: {hansen_ids}"
        )
    # Pixels outside all Hansen tiles get the NoData sentinel
    merged[~covered] = _NODATA

    profile: dict[str, Any] = {
        "driver": "GTiff",
        "dtype": "uint8",
        "width": W,
        "height": H,
        "count": 1,
        "crs": target_utm_crs,
        "transform": target_transform,
        "compress": "deflate",
        "nodata": _NODATA,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(merged, 1)
    LOGGER.info("Hansen treecover written: tile=%s  path=%s", tile_id, out_path)


def build_forest_mask(
    treecover_path: Path,
    threshold: int = 30,
) -> np.ndarray:
    """Build a boolean forest mask from a reprojected treecover2020 GeoTIFF.

    Args:
        treecover_path: uint8 GeoTIFF where pixel values are tree-cover
            percentage (0–100) and 255 is NoData.
        threshold: Minimum tree-cover percentage to classify as forest
            (default 30 % — GFC standard definition).

    Returns:
        ``bool`` array of shape ``(H, W)``; ``True`` where treecover ≥ threshold.
    """
    with rasterio.open(treecover_path) as src:
        tc = src.read(1)
    return (tc != _NODATA) & (tc >= np.uint8(threshold))


def get_or_build_forest_mask(
    tile_id: str,
    tile_meta: dict[str, Any],
    cache_dir: Path,
    threshold: int = 30,
    ndvi_fallback_observations: list | None = None,
) -> np.ndarray:
    """Return the Hansen-based 2020 forest mask for ``tile_id``.

    Checks for a cached ``{tile_id}_forest2020.tif`` before downloading.  A
    ``FileLock`` serialises concurrent downloads within a ``multiprocessing.Pool``
    so only one worker fetches each tile.

    If Hansen GFC is unreachable the function falls back to:
    1. An NDVI-based mask computed from ``ndvi_fallback_observations`` (if supplied).
    2. An all-``True`` mask as a last resort (conservative — retains all forest pixels).

    Args:
        tile_id: Tile identifier used for cache file naming and log messages.
        tile_meta: Dict with keys:

            - ``"crs"``       — :class:`rasterio.crs.CRS` of the UTM grid.
            - ``"transform"`` — :class:`~rasterio.transform.Affine`.
            - ``"shape"``     — ``(height, width)`` tuple.

        cache_dir: Directory for the ``{tile_id}_forest2020.tif`` cache.
        threshold: Treecover percentage threshold (default 30 %).
        ndvi_fallback_observations: Optional list of
            :class:`~src.index.RasterTimePoint` used for the NDVI fallback.

    Returns:
        Boolean ``np.ndarray`` of shape ``(H, W)``.
    """
    crs: CRS           = tile_meta["crs"]
    transform: Affine  = tile_meta["transform"]
    H, W               = tile_meta["shape"]

    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path   = cache_dir / f"{tile_id}_forest2020.tif"
    lock_path  = out_path.with_suffix(".tif.lock")

    with FileLock(str(lock_path)):
        if not out_path.exists():
            try:
                download_hansen_tile(tile_id, crs, transform, (H, W), out_path)
            except RuntimeError as exc:
                LOGGER.warning(
                    "Hansen download failed for %s — %s. Falling back to %s.",
                    tile_id,
                    exc,
                    "NDVI mask" if ndvi_fallback_observations else "all-True mask",
                )
                return _ndvi_or_alltrue_fallback(
                    H, W, crs, transform, ndvi_fallback_observations
                )

    return build_forest_mask(out_path, threshold=threshold)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ndvi_or_alltrue_fallback(
    height: int,
    width: int,
    crs: CRS,
    transform: Affine,
    observations: list | None,
) -> np.ndarray:
    """Compute NDVI-based mask from observations, or return all-True as last resort."""
    if observations is not None:
        try:
            from ..features import compute_forest_mask as _ndvi_mask
            from ..raster_utils import RasterGrid

            grid = RasterGrid(width=width, height=height, crs=crs, transform=transform)
            LOGGER.info("Using NDVI fallback for forest mask (%d observations).", len(observations))
            return _ndvi_mask(observations, grid)
        except Exception as exc:
            LOGGER.warning("NDVI fallback failed (%s); returning all-True mask.", exc)

    LOGGER.warning(
        "Returning all-True forest mask (%d×%d) — Hansen unreachable and no S2 fallback.",
        height,
        width,
    )
    return np.ones((height, width), dtype=bool)
