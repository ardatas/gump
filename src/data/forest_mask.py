"""Hansen GFC treecover2000 + lossyear → forest_2020 mask.

Hansen Global Forest Change v1.8 (GFC-2020-v1.8) provides two layers used here:

* ``treecover2000`` — percent tree cover in the year 2000 (uint8, 0–100, 255=NoData)
* ``lossyear``      — year of first forest loss (uint8, 1–20 for 2001–2020, 0=no loss,
                       255=NoData)

Forest in 2020 is defined as:
    treecover2000 >= threshold  AND  lossyear == 0

Both layers are streamed via ``/vsicurl/`` (GDAL VSI), reprojected to the target
UTM grid in-memory, and the result is cached as a binary uint8 GeoTIFF.

Public API
----------
- :func:`get_hansen_tile_ids`        — map WGS84 bbox → Hansen 10°×10° tile IDs
- :func:`download_hansen_tile`       — download + reproject + compute forest_2020
- :func:`build_forest_mask`          — read cached GeoTIFF → bool mask
- :func:`get_or_build_forest_mask`   — caching + fallback wrapper
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

_GCS_BASE = (
    "https://storage.googleapis.com/earthenginepartners-hansen/GFC-2020-v1.8"
)
_WGS84: CRS = CRS.from_epsg(4326)
# treecover2000 NoData; lossyear NoData (in practice absent but safe to mask)
_NODATA_TC  = 255
_NODATA_LY  = 255
# Output GeoTIFF encoding: 1=forest 2020, 0=non-forest, 255=outside coverage
_OUT_NODATA = 255


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def get_hansen_tile_ids(bbox_lonlat: tuple[float, float, float, float]) -> list[str]:
    """Return Hansen 10°×10° tile IDs that overlap a WGS84 bounding box.

    Hansen tiles are identified by their **NW corner**.  ``10N_080W`` has its
    north edge at 10°N and west edge at 80°W, covering 0°–10°N, 80°–70°W::

        north_edge = ceil(lat / 10) * 10
        west_edge  = floor(lon / 10) * 10

    Args:
        bbox_lonlat: ``(west, south, east, north)`` in WGS84 decimal degrees.

    Returns:
        List of tile ID strings, e.g. ``["00N_080W"]``.
    """
    west, south, east, north = bbox_lonlat

    lat_start = int(math.ceil(south / 10.0)) * 10
    lat_stop  = int(math.ceil(north / 10.0)) * 10
    lon_start = int(math.floor(west  / 10.0)) * 10
    lon_stop  = int(math.floor(east  / 10.0)) * 10

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
    threshold: int = 30,
) -> None:
    """Download, reproject, and compute the 2020 forest mask for one UTM tile.

    Streams ``treecover2000`` and ``lossyear`` via ``/vsicurl/``.
    Forest_2020 = treecover2000 ≥ threshold AND lossyear == 0 (no loss 2001–2020).

    The result is written as a uint8 GeoTIFF: ``1`` = forest 2020,
    ``0`` = non-forest, ``255`` = outside Hansen coverage.

    Args:
        tile_id: Sentinel-2 tile ID (used for logging only).
        target_utm_crs: Destination CRS (UTM).
        target_transform: Destination affine transform.
        target_shape: ``(height, width)`` in pixels.
        out_path: Destination GeoTIFF path.
        threshold: Minimum treecover2000 % to qualify as forest (default 30).

    Raises:
        RuntimeError: If a required Hansen tile is inaccessible.
    """
    H, W = target_shape

    # Derive WGS84 bbox from UTM grid corners
    left   = target_transform.c
    top    = target_transform.f
    right  = left + W * target_transform.a
    bottom = top  + H * target_transform.e   # .e negative for north-up
    bbox_wgs84 = transform_bounds(target_utm_crs, _WGS84, left, bottom, right, top)
    hansen_ids = get_hansen_tile_ids(bbox_wgs84)
    LOGGER.debug("tile=%s  WGS84=%s  Hansen tiles=%s", tile_id, bbox_wgs84, hansen_ids)

    # Accumulate treecover2000 (bilinear %) and lossyear (nearest, categorical)
    tc2000   = np.zeros((H, W), dtype=np.float32)   # treecover %
    lossyr   = np.zeros((H, W), dtype=np.float32)   # 0 = no loss; 1–20 = loss year
    covered  = np.zeros((H, W), dtype=bool)

    for hid in hansen_ids:
        tc_url = f"/vsicurl/{_GCS_BASE}/Hansen_GFC-2020-v1.8_treecover2000_{hid}.tif"
        ly_url = f"/vsicurl/{_GCS_BASE}/Hansen_GFC-2020-v1.8_lossyear_{hid}.tif"

        try:
            tc_buf = np.full((H, W), np.nan, dtype=np.float32)
            with rasterio.open(tc_url) as src:
                _rasterio_reproject(
                    source=rasterio.band(src, 1),
                    destination=tc_buf,
                    src_crs=src.crs,
                    src_transform=src.transform,
                    src_nodata=float(_NODATA_TC),
                    dst_crs=target_utm_crs,
                    dst_transform=target_transform,
                    dst_nodata=np.nan,
                    resampling=Resampling.bilinear,
                )

            ly_buf = np.full((H, W), np.nan, dtype=np.float32)
            with rasterio.open(ly_url) as src:
                _rasterio_reproject(
                    source=rasterio.band(src, 1),
                    destination=ly_buf,
                    src_crs=src.crs,
                    src_transform=src.transform,
                    src_nodata=float(_NODATA_LY),
                    dst_crs=target_utm_crs,
                    dst_transform=target_transform,
                    dst_nodata=np.nan,
                    resampling=Resampling.nearest,   # categorical: no interpolation
                )
        except Exception as exc:
            raise RuntimeError(
                f"Cannot access Hansen tile {hid!r} for S2 tile {tile_id!r}: {exc}"
            ) from exc

        valid = np.isfinite(tc_buf) & np.isfinite(ly_buf)
        # Take per-pixel maximum for compositing (non-overlapping tiles give 0 on
        # the outside; within-tile values are consistent)
        tc2000[valid]  = np.maximum(tc2000[valid],  np.clip(tc_buf[valid], 0, 100))
        lossyr[valid]  = np.maximum(lossyr[valid],  np.clip(ly_buf[valid], 0, 255))
        covered       |= valid

    if not covered.any():
        raise RuntimeError(
            f"No Hansen data projected onto tile {tile_id!r}; tried: {hansen_ids}"
        )

    # forest_2020: forest in 2000 AND no recorded loss through 2020
    forest_2020 = (tc2000 >= threshold) & (lossyr == 0) & covered

    result = np.full((H, W), _OUT_NODATA, dtype=np.uint8)
    result[covered] = forest_2020[covered].astype(np.uint8)  # 0 or 1

    profile: dict[str, Any] = {
        "driver": "GTiff",
        "dtype": "uint8",
        "width": W,
        "height": H,
        "count": 1,
        "crs": target_utm_crs,
        "transform": target_transform,
        "compress": "deflate",
        "nodata": _OUT_NODATA,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(result, 1)

    n_forest = int(forest_2020[covered].sum())
    n_total  = int(covered.sum())
    LOGGER.info(
        "Hansen forest_2020 written: tile=%s  forest_pct=%.1f%%  path=%s",
        tile_id, 100.0 * n_forest / max(n_total, 1), out_path,
    )


def build_forest_mask(
    forest2020_path: Path,
    threshold: int = 30,  # kept for API compatibility; baked in at download time
) -> np.ndarray:
    """Load a precomputed forest_2020 GeoTIFF and return a boolean mask.

    Args:
        forest2020_path: uint8 GeoTIFF written by :func:`download_hansen_tile`
            (``1`` = forest 2020, ``0`` = non-forest, ``255`` = NoData).
        threshold: Unused at load time (already applied during download).
            Kept for API compatibility.

    Returns:
        ``bool`` array of shape ``(H, W)``; ``True`` where forest in 2020.
    """
    with rasterio.open(forest2020_path) as src:
        data = src.read(1)
    return data == 1


def get_or_build_forest_mask(
    tile_id: str,
    tile_meta: dict[str, Any],
    cache_dir: Path,
    threshold: int = 30,
    ndvi_fallback_observations: list | None = None,
) -> np.ndarray:
    """Return the Hansen 2020 forest mask, downloading and caching as needed.

    Uses a ``FileLock`` so parallel ``multiprocessing.Pool`` workers do not race.
    Falls back to an NDVI-based mask (or all-True) when Hansen GFC is unreachable.

    Args:
        tile_id: Tile identifier for cache file naming and logging.
        tile_meta: Dict with ``"crs"`` (CRS), ``"transform"`` (Affine),
            ``"shape"`` (H, W).
        cache_dir: Directory for ``{tile_id}_forest2020.tif``.
        threshold: Treecover % threshold applied during download (default 30).
        ndvi_fallback_observations: Optional S2 observations for NDVI fallback.

    Returns:
        Boolean ``np.ndarray`` of shape ``(H, W)``.
    """
    crs: CRS          = tile_meta["crs"]
    transform: Affine = tile_meta["transform"]
    H, W              = tile_meta["shape"]

    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path  = cache_dir / f"{tile_id}_forest2020.tif"
    lock_path = out_path.with_suffix(".tif.lock")

    with FileLock(str(lock_path)):
        if not out_path.exists():
            try:
                download_hansen_tile(
                    tile_id, crs, transform, (H, W), out_path, threshold=threshold
                )
            except RuntimeError as exc:
                LOGGER.warning(
                    "Hansen download failed for %s — %s. Falling back to %s.",
                    tile_id, exc,
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
    if observations is not None:
        try:
            from ..features import compute_forest_mask as _ndvi_mask
            from ..raster_utils import RasterGrid

            grid = RasterGrid(width=width, height=height, crs=crs, transform=transform)
            LOGGER.info("NDVI fallback: computing forest mask from %d observations.", len(observations))
            return _ndvi_mask(observations, grid)
        except Exception as exc:
            LOGGER.warning("NDVI fallback failed (%s); returning all-True mask.", exc)

    LOGGER.warning(
        "Returning all-True forest mask (%d×%d) — Hansen unreachable, no S2 fallback.",
        height, width,
    )
    return np.ones((height, width), dtype=bool)
