#!/usr/bin/env python3
"""Download African training tiles for deforestation detection.

Acquires data for 4 training sub-tiles in MGRS square 33NTE (Congo Basin,
Cameroon ~3.7-4.4°N 12.4-13.1°E) and arranges it in the directory layout
expected by src/preprocess.py.

Data sources:
  - Sentinel-2 L2A monthly composites: Element84 STAC (free public COGs)
  - Sentinel-1 GRD VV monthly composites: Element84 STAC (free public COGs)
  - AEF embeddings: zero-filled stubs (64-channel; model handles gracefully)
  - Deforestation labels: Hansen GFC-2022 lossyear → synthetic RADD + GLAD-L files

Output:
  {data_root}/sentinel-2/train/{tile}__s2_l2a/{tile}__s2_l2a_{year}_{month}.tif
  {data_root}/sentinel-1/train/{tile}__s1_rtc/{tile}__s1_rtc_{year}_{month}_{orbit}.tif
  {data_root}/aef-embeddings/train/{tile}_{year}.tiff
  {data_root}/labels/train/radd/radd_{tile}_labels.tif
  {data_root}/labels/train/gladl/gladl_{tile}_alert{YY}.tif
  {data_root}/labels/train/gladl/gladl_{tile}_alertDate{YY}.tif
"""

from __future__ import annotations

import argparse
import logging
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pystac_client
import rasterio
import rasterio.transform
import rasterio.warp
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.windows import from_bounds

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Target tile definitions (MGRS 33NTE, Congo Basin, Cameroon)
# Origin derived from S2 scene: left=199980, top=500040, CRS=EPSG:32633
# Sub-tile naming: col from left, row from top (both 0-9, 10km each)
# ---------------------------------------------------------------------------
TE_LEFT = 199980
TE_TOP = 500040
TILE_SIZE_M = 10000

AFRICA_TILES: dict[str, tuple[float, float, float, float]] = {
    "33NTE_2_3": (
        TE_LEFT + 2 * TILE_SIZE_M, TE_TOP - 4 * TILE_SIZE_M,
        TE_LEFT + 3 * TILE_SIZE_M, TE_TOP - 3 * TILE_SIZE_M,
    ),
    "33NTE_7_5": (
        TE_LEFT + 7 * TILE_SIZE_M, TE_TOP - 6 * TILE_SIZE_M,
        TE_LEFT + 8 * TILE_SIZE_M, TE_TOP - 5 * TILE_SIZE_M,
    ),
    "33NTE_1_8": (
        TE_LEFT + 1 * TILE_SIZE_M, TE_TOP - 9 * TILE_SIZE_M,
        TE_LEFT + 2 * TILE_SIZE_M, TE_TOP - 8 * TILE_SIZE_M,
    ),
    "33NTE_8_1": (
        TE_LEFT + 8 * TILE_SIZE_M, TE_TOP - 2 * TILE_SIZE_M,
        TE_LEFT + 9 * TILE_SIZE_M, TE_TOP - 1 * TILE_SIZE_M,
    ),
}

UTM_CRS = CRS.from_epsg(32633)
WGS84_CRS = CRS.from_epsg(4326)

# S2 band asset names on Element84 → output band order matching challenge format
S2_ASSETS = [
    "coastal",   # B01
    "blue",      # B02
    "green",     # B03
    "red",       # B04
    "rededge1",  # B05
    "rededge2",  # B06
    "rededge3",  # B07
    "nir",       # B08
    "nir08",     # B8A
    "nir09",     # B09
    "swir16",    # B11
    "swir22",    # B12
]
SCL_VALID_VALUES = {4, 5, 6, 7, 11}  # vegetation, bare, water, unclassified, snow

# Hansen GFC lossyear URL (10°×10° tile containing ~4°N 12-13°E)
HANSEN_LOSSYEAR_URL = (
    "https://storage.googleapis.com/earthenginepartners-hansen/"
    "GFC-2022-v1.10/Hansen_GFC-2022-v1.10_lossyear_10N_010E.tif"
)
RADD_BASE_DATE = np.datetime64("2014-12-31")

# Label years to generate
LABEL_YEARS = [2021, 2022]  # Hansen 2022 covers through 2022


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def utm_bounds_to_wgs84(xmin: float, ymin: float, xmax: float, ymax: float) -> tuple[float, float, float, float]:
    """Return (lon_min, lat_min, lon_max, lat_max) for a UTM EPSG:32633 box."""
    xs = [xmin, xmax, xmin, xmax]
    ys = [ymin, ymin, ymax, ymax]
    lons, lats = rasterio.warp.transform(UTM_CRS, WGS84_CRS, xs, ys)
    return min(lons), min(lats), max(lons), max(lats)


def make_utm_transform(xmin: float, ymax: float, pixel_size: float = 10.0) -> rasterio.Affine:
    return rasterio.transform.from_origin(xmin, ymax, pixel_size, pixel_size)


def read_window_safe(href: str, xmin: float, ymin: float, xmax: float, ymax: float) -> np.ndarray | None:
    """Read a windowed portion of a COG, return None on failure."""
    try:
        with rasterio.open(href) as src:
            window = from_bounds(xmin, ymin, xmax, ymax, src.transform)
            if window.width <= 0 or window.height <= 0:
                return None
            return src.read(window=window, out_shape=(src.count, 1000, 1000),
                            resampling=Resampling.bilinear).astype(np.float32)
    except Exception as exc:
        LOGGER.warning("Failed to read %s: %s", href[:60], exc)
        return None


# ---------------------------------------------------------------------------
# Sentinel-2 monthly composites
# ---------------------------------------------------------------------------

def download_s2_month(
    client: pystac_client.Client,
    tile_id: str,
    bounds_utm: tuple[float, float, float, float],
    year: int,
    month: int,
    out_dir: Path,
) -> Path | None:
    out_path = out_dir / f"{tile_id}__s2_l2a_{year}_{month}.tif"
    if out_path.exists():
        LOGGER.info("S2 already exists: %s", out_path.name)
        return out_path

    xmin, ymin, xmax, ymax = bounds_utm
    date_range = f"{year}-{month:02d}-01/{year}-{month:02d}-{_last_day(year, month):02d}"

    # Fetch up to 6 scenes; sort by cloud cover so clearest scenes are composited first.
    # No hard cloud-cover pre-filter: the full 100km tile may be >90% cloudy even when
    # our 10km window has clear pixels (tropical rainy season).
    items = sorted(
        client.search(
            collections=["sentinel-2-l2a"],
            query={"grid:code": {"eq": "MGRS-33NTE"}},
            datetime=date_range,
            max_items=6,
        ).items(),
        key=lambda i: i.properties.get("eo:cloud_cover", 100),
    )

    if not items:
        LOGGER.debug("No S2 scenes for %s %d-%02d", tile_id, year, month)
        return None

    LOGGER.info("S2 %d-%02d: %d scenes for %s (cloud: %.0f%%)",
                year, month, len(items),
                tile_id, items[0].properties.get("eo:cloud_cover", -1))

    def _read_scene(item) -> np.ndarray | None:
        """Download one S2 scene and return (12, 1000, 1000) float32 or None."""
        scl_obj = item.assets.get("scl")
        scl_href = scl_obj.href if scl_obj else None
        if scl_href:
            scl_data = read_window_safe(scl_href, xmin, ymin, xmax, ymax)
            valid_mask = (
                np.isin(scl_data[0].astype(int), list(SCL_VALID_VALUES))
                if scl_data is not None
                else np.ones((1000, 1000), dtype=bool)
            )
        else:
            valid_mask = np.ones((1000, 1000), dtype=bool)

        if valid_mask.sum() < 100:  # <100 valid pixels in our 10km window → skip
            return None

        def _read_band(asset_name: str) -> np.ndarray | None:
            asset = item.assets.get(asset_name)
            if asset is None:
                return None
            band = read_window_safe(asset.href, xmin, ymin, xmax, ymax)
            if band is None:
                return None
            arr = band[0].astype(np.float32)
            arr[~valid_mask] = np.nan
            return arr

        with ThreadPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(_read_band, S2_ASSETS))

        if any(r is None for r in results):
            return None
        return np.stack(results, axis=0)  # (12, 1000, 1000)

    band_stacks: list[np.ndarray] = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(_read_scene, item): item for item in items}
        for fut in as_completed(futures):
            result = fut.result()
            if result is not None:
                band_stacks.append(result)

    if not band_stacks:
        LOGGER.warning("No usable S2 data for %s %d-%02d", tile_id, year, month)
        return None

    # Median composite over all valid scenes
    cube = np.stack(band_stacks, axis=0)  # (N, 12, 1000, 1000)
    composite = np.nanmedian(cube, axis=0).astype(np.float32)  # (12, 1000, 1000)
    composite_uint16 = np.nan_to_num(composite, nan=0.0).clip(0, 65535).astype(np.uint16)

    transform = make_utm_transform(xmin, ymax)
    out_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out_path, "w",
        driver="GTiff", crs=UTM_CRS, transform=transform,
        width=1000, height=1000, count=12, dtype="uint16",
        compress="lzw",
    ) as dst:
        dst.write(composite_uint16)

    LOGGER.info("Wrote S2: %s", out_path.name)
    return out_path


# ---------------------------------------------------------------------------
# Sentinel-1 monthly composites
# ---------------------------------------------------------------------------

def download_s1_month(
    client: pystac_client.Client,
    tile_id: str,
    bounds_utm: tuple[float, float, float, float],
    bounds_wgs84: tuple[float, float, float, float],
    year: int,
    month: int,
    orbit: str,
    out_dir: Path,
) -> Path | None:
    """Create a zero-filled S1 stub.

    Public S1 GRD data from Element84 requires requester-pays AWS credentials.
    Until a free S1 RTC source is identified, we write zero stubs so the
    preprocessing pipeline does not fail.  The model degrades gracefully:
    S1-derived features (backscatter delta, availability) will all be zero
    for African tiles.
    """
    out_path = out_dir / f"{tile_id}__s1_rtc_{year}_{month}_{orbit}.tif"
    if out_path.exists():
        return out_path
    return _write_s1_zeros(out_path, bounds_utm[0], bounds_utm[3])


def _write_s1_zeros(out_path: Path, xmin: float, ymax: float) -> Path:
    transform = make_utm_transform(xmin, ymax)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out_path, "w",
        driver="GTiff", crs=UTM_CRS, transform=transform,
        width=1000, height=1000, count=1, dtype="float32",
        compress="lzw",
    ) as dst:
        dst.write(np.zeros((1, 1000, 1000), dtype=np.float32))
    LOGGER.warning("Wrote zero S1: %s", out_path.name)
    return out_path


# ---------------------------------------------------------------------------
# AEF embedding stubs
# ---------------------------------------------------------------------------

def create_aef_stub(tile_id: str, bounds_utm: tuple, year: int, out_dir: Path) -> Path:
    out_path = out_dir / f"{tile_id}_{year}.tiff"
    if out_path.exists():
        return out_path

    xmin, ymin, xmax, ymax = bounds_utm
    lon_min, lat_min, lon_max, lat_max = utm_bounds_to_wgs84(xmin, ymin, xmax, ymax)

    # AEF is in EPSG:4326 at ~0.0001 degree resolution
    pixel_size = 0.0001
    width = math.ceil((lon_max - lon_min) / pixel_size)
    height = math.ceil((lat_max - lat_min) / pixel_size)
    transform = rasterio.transform.from_origin(lon_min, lat_max, pixel_size, pixel_size)

    out_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out_path, "w",
        driver="GTiff", crs=WGS84_CRS, transform=transform,
        width=width, height=height, count=64, dtype="float32",
        compress="lzw",
    ) as dst:
        dst.write(np.zeros((64, height, width), dtype=np.float32))

    LOGGER.info("Wrote AEF stub: %s", out_path.name)
    return out_path


# ---------------------------------------------------------------------------
# Labels from Hansen GFC lossyear
# ---------------------------------------------------------------------------

def build_labels_from_hansen(
    tile_id: str,
    bounds_utm: tuple[float, float, float, float],
    labels_root: Path,
) -> None:
    xmin, ymin, xmax, ymax = bounds_utm
    lon_min, lat_min, lon_max, lat_max = utm_bounds_to_wgs84(xmin, ymin, xmax, ymax)

    # Download lossyear for our tile extent via VSICURL
    vsicurl_url = f"/vsicurl/{HANSEN_LOSSYEAR_URL}"
    LOGGER.info("Reading Hansen lossyear for %s via VSICURL", tile_id)

    try:
        with rasterio.open(vsicurl_url) as src:
            window = from_bounds(lon_min, lat_min, lon_max, lat_max, src.transform)
            if window.width <= 0 or window.height <= 0:
                LOGGER.warning("Invalid window for %s, skipping labels", tile_id)
                return
            # Read at 0.0001° resolution
            out_h = max(1, math.ceil((lat_max - lat_min) / 0.0001))
            out_w = max(1, math.ceil((lon_max - lon_min) / 0.0001))
            lossyear = src.read(
                1, window=window,
                out_shape=(out_h, out_w),
                resampling=Resampling.nearest,
            )
    except Exception as exc:
        LOGGER.error("Failed to read Hansen lossyear: %s", exc)
        return

    transform_wgs84 = rasterio.transform.from_origin(lon_min, lat_max, 0.0001, 0.0001)
    h, w = lossyear.shape

    # --- RADD synthetic labels ---
    # Encode: confidence * 10000 + days_since_2014-12-31
    # lossyear >= 21 (2021+) = deforested after 2020
    radd = np.zeros((h, w), dtype=np.uint16)
    for yy_offset, (year, midday) in enumerate([(2021, 2192), (2022, 2557)]):  # days from 2014-12-31
        mask = lossyear == (year - 2000)
        radd[mask] = 30000 + midday  # confidence=3 (high), mid-year approximation

    radd_dir = labels_root / "radd"
    radd_dir.mkdir(parents=True, exist_ok=True)
    radd_path = radd_dir / f"radd_{tile_id}_labels.tif"
    with rasterio.open(
        radd_path, "w",
        driver="GTiff", crs=WGS84_CRS, transform=transform_wgs84,
        width=w, height=h, count=1, dtype="uint16",
        compress="lzw", nodata=0,
    ) as dst:
        dst.write(radd[np.newaxis])
    LOGGER.info("Wrote RADD labels: %s (positive pixels: %d)", radd_path.name, (radd > 0).sum())

    # --- GLAD-L synthetic labels ---
    gladl_dir = labels_root / "gladl"
    gladl_dir.mkdir(parents=True, exist_ok=True)
    for year in [2021, 2022]:
        yy = year - 2000
        mask = lossyear == yy
        alert = np.where(mask, 3, 0).astype(np.uint8)   # 3 = confirmed
        alertdate = np.where(mask, 182, 0).astype(np.uint16)  # day 182 = July 1

        for suffix, data, dtype in [
            (f"alert{yy:02d}", alert, "uint8"),
            (f"alertDate{yy:02d}", alertdate, "uint16"),
        ]:
            path = gladl_dir / f"gladl_{tile_id}_{suffix}.tif"
            with rasterio.open(
                path, "w",
                driver="GTiff", crs=WGS84_CRS, transform=transform_wgs84,
                width=w, height=h, count=1, dtype=dtype,
                compress="lzw", nodata=0,
            ) as dst:
                dst.write(data[np.newaxis])
        LOGGER.info("Wrote GLAD-L %d labels for %s (positives: %d)", year, tile_id, mask.sum())

    # --- GLAD-S2 synthetic labels (same as GLAD-L but single-file format) ---
    # Only covers 2019+ so valid for our years
    glads2_dir = labels_root / "glads2"
    glads2_dir.mkdir(parents=True, exist_ok=True)

    # GLAD-S2 days since 2019-01-01
    glads2_base = np.datetime64("2019-01-01")
    glads2_alert = np.zeros((h, w), dtype=np.uint8)
    glads2_alertdate = np.zeros((h, w), dtype=np.uint16)

    for year in [2021, 2022]:
        mask = lossyear == (year - 2000)
        midpoint = np.datetime64(f"{year}-07-01")
        days = int((midpoint - glads2_base) / np.timedelta64(1, "D"))
        glads2_alert[mask] = 3
        glads2_alertdate[mask] = days

    for suffix, data, dtype in [
        ("alert", glads2_alert, "uint8"),
        ("alertDate", glads2_alertdate, "uint16"),
    ]:
        path = glads2_dir / f"glads2_{tile_id}_{suffix}.tif"
        with rasterio.open(
            path, "w",
            driver="GTiff", crs=WGS84_CRS, transform=transform_wgs84,
            width=w, height=h, count=1, dtype=dtype,
            compress="lzw", nodata=0,
        ) as dst:
            dst.write(data[np.newaxis])
    LOGGER.info("Wrote GLAD-S2 labels for %s", tile_id)


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def process_tile(
    tile_id: str,
    bounds_utm: tuple,
    data_root: Path,
    years: list[int],
    months: list[int],
    client: pystac_client.Client,
) -> None:
    bounds_wgs84 = utm_bounds_to_wgs84(*bounds_utm)
    LOGGER.info("=== Processing tile %s (%.3f,%.3f -> %.3f,%.3f) ===",
                tile_id, *bounds_wgs84)

    s2_dir = data_root / "sentinel-2" / "train" / f"{tile_id}__s2_l2a"
    s1_dir = data_root / "sentinel-1" / "train" / f"{tile_id}__s1_rtc"
    aef_dir = data_root / "aef-embeddings" / "train"
    labels_root = data_root / "labels" / "train"

    # S2 monthly composites
    for year in years:
        for month in months:
            download_s2_month(client, tile_id, bounds_utm, year, month, s2_dir)

    # S1 monthly stubs (zero-filled; real S1 requires auth)
    for orbit in ["ascending", "descending"]:
        for year in years:
            for month in months:
                download_s1_month(client, tile_id, bounds_utm, bounds_wgs84,
                                  year, month, orbit, s1_dir)

    # AEF stubs
    for year in years:
        create_aef_stub(tile_id, bounds_utm, year, aef_dir)

    # Labels from Hansen GFC
    build_labels_from_hansen(tile_id, bounds_utm, labels_root)


def _last_day(year: int, month: int) -> int:
    import calendar
    return calendar.monthrange(year, month)[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download African training tiles")
    parser.add_argument("--data-root", type=Path, default=Path("/data/africa"),
                        help="Root directory for raw data output")
    parser.add_argument("--years", type=int, nargs="+", default=[2020, 2021, 2022],
                        help="Years to download (default: 2020 2021 2022)")
    parser.add_argument("--months", type=int, nargs="+",
                        default=list(range(1, 13)),
                        help="Months to download (default: 1-12)")
    parser.add_argument("--tiles", nargs="+", default=list(AFRICA_TILES.keys()),
                        choices=list(AFRICA_TILES.keys()),
                        help="Tiles to process")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    LOGGER.info("Target data root: %s", args.data_root)
    LOGGER.info("Tiles: %s", args.tiles)
    LOGGER.info("Years: %s", args.years)
    LOGGER.info("Months: %s", args.months)

    client = pystac_client.Client.open("https://earth-search.aws.element84.com/v1")

    for tile_id in args.tiles:
        if tile_id not in AFRICA_TILES:
            LOGGER.error("Unknown tile: %s", tile_id)
            continue
        process_tile(
            tile_id=tile_id,
            bounds_utm=AFRICA_TILES[tile_id],
            data_root=args.data_root,
            years=args.years,
            months=args.months,
            client=client,
        )

    LOGGER.info("Download complete. Now run:")
    LOGGER.info("  python -m src.preprocess --data-root %s --cache-root /data/africa_cache --split train",
                args.data_root)


if __name__ == "__main__":
    main()
