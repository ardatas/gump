from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import shutil

import numpy as np
from numcodecs import Blosc
from rasterio.enums import Resampling
from tqdm import tqdm
import zarr

from .data.forest_mask import get_or_build_forest_mask   # --- NEU ---
from .features import compute_forest_mask, linear_backscatter_to_db
from .index import TileSources, discover_tiles
from .labels import build_label_products
from .raster_utils import grid_from_raster, read_reprojected, to_jsonable_transform


LOGGER = logging.getLogger(__name__)
COMPRESSOR = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
S2_BAND_NAMES = [
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B11",
    "B12",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess challenge tiles into training-ready Zarr caches."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "test", "all"], default="train")
    parser.add_argument("--tile", action="append", dest="tiles", default=[])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--forest-mask-year", type=int, default=2020)
    parser.add_argument("--forest-threshold", type=float, default=0.6)
    parser.add_argument("--forest-min-observations", type=int, default=3)
    # --- NEU: Hansen GFC ---
    parser.add_argument(
        "--hansen-cache-dir",
        type=Path,
        default=None,
        help="Directory for Hansen treecover2020 GeoTIFFs. "
             "Defaults to <cache-root>/forest_masks.",
    )
    parser.add_argument(
        "--hansen-threshold",
        type=int,
        default=30,
        help="Minimum treecover %% to classify as forest (default: 30).",
    )
    parser.add_argument(
        "--no-hansen",
        action="store_true",
        help="Skip Hansen GFC download; fall back to NDVI threshold mask.",
    )
    parser.add_argument("--red-band-index", type=int, default=4)
    parser.add_argument("--nir-band-index", type=int, default=8)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    tiles = discover_tiles(args.data_root, split=args.split)
    if args.tiles:
        selected = set(args.tiles)
        tiles = [tile for tile in tiles if tile.tile_id in selected]
    if args.limit is not None:
        tiles = tiles[: args.limit]

    if not tiles:
        raise SystemExit("No tiles discovered for preprocessing.")

    manifest_entries: list[dict[str, object]] = []
    for tile in tqdm(tiles, desc="Preprocessing tiles"):
        out_path = preprocess_tile(tile, args)
        manifest_entries.append(
            {
                "tile_id": tile.tile_id,
                "split": tile.split,
                "cache_path": str(out_path),
            }
        )

    manifest_path = args.cache_root / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    merged_manifest = merge_manifest_entries(manifest_path, manifest_entries)
    manifest_path.write_text(json.dumps(merged_manifest, indent=2))
    LOGGER.info("Wrote manifest: %s", manifest_path)


def preprocess_tile(tile: TileSources, args: argparse.Namespace) -> Path:
    grid = grid_from_raster(tile.s2[0].path)
    out_dir = args.cache_root / tile.split / f"{tile.tile_id}.zarr"
    if out_dir.exists():
        if not args.overwrite:
            LOGGER.info("Skipping existing cache: %s", out_dir)
            return out_dir
        shutil.rmtree(out_dir)

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(out_dir, mode="w")
    root.attrs.update(
        {
            "tile_id": tile.tile_id,
            "split": tile.split,
            "crs": grid.crs.to_string(),
            "transform": to_jsonable_transform(grid.transform),
            "width": grid.width,
            "height": grid.height,
        }
    )

    write_s2_group(root, tile, grid)
    write_s1_group(root, "ascending", tile.s1_ascending, grid)
    write_s1_group(root, "descending", tile.s1_descending, grid)
    write_aef_group(root, tile, grid)

    # --- GEÄNDERT: Hansen GFC treecover2020 ersetzt NDVI-Approximation ---
    if args.no_hansen:
        forest_mask = compute_forest_mask(
            tile.s2,
            grid,
            year=args.forest_mask_year,
            red_band_index=args.red_band_index,
            nir_band_index=args.nir_band_index,
            ndvi_threshold=args.forest_threshold,
            min_observations=args.forest_min_observations,
        )
    else:
        hansen_cache = args.hansen_cache_dir or (args.cache_root / "forest_masks")
        tile_meta = {
            "crs":       grid.crs,
            "transform": grid.transform,
            "shape":     (grid.height, grid.width),
        }
        forest_mask = get_or_build_forest_mask(
            tile.tile_id,
            tile_meta,
            cache_dir=hansen_cache,
            threshold=args.hansen_threshold,
            ndvi_fallback_observations=tile.s2,   # NDVI fallback if Hansen unreachable
        )
    # --- ENDE GEÄNDERT ---
    root.create_dataset(
        "forest_2020",
        data=forest_mask.astype(bool),
        shape=forest_mask.shape,
        chunks=_spatial_chunks(grid.height, grid.width),
        compressor=COMPRESSOR,
        overwrite=True,
    )

    if tile.split == "train":
        label_products = build_label_products(tile, grid, forest_mask)
        labels = root.create_group("labels")
        _write_array(labels, "consensus", label_products.consensus, np.uint8)
        _write_array(labels, "weight", label_products.weight, np.float32)
        _write_array(labels, "votes", label_products.votes, np.uint8)
        _write_array(labels, "source_mask", label_products.source_mask, np.uint8)
        _write_array(labels, "alert_year", label_products.alert_year, np.uint16)

    LOGGER.info("Preprocessed tile %s -> %s", tile.tile_id, out_dir)
    return out_dir


def write_s2_group(root, tile: TileSources, grid) -> None:
    group = root.create_group("s2")
    years = np.asarray([item.year for item in tile.s2], dtype=np.int16)
    months = np.asarray([item.month for item in tile.s2], dtype=np.int8)
    _write_array(group, "year", years, np.int16)
    _write_array(group, "month", months, np.int8)
    group.attrs["band_names"] = S2_BAND_NAMES

    band_count = len(S2_BAND_NAMES)
    data = group.create_dataset(
        "data",
        shape=(len(tile.s2), band_count, grid.height, grid.width),
        chunks=(1, band_count, *_spatial_chunks(grid.height, grid.width)),
        dtype=np.uint16,
        compressor=COMPRESSOR,
        overwrite=True,
    )
    for index, observation in enumerate(tile.s2):
        raster = read_reprojected(
            observation.path,
            grid,
            resampling=Resampling.bilinear,
            out_dtype=np.float32,
        )
        data[index] = np.nan_to_num(raster, nan=0.0).round().astype(np.uint16)


def write_s1_group(root, name: str, observations, grid) -> None:
    group = root.create_group(f"s1_{name}")
    years = np.asarray([item.year for item in observations], dtype=np.int16)
    months = np.asarray([item.month for item in observations], dtype=np.int8)
    _write_array(group, "year", years, np.int16)
    _write_array(group, "month", months, np.int8)

    data = group.create_dataset(
        "data",
        shape=(len(observations), 1, grid.height, grid.width),
        chunks=(1, 1, *_spatial_chunks(grid.height, grid.width)),
        dtype=np.float16,
        compressor=COMPRESSOR,
        overwrite=True,
    )
    for index, observation in enumerate(observations):
        raster = read_reprojected(
            observation.path,
            grid,
            resampling=Resampling.bilinear,
            out_dtype=np.float32,
        )[0]
        db = linear_backscatter_to_db(raster).astype(np.float16)
        data[index, 0] = db


def write_aef_group(root, tile: TileSources, grid) -> None:
    group = root.create_group("aef")
    years = np.asarray([item.year for item in tile.aef], dtype=np.int16)
    _write_array(group, "year", years, np.int16)

    band_count = 64
    data = group.create_dataset(
        "data",
        shape=(len(tile.aef), band_count, grid.height, grid.width),
        chunks=(1, band_count, *_spatial_chunks(grid.height, grid.width)),
        dtype=np.float16,
        compressor=COMPRESSOR,
        overwrite=True,
    )
    for index, observation in enumerate(tile.aef):
        raster = read_reprojected(
            observation.path,
            grid,
            resampling=Resampling.bilinear,
            out_dtype=np.float32,
        )
        data[index] = raster.astype(np.float16)


def _write_array(group, name: str, data: np.ndarray, dtype) -> None:
    array = np.asarray(data, dtype=dtype)
    group.create_dataset(
        name,
        data=array,
        shape=array.shape,
        chunks=_chunks_for_array(array),
        dtype=dtype,
        compressor=COMPRESSOR,
        overwrite=True,
    )


def _spatial_chunks(height: int, width: int) -> tuple[int, int]:
    return min(256, height), min(256, width)


def _chunks_for_array(array: np.ndarray) -> tuple[int, ...]:
    if array.ndim == 0:
        return ()
    if array.ndim == 1:
        return (min(max(array.shape[0], 1), 512),)
    if array.ndim == 2:
        return _spatial_chunks(array.shape[0], array.shape[1])
    return tuple(max(min(size, 256), 1) for size in array.shape)


def merge_manifest_entries(
    manifest_path: Path,
    new_entries: list[dict[str, object]],
) -> list[dict[str, object]]:
    entries_by_key: dict[tuple[str, str], dict[str, object]] = {}

    if manifest_path.exists():
        existing_entries = json.loads(manifest_path.read_text())
        for entry in existing_entries:
            key = (str(entry["split"]), str(entry["tile_id"]))
            entries_by_key[key] = entry

    for entry in new_entries:
        key = (str(entry["split"]), str(entry["tile_id"]))
        entries_by_key[key] = entry

    return sorted(
        entries_by_key.values(),
        key=lambda entry: (str(entry["split"]), str(entry["tile_id"])),
    )


if __name__ == "__main__":
    main()
