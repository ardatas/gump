"""Patch the forest_2020 layer in existing Zarr caches with Hansen GFC treecover2020.

Does NOT re-warp S2/S1/AEF data.  Reads the grid metadata already in the Zarr,
downloads/reprojects the Hansen mask, and overwrites forest_2020 in-place.
Also deletes any stale .features.zarr sibling caches (they contain
persistence_mask/weights computed from the old NDVI-based mask).

Usage
-----
python -m src.patch_forest_masks \\
    --cache-root /workspace/cache/gump \\
    --split both \\
    --hansen-cache-dir /workspace/cache/forest_masks \\
    --threshold 30
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import numpy as np
import zarr
from rasterio.crs import CRS
from rasterio.transform import Affine

from .cache import list_tile_paths
from .data.forest_mask import get_or_build_forest_mask

LOGGER = logging.getLogger(__name__)


def patch_tile(
    tile_path: Path,
    hansen_cache_dir: Path,
    threshold: int,
) -> str:
    # --- Read grid metadata from the existing Zarr store ---
    root = zarr.open_group(str(tile_path), mode="r")
    tile_id   = str(root.attrs["tile_id"])
    crs       = CRS.from_string(str(root.attrs["crs"]))
    transform = Affine(*root.attrs["transform"])
    H         = int(root.attrs["height"])
    W         = int(root.attrs["width"])

    tile_meta = {"crs": crs, "transform": transform, "shape": (H, W)}

    # --- Download / read from cache ---
    forest_mask = get_or_build_forest_mask(
        tile_id,
        tile_meta,
        cache_dir=hansen_cache_dir,
        threshold=threshold,
    )

    # --- Overwrite forest_2020 in the Zarr store ---
    root_rw = zarr.open_group(str(tile_path), mode="r+")
    root_rw["forest_2020"][:] = forest_mask.astype(bool)

    # --- Invalidate stale feature cache (computed with old NDVI mask) ---
    feature_cache = tile_path.parent / (tile_path.stem + ".features.zarr")
    if feature_cache.exists():
        shutil.rmtree(feature_cache)
        LOGGER.info("Deleted stale feature cache: %s", feature_cache)

    pos_pct = 100.0 * forest_mask.mean()
    return f"patched {tile_id}: forest_coverage={pos_pct:.1f}%"


def main() -> None:
    ap = argparse.ArgumentParser(description="Patch forest_2020 in Zarr caches with Hansen GFC.")
    ap.add_argument("--cache-root", type=Path, required=True)
    ap.add_argument("--split", choices=["train", "test", "both"], default="both")
    ap.add_argument("--hansen-cache-dir", type=Path, default=None)
    ap.add_argument("--threshold", type=int, default=30)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    hansen_cache = args.hansen_cache_dir or (args.cache_root / "forest_masks")
    splits = ["train", "test"] if args.split == "both" else [args.split]

    for split in splits:
        for tile_path in list_tile_paths(args.cache_root, split):
            try:
                msg = patch_tile(tile_path, hansen_cache, args.threshold)
                LOGGER.info(msg)
            except Exception as exc:
                LOGGER.error("FAIL %s: %s", tile_path.name, exc)


if __name__ == "__main__":
    main()
