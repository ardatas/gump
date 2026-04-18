"""Compute binary 2020 forest mask from S2 NDVI + NDWI median."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def build_forest_mask(
    cache_tile_dir: Path,
    ndvi_threshold: float = 0.6,
    ndwi_threshold: float = 0.0,
) -> np.ndarray:
    """Return bool (H, W): True where 2020 forest (dense veg, not water).

    Uses median NDVI > ndvi_threshold AND median NDWI < ndwi_threshold over
    all 2020 S2 scenes (fallback: 2019-2021 if no 2020 scenes present).
    """
    meta = json.loads((cache_tile_dir / "meta.json").read_text())
    s2 = np.load(cache_tile_dir / "s2.npy", mmap_mode="r")  # (T, 12, H, W) f16

    months = [tuple(x) for x in meta["months_s2"]]
    idx_2020 = [i for i, (y, _) in enumerate(months) if y == 2020]
    if not idx_2020:
        idx_2020 = [i for i, (y, _) in enumerate(months) if y in (2019, 2020, 2021)]
    if not idx_2020:
        # Last-resort: use all available frames
        idx_2020 = list(range(len(months)))

    frames = s2[idx_2020].astype(np.float32)  # (T', 12, H, W)

    # B04=index 3 (Red), B08=index 7 (NIR), B03=index 2 (Green)
    nir = frames[:, 7]
    red = frames[:, 3]
    grn = frames[:, 2]

    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi = (nir - red) / (nir + red + 1e-6)
        ndwi = (grn - nir) / (grn + nir + 1e-6)

    ndvi_med = np.nanmedian(ndvi, axis=0)
    ndwi_med = np.nanmedian(ndwi, axis=0)

    forest = (ndvi_med > ndvi_threshold) & (ndwi_med < ndwi_threshold)
    return forest.astype(bool)
