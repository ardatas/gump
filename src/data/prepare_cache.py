"""Build per-tile mmap-able cache of S2, S1, AEF on the S2 UTM 10 m grid.

Output layout::

    {cache_root}/{tile}/
        meta.json
        s2.npy        (T_s2, 12, H, W)  float16
        s1_asc.npy    (T_s1a, 1, H, W)  float16  — dB, clipped to s1_clip_db
        s1_desc.npy   (T_s1d, 1, H, W)  float16
        aef.npy       (Y, 64, H, W)     float16  — warped EPSG:4326 → S2 UTM
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
from pathlib import Path

import numpy as np
import rasterio
import yaml
from rasterio.enums import Resampling

from src.data.forest_mask import get_or_build_forest_mask   # --- NEU ---
from src.data.io import aef_paths, list_tiles, s1_paths, s2_paths
from src.data.reproject import open_reference_grid, read_on_grid

logger = logging.getLogger(__name__)


def _area(p: Path) -> int:
    with rasterio.open(p) as s:
        return s.height * s.width


# --- NEU ---
def extract_tile_meta(tile_id: str, raw_data_dir: Path, split: str = "train") -> dict:
    """Return ``{crs, transform, shape}`` from the first available S2 TIFF.

    Args:
        tile_id: Tile name, e.g. ``"18NWH_1_4"``.
        raw_data_dir: Root of the raw data tree (contains ``sentinel-2/``).
        split: ``"train"`` or ``"test"`` — tried in that order if the first fails.

    Returns:
        Dict with keys ``"crs"`` (rasterio CRS), ``"transform"`` (Affine),
        ``"shape"`` (tuple[int, int] of H × W).

    Raises:
        RuntimeError: If no S2 files are found for ``tile_id``.
    """
    s2_list = s2_paths(raw_data_dir, tile_id, split)
    if not s2_list:
        other_split = "test" if split == "train" else "train"
        s2_list = s2_paths(raw_data_dir, tile_id, other_split)
    if not s2_list:
        raise RuntimeError(f"No S2 files found for tile {tile_id!r} in {raw_data_dir}")

    # Use the largest frame as reference (same logic as prepare_tile)
    ref_path = max((p for _, p in s2_list), key=lambda p: _area(p))
    transform, crs, H, W = open_reference_grid(ref_path)
    return {"crs": crs, "transform": transform, "shape": (H, W)}


def _write_meta(
    out_dir: Path,
    tile: str,
    split: str,
    crs: rasterio.crs.CRS,
    transform: rasterio.Affine,
    height: int,
    width: int,
    months_s2: list[tuple[int, int]],
    months_s1_asc: list[tuple[int, int]],
    months_s1_desc: list[tuple[int, int]],
    years_aef: list[int],
    clip_s1_db: tuple[float, float],
    has_nan_aef: bool,
) -> None:
    meta = {
        "tile": tile,
        "split": split,
        "crs": crs.to_string() if crs is not None else None,
        "transform": list(transform)[:6],
        "height": height,
        "width": width,
        "months_s2": [list(x) for x in months_s2],
        "months_s1_asc": [list(x) for x in months_s1_asc],
        "months_s1_desc": [list(x) for x in months_s1_desc],
        "years_aef": years_aef,
        "clip_s1_db": list(clip_s1_db),
        "has_nan_aef": bool(has_nan_aef),
        "dtype": "float16",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def _stack_s2(
    s2_list: list[tuple[tuple[int, int], Path]],
    transform: rasterio.Affine,
    crs: rasterio.crs.CRS,
    H: int,
    W: int,
) -> np.ndarray:
    """S2 is already on the target grid — straight read, no warp."""
    out = np.empty((len(s2_list), 12, H, W), dtype=np.float16)
    for i, (_, p) in enumerate(s2_list):
        with rasterio.open(p) as src:
            arr = src.read(out_dtype=np.float32)
        if arr.shape != (12, H, W):
            # Fallback: if an S2 frame has a different shape, warp it onto the grid.
            arr = read_on_grid(p, transform, crs, (H, W), Resampling.bilinear, np.nan)
        out[i] = arr.astype(np.float16)
    return out


def _stack_s1(
    s1_list: list[tuple[tuple[int, int], Path]],
    transform: rasterio.Affine,
    crs: rasterio.crs.CRS,
    H: int,
    W: int,
    clip_db: tuple[float, float],
    epsilon: float,
) -> np.ndarray:
    """Warp each S1 frame onto the grid, convert linear→dB, clip, cast to fp16."""
    out = np.empty((len(s1_list), 1, H, W), dtype=np.float16)
    lo, hi = clip_db
    for i, (_, p) in enumerate(s1_list):
        lin = read_on_grid(p, transform, crs, (H, W), Resampling.bilinear, np.nan)
        # lin: (1, H, W) float32 in linear power units
        with np.errstate(invalid="ignore", divide="ignore"):
            db = 10.0 * np.log10(np.clip(lin, epsilon, None))
        db = np.clip(db, lo, hi)
        out[i] = db.astype(np.float16)
    return out


def _stack_aef(
    aef_list: list[tuple[int, Path]],
    transform: rasterio.Affine,
    crs: rasterio.crs.CRS,
    H: int,
    W: int,
) -> tuple[np.ndarray, bool]:
    """Warp AEF (EPSG:4326, 64 bands) onto the S2 UTM grid."""
    out = np.empty((len(aef_list), 64, H, W), dtype=np.float16)
    has_nan = False
    for i, (_, p) in enumerate(aef_list):
        arr = read_on_grid(p, transform, crs, (H, W), Resampling.bilinear, np.nan)
        if not has_nan and np.isnan(arr).any():
            has_nan = True
        out[i] = arr.astype(np.float16)
    return out, has_nan


def prepare_tile(
    tile: str,
    split: str,
    data_root: Path,
    cache_root: Path,
    cfg: dict,
) -> str:
    """Build the cache directory for one tile. Idempotent by default."""
    out_dir = cache_root / tile
    meta_path = out_dir / "meta.json"
    if meta_path.exists() and not cfg.get("force", False):
        return f"skip {tile}"

    s2_list = s2_paths(data_root, tile, split)
    s1a_list = s1_paths(data_root, tile, split, "ascending")
    s1d_list = s1_paths(data_root, tile, split, "descending")
    aef_list = aef_paths(data_root, tile, split)
    if not s2_list:
        raise RuntimeError(f"No S2 files found for tile={tile} split={split}")

    # Pick the S2 frame with the largest HxW as reference. Some tiles have a
    # truncated first month (e.g. 6x4) alongside full-size later frames.
    ref_path = max((p for _, p in s2_list), key=_area)
    transform, crs, H, W = open_reference_grid(ref_path)

    s2 = _stack_s2(s2_list, transform, crs, H, W)
    s1a = _stack_s1(
        s1a_list, transform, crs, H, W,
        tuple(cfg["s1_clip_db"]), float(cfg["s1_epsilon_linear"]),
    )
    s1d = _stack_s1(
        s1d_list, transform, crs, H, W,
        tuple(cfg["s1_clip_db"]), float(cfg["s1_epsilon_linear"]),
    )
    aef, has_nan_aef = _stack_aef(aef_list, transform, crs, H, W)

    # --- NEU: Hansen GFC forest mask ---
    tile_meta = {"crs": crs, "transform": transform, "shape": (H, W)}
    hansen_cache = Path(cfg.get("hansen_cache_dir") or (cache_root / "forest_masks"))
    forest_mask = get_or_build_forest_mask(
        tile,
        tile_meta,
        cache_dir=hansen_cache,
        threshold=int(cfg.get("hansen_threshold", 30)),
    )
    # --- ENDE NEU ---

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "s2.npy", s2)
    np.save(out_dir / "s1_asc.npy", s1a)
    np.save(out_dir / "s1_desc.npy", s1d)
    np.save(out_dir / "aef.npy", aef)
    np.save(out_dir / "forest_2020.npy", forest_mask)  # --- NEU ---

    _write_meta(
        out_dir, tile, split, crs, transform, H, W,
        months_s2=[x[0] for x in s2_list],
        months_s1_asc=[x[0] for x in s1a_list],
        months_s1_desc=[x[0] for x in s1d_list],
        years_aef=[x[0] for x in aef_list],
        clip_s1_db=tuple(cfg["s1_clip_db"]),
        has_nan_aef=has_nan_aef,
    )
    return f"done {tile} ({split}): s2={s2.shape} s1_asc={s1a.shape} s1_desc={s1d.shape} aef={aef.shape}"


def _worker(args):
    tile, split, data_root, cache_root, cfg = args
    try:
        return prepare_tile(tile, split, Path(data_root), Path(cache_root), cfg)
    except Exception as e:
        return f"FAIL {tile} ({split}): {type(e).__name__}: {e}"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/preprocess.yaml"))
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--cache-root", type=Path, default=None)
    ap.add_argument("--split", choices=("train", "test", "both"), default="train")
    ap.add_argument("--tiles", nargs="*", default=None, help="Tile names; default = all in split")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    # --- NEU ---
    ap.add_argument("--hansen-cache-dir", type=Path, default=None)
    ap.add_argument("--hansen-threshold", type=int, default=30)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text()) if args.config.exists() else {}
    data_root = Path(args.data_root or cfg.get("data_root", "/workspace/data"))
    cache_root = Path(args.cache_root or cfg.get("cache_root", "/workspace/cache"))
    workers = args.workers or int(cfg.get("workers", 8))
    cfg_worker = {
        "force": args.force,
        "s1_clip_db": cfg.get("s1_clip_db", [-30.0, 0.0]),
        "s1_epsilon_linear": cfg.get("s1_epsilon_linear", 1e-6),
        # --- NEU ---
        "hansen_cache_dir": str(args.hansen_cache_dir or cache_root / "forest_masks"),
        "hansen_threshold": args.hansen_threshold,
    }

    splits = ["train", "test"] if args.split == "both" else [args.split]
    jobs: list[tuple[str, str, str, str, dict]] = []
    for split in splits:
        tiles = args.tiles if args.tiles else list_tiles(data_root, split)
        for t in tiles:
            jobs.append((t, split, str(data_root), str(cache_root), cfg_worker))

    logger.info("preparing %d (tile, split) pairs with %d workers", len(jobs), workers)
    if workers <= 1:
        for j in jobs:
            logger.info(_worker(j))
    else:
        with mp.Pool(workers) as pool:
            for msg in pool.imap_unordered(_worker, jobs):
                logger.info(msg)


if __name__ == "__main__":
    main()
