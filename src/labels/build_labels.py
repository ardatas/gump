"""CLI orchestrator: build label artefacts for all tiles.

Usage::

    python -m src.labels.build_labels \\
        --config configs/labels.yaml \\
        --split train \\
        [--tiles 18NWG_6_6 ...] \\
        [--workers 20] \\
        [--force]
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
import yaml

from src.labels.decoders import decode_gladl, decode_glads2, decode_radd
from src.labels.forest_mask import build_forest_mask
from src.labels.harmonize import harmonize
from src.labels.io import gladl_paths, glads2_paths, radd_path
from src.labels.warp import warp_label

logger = logging.getLogger(__name__)


def _load_grid(cache_tile_dir: Path) -> tuple[rasterio.Affine, rasterio.crs.CRS, int, int]:
    meta = json.loads((cache_tile_dir / "meta.json").read_text())
    transform = rasterio.Affine(*meta["transform"])
    crs = rasterio.crs.CRS.from_string(meta["crs"])
    return transform, crs, meta["height"], meta["width"]


def process_tile(
    tile: str,
    split: str,
    cache_root: Path,
    labels_root: Path,
    cfg: dict,
) -> str:
    out_dir = Path(cache_root) / tile
    done_flag = out_dir / "labels_meta.json"
    if done_flag.exists() and not cfg.get("force", False):
        return f"skip {tile}"

    transform, crs, H, W = _load_grid(out_dir)
    dst_shape = (H, W)
    labels_root = Path(labels_root)
    ndvi_thr = float(cfg.get("ndvi_threshold", 0.6))
    ndwi_thr = float(cfg.get("ndwi_threshold", 0.0))
    w_single = float(cfg.get("w_single", 0.3))

    # ── Forest mask (all splits need this) ───────────────────────────────────
    forest_2020 = build_forest_mask(out_dir, ndvi_threshold=ndvi_thr, ndwi_threshold=ndwi_thr)
    np.save(out_dir / "forest_2020.npy", forest_2020)

    if split != "train":
        # Test tiles: only forest mask, no labels
        meta_out = {"tile": tile, "split": split, "built_at": datetime.utcnow().isoformat(),
                    "forest_only": True}
        done_flag.write_text(json.dumps(meta_out, indent=2))
        return f"done {tile} (test, forest-mask only)"

    # ── Decode label sources ──────────────────────────────────────────────────
    sources: list[tuple[np.ndarray, np.ndarray]] = []
    sources_used: list[str] = []

    # RADD
    rp = radd_path(labels_root, tile)
    if rp:
        try:
            raw = warp_label(rp, transform, crs, dst_shape)
            is_defo, ay = decode_radd(raw)
            sources.append((is_defo, ay))
            sources_used.append("radd")
        except Exception as e:
            logger.warning("RADD failed for %s: %s", tile, e)
    else:
        logger.info("RADD not found for %s", tile)

    # GLAD-L
    gl_list = gladl_paths(labels_root, tile)
    if gl_list:
        try:
            gladl_defo = np.zeros((H, W), dtype=bool)
            gladl_year = np.full((H, W), 9999, dtype=np.int32)
            for yy, alert_p, date_p in gl_list:
                raw_a = warp_label(alert_p, transform, crs, dst_shape)
                raw_d = warp_label(date_p, transform, crs, dst_shape)
                is_d, ay = decode_gladl(raw_a, raw_d, yy)
                gladl_defo |= is_d
                # Keep earliest year
                gladl_year = np.where(is_d & (ay.astype(np.int32) < gladl_year),
                                      ay.astype(np.int32), gladl_year)
            gladl_year_out = np.where(gladl_year < 9999, gladl_year, 0).astype(np.uint16)
            sources.append((gladl_defo, gladl_year_out))
            sources_used.append(f"gladl_yy={[x[0] for x in gl_list]}")
        except Exception as e:
            logger.warning("GLAD-L failed for %s: %s", tile, e)
    else:
        logger.info("GLAD-L not found for %s", tile)

    # GLAD-S2
    gs = glads2_paths(labels_root, tile)
    if gs:
        try:
            alert_p, date_p = gs
            raw_a = warp_label(alert_p, transform, crs, dst_shape)
            raw_d = warp_label(date_p, transform, crs, dst_shape)
            is_d, ay = decode_glads2(raw_a, raw_d)
            sources.append((is_d, ay))
            sources_used.append("glads2")
        except Exception as e:
            logger.warning("GLAD-S2 failed for %s: %s", tile, e)
    else:
        logger.info("GLAD-S2 not found for %s", tile)

    # ── Harmonize ────────────────────────────────────────────────────────────
    consensus, weight, alert_year = harmonize(sources, forest_2020, w_single=w_single)

    np.save(out_dir / "label_consensus.npy", consensus)
    np.save(out_dir / "label_weight.npy", weight)
    np.save(out_dir / "alert_year.npy", alert_year)

    # labels_meta.json written last (idempotency sentinel)
    meta_out = {
        "tile": tile,
        "split": split,
        "built_at": datetime.utcnow().isoformat(),
        "sources_used": sources_used,
        "n_sources": len(sources),
        "ndvi_threshold": ndvi_thr,
        "ndwi_threshold": ndwi_thr,
        "w_single": w_single,
        "consensus_mean": float(consensus.mean()),
        "forest_fraction": float(forest_2020.mean()),
    }
    done_flag.write_text(json.dumps(meta_out, indent=2))
    return (f"done {tile}: sources={sources_used} "
            f"consensus={meta_out['consensus_mean']:.4f} "
            f"forest={meta_out['forest_fraction']:.3f}")


def _worker(args):
    tile, split, cache_root, labels_root, cfg = args
    try:
        return process_tile(tile, split, Path(cache_root), Path(labels_root), cfg)
    except Exception as e:
        return f"FAIL {tile}: {type(e).__name__}: {e}"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/labels.yaml"))
    ap.add_argument("--cache-root", type=Path, default=None)
    ap.add_argument("--labels-root", type=Path, default=None)
    ap.add_argument("--split", choices=("train", "test", "both"), default="train")
    ap.add_argument("--tiles", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text()) if args.config.exists() else {}
    cache_root = Path(args.cache_root or cfg.get("cache_root", "/workspace/cache"))
    labels_root = Path(args.labels_root or cfg.get("labels_root", "/workspace/data/labels"))
    workers = args.workers or int(cfg.get("workers", 8))
    cfg_worker = {
        "force": args.force,
        "ndvi_threshold": cfg.get("ndvi_threshold", 0.6),
        "ndwi_threshold": cfg.get("ndwi_threshold", 0.0),
        "w_single": cfg.get("w_single", 0.3),
    }

    splits = ["train", "test"] if args.split == "both" else [args.split]
    jobs = []
    for split in splits:
        split_labels = labels_root / split
        if args.tiles:
            tiles = args.tiles
        else:
            # Discover from cache (all dirs with meta.json matching split)
            import json as _json
            tiles = []
            for d in sorted(cache_root.iterdir()):
                meta_p = d / "meta.json"
                if meta_p.exists():
                    m = _json.loads(meta_p.read_text())
                    if m.get("split") == split:
                        tiles.append(d.name)
        for t in tiles:
            jobs.append((t, split, str(cache_root), str(split_labels), cfg_worker))

    logger.info("processing %d tiles with %d workers", len(jobs), workers)
    results = []
    if workers <= 1:
        for j in jobs:
            r = _worker(j)
            logger.info(r)
            results.append(r)
    else:
        with mp.Pool(workers) as pool:
            for r in pool.imap_unordered(_worker, jobs):
                logger.info(r)
                results.append(r)

    failed = [r for r in results if r.startswith("FAIL")]
    logger.info("done. %d ok, %d failed", len(results) - len(failed), len(failed))
    for f in failed:
        logger.error(f)


if __name__ == "__main__":
    main()
