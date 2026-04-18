"""Compute per-band mean/std normalization stats from the cache.

Reads the per-tile ``.npy`` arrays via mmap and aggregates on-the-fly using
chunked sum / sum-of-squares (NaN-safe). Writes a single ``norm_stats.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

ARRAYS = ("s2", "s1_asc", "s1_desc", "aef")


def _accumulate(arr: np.ndarray, n_bands: int, chunk: int = 8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (sum, sum_sq, count) per band over all T, H, W — NaN-safe.

    Processes along axis=0 in chunks to keep the working set bounded when the
    array is mmap-ed float16 and we upcast to float64 for numerical stability.
    """
    s = np.zeros(n_bands, dtype=np.float64)
    ss = np.zeros(n_bands, dtype=np.float64)
    cnt = np.zeros(n_bands, dtype=np.int64)
    T = arr.shape[0]
    for t0 in range(0, T, chunk):
        block = arr[t0 : t0 + chunk].astype(np.float64)   # (t, B, H, W)
        mask = ~np.isnan(block)
        block0 = np.where(mask, block, 0.0)
        # Sum over (t, H, W) per band
        s += block0.sum(axis=(0, 2, 3))
        ss += (block0 * block0).sum(axis=(0, 2, 3))
        cnt += mask.sum(axis=(0, 2, 3))
    return s, ss, cnt


def _finalize(s: np.ndarray, ss: np.ndarray, cnt: np.ndarray) -> tuple[list[float], list[float]]:
    cnt_safe = np.where(cnt > 0, cnt, 1)
    mean = s / cnt_safe
    var = np.clip(ss / cnt_safe - mean * mean, 0.0, None)
    std = np.sqrt(var)
    mean = np.where(cnt > 0, mean, 0.0)
    std = np.where(cnt > 0, std, 1.0)
    return mean.tolist(), std.tolist()


def compute_stats(cache_root: Path, split: str) -> dict:
    tiles: list[str] = []
    for p in sorted(cache_root.iterdir()):
        meta_p = p / "meta.json"
        if not meta_p.exists():
            continue
        meta = json.loads(meta_p.read_text())
        if meta.get("split") == split:
            tiles.append(p.name)

    if not tiles:
        raise RuntimeError(f"No cached tiles with split={split} found under {cache_root}")

    logger.info("computing stats from %d tiles: %s", len(tiles), tiles)

    totals = {k: None for k in ARRAYS}  # (s, ss, cnt) per array key
    for tile in tiles:
        for key in ARRAYS:
            fp = cache_root / tile / f"{key}.npy"
            if not fp.exists():
                continue
            arr = np.load(fp, mmap_mode="r")
            n_bands = arr.shape[1]
            s, ss, cnt = _accumulate(arr, n_bands)
            if totals[key] is None:
                totals[key] = (s, ss, cnt)
            else:
                S, SS, C = totals[key]
                totals[key] = (S + s, SS + ss, C + cnt)
            logger.info("  %s / %s: T=%d B=%d (cumulative pixels=%d)",
                        tile, key, arr.shape[0], n_bands, int(totals[key][2].sum()))

    out = {"built_from": tiles, "split": split}
    for key, trip in totals.items():
        if trip is None:
            continue
        mean, std = _finalize(*trip)
        out[key] = {"mean": mean, "std": std}
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", type=Path, default=Path("/workspace/cache"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    stats = compute_stats(args.cache_root, args.split)
    out = args.out or (args.cache_root / "norm_stats.json")
    out.write_text(json.dumps(stats, indent=2))
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
