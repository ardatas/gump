"""Compute and save per-feature normalisation statistics.

Usage
-----
python -m src.compute_feature_norms \
    --cache-root /workspace/cache/gump \
    --output /workspace/cache/feature_norm_stats.json

The output JSON contains ``mean`` and ``std`` arrays (one value per feature
dimension) computed from all valid train pixels.  The training script can load
these instead of recomputing them from scratch every run.

Loading example::

    import json, numpy as np
    from src.train import Normalizer

    stats = json.loads(Path("feature_norm_stats.json").read_text())
    normalizer = Normalizer(
        mean=np.array(stats["mean"], dtype=np.float32),
        std=np.array(stats["std"], dtype=np.float32),
    )
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .cache import discover_target_years, list_tile_paths
from .feature_bank import compute_feature_norm_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute feature-level norm stats over all train tiles.")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None,
                        help="Output JSON path. Defaults to <cache-root>/feature_norm_stats.json")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    output = args.output or args.cache_root / "feature_norm_stats.json"
    tile_paths = list_tile_paths(args.cache_root, "train")
    if not tile_paths:
        raise SystemExit("No train tiles found in cache root.")
    target_years = discover_target_years(args.cache_root)
    compute_feature_norm_stats(tile_paths, target_years, output)


if __name__ == "__main__":
    main()
