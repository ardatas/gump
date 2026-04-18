"""Mine hard-negative pixels from training tiles using an existing checkpoint.

Runs the given checkpoint(s) over training tiles, scores all forest pixels, and
records the top-K pixels per tile that are labelled ``0`` but predicted
positive with the highest confidence. These are the classic false positives the
model gets most wrong — re-injecting them as negatives in a second training run
targets the FPR directly.

Output is a JSON file of the form::

    {
        "checkpoints": [...],
        "threshold": 0.5,
        "tiles": {
            "<tile_id>": [[row, col], ...],
            ...
        }
    }

``train.py`` consumes this via ``--hard-negatives-json``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from .cache import list_tile_paths, load_tile_data, load_tile_metadata
from .dataset import build_eval_matrix, gather_eval_sequences
from .feature_bank import build_monthly_sequence, build_tile_features
from .infer import load_checkpoint_bundle
from .train import RuntimeConfig, normalize_features, predict_probabilities, resolve_amp_dtype


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine hard negatives from train tiles.")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["train"], default="train")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Pixels with prob >= this and label==0 are candidates.")
    parser.add_argument("--top-k-per-tile", type=int, default=200_000,
                        help="Cap of hard negatives per tile (highest-prob first).")
    parser.add_argument("--exclude-region", action="append", dest="exclude_regions", default=[],
                        help="Regions to skip (e.g. the val region used for the checkpoint).")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp-dtype", choices=["none", "bf16", "fp16"], default="bf16")
    parser.add_argument("--eval-batch-size", type=int, default=131072)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    runtime_config = RuntimeConfig(
        device=device,
        amp_dtype=resolve_amp_dtype(args.amp_dtype, device),
        grad_clip_norm=0.0,
    )

    bundles = [load_checkpoint_bundle(path, device) for path in args.checkpoint]
    if not bundles:
        raise SystemExit("At least one --checkpoint is required.")

    primary = bundles[0]
    target_years = primary.target_years
    target_months = primary.target_months
    needs_sequence = any(bundle.sequence_channels > 0 for bundle in bundles)

    exclude_set = set(args.exclude_regions or [])
    tile_paths = list_tile_paths(args.cache_root, args.split)

    tiles_payload: dict[str, list[list[int]]] = {}
    total_found = 0

    for tile_path in tile_paths:
        metadata = load_tile_metadata(tile_path)
        if metadata.region in exclude_set:
            LOGGER.info("Skipping tile %s (region %s excluded)", metadata.tile_id, metadata.region)
            continue

        tile = load_tile_data(tile_path)
        tile_features = build_tile_features(tile, target_years)
        try:
            feature_matrix, labels, weights, eval_mask = build_eval_matrix(tile_features)
        except ValueError as exc:
            LOGGER.warning("Skipping tile %s: %s", metadata.tile_id, exc)
            continue

        if feature_matrix.shape[0] == 0:
            continue

        sequence_matrix = None
        if needs_sequence:
            sequence = build_monthly_sequence(tile, target_months)
            sequence_matrix = gather_eval_sequences(tile_features, sequence)

        ensemble = np.zeros(feature_matrix.shape[0], dtype=np.float32)
        for bundle in bundles:
            normalized = normalize_features(feature_matrix, bundle.normalizer)
            bundle_sequence = sequence_matrix if bundle.sequence_channels > 0 else None
            ensemble += predict_probabilities(
                model=bundle.model,
                features=normalized,
                runtime_config=runtime_config,
                batch_size=args.eval_batch_size,
                sequences=bundle_sequence,
            )
        ensemble /= len(bundles)

        negative_mask = (labels == 0) & (weights > 0)
        candidate_mask = negative_mask & (ensemble >= args.threshold)
        if not candidate_mask.any():
            LOGGER.info("Tile %s: no hard negatives above threshold.", metadata.tile_id)
            continue

        candidate_probs = ensemble[candidate_mask]
        eval_rows, eval_cols = np.nonzero(eval_mask)
        candidate_rows = eval_rows[candidate_mask]
        candidate_cols = eval_cols[candidate_mask]

        if candidate_probs.shape[0] > args.top_k_per_tile:
            top_idx = np.argpartition(-candidate_probs, args.top_k_per_tile)[: args.top_k_per_tile]
            candidate_rows = candidate_rows[top_idx]
            candidate_cols = candidate_cols[top_idx]

        coords = np.stack([candidate_rows, candidate_cols], axis=1).astype(int).tolist()
        tiles_payload[metadata.tile_id] = coords
        total_found += len(coords)
        LOGGER.info("Tile %s: %d hard negatives mined.", metadata.tile_id, len(coords))

    payload = {
        "checkpoints": [str(path) for path in args.checkpoint],
        "threshold": float(args.threshold),
        "top_k_per_tile": int(args.top_k_per_tile),
        "excluded_regions": sorted(exclude_set),
        "tiles": tiles_payload,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    LOGGER.info("Wrote %d hard negatives across %d tiles to %s",
                total_found, len(tiles_payload), args.output)


if __name__ == "__main__":
    main()
