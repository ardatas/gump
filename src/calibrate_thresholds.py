"""Calibrate global and per-region binarisation thresholds on validation tiles.

Loads one or more checkpoints (averaged as an ensemble), runs them over the
held-out validation tiles in the cache, and writes a JSON containing the best
global threshold plus a per-region mapping for downstream inference.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from .cache import list_tile_paths, load_tile_data, load_tile_metadata
from .dataset import (
    build_eval_matrix,
    gather_eval_sequences,
)
from .feature_bank import build_monthly_sequence, build_tile_features
from .infer import load_checkpoint_bundle
from .metrics import find_best_threshold, find_best_threshold_per_region, weighted_average_precision
from .train import RuntimeConfig, normalize_features, predict_probabilities, resolve_amp_dtype


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate per-region thresholds on val tiles.")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--region", action="append", dest="regions", default=[])
    parser.add_argument("--split", choices=["train", "test"], default="train")
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

    primary_bundle = bundles[0]
    target_years = primary_bundle.target_years
    target_months = primary_bundle.target_months
    needs_sequence = any(bundle.sequence_channels > 0 for bundle in bundles)

    val_regions = tuple(args.regions) if args.regions else tuple(
        sorted({region for bundle in bundles for region in _bundle_val_regions(bundle.path)})
    )
    if not val_regions:
        raise SystemExit("No validation regions specified or recoverable from checkpoints.")

    tile_paths = [
        path for path in list_tile_paths(args.cache_root, args.split)
        if load_tile_metadata(path).region in val_regions
    ]
    if not tile_paths:
        raise SystemExit(f"No tiles found in regions {val_regions} for split {args.split}.")

    all_probabilities: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_weights: list[np.ndarray] = []
    all_regions: list[np.ndarray] = []

    for tile_path in tile_paths:
        metadata = load_tile_metadata(tile_path)
        tile = load_tile_data(tile_path)
        tile_features = build_tile_features(tile, target_years)
        try:
            feature_matrix, tile_labels, tile_weights, _ = build_eval_matrix(tile_features)
        except ValueError as exc:
            LOGGER.warning("Skipping tile %s: %s", metadata.tile_id, exc)
            continue
        sequence_matrix = None
        if needs_sequence and feature_matrix.shape[0] > 0:
            sequence = build_monthly_sequence(tile, target_months)
            sequence_matrix = gather_eval_sequences(tile_features, sequence)

        ensemble_probabilities = np.zeros(feature_matrix.shape[0], dtype=np.float32)
        for bundle in bundles:
            normalized = normalize_features(feature_matrix, bundle.normalizer)
            bundle_sequence = sequence_matrix if bundle.sequence_channels > 0 else None
            ensemble_probabilities += predict_probabilities(
                model=bundle.model,
                features=normalized,
                runtime_config=runtime_config,
                batch_size=args.eval_batch_size,
                sequences=bundle_sequence,
            )
        ensemble_probabilities /= len(bundles)

        all_probabilities.append(ensemble_probabilities)
        all_labels.append(tile_labels)
        all_weights.append(tile_weights)
        all_regions.append(np.full(ensemble_probabilities.shape[0], metadata.region))

    if not all_probabilities:
        raise SystemExit("No probabilities produced; check your validation tiles.")

    probabilities = np.concatenate(all_probabilities, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    weights = np.concatenate(all_weights, axis=0)
    regions = np.concatenate(all_regions, axis=0)

    global_threshold, global_metrics = find_best_threshold(probabilities, labels, weights)
    average_precision = weighted_average_precision(labels, probabilities, weights)
    per_region = find_best_threshold_per_region(probabilities, labels, weights, regions)

    payload = {
        "checkpoints": [str(path) for path in args.checkpoint],
        "regions": list(val_regions),
        "global_threshold": global_threshold,
        "global_metrics": global_metrics,
        "global_average_precision": average_precision,
        "per_region": {
            region: {"threshold": threshold, "metrics": metrics}
            for region, (threshold, metrics) in per_region.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    LOGGER.info(
        "Wrote thresholds to %s — global f1=%.4f, average_precision=%.4f",
        args.output,
        global_metrics["f1"],
        average_precision,
    )


def _bundle_val_regions(path: Path) -> tuple[str, ...]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return tuple(str(region) for region in payload.get("val_regions", ()) or ())


if __name__ == "__main__":
    main()
