from __future__ import annotations

import argparse
from itertools import product
import json
import logging
from pathlib import Path

import numpy as np
import rasterio
import torch

from .cache import list_tile_paths, load_tile_data, load_tile_metadata
from .dataset import (
    build_prediction_matrix,
    gather_prediction_sequences,
    scatter_predictions,
)
from .feature_bank import build_monthly_sequence, build_tile_features
from .metrics import weighted_precision_recall_f1
from .models import load_model_from_checkpoint
from .postprocess import PostprocessConfig, postprocess_probability_map
from .train import RuntimeConfig, load_normalizer, normalize_features, predict_probabilities, resolve_amp_dtype


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate inference/postprocessing configs on validation tiles.")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--region", action="append", dest="regions", default=[])
    parser.add_argument("--eval-batch-size", type=int, default=131072)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp-dtype", choices=["none", "bf16", "fp16"], default="bf16")
    parser.add_argument("--threshold", action="append", type=float, dest="thresholds")
    parser.add_argument("--closing-radius", action="append", type=int, dest="closing_radii")
    parser.add_argument("--opening-radius", action="append", type=int, dest="opening_radii")
    parser.add_argument("--min-area-ha", action="append", type=float, dest="min_area_has")
    parser.add_argument("--min-hole-area-ha", action="append", type=float, dest="min_hole_area_has")
    parser.add_argument("--persistence-threshold", action="append", type=float, dest="persistence_thresholds")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = load_model_from_checkpoint(checkpoint)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    runtime_config = RuntimeConfig(
        device=device,
        amp_dtype=resolve_amp_dtype(args.amp_dtype, device),
        grad_clip_norm=0.0,
    )
    model.to(device)
    model.eval()

    normalizer = load_normalizer(checkpoint)
    target_years = tuple(int(year) for year in checkpoint["target_years"])
    raw_months = checkpoint.get("target_months") or []
    target_months = tuple((int(year), int(month)) for year, month in raw_months)
    sequence_channels = int(checkpoint.get("sequence_channels", 0) or 0)
    default_regions = tuple(str(value) for value in checkpoint.get("val_regions", []))
    regions = tuple(args.regions) if args.regions else default_regions

    tile_paths = list_tile_paths(args.cache_root, args.split)
    if regions:
        tile_paths = [path for path in tile_paths if load_tile_metadata(path).region in regions]
    if not tile_paths:
        raise SystemExit("No evaluation tiles found for the requested split/regions.")

    thresholds = args.thresholds or [float(checkpoint["threshold"])]
    closing_radii = args.closing_radii or [1]
    opening_radii = args.opening_radii or [1]
    min_area_has = args.min_area_has or [0.5]
    min_hole_area_has = args.min_hole_area_has or [0.0]
    persistence_thresholds = args.persistence_thresholds or [0.0]

    tile_predictions = [
        precompute_tile_predictions(
            path,
            target_years,
            model,
            normalizer,
            runtime_config,
            args.eval_batch_size,
            target_months=target_months,
            uses_sequence=sequence_channels > 0,
        )
        for path in tile_paths
    ]

    results: list[dict[str, object]] = []
    for threshold, closing_radius, opening_radius, min_area_ha, min_hole_area_ha, persistence_threshold in product(
        thresholds,
        closing_radii,
        opening_radii,
        min_area_has,
        min_hole_area_has,
        persistence_thresholds,
    ):
        config = PostprocessConfig(
            threshold=float(threshold),
            closing_radius=max(0, int(closing_radius)),
            opening_radius=max(0, int(opening_radius)),
            min_area_ha=max(0.0, float(min_area_ha)),
            min_hole_area_ha=max(0.0, float(min_hole_area_ha)),
            persistence_threshold=max(0.0, float(persistence_threshold)),
        )
        metrics = evaluate_config(tile_predictions, config)
        results.append(
            {
                "config": {
                    "threshold": config.threshold,
                    "closing_radius": config.closing_radius,
                    "opening_radius": config.opening_radius,
                    "min_area_ha": config.min_area_ha,
                    "min_hole_area_ha": config.min_hole_area_ha,
                    "persistence_threshold": config.persistence_threshold,
                },
                "metrics": metrics,
            }
        )
        LOGGER.info(
            "cfg threshold=%.2f close=%s open=%s min_area=%.2f persistence=%.2f -> f1=%.4f precision=%.4f recall=%.4f",
            config.threshold,
            config.closing_radius,
            config.opening_radius,
            config.min_area_ha,
            config.persistence_threshold,
            metrics["f1"],
            metrics["precision"],
            metrics["recall"],
        )

    results.sort(key=lambda item: item["metrics"]["f1"], reverse=True)
    payload = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "regions": list(regions),
        "top_results": results[:20],
        "num_evaluated": len(results),
    }
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(payload, indent=2))
    LOGGER.info("Wrote evaluation report to %s", args.report_path)


def precompute_tile_predictions(
    tile_path: Path,
    target_years: tuple[int, ...],
    model: torch.nn.Module,
    normalizer,
    runtime_config: RuntimeConfig,
    batch_size: int,
    *,
    target_months: tuple[tuple[int, int], ...] = (),
    uses_sequence: bool = False,
) -> dict[str, object]:
    tile = load_tile_data(tile_path)
    tile_features = build_tile_features(tile, target_years)
    if tile_features.labels is None or tile_features.weights is None:
        raise ValueError(f"Tile {tile.metadata.tile_id} does not contain labels for evaluation.")

    feature_matrix, prediction_mask = build_prediction_matrix(tile_features)
    raw_feature_matrix = feature_matrix
    feature_matrix = normalize_features(feature_matrix, normalizer)
    sequence_matrix = None
    if uses_sequence and raw_feature_matrix.shape[0] > 0:
        sequence = build_monthly_sequence(tile, target_months)
        sequence_matrix = gather_prediction_sequences(sequence, prediction_mask)
    probabilities = predict_probabilities(
        model=model,
        features=feature_matrix,
        runtime_config=runtime_config,
        batch_size=batch_size,
        sequences=sequence_matrix,
    )
    probability_map = scatter_predictions(probabilities, prediction_mask, dtype=np.float32)
    eval_mask = tile_features.forest_mask & (tile_features.weights > 0)
    return {
        "tile_id": tile.metadata.tile_id,
        "transform": rasterio.Affine(*tile.metadata.transform),
        "forest_mask": tile_features.forest_mask,
        "eval_mask": eval_mask,
        "labels": tile_features.labels,
        "weights": tile_features.weights,
        "probability_map": probability_map,
        "persistence_score": tile_features.persistence_score,
    }


def evaluate_config(tile_predictions: list[dict[str, object]], config: PostprocessConfig) -> dict[str, float]:
    labels: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    weights: list[np.ndarray] = []

    for item in tile_predictions:
        prediction_map = postprocess_probability_map(
            item["probability_map"],
            forest_mask=item["forest_mask"],
            transform=item["transform"],
            config=config,
            persistence_score=item["persistence_score"],
        )
        eval_mask = item["eval_mask"]
        labels.append(item["labels"][eval_mask].astype(np.float32))
        predictions.append(prediction_map[eval_mask].astype(np.float32))
        weights.append(item["weights"][eval_mask].astype(np.float32))

    return weighted_precision_recall_f1(
        np.concatenate(labels, axis=0),
        np.concatenate(predictions, axis=0),
        np.concatenate(weights, axis=0),
    )


if __name__ == "__main__":
    main()
