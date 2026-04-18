from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import rasterio
import torch

from .cache import list_tile_paths, load_tile_data
from .dataset import build_prediction_matrix, scatter_predictions
from .feature_bank import build_tile_features
from .models import load_model_from_checkpoint
from .postprocess import PostprocessConfig, postprocess_probability_map
from .train import RuntimeConfig, load_normalizer, normalize_features, predict_probabilities, resolve_amp_dtype
from submission_utils import raster_to_geojson


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on cached tiles.")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", choices=["test", "train"], default="test")
    parser.add_argument("--eval-batch-size", type=int, default=131072)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp-dtype", choices=["none", "bf16", "fp16"], default="bf16")
    parser.add_argument("--threshold-override", type=float)
    parser.add_argument("--save-probabilities", action="store_true")
    parser.add_argument("--closing-radius", type=int, default=1)
    parser.add_argument("--opening-radius", type=int, default=1)
    parser.add_argument("--min-area-ha", type=float, default=0.5)
    parser.add_argument("--min-hole-area-ha", type=float, default=0.0)
    parser.add_argument("--persistence-threshold", type=float, default=0.0)
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
    threshold = float(args.threshold_override) if args.threshold_override is not None else float(checkpoint["threshold"])
    postprocess_config = PostprocessConfig(
        threshold=threshold,
        closing_radius=max(0, args.closing_radius),
        opening_radius=max(0, args.opening_radius),
        min_area_ha=max(0.0, args.min_area_ha),
        min_hole_area_ha=max(0.0, args.min_hole_area_ha),
        persistence_threshold=max(0.0, args.persistence_threshold),
    )

    raster_dir = args.output_root / "rasters"
    geojson_dir = args.output_root / "geojson"
    probability_dir = args.output_root / "probabilities"
    raster_dir.mkdir(parents=True, exist_ok=True)
    geojson_dir.mkdir(parents=True, exist_ok=True)
    if args.save_probabilities:
        probability_dir.mkdir(parents=True, exist_ok=True)

    for tile_path in list_tile_paths(args.cache_root, args.split):
        tile = load_tile_data(tile_path)
        tile_features = build_tile_features(tile, target_years)
        feature_matrix, prediction_mask = build_prediction_matrix(tile_features)
        feature_matrix = normalize_features(feature_matrix, normalizer)
        probabilities = predict_probabilities(
            model=model,
            features=feature_matrix,
            runtime_config=runtime_config,
            batch_size=args.eval_batch_size,
        )
        probability_map = scatter_predictions(probabilities, prediction_mask, dtype=np.float32)
        prediction = postprocess_probability_map(
            probability_map,
            forest_mask=tile.forest_mask,
            transform=rasterio.Affine(*tile.metadata.transform),
            config=postprocess_config,
            persistence_score=tile_features.persistence_score,
        )

        raster_path = raster_dir / f"{tile.metadata.tile_id}.tif"
        write_raster(tile, prediction.astype(np.uint8), raster_path, dtype="uint8", nodata=0)

        if args.save_probabilities:
            probability_path = probability_dir / f"{tile.metadata.tile_id}.tif"
            write_raster(tile, probability_map.astype(np.float32), probability_path, dtype="float32", nodata=np.nan)

        geojson_path = geojson_dir / f"{tile.metadata.tile_id}.geojson"
        if prediction.any():
            raster_to_geojson(
                raster_path,
                output_path=geojson_path,
                min_area_ha=postprocess_config.min_area_ha,
            )
        else:
            geojson_path.write_text(json.dumps(empty_feature_collection()))
        LOGGER.info("Wrote predictions for %s", tile.metadata.tile_id)

    (args.output_root / "inference_config.json").write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "split": args.split,
                "postprocess": {
                    "threshold": postprocess_config.threshold,
                    "closing_radius": postprocess_config.closing_radius,
                    "opening_radius": postprocess_config.opening_radius,
                    "min_area_ha": postprocess_config.min_area_ha,
                    "min_hole_area_ha": postprocess_config.min_hole_area_ha,
                    "persistence_threshold": postprocess_config.persistence_threshold,
                },
            },
            indent=2,
        )
    )


def write_raster(tile, data: np.ndarray, output_path: Path, *, dtype: str, nodata: float | int | None) -> None:
    transform = rasterio.Affine(*tile.metadata.transform)
    profile = {
        "driver": "GTiff",
        "height": tile.metadata.height,
        "width": tile.metadata.width,
        "count": 1,
        "dtype": dtype,
        "crs": tile.metadata.crs,
        "transform": transform,
        "compress": "deflate",
    }
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(data, 1)


def empty_feature_collection() -> dict[str, object]:
    return {"type": "FeatureCollection", "features": []}


if __name__ == "__main__":
    main()
