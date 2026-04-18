from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import rasterio
import torch

from .cache import list_tile_paths, load_tile_data
from .feature_bank import build_tile_features
from .models import MLPBaseline
from .train import Normalizer, normalize_features, predict_probabilities
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
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = MLPBaseline(input_dim=int(checkpoint["input_dim"]))
    model.load_state_dict(checkpoint["model_state"])

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    normalizer = Normalizer(
        mean=np.asarray(checkpoint["normalizer"]["mean"], dtype=np.float32),
        std=np.asarray(checkpoint["normalizer"]["std"], dtype=np.float32),
    )
    target_years = tuple(int(year) for year in checkpoint["target_years"])
    threshold = float(checkpoint["threshold"])

    raster_dir = args.output_root / "rasters"
    geojson_dir = args.output_root / "geojson"
    raster_dir.mkdir(parents=True, exist_ok=True)
    geojson_dir.mkdir(parents=True, exist_ok=True)

    for tile_path in list_tile_paths(args.cache_root, args.split):
        tile = load_tile_data(tile_path)
        tile_features = build_tile_features(tile, target_years)
        feature_matrix = normalize_features(
            tile_features.feature_cube.reshape(tile_features.feature_cube.shape[0], -1).T,
            normalizer,
        )
        probabilities = predict_probabilities(model, feature_matrix, device, args.eval_batch_size)
        probability_map = probabilities.reshape(tile.metadata.height, tile.metadata.width)
        prediction = (probability_map >= threshold) & tile.forest_mask

        raster_path = raster_dir / f"{tile.metadata.tile_id}.tif"
        write_prediction_raster(tile, prediction.astype(np.uint8), raster_path)

        geojson_path = geojson_dir / f"{tile.metadata.tile_id}.geojson"
        if prediction.any():
            raster_to_geojson(raster_path, output_path=geojson_path)
        else:
            geojson_path.write_text(json.dumps(empty_feature_collection()))
        LOGGER.info("Wrote predictions for %s", tile.metadata.tile_id)


def write_prediction_raster(tile, data: np.ndarray, output_path: Path) -> None:
    transform = rasterio.Affine(*tile.metadata.transform)
    profile = {
        "driver": "GTiff",
        "height": tile.metadata.height,
        "width": tile.metadata.width,
        "count": 1,
        "dtype": "uint8",
        "crs": tile.metadata.crs,
        "transform": transform,
        "nodata": 0,
        "compress": "deflate",
    }
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(data.astype(np.uint8), 1)


def empty_feature_collection() -> dict[str, object]:
    return {"type": "FeatureCollection", "features": []}


if __name__ == "__main__":
    main()
