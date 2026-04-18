from __future__ import annotations

import argparse
from dataclasses import dataclass
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
from .feature_bank import build_monthly_sequence, build_tile_features, compute_year_heuristic_map
from .models import load_model_from_checkpoint
from .postprocess import PostprocessConfig, postprocess_probability_map
from .train import (
    Normalizer,
    RuntimeConfig,
    load_normalizer,
    normalize_features,
    predict_probabilities,
    predict_year_logits,
    resolve_amp_dtype,
)
from submission_utils import raster_to_geojson


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CheckpointBundle:
    path: Path
    model: torch.nn.Module
    normalizer: Normalizer
    target_years: tuple[int, ...]
    target_months: tuple[tuple[int, int], ...]
    sequence_channels: int
    threshold: float
    year_min: int | None
    num_year_classes: int
    feature_names: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on cached tiles.")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", type=Path, required=True, help="May be passed multiple times for ensembling.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", choices=["test", "train"], default="test")
    parser.add_argument("--eval-batch-size", type=int, default=131072)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp-dtype", choices=["none", "bf16", "fp16"], default="bf16")
    parser.add_argument("--threshold-override", type=float)
    parser.add_argument("--threshold-offset", type=float, default=0.0,
                        help="Additive offset applied to the base threshold (negative = more recall, positive = more precision).")
    parser.add_argument("--per-region-thresholds", type=Path, help="Optional JSON mapping region -> threshold.")
    parser.add_argument("--save-probabilities", action="store_true")
    parser.add_argument("--closing-radius", type=int, default=2)
    parser.add_argument("--opening-radius", type=int, default=1)
    parser.add_argument("--min-area-ha", type=float, default=0.5)
    parser.add_argument("--min-hole-area-ha", type=float, default=0.0)
    parser.add_argument("--persistence-threshold", type=float, default=0.0)
    parser.add_argument("--predict-time-step", action="store_true",
                        help="Fill time_step GeoJSON field. Uses the alert_year head if present, otherwise falls back to a per-pixel NDVI/NBR heuristic.")
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
    if any(bundle.target_years != primary_bundle.target_years for bundle in bundles):
        raise SystemExit("All ensembled checkpoints must share target_years.")
    target_years = primary_bundle.target_years
    target_months = primary_bundle.target_months
    needs_sequence = any(bundle.sequence_channels > 0 for bundle in bundles)
    if needs_sequence and not target_months:
        raise SystemExit("Temporal checkpoint missing target_months metadata.")

    per_region_thresholds = _load_per_region_thresholds(args.per_region_thresholds)
    base_threshold = (
        float(args.threshold_override)
        if args.threshold_override is not None
        else float(np.mean([bundle.threshold for bundle in bundles]))
    )
    base_threshold = float(np.clip(base_threshold + args.threshold_offset, 0.0, 1.0))

    raster_dir = args.output_root / "rasters"
    geojson_dir = args.output_root / "geojson"
    probability_dir = args.output_root / "probabilities"
    raster_dir.mkdir(parents=True, exist_ok=True)
    geojson_dir.mkdir(parents=True, exist_ok=True)
    if args.save_probabilities:
        probability_dir.mkdir(parents=True, exist_ok=True)

    year_bundle = next((bundle for bundle in bundles if bundle.num_year_classes > 0), None)
    # Auto-enable time_step: use year head when available, else NDVI/NBR heuristic when ≥2 target years.
    predict_time_step = args.predict_time_step or year_bundle is not None or len(target_years) >= 2
    heuristic_year_fallback = predict_time_step and year_bundle is None and len(target_years) >= 2

    for tile_path in list_tile_paths(args.cache_root, args.split):
        tile = load_tile_data(tile_path)
        metadata = load_tile_metadata(tile_path)
        tile_features = build_tile_features(tile, target_years)
        feature_matrix, prediction_mask = build_prediction_matrix(tile_features)

        sequence_matrix = None
        if needs_sequence and feature_matrix.shape[0] > 0:
            sequence = build_monthly_sequence(tile, target_months)
            sequence_matrix = gather_prediction_sequences(sequence, prediction_mask)

        ensemble_probabilities = np.zeros(feature_matrix.shape[0], dtype=np.float32)
        for bundle in bundles:
            bundle_features = _select_checkpoint_features(
                feature_matrix,
                current_feature_names=tile_features.feature_names,
                checkpoint_feature_names=bundle.feature_names,
                checkpoint_path=bundle.path,
            )
            normalized = normalize_features(bundle_features, bundle.normalizer)
            bundle_sequence = sequence_matrix if bundle.sequence_channels > 0 else None
            tile_probabilities = predict_probabilities(
                model=bundle.model,
                features=normalized,
                runtime_config=runtime_config,
                batch_size=args.eval_batch_size,
                sequences=bundle_sequence,
            )
            ensemble_probabilities += tile_probabilities
        if bundles:
            ensemble_probabilities /= len(bundles)

        probability_map = scatter_predictions(ensemble_probabilities, prediction_mask, dtype=np.float32)
        threshold = per_region_thresholds.get(metadata.region, base_threshold)
        postprocess_config = PostprocessConfig(
            threshold=float(threshold),
            closing_radius=max(0, args.closing_radius),
            opening_radius=max(0, args.opening_radius),
            min_area_ha=max(0.0, args.min_area_ha),
            min_hole_area_ha=max(0.0, args.min_hole_area_ha),
            persistence_threshold=max(0.0, args.persistence_threshold),
        )
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

        time_step_map = None
        if predict_time_step and year_bundle is not None and feature_matrix.shape[0] > 0:
            year_features = _select_checkpoint_features(
                feature_matrix,
                current_feature_names=tile_features.feature_names,
                checkpoint_feature_names=year_bundle.feature_names,
                checkpoint_path=year_bundle.path,
            )
            normalized_year = normalize_features(year_features, year_bundle.normalizer)
            year_logits = predict_year_logits(
                model=year_bundle.model,
                features=normalized_year,
                sequences=sequence_matrix if year_bundle.sequence_channels > 0 else None,
                runtime_config=runtime_config,
                batch_size=args.eval_batch_size,
            )
            if year_logits is not None and year_bundle.year_min is not None:
                predicted_class = year_logits.argmax(axis=-1)
                predicted_year = predicted_class + int(year_bundle.year_min)
                time_step_map = scatter_predictions(
                    predicted_year.astype(np.float32),
                    prediction_mask,
                    fill_value=np.nan,
                    dtype=np.float32,
                )
        elif heuristic_year_fallback:
            time_step_map = compute_year_heuristic_map(tile, target_years)

        geojson_path = geojson_dir / f"{tile.metadata.tile_id}.geojson"
        if prediction.any():
            raster_to_geojson(
                raster_path,
                output_path=geojson_path,
                min_area_ha=postprocess_config.min_area_ha,
            )
            if time_step_map is not None:
                _attach_time_step_to_geojson(
                    geojson_path,
                    time_step_map=time_step_map,
                    prediction_mask=prediction.astype(bool),
                    transform=rasterio.Affine(*tile.metadata.transform),
                    crs=tile.metadata.crs,
                )
        else:
            geojson_path.write_text(json.dumps(empty_feature_collection()))
        LOGGER.info("Wrote predictions for %s", tile.metadata.tile_id)

    (args.output_root / "inference_config.json").write_text(
        json.dumps(
            {
                "checkpoints": [str(bundle.path) for bundle in bundles],
                "split": args.split,
                "ensemble": len(bundles) > 1,
                "per_region_thresholds": per_region_thresholds,
                "base_threshold": base_threshold,
                "postprocess": {
                    "closing_radius": max(0, args.closing_radius),
                    "opening_radius": max(0, args.opening_radius),
                    "min_area_ha": max(0.0, args.min_area_ha),
                    "min_hole_area_ha": max(0.0, args.min_hole_area_ha),
                    "persistence_threshold": max(0.0, args.persistence_threshold),
                },
                "predict_time_step": predict_time_step and (year_bundle is not None or heuristic_year_fallback),
                "time_step_source": (
                    "year_head" if (predict_time_step and year_bundle is not None)
                    else ("ndvi_nbr_heuristic" if heuristic_year_fallback else "none")
                ),
                "threshold_offset": float(args.threshold_offset),
            },
            indent=2,
        )
    )


def load_checkpoint_bundle(path: Path, device: torch.device) -> CheckpointBundle:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = load_model_from_checkpoint(payload).to(device)
    model.eval()
    normalizer = load_normalizer(payload)
    target_years = tuple(int(year) for year in payload["target_years"])
    raw_months = payload.get("target_months") or []
    target_months = tuple((int(year), int(month)) for year, month in raw_months)
    sequence_channels = int(payload.get("sequence_channels", 0) or 0)
    threshold = float(payload.get("best_threshold", payload.get("threshold", 0.5)))
    year_min_value = payload.get("year_min")
    year_min = int(year_min_value) if year_min_value is not None else None
    num_year_classes = int(payload.get("num_year_classes", 0) or 0)
    feature_names = tuple(str(name) for name in payload.get("feature_names", ()))
    return CheckpointBundle(
        path=path,
        model=model,
        normalizer=normalizer,
        target_years=target_years,
        target_months=target_months,
        sequence_channels=sequence_channels,
        threshold=threshold,
        year_min=year_min,
        num_year_classes=num_year_classes,
        feature_names=feature_names,
    )


def _load_per_region_thresholds(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    raw = json.loads(path.read_text())
    return {str(region): float(value) for region, value in raw.items()}


def _select_checkpoint_features(
    feature_matrix: np.ndarray,
    *,
    current_feature_names: tuple[str, ...],
    checkpoint_feature_names: tuple[str, ...],
    checkpoint_path: Path,
) -> np.ndarray:
    if not checkpoint_feature_names:
        return feature_matrix
    if checkpoint_feature_names == current_feature_names:
        return feature_matrix

    index_by_name = {name: index for index, name in enumerate(current_feature_names)}
    missing = [name for name in checkpoint_feature_names if name not in index_by_name]
    if missing:
        missing_preview = ", ".join(missing[:8])
        if len(missing) > 8:
            missing_preview += ", ..."
        raise SystemExit(
            f"Checkpoint {checkpoint_path} expects missing features: {missing_preview}"
        )

    indices = np.asarray(
        [index_by_name[name] for name in checkpoint_feature_names],
        dtype=np.int64,
    )
    return feature_matrix[:, indices]


def _attach_time_step_to_geojson(
    geojson_path: Path,
    *,
    time_step_map: np.ndarray,
    prediction_mask: np.ndarray,
    transform: rasterio.Affine,
    crs: str,
) -> None:
    """Re-open the written GeoJSON and fill the time_step field for each polygon."""
    import geopandas as gpd
    from shapely.geometry import shape

    gdf = gpd.read_file(geojson_path)
    if gdf.empty:
        return
    masked_years = np.where(prediction_mask, time_step_map, np.nan)

    # Project polygons back into the raster CRS to sample years per polygon.
    raster_gdf = gdf.to_crs(crs)
    inv_transform = ~transform
    height, width = prediction_mask.shape

    median_years: list[int | None] = []
    for geometry in raster_gdf.geometry:
        if geometry is None or geometry.is_empty:
            median_years.append(None)
            continue
        rows, cols = _polygon_pixel_indices(geometry, inv_transform, height, width)
        if rows.size == 0:
            median_years.append(None)
            continue
        years = masked_years[rows, cols]
        years = years[np.isfinite(years)]
        if years.size == 0:
            median_years.append(None)
        else:
            median_years.append(int(np.round(np.median(years))))

    gdf["time_step"] = median_years
    gdf.to_file(geojson_path, driver="GeoJSON")


def _polygon_pixel_indices(
    geometry,
    inv_transform: rasterio.Affine,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    minx, miny, maxx, maxy = geometry.bounds
    col_start, row_start = inv_transform * (minx, maxy)
    col_stop, row_stop = inv_transform * (maxx, miny)
    row_low = int(np.clip(np.floor(min(row_start, row_stop)), 0, height - 1))
    row_high = int(np.clip(np.ceil(max(row_start, row_stop)), 0, height - 1))
    col_low = int(np.clip(np.floor(min(col_start, col_stop)), 0, width - 1))
    col_high = int(np.clip(np.ceil(max(col_start, col_stop)), 0, width - 1))
    if row_high < row_low or col_high < col_low:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    rows = np.arange(row_low, row_high + 1)
    cols = np.arange(col_low, col_high + 1)
    grid_rows, grid_cols = np.meshgrid(rows, cols, indexing="ij")
    flat_rows = grid_rows.ravel()
    flat_cols = grid_cols.ravel()
    return flat_rows, flat_cols


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
