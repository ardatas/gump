from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import yaml

import numpy as np
import rasterio
import torch
from tqdm import tqdm

from src.cache import list_tile_paths, load_tile_data, load_tile_metadata
from src.dataset import build_prediction_matrix, gather_prediction_sequences, scatter_predictions
from src.feature_bank import build_monthly_sequence, build_tile_features
from src.metrics import weighted_precision_recall_f1
from src.models import load_model_from_checkpoint
from src.postprocess import PostprocessConfig, postprocess_probability_map
from src.train import RuntimeConfig, load_normalizer, normalize_features, predict_probabilities, resolve_amp_dtype


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate multiple models from a runs directory on the test set using IoU.")
    parser.add_argument("--runs-dir", type=Path, help="Override directory containing run folders")
    parser.add_argument("--train-config", type=Path, default=Path("configs/training.yaml"), help="Path to training config to read runs_dir from")
    parser.add_argument("--config", type=Path, default=Path("configs/preprocess.yaml"), help="Path to preprocessing config to read cache_root from")
    parser.add_argument("--cache-root", type=Path, help="Override cache root path")
    parser.add_argument("--output-dir", type=Path, default=Path("/workspace/evaluation"), help="Directory to save evaluation results")
    parser.add_argument("--split", choices=["train", "test"], default="train", help="Dataset split to evaluate on (use 'train' to evaluate on the model's validation regions since 'test' has no labels)")
    parser.add_argument("--eval-batch-size", type=int, default=131072)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp-dtype", choices=["none", "bf16", "fp16"], default="bf16")
    parser.add_argument("--threshold-override", type=float, help="Optional global threshold override, otherwise reads from checkpoint")
    parser.add_argument("--closing-radius", type=int, default=1)
    parser.add_argument("--opening-radius", type=int, default=1)
    parser.add_argument("--min-area-ha", type=float, default=0.5)
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

    if args.cache_root:
        cache_root = args.cache_root
    else:
        if not args.config.exists():
            raise SystemExit(f"Configuration file {args.config} not found and --cache-root not provided.")
        with open(args.config, "r") as f:
            config_data = yaml.safe_load(f)
            if "cache_root" not in config_data:
                raise SystemExit(f"'cache_root' key missing in {args.config}")
            cache_root = Path(config_data["cache_root"])
            LOGGER.info("Loaded cache_root=%s from %s", cache_root, args.config)

    if args.runs_dir:
        runs_dir = args.runs_dir
    else:
        if not args.train_config.exists():
            raise SystemExit(f"Configuration file {args.train_config} not found and --runs-dir not provided.")
        with open(args.train_config, "r") as f:
            train_config_data = yaml.safe_load(f)
            if "runs_dir" not in train_config_data:
                raise SystemExit(f"'runs_dir' key missing in {args.train_config}")
            runs_dir = Path(train_config_data["runs_dir"])
            LOGGER.info("Loaded runs_dir=%s from %s", runs_dir, args.train_config)

    # Find all best checkpoints
    checkpoints = list(runs_dir.rglob("best.pt"))
    if not checkpoints:
        LOGGER.error("No best.pt checkpoints found in %s", runs_dir)
        return

    # Check if 'gump' subdirectory exists in cache, as the pipeline creates them there
    if (cache_root / "gump").exists():
        actual_cache_root = cache_root / "gump"
    else:
        actual_cache_root = cache_root

    tile_paths = list_tile_paths(actual_cache_root, args.split)

    if not tile_paths:
        raise SystemExit(f"No evaluation tiles found for split '{args.split}' in {actual_cache_root}.")

    # Load all models into memory first
    models = []
    model_names = []
    payloads = []
    
    LOGGER.info("Loading %d checkpoints into memory...", len(checkpoints))
    for ckpt_path in checkpoints:
        model_name = ckpt_path.parent.name
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = load_model_from_checkpoint(payload).to(device)
        model.eval()
        
        models.append(model)
        model_names.append(model_name)
        payloads.append(payload)

    # Prepare dictionaries to accumulate results per model
    all_labels = {name: [] for name in model_names}
    all_predictions = {name: [] for name in model_names}
    all_weights = {name: [] for name in model_names}

    # Gather the union of all validation regions across all models
    all_val_regions = set()
    for payload in payloads:
        all_val_regions.update(str(r) for r in payload.get("val_regions", []))
    
    if args.split == "train" and all_val_regions:
        LOGGER.info("Filtering tile paths to only load regions used for validation: %s", sorted(list(all_val_regions)))
        filtered_paths = []
        for path in tile_paths:
            try:
                meta = load_tile_metadata(path)
                if meta.region in all_val_regions:
                    filtered_paths.append(path)
            except Exception:
                pass
        tile_paths = filtered_paths

    LOGGER.info("Evaluating models on %d test tiles...", len(tile_paths))

    for tile_path in tqdm(tile_paths, desc="Processing Tiles", position=0):
        tile = load_tile_data(tile_path)
        
        # We need to build features for the superset of target_years needed by any model.
        # For simplicity, if all models use the same years, we build it once.
        # Otherwise, we track built features by their target_years tuple.
        built_features_cache = {}
        
        for i, model in enumerate(models):
            model_name = model_names[i]
            payload = payloads[i]
            
            val_regions = tuple(str(value) for value in payload.get("val_regions", []))
            
            # If we're evaluating on the training split, we only want to test on the validation regions 
            # to get a fair score.
            if args.split == "train" and val_regions and tile.metadata.region not in val_regions:
                continue
                
            target_years = tuple(int(year) for year in payload["target_years"])
            
            # Build or get features for these specific target years
            # We must use use_cache=False, because the on-disk feature cube might have been built
            # for a different set of target_years (e.g. 3 years = 682 feats) and loading it for a 1-year model crashes.
            if target_years not in built_features_cache:
                built_features_cache[target_years] = build_tile_features(tile, target_years, use_cache=False)
            
            tile_features = built_features_cache[target_years]
            
            if tile_features.labels is None or tile_features.weights is None:
                continue

            normalizer = load_normalizer(payload)
            raw_months = payload.get("target_months") or []
            target_months = tuple((int(year), int(month)) for year, month in raw_months)
            sequence_channels = int(payload.get("sequence_channels", 0) or 0)
            
            default_threshold = float(payload.get("best_threshold", payload.get("threshold", 0.5)))
            threshold = args.threshold_override if args.threshold_override is not None else default_threshold
            
            postprocess_config = PostprocessConfig(
                threshold=threshold,
                closing_radius=max(0, args.closing_radius),
                opening_radius=max(0, args.opening_radius),
                min_area_ha=max(0.0, args.min_area_ha),
                min_hole_area_ha=0.0,
                persistence_threshold=0.0,
            )

            # Extract the specific feature names this model expects
            expected_feature_names = tuple(str(name) for name in payload.get("feature_names", ()))

            feature_matrix, prediction_mask = build_prediction_matrix(tile_features)
            
            # Sub-select only the features this model expects, in the correct order
            if expected_feature_names and hasattr(tile_features, "feature_names") and tile_features.feature_names:
                # Find indices of expected features in the generated feature cube
                feature_indices = []
                missing_features = []
                for expected_name in expected_feature_names:
                    try:
                        idx = tile_features.feature_names.index(expected_name)
                        feature_indices.append(idx)
                    except ValueError:
                        missing_features.append(expected_name)
                
                if missing_features:
                    LOGGER.error("Model %s expects features not present in tile: %s", model_name, missing_features[:5])
                    continue
                
                feature_matrix = feature_matrix[:, feature_indices]
            elif feature_matrix.shape[1] != normalizer.mean.shape[0]:
                LOGGER.error(
                    "Model %s normalizer expects %d features, but generated matrix has %d and no feature_names mapping was found.", 
                    model_name, normalizer.mean.shape[0], feature_matrix.shape[1]
                )
                continue
            
            sequence_matrix = None
            if sequence_channels > 0 and feature_matrix.shape[0] > 0:
                sequence = build_monthly_sequence(tile, target_months)
                sequence_matrix = gather_prediction_sequences(sequence, prediction_mask)

            normalized_features = normalize_features(feature_matrix, normalizer)
            probabilities = predict_probabilities(
                model=model,
                features=normalized_features,
                runtime_config=runtime_config,
                batch_size=args.eval_batch_size,
                sequences=sequence_matrix,
            )

            probability_map = scatter_predictions(probabilities, prediction_mask, dtype=np.float32)
            prediction_map = postprocess_probability_map(
                probability_map,
                forest_mask=tile.forest_mask,
                transform=rasterio.Affine(*tile.metadata.transform),
                config=postprocess_config,
                persistence_score=tile_features.persistence_score,
            )

            eval_mask = tile_features.forest_mask & (tile_features.weights > 0)
            all_labels[model_name].append(tile_features.labels[eval_mask].astype(np.float32))
            all_predictions[model_name].append(prediction_map[eval_mask].astype(np.float32))
            all_weights[model_name].append(tile_features.weights[eval_mask].astype(np.float32))
            
    results = []
    
    for i, model_name in enumerate(model_names):
        payload = payloads[i]
        threshold = args.threshold_override if args.threshold_override is not None else float(payload.get("best_threshold", payload.get("threshold", 0.5)))
        
        if not all_labels[model_name]:
            LOGGER.warning("No valid evaluation pixels found for %s", model_name)
            continue
            
        labels_cat = np.concatenate(all_labels[model_name], axis=0)
        predictions_cat = np.concatenate(all_predictions[model_name], axis=0)
        weights_cat = np.concatenate(all_weights[model_name], axis=0)

        metrics = weighted_precision_recall_f1(labels_cat, predictions_cat, weights_cat)

        results.append({
            "model_name": model_name,
            "checkpoint": str(checkpoints[i]),
            "threshold_used": float(threshold),
            "metrics": metrics,
        })
        
    tqdm.write("\nEvaluation Summary:")
    for r in results:
        m = r["metrics"]
        tqdm.write(f"Model {r['model_name']}: u-IoU={m['unweighted_iou']:.4f} (w-IoU={m['iou']:.4f}), F1={m['f1']:.4f}")

    if not results:
        LOGGER.warning("No valid results computed.")
        return

    # Sort results by unweighted IoU descending
    results.sort(key=lambda r: r["metrics"]["unweighted_iou"], reverse=True)
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "test_results.json"
    report_path.write_text(json.dumps({
        "split": args.split,
        "results": results
    }, indent=2))
    
    LOGGER.info("Evaluation complete. Results saved to %s", report_path)
    
    print("\n" + "=" * 85)
    print(f"{'Model':<30} | {'u-IoU':<8} | {'w-IoU':<8} | {'F1':<8} | {'Prec':<8} | {'Rec':<8}")
    print("-" * 85)
    for r in results:
        m = r["metrics"]
        print(
            f"{r['model_name']:<30} | {m['unweighted_iou']:.4f}   | {m['iou']:.4f}   | "
            f"{m['f1']:.4f}   | {m['precision']:.4f}   | {m['recall']:.4f}"
        )
    print("=" * 85 + "\n")


if __name__ == "__main__":
    main()
