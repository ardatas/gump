from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import rasterio
import torch

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
    parser.add_argument("--runs-dir", type=Path, required=True, help="Directory containing run folders (e.g. /workspace/runs/gump)")
    parser.add_argument("--cache-root", type=Path, required=True, help="Path to tile cache directory")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON report file")
    parser.add_argument("--split", choices=["train", "test"], default="test", help="Dataset split to evaluate on")
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

    # Find all best checkpoints
    checkpoints = list(args.runs_dir.rglob("best.pt"))
    if not checkpoints:
        LOGGER.error("No best.pt checkpoints found in %s", args.runs_dir)
        return

    tile_paths = list_tile_paths(args.cache_root, args.split)
    if not tile_paths:
        raise SystemExit(f"No evaluation tiles found for split '{args.split}' in {args.cache_root}.")

    results = []

    for ckpt_path in checkpoints:
        model_name = ckpt_path.parent.name
        LOGGER.info("Evaluating model: %s", model_name)
        
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = load_model_from_checkpoint(payload).to(device)
        model.eval()

        normalizer = load_normalizer(payload)
        target_years = tuple(int(year) for year in payload["target_years"])
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

        labels_list: list[np.ndarray] = []
        predictions_list: list[np.ndarray] = []
        weights_list: list[np.ndarray] = []

        for tile_path in tile_paths:
            tile = load_tile_data(tile_path)
            tile_features = build_tile_features(tile, target_years)
            if tile_features.labels is None or tile_features.weights is None:
                continue

            feature_matrix, prediction_mask = build_prediction_matrix(tile_features)
            
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
            labels_list.append(tile_features.labels[eval_mask].astype(np.float32))
            predictions_list.append(prediction_map[eval_mask].astype(np.float32))
            weights_list.append(tile_features.weights[eval_mask].astype(np.float32))

        if not labels_list:
            LOGGER.warning("No valid evaluation pixels found for %s", model_name)
            continue

        all_labels = np.concatenate(labels_list, axis=0)
        all_predictions = np.concatenate(predictions_list, axis=0)
        all_weights = np.concatenate(weights_list, axis=0)

        metrics = weighted_precision_recall_f1(all_labels, all_predictions, all_weights)

        results.append({
            "model_name": model_name,
            "checkpoint": str(ckpt_path),
            "threshold_used": float(threshold),
            "metrics": metrics,
        })
        
        LOGGER.info(
            "Model %s: u-IoU=%.4f (w-IoU=%.4f), F1=%.4f",
            model_name,
            metrics["unweighted_iou"],
            metrics["iou"],
            metrics["f1"],
        )

    if not results:
        LOGGER.warning("No valid results computed.")
        return

    # Sort results by unweighted IoU descending
    results.sort(key=lambda r: r["metrics"]["unweighted_iou"], reverse=True)
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "split": args.split,
        "results": results
    }, indent=2))
    
    LOGGER.info("Evaluation complete. Results saved to %s", args.output)
    
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
