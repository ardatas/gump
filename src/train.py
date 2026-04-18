from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import logging
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from .cache import choose_validation_regions, discover_target_years, list_tile_paths, load_tile_data, load_tile_metadata
from .dataset import PixelSampleDataset, build_eval_matrix, sample_tile_training_data
from .feature_bank import build_tile_features
from .metrics import find_best_threshold, weighted_precision_recall_f1
from .models import MLPBaseline


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Normalizer:
    mean: np.ndarray
    std: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a baseline MLP on preprocessed Zarr caches.")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--eval-batch-size", type=int, default=131072)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--negative-ratio", type=float, default=3.0)
    parser.add_argument("--max-samples-per-tile", type=int, default=500_000)
    parser.add_argument("--val-region", action="append", dest="val_regions", default=[])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    set_seed(args.seed)

    train_tile_paths = list_tile_paths(args.cache_root, "train")
    if not train_tile_paths:
        raise SystemExit("No preprocessed train caches found.")

    metadata = [load_tile_metadata(path) for path in train_tile_paths]
    val_regions = tuple(args.val_regions) if args.val_regions else choose_validation_regions(metadata)
    train_metadata = [item for item in metadata if item.region not in val_regions]
    val_metadata = [item for item in metadata if item.region in val_regions]
    if not train_metadata or not val_metadata:
        raise SystemExit(f"Invalid split. train={len(train_metadata)} val={len(val_metadata)}")

    target_years = discover_target_years(args.cache_root)
    LOGGER.info("Target years: %s", target_years)
    LOGGER.info("Validation regions: %s", val_regions)
    LOGGER.info("Train tiles: %s", [item.tile_id for item in train_metadata])
    LOGGER.info("Val tiles: %s", [item.tile_id for item in val_metadata])

    rng = np.random.default_rng(args.seed)
    train_feature_chunks: list[np.ndarray] = []
    train_label_chunks: list[np.ndarray] = []
    train_weight_chunks: list[np.ndarray] = []
    feature_names: tuple[str, ...] | None = None

    for item in train_metadata:
        LOGGER.info("Building train samples for %s", item.tile_id)
        tile = load_tile_data(item.path)
        tile_features = build_tile_features(tile, target_years)
        feature_names = tile_features.feature_names
        try:
            sampled = sample_tile_training_data(
                item,
                tile_features,
                rng,
                negative_ratio=args.negative_ratio,
                max_samples_per_tile=args.max_samples_per_tile,
            )
        except ValueError as exc:
            LOGGER.warning("Skipping train tile %s: %s", item.tile_id, exc)
            continue
        LOGGER.info(
            "Tile %s sampled %s rows (%s positive candidates, %s negative candidates)",
            item.tile_id,
            sampled.features.shape[0],
            sampled.positive_count,
            sampled.negative_count,
        )
        train_feature_chunks.append(sampled.features)
        train_label_chunks.append(sampled.labels)
        train_weight_chunks.append(sampled.weights)

    if feature_names is None:
        raise SystemExit("No features were built for training.")
    if not train_feature_chunks:
        raise SystemExit("No trainable samples were collected from the cache.")

    train_features = np.concatenate(train_feature_chunks, axis=0)
    train_labels = np.concatenate(train_label_chunks, axis=0)
    train_weights = np.concatenate(train_weight_chunks, axis=0)

    normalizer = fit_normalizer(train_features)
    train_features = normalize_features(train_features, normalizer)

    dataset = PixelSampleDataset(train_features, train_labels, train_weights)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = MLPBaseline(input_dim=train_features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    run_dir = args.runs_dir / datetime.utcnow().strftime("%Y%m%d-%H%M%S-baseline")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "val_regions": list(val_regions),
                "target_years": list(target_years),
                "train_tiles": [item.tile_id for item in train_metadata],
                "val_tiles": [item.tile_id for item in val_metadata],
            },
            indent=2,
        )
    )

    best_f1 = -1.0
    best_threshold = 0.5
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, loader, optimizer, device)
        val_result = evaluate_on_tiles(
            model,
            val_metadata,
            target_years,
            normalizer,
            device,
            args.eval_batch_size,
        )
        threshold, threshold_metrics = find_best_threshold(
            val_result["probabilities"],
            val_result["labels"],
            val_result["weights"],
        )
        summary = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_f1": threshold_metrics["f1"],
            "val_precision": threshold_metrics["precision"],
            "val_recall": threshold_metrics["recall"],
            "threshold": threshold,
        }
        history.append(summary)
        LOGGER.info(
            "Epoch %s train_loss=%.5f val_f1=%.4f precision=%.4f recall=%.4f threshold=%.2f",
            epoch,
            train_loss,
            threshold_metrics["f1"],
            threshold_metrics["precision"],
            threshold_metrics["recall"],
            threshold,
        )

        if threshold_metrics["f1"] > best_f1:
            best_f1 = threshold_metrics["f1"]
            best_threshold = threshold
            save_checkpoint(
                run_dir / "best.pt",
                model,
                normalizer,
                feature_names,
                target_years,
                threshold,
                val_regions,
                summary,
            )

    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "best_f1": best_f1,
                "best_threshold": best_threshold,
                "history": history,
            },
            indent=2,
        )
    )
    LOGGER.info("Saved run to %s", run_dir)


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_weight = 0.0
    for features, labels, weights in loader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        weights = weights.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss_per_item = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            labels,
            reduction="none",
        )
        loss = (loss_per_item * weights).sum() / weights.sum().clamp_min(1e-6)
        loss.backward()
        optimizer.step()

        total_loss += float((loss_per_item * weights).sum().detach().cpu())
        total_weight += float(weights.sum().detach().cpu())
    return total_loss / max(total_weight, 1e-6)


def evaluate_on_tiles(
    model: torch.nn.Module,
    metadata: list,
    target_years: tuple[int, ...],
    normalizer: Normalizer,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    weights: list[np.ndarray] = []

    for item in metadata:
        LOGGER.info("Evaluating tile %s", item.tile_id)
        tile = load_tile_data(item.path)
        tile_features = build_tile_features(tile, target_years)
        try:
            feature_matrix, tile_labels, tile_weights, _ = build_eval_matrix(tile_features)
        except ValueError as exc:
            LOGGER.warning("Skipping eval tile %s: %s", item.tile_id, exc)
            continue
        feature_matrix = normalize_features(feature_matrix, normalizer)
        tile_probabilities = predict_probabilities(model, feature_matrix, device, batch_size)
        probabilities.append(tile_probabilities)
        labels.append(tile_labels)
        weights.append(tile_weights)

    if not probabilities:
        raise ValueError("No validation tiles produced evaluation pixels.")

    return {
        "probabilities": np.concatenate(probabilities, axis=0),
        "labels": np.concatenate(labels, axis=0),
        "weights": np.concatenate(weights, axis=0),
    }


def predict_probabilities(
    model: torch.nn.Module,
    features: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            stop = min(start + batch_size, features.shape[0])
            batch = torch.from_numpy(features[start:stop]).to(device)
            probabilities = torch.sigmoid(model(batch)).cpu().numpy().astype(np.float32)
            outputs.append(probabilities)
    return np.concatenate(outputs, axis=0)


def fit_normalizer(features: np.ndarray) -> Normalizer:
    mean = features.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = features.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    return Normalizer(mean=mean, std=std)


def normalize_features(features: np.ndarray, normalizer: Normalizer) -> np.ndarray:
    return ((features - normalizer.mean) / normalizer.std).astype(np.float32)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    normalizer: Normalizer,
    feature_names: tuple[str, ...],
    target_years: tuple[int, ...],
    threshold: float,
    val_regions: tuple[str, ...],
    summary: dict[str, float],
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "input_dim": len(feature_names),
            "feature_names": list(feature_names),
            "target_years": list(target_years),
            "threshold": threshold,
            "val_regions": list(val_regions),
            "normalizer": {
                "mean": normalizer.mean,
                "std": normalizer.std,
            },
            "summary": summary,
        },
        path,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
