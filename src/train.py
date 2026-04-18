from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
from datetime import datetime
import json
import logging
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from .cache import TileMetadata, choose_validation_regions, discover_target_years, list_tile_paths, load_tile_data, load_tile_metadata
from .dataset import PixelSampleDataset, build_eval_matrix, sample_tile_training_data
from .feature_bank import build_tile_features
from .metrics import find_best_threshold, weighted_average_precision
from .models import build_model, load_model_from_checkpoint


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Normalizer:
    mean: np.ndarray
    std: np.ndarray


@dataclass(frozen=True)
class LossConfig:
    name: str
    focal_gamma: float
    positive_class_weight: float | None


@dataclass(frozen=True)
class RuntimeConfig:
    device: torch.device
    amp_dtype: torch.dtype | None
    grad_clip_norm: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a baseline model on preprocessed Zarr caches.")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--eval-batch-size", type=int, default=131072)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--negative-ratio", type=float, default=3.0)
    parser.add_argument("--max-samples-per-tile", type=int, default=500_000)
    parser.add_argument("--val-region", action="append", dest="val_regions", default=[])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--model-name", choices=["mlp", "resmlp"], default="resmlp")
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[512, 512, 256, 256, 128])
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--loss", choices=["bce", "focal"], default="focal")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--positive-class-weight", type=float, default=0.0)
    parser.add_argument("--auto-positive-class-weight", action="store_true")
    parser.add_argument("--amp-dtype", choices=["none", "bf16", "fp16"], default="bf16")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--resume", type=Path)
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

    raw_train_features = np.concatenate(train_feature_chunks, axis=0)
    train_labels = np.concatenate(train_label_chunks, axis=0)
    train_weights = np.concatenate(train_weight_chunks, axis=0)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    amp_dtype = resolve_amp_dtype(args.amp_dtype, device)
    runtime_config = RuntimeConfig(
        device=device,
        amp_dtype=amp_dtype,
        grad_clip_norm=args.grad_clip_norm,
    )

    checkpoint: dict[str, object] | None = None
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)

    positive_class_weight = resolve_positive_class_weight(
        train_labels,
        train_weights,
        user_value=args.positive_class_weight,
        auto_enabled=args.auto_positive_class_weight,
    )
    loss_config = LossConfig(
        name=args.loss,
        focal_gamma=args.focal_gamma,
        positive_class_weight=positive_class_weight,
    )

    if checkpoint is None:
        normalizer = fit_normalizer(raw_train_features)
    else:
        normalizer = load_normalizer(checkpoint)
    train_features = normalize_features(raw_train_features, normalizer)

    dataset = PixelSampleDataset(train_features, train_labels, train_weights)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )

    if checkpoint is None:
        model = build_model(
            args.model_name,
            input_dim=train_features.shape[1],
            hidden_dims=args.hidden_dims,
            dropout=args.dropout,
        ).to(device)
    else:
        model = load_model_from_checkpoint(checkpoint).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs),
        eta_min=args.min_learning_rate,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and amp_dtype == torch.float16)

    run_dir = args.runs_dir / datetime.utcnow().strftime("%Y%m%d-%H%M%S-baseline")
    history: list[dict[str, float]] = []
    start_epoch = 1
    best_f1 = -1.0
    best_threshold = 0.5

    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        if scaler.is_enabled() and checkpoint.get("scaler_state") is not None:
            scaler.load_state_dict(checkpoint["scaler_state"])
        run_dir = args.resume.parent
        history = list(checkpoint.get("history", []))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_f1 = float(checkpoint.get("best_f1", checkpoint.get("threshold_metrics", {}).get("f1", -1.0)))
        best_threshold = float(checkpoint.get("best_threshold", checkpoint.get("threshold", 0.5)))
        LOGGER.info("Resuming from %s at epoch %s", args.resume, start_epoch)

    run_dir.mkdir(parents=True, exist_ok=True)
    write_run_config(
        run_dir=run_dir,
        args=args,
        val_regions=val_regions,
        target_years=target_years,
        train_metadata=train_metadata,
        val_metadata=val_metadata,
        feature_names=feature_names,
        positive_class_weight=positive_class_weight,
    )

    epochs_without_improvement = 0
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            runtime_config=runtime_config,
            loss_config=loss_config,
            scaler=scaler,
        )
        scheduler.step()

        val_result = evaluate_on_tiles(
            model=model,
            metadata=val_metadata,
            target_years=target_years,
            normalizer=normalizer,
            runtime_config=runtime_config,
            batch_size=args.eval_batch_size,
        )
        threshold, threshold_metrics = find_best_threshold(
            val_result["probabilities"],
            val_result["labels"],
            val_result["weights"],
        )
        average_precision = weighted_average_precision(
            val_result["labels"],
            val_result["probabilities"],
            val_result["weights"],
        )
        summary = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_loss": train_loss,
            "val_f1": threshold_metrics["f1"],
            "val_precision": threshold_metrics["precision"],
            "val_recall": threshold_metrics["recall"],
            "val_average_precision": average_precision,
            "threshold": threshold,
        }
        history.append(summary)
        LOGGER.info(
            "Epoch %s lr=%.6f train_loss=%.5f val_f1=%.4f val_ap=%.4f precision=%.4f recall=%.4f threshold=%.2f",
            epoch,
            summary["learning_rate"],
            train_loss,
            threshold_metrics["f1"],
            average_precision,
            threshold_metrics["precision"],
            threshold_metrics["recall"],
            threshold,
        )

        improved = threshold_metrics["f1"] > best_f1
        checkpoint_best_f1 = threshold_metrics["f1"] if improved else best_f1
        checkpoint_best_threshold = threshold if improved else best_threshold

        save_checkpoint(
            path=run_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler if scaler.is_enabled() else None,
            normalizer=normalizer,
            feature_names=feature_names,
            target_years=target_years,
            threshold=threshold,
            val_regions=val_regions,
            summary=summary,
            args=args,
            epoch=epoch,
            best_f1=checkpoint_best_f1,
            best_threshold=checkpoint_best_threshold,
            history=history,
        )

        if improved:
            best_f1 = checkpoint_best_f1
            best_threshold = checkpoint_best_threshold
            epochs_without_improvement = 0
            save_checkpoint(
                path=run_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler if scaler.is_enabled() else None,
                normalizer=normalizer,
                feature_names=feature_names,
                target_years=target_years,
                threshold=threshold,
                val_regions=val_regions,
                summary=summary,
                args=args,
                epoch=epoch,
                best_f1=best_f1,
                best_threshold=best_threshold,
                history=history,
            )
        else:
            epochs_without_improvement += 1

        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            LOGGER.info("Early stopping after %s epochs without improvement.", epochs_without_improvement)
            break

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
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    runtime_config: RuntimeConfig,
    loss_config: LossConfig,
    scaler: torch.cuda.amp.GradScaler,
) -> float:
    model.train()
    total_loss = 0.0
    total_weight = 0.0

    for features, labels, weights in loader:
        features = features.to(runtime_config.device, non_blocking=True)
        labels = labels.to(runtime_config.device, non_blocking=True)
        weights = weights.to(runtime_config.device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(runtime_config):
            logits = model(features)
            loss, weighted_loss_sum, weight_sum = compute_weighted_loss(
                logits=logits,
                labels=labels,
                weights=weights,
                loss_config=loss_config,
            )

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            if runtime_config.grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), runtime_config.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if runtime_config.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), runtime_config.grad_clip_norm)
            optimizer.step()

        total_loss += float(weighted_loss_sum.detach().cpu())
        total_weight += float(weight_sum.detach().cpu())

    return total_loss / max(total_weight, 1e-6)


def compute_weighted_loss(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    loss_config: LossConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pos_weight = None
    if loss_config.positive_class_weight is not None:
        pos_weight = torch.tensor(
            loss_config.positive_class_weight,
            device=logits.device,
            dtype=logits.dtype,
        )

    loss_per_item = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        labels,
        reduction="none",
        pos_weight=pos_weight,
    )

    if loss_config.name == "focal":
        probabilities = torch.sigmoid(logits)
        pt = torch.where(labels > 0.5, probabilities, 1.0 - probabilities)
        focal_factor = (1.0 - pt).pow(loss_config.focal_gamma)
        loss_per_item = loss_per_item * focal_factor

    weighted_loss_sum = (loss_per_item * weights).sum()
    weight_sum = weights.sum().clamp_min(1e-6)
    loss = weighted_loss_sum / weight_sum
    return loss, weighted_loss_sum, weight_sum


def evaluate_on_tiles(
    *,
    model: torch.nn.Module,
    metadata: list[TileMetadata],
    target_years: tuple[int, ...],
    normalizer: Normalizer,
    runtime_config: RuntimeConfig,
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
        tile_probabilities = predict_probabilities(
            model=model,
            features=feature_matrix,
            runtime_config=runtime_config,
            batch_size=batch_size,
        )
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
    *,
    model: torch.nn.Module,
    features: np.ndarray,
    runtime_config: RuntimeConfig,
    batch_size: int,
) -> np.ndarray:
    if features.shape[0] == 0:
        return np.empty((0,), dtype=np.float32)
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            stop = min(start + batch_size, features.shape[0])
            batch = torch.from_numpy(features[start:stop]).to(runtime_config.device, non_blocking=True)
            with autocast_context(runtime_config):
                probabilities = torch.sigmoid(model(batch))
            outputs.append(probabilities.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(outputs, axis=0)


def fit_normalizer(features: np.ndarray) -> Normalizer:
    mean = features.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = features.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    return Normalizer(mean=mean, std=std)


def load_normalizer(checkpoint: dict[str, object]) -> Normalizer:
    return Normalizer(
        mean=np.asarray(checkpoint["normalizer"]["mean"], dtype=np.float32),
        std=np.asarray(checkpoint["normalizer"]["std"], dtype=np.float32),
    )


def normalize_features(features: np.ndarray, normalizer: Normalizer) -> np.ndarray:
    return ((features - normalizer.mean) / normalizer.std).astype(np.float32)


def write_run_config(
    *,
    run_dir: Path,
    args: argparse.Namespace,
    val_regions: tuple[str, ...],
    target_years: tuple[int, ...],
    train_metadata: list[TileMetadata],
    val_metadata: list[TileMetadata],
    feature_names: tuple[str, ...],
    positive_class_weight: float | None,
) -> None:
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "val_regions": list(val_regions),
                "target_years": list(target_years),
                "train_tiles": [item.tile_id for item in train_metadata],
                "val_tiles": [item.tile_id for item in val_metadata],
                "feature_names": list(feature_names),
                "positive_class_weight": positive_class_weight,
            },
            indent=2,
        )
    )


def save_checkpoint(
    *,
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler | None,
    normalizer: Normalizer,
    feature_names: tuple[str, ...],
    target_years: tuple[int, ...],
    threshold: float,
    val_regions: tuple[str, ...],
    summary: dict[str, float],
    args: argparse.Namespace,
    epoch: int,
    best_f1: float,
    best_threshold: float,
    history: list[dict[str, float]],
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": None if scaler is None else scaler.state_dict(),
            "model_name": args.model_name,
            "hidden_dims": [int(value) for value in args.hidden_dims],
            "dropout": float(args.dropout),
            "input_dim": len(feature_names),
            "feature_names": list(feature_names),
            "target_years": list(target_years),
            "threshold": threshold,
            "best_threshold": best_threshold,
            "best_f1": best_f1,
            "val_regions": list(val_regions),
            "normalizer": {
                "mean": normalizer.mean,
                "std": normalizer.std,
            },
            "summary": summary,
            "epoch": epoch,
            "history": history,
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        },
        path,
    )


def resolve_positive_class_weight(
    labels: np.ndarray,
    weights: np.ndarray,
    *,
    user_value: float,
    auto_enabled: bool,
) -> float | None:
    if user_value > 0:
        return float(user_value)
    if not auto_enabled:
        return None

    positive_weight = float(weights[labels > 0.5].sum())
    negative_weight = float(weights[labels <= 0.5].sum())
    if positive_weight <= 0 or negative_weight <= 0:
        return None
    return negative_weight / positive_weight


def resolve_amp_dtype(value: str, device: torch.device) -> torch.dtype | None:
    if device.type != "cuda":
        return None
    if value == "none":
        return None
    if value == "bf16":
        return torch.bfloat16
    if value == "fp16":
        return torch.float16
    raise ValueError(f"Unsupported amp dtype: {value}")


def autocast_context(runtime_config: RuntimeConfig) -> contextlib.AbstractContextManager[None]:
    if runtime_config.device.type != "cuda" or runtime_config.amp_dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=runtime_config.amp_dtype)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
