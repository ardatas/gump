from __future__ import annotations

import argparse
import contextlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime
import hashlib
from itertools import repeat
import json
import logging
from pathlib import Path
import random
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader

from .cache import TileData, TileMetadata, choose_validation_regions, discover_target_years, list_tile_paths, load_tile_data, load_tile_metadata
from .dataset import (
    PixelSampleDataset,
    build_eval_matrix,
    gather_eval_sequences,
    sample_tile_training_data,
)
from .feature_bank import (
    LABEL_STRATEGIES,
    SEQUENCE_CHANNELS,
    TileFeatures,
    apply_label_strategy,
    build_monthly_sequence,
    build_target_months,
    build_tile_features,
)
from .metrics import find_best_threshold, weighted_average_precision
from .models import ModelOutputs, build_model, load_model_from_checkpoint


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
    gce_q: float
    label_smoothing: float


@dataclass(frozen=True)
class RuntimeConfig:
    device: torch.device
    amp_dtype: torch.dtype | None
    grad_clip_norm: float


@dataclass(frozen=True)
class TemporalConfig:
    enabled: bool
    target_months: tuple[tuple[int, int], ...]
    channels: tuple[str, ...]
    year_min: int | None
    num_year_classes: int
    aux_year_loss_weight: float


@dataclass(frozen=True)
class PreparedTrainTile:
    tile_id: str
    feature_names: tuple[str, ...] | None
    features: np.ndarray | None
    labels: np.ndarray | None
    weights: np.ndarray | None
    positive_count: int
    negative_count: int
    sequences: np.ndarray | None = None
    year_classes: np.ndarray | None = None
    year_mask: np.ndarray | None = None
    skipped_reason: str | None = None


@dataclass(frozen=True)
class PreparedTrainData:
    features: np.ndarray | torch.Tensor
    labels: np.ndarray | torch.Tensor
    weights: np.ndarray | torch.Tensor
    feature_names: tuple[str, ...]
    sequences: np.ndarray | torch.Tensor | None = None
    year_classes: np.ndarray | torch.Tensor | None = None
    year_mask: np.ndarray | torch.Tensor | None = None
    device_resident: bool = False


@dataclass(frozen=True)
class PreparedEvalTile:
    tile_id: str
    features: np.ndarray | None
    labels: np.ndarray | None
    weights: np.ndarray | None
    sequences: np.ndarray | None = None
    skipped_reason: str | None = None


@dataclass(frozen=True)
class PreparedEvalData:
    features: np.ndarray | torch.Tensor
    labels: np.ndarray
    weights: np.ndarray
    sequences: np.ndarray | torch.Tensor | None = None
    device_resident: bool = False


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
    parser.add_argument("--prep-workers", type=int, default=4)
    parser.add_argument("--gpu-resident-data", choices=["auto", "always", "never"], default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--model-name", choices=["mlp", "resmlp", "tcnn"], default="resmlp")
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[512, 512, 256, 256, 128])
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--loss", choices=["bce", "focal", "gce"], default="focal")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--gce-q", type=float, default=0.7)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--positive-class-weight", type=float, default=0.0)
    parser.add_argument("--auto-positive-class-weight", action="store_true")
    parser.add_argument("--amp-dtype", choices=["none", "bf16", "fp16"], default="bf16")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--use-temporal", action="store_true", help="Build monthly sequences (auto when --model-name=tcnn).")
    parser.add_argument("--aux-year-loss-weight", type=float, default=0.0, help="Multi-task weight for year classification head.")
    parser.add_argument(
        "--label-strategy",
        choices=LABEL_STRATEGIES,
        default="any_source",
        help=(
            "How to derive training_labels/weights from the cached tile labels. "
            "any_source (default) treats any label source (RADD/GLAD-L/GLAD-S2) "
            "firing as positive, matching the leaderboard's broad GT better than "
            "the persistence-refined gate that was optimising for our own heuristic."
        ),
    )
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

    use_temporal = args.use_temporal or args.model_name == "tcnn"
    target_months = build_target_months(target_years) if use_temporal else ()
    sequence_length = len(target_months)
    sequence_channels = len(SEQUENCE_CHANNELS) if use_temporal else 0

    year_min = min(target_years) if target_years else None
    num_year_classes = (max(target_years) - min(target_years) + 1) if (target_years and args.aux_year_loss_weight > 0) else 0
    temporal_config = TemporalConfig(
        enabled=use_temporal,
        target_months=target_months,
        channels=SEQUENCE_CHANNELS if use_temporal else (),
        year_min=year_min,
        num_year_classes=num_year_classes,
        aux_year_loss_weight=float(args.aux_year_loss_weight),
    )

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

    label_strategy = str(args.label_strategy)
    LOGGER.info("Label strategy: %s", label_strategy)

    train_prep_start = perf_counter()
    train_data = prepare_train_data(
        metadata=train_metadata,
        target_years=target_years,
        temporal_config=temporal_config,
        negative_ratio=args.negative_ratio,
        max_samples_per_tile=args.max_samples_per_tile,
        seed=args.seed,
        prep_workers=args.prep_workers,
        label_strategy=label_strategy,
    )
    LOGGER.info(
        "Prepared %s training rows in %.1fs",
        train_data.features.shape[0],
        perf_counter() - train_prep_start,
    )

    val_prep_start = perf_counter()
    raw_val_data = prepare_eval_data(
        metadata=val_metadata,
        target_years=target_years,
        temporal_config=temporal_config,
        prep_workers=args.prep_workers,
        label_strategy=label_strategy,
    )
    LOGGER.info(
        "Prepared %s validation rows in %.1fs",
        raw_val_data.features.shape[0],
        perf_counter() - val_prep_start,
    )

    positive_class_weight = resolve_positive_class_weight(
        np.asarray(train_data.labels),
        np.asarray(train_data.weights),
        user_value=args.positive_class_weight,
        auto_enabled=args.auto_positive_class_weight,
    )
    loss_config = LossConfig(
        name=args.loss,
        focal_gamma=args.focal_gamma,
        positive_class_weight=positive_class_weight,
        gce_q=float(args.gce_q),
        label_smoothing=float(args.label_smoothing),
    )

    if checkpoint is None:
        normalizer = fit_normalizer(np.asarray(train_data.features))
    else:
        normalizer = load_normalizer(checkpoint)
    train_data = PreparedTrainData(
        features=normalize_features(np.asarray(train_data.features), normalizer),
        labels=np.asarray(train_data.labels),
        weights=np.asarray(train_data.weights),
        feature_names=train_data.feature_names,
        sequences=(
            None
            if train_data.sequences is None
            else np.asarray(train_data.sequences, dtype=np.float32)
        ),
        year_classes=(
            None
            if train_data.year_classes is None
            else np.asarray(train_data.year_classes, dtype=np.int64)
        ),
        year_mask=(
            None
            if train_data.year_mask is None
            else np.asarray(train_data.year_mask, dtype=np.float32)
        ),
    )
    val_data = PreparedEvalData(
        features=normalize_features(np.asarray(raw_val_data.features), normalizer),
        labels=raw_val_data.labels,
        weights=raw_val_data.weights,
        sequences=(
            None
            if raw_val_data.sequences is None
            else np.asarray(raw_val_data.sequences, dtype=np.float32)
        ),
    )

    use_gpu_resident_data = should_use_gpu_resident_data(
        mode=args.gpu_resident_data,
        model_name=args.model_name,
        temporal_config=temporal_config,
        device=device,
        train_data=train_data,
        val_data=val_data,
    )
    if use_gpu_resident_data:
        train_data = move_train_data_to_device(train_data, device)
        val_data = move_eval_data_to_device(val_data, device)
        loader = None
    else:
        loader = build_train_loader(
            train_data=train_data,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )

    if checkpoint is None:
        model = build_model(
            args.model_name,
            input_dim=len(train_data.feature_names),
            hidden_dims=args.hidden_dims,
            dropout=args.dropout,
            sequence_channels=sequence_channels,
            sequence_length=sequence_length,
            num_year_classes=num_year_classes,
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
        feature_names=train_data.feature_names,
        positive_class_weight=positive_class_weight,
        temporal_config=temporal_config,
    )

    epochs_without_improvement = 0
    for epoch in range(start_epoch, args.epochs + 1):
        train_start = perf_counter()
        if train_data.device_resident:
            train_loss = train_one_epoch_resident(
                model=model,
                train_data=train_data,
                batch_size=args.batch_size,
                optimizer=optimizer,
                runtime_config=runtime_config,
                loss_config=loss_config,
                scaler=scaler,
                temporal_config=temporal_config,
            )
        else:
            if loader is None:
                raise RuntimeError("CPU-resident training path requires a DataLoader.")
            train_loss = train_one_epoch(
                model=model,
                loader=loader,
                optimizer=optimizer,
                runtime_config=runtime_config,
                loss_config=loss_config,
                scaler=scaler,
                temporal_config=temporal_config,
            )
        train_duration = perf_counter() - train_start
        scheduler.step()

        eval_start = perf_counter()
        val_result = evaluate_prepared_data(
            model=model,
            eval_data=val_data,
            runtime_config=runtime_config,
            batch_size=args.eval_batch_size,
        )
        eval_duration = perf_counter() - eval_start
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
        LOGGER.info(
            "Epoch %s timings train=%.1fs eval=%.1fs",
            epoch,
            train_duration,
            eval_duration,
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
            feature_names=train_data.feature_names,
            target_years=target_years,
            threshold=threshold,
            val_regions=val_regions,
            summary=summary,
            args=args,
            epoch=epoch,
            best_f1=checkpoint_best_f1,
            best_threshold=checkpoint_best_threshold,
            history=history,
            temporal_config=temporal_config,
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
                feature_names=train_data.feature_names,
                target_years=target_years,
                threshold=threshold,
                val_regions=val_regions,
                summary=summary,
                args=args,
                epoch=epoch,
                best_f1=best_f1,
                best_threshold=best_threshold,
                history=history,
                temporal_config=temporal_config,
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


def _stable_tile_seed(seed: int, tile_id: str) -> int:
    digest = hashlib.blake2b(f"{seed}:{tile_id}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def _tile_features_with_strategy(
    tile: TileData,
    target_years: tuple[int, ...],
    label_strategy: str,
) -> TileFeatures:
    """Build tile features and override labels/weights according to ``label_strategy``.

    ``build_tile_features`` can either compute ``persistence.training_labels``
    (cache miss) or fall back to ``tile.label_consensus`` (cache hit); both are
    inconsistent with each other and with what we actually want to train on.
    This helper re-derives labels/weights directly from ``tile.label_votes`` and
    the tile's persistence score so the training signal no longer depends on
    which code path happened to populate the cache.
    """
    tile_features = build_tile_features(tile, target_years)
    if tile.label_votes is None:
        # Test-split tile or unlabelled cache; nothing to override.
        return tile_features
    labels, weights = apply_label_strategy(tile, tile_features, label_strategy)
    return replace(tile_features, labels=labels, weights=weights)


def _resolve_prep_workers(prep_workers: int, item_count: int) -> int:
    if item_count <= 1:
        return 1
    return max(1, min(prep_workers, item_count))


def _prepare_train_tile(
    item: TileMetadata,
    target_years: tuple[int, ...],
    temporal_config: TemporalConfig,
    negative_ratio: float,
    max_samples_per_tile: int | None,
    seed: int,
    label_strategy: str,
) -> PreparedTrainTile:
    tile = load_tile_data(item.path)
    tile_features = _tile_features_with_strategy(tile, target_years, label_strategy)
    sequence = build_monthly_sequence(tile, temporal_config.target_months) if temporal_config.enabled else None
    rng = np.random.default_rng(_stable_tile_seed(seed, item.tile_id))
    try:
        sampled = sample_tile_training_data(
            item,
            tile_features,
            rng,
            negative_ratio=negative_ratio,
            max_samples_per_tile=max_samples_per_tile,
            sequence=sequence,
            year_min=temporal_config.year_min,
            num_year_classes=temporal_config.num_year_classes,
        )
    except ValueError as exc:
        return PreparedTrainTile(
            tile_id=item.tile_id,
            feature_names=tile_features.feature_names,
            features=None,
            labels=None,
            weights=None,
            positive_count=0,
            negative_count=0,
            skipped_reason=str(exc),
        )

    return PreparedTrainTile(
        tile_id=item.tile_id,
        feature_names=tile_features.feature_names,
        features=sampled.features,
        labels=sampled.labels,
        weights=sampled.weights,
        positive_count=sampled.positive_count,
        negative_count=sampled.negative_count,
        sequences=sampled.sequences,
        year_classes=sampled.year_classes,
        year_mask=sampled.year_mask,
    )


def prepare_train_data(
    *,
    metadata: list[TileMetadata],
    target_years: tuple[int, ...],
    temporal_config: TemporalConfig,
    negative_ratio: float,
    max_samples_per_tile: int | None,
    seed: int,
    prep_workers: int,
    label_strategy: str = "persistence_refined",
) -> PreparedTrainData:
    worker_count = _resolve_prep_workers(prep_workers, len(metadata))
    LOGGER.info(
        "Preparing %s train tiles with %s worker(s) [label_strategy=%s]",
        len(metadata),
        worker_count,
        label_strategy,
    )

    results: list[PreparedTrainTile]
    if worker_count == 1:
        results = [
            _prepare_train_tile(
                item,
                target_years,
                temporal_config,
                negative_ratio,
                max_samples_per_tile,
                seed,
                label_strategy,
            )
            for item in metadata
        ]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            results = list(
                executor.map(
                    _prepare_train_tile,
                    metadata,
                    repeat(target_years),
                    repeat(temporal_config),
                    repeat(negative_ratio),
                    repeat(max_samples_per_tile),
                    repeat(seed),
                    repeat(label_strategy),
                )
            )

    feature_names: tuple[str, ...] | None = None
    feature_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    weight_chunks: list[np.ndarray] = []
    sequence_chunks: list[np.ndarray] = []
    year_class_chunks: list[np.ndarray] = []
    year_mask_chunks: list[np.ndarray] = []

    for result in results:
        if result.feature_names is not None and feature_names is None:
            feature_names = result.feature_names
        if result.skipped_reason is not None:
            LOGGER.warning("Skipping train tile %s: %s", result.tile_id, result.skipped_reason)
            continue
        if feature_names is not None and result.feature_names != feature_names:
            raise ValueError(f"Feature names mismatch for train tile {result.tile_id}")
        if result.features is None or result.labels is None or result.weights is None:
            raise ValueError(f"Train tile {result.tile_id} did not return sample arrays.")
        LOGGER.info(
            "Tile %s sampled %s rows (%s positive candidates, %s negative candidates)",
            result.tile_id,
            result.features.shape[0],
            result.positive_count,
            result.negative_count,
        )
        feature_chunks.append(result.features)
        label_chunks.append(result.labels)
        weight_chunks.append(result.weights)
        if result.sequences is not None:
            sequence_chunks.append(result.sequences)
        if result.year_classes is not None and result.year_mask is not None:
            year_class_chunks.append(result.year_classes)
            year_mask_chunks.append(result.year_mask)

    if feature_names is None:
        raise SystemExit("No features were built for training.")
    if not feature_chunks:
        raise SystemExit("No trainable samples were collected from the cache.")

    return PreparedTrainData(
        features=np.concatenate(feature_chunks, axis=0),
        labels=np.concatenate(label_chunks, axis=0),
        weights=np.concatenate(weight_chunks, axis=0),
        feature_names=feature_names,
        sequences=np.concatenate(sequence_chunks, axis=0) if sequence_chunks else None,
        year_classes=np.concatenate(year_class_chunks, axis=0) if year_class_chunks else None,
        year_mask=np.concatenate(year_mask_chunks, axis=0) if year_mask_chunks else None,
    )


def _prepare_eval_tile(
    item: TileMetadata,
    target_years: tuple[int, ...],
    temporal_config: TemporalConfig,
    label_strategy: str,
) -> PreparedEvalTile:
    tile = load_tile_data(item.path)
    tile_features = _tile_features_with_strategy(tile, target_years, label_strategy)
    try:
        feature_matrix, tile_labels, tile_weights, _ = build_eval_matrix(tile_features)
    except ValueError as exc:
        return PreparedEvalTile(
            tile_id=item.tile_id,
            features=None,
            labels=None,
            weights=None,
            skipped_reason=str(exc),
        )

    sequence_matrix = None
    if temporal_config.enabled:
        sequence = build_monthly_sequence(tile, temporal_config.target_months)
        sequence_matrix = gather_eval_sequences(tile_features, sequence)

    return PreparedEvalTile(
        tile_id=item.tile_id,
        features=feature_matrix,
        labels=tile_labels,
        weights=tile_weights,
        sequences=sequence_matrix,
    )


def prepare_eval_data(
    *,
    metadata: list[TileMetadata],
    target_years: tuple[int, ...],
    temporal_config: TemporalConfig,
    prep_workers: int,
    label_strategy: str = "persistence_refined",
) -> PreparedEvalData:
    worker_count = _resolve_prep_workers(prep_workers, len(metadata))
    LOGGER.info(
        "Preparing %s validation tiles with %s worker(s) [label_strategy=%s]",
        len(metadata),
        worker_count,
        label_strategy,
    )

    results: list[PreparedEvalTile]
    if worker_count == 1:
        results = [
            _prepare_eval_tile(item, target_years, temporal_config, label_strategy)
            for item in metadata
        ]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            results = list(
                executor.map(
                    _prepare_eval_tile,
                    metadata,
                    repeat(target_years),
                    repeat(temporal_config),
                    repeat(label_strategy),
                )
            )

    feature_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    weight_chunks: list[np.ndarray] = []
    sequence_chunks: list[np.ndarray] = []

    for result in results:
        if result.skipped_reason is not None:
            LOGGER.warning("Skipping eval tile %s: %s", result.tile_id, result.skipped_reason)
            continue
        if result.features is None or result.labels is None or result.weights is None:
            raise ValueError(f"Validation tile {result.tile_id} did not return eval arrays.")
        LOGGER.info("Prepared eval tile %s with %s rows", result.tile_id, result.features.shape[0])
        feature_chunks.append(result.features)
        label_chunks.append(result.labels)
        weight_chunks.append(result.weights)
        if result.sequences is not None:
            sequence_chunks.append(result.sequences)

    if not feature_chunks:
        raise ValueError("No validation tiles produced evaluation pixels.")

    return PreparedEvalData(
        features=np.concatenate(feature_chunks, axis=0),
        labels=np.concatenate(label_chunks, axis=0),
        weights=np.concatenate(weight_chunks, axis=0),
        sequences=np.concatenate(sequence_chunks, axis=0) if sequence_chunks else None,
    )


def build_train_loader(
    *,
    train_data: PreparedTrainData,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    dataset = PixelSampleDataset(
        np.asarray(train_data.features),
        np.asarray(train_data.labels),
        np.asarray(train_data.weights),
        sequences=(
            None
            if train_data.sequences is None
            else np.asarray(train_data.sequences)
        ),
        year_classes=(
            None
            if train_data.year_classes is None
            else np.asarray(train_data.year_classes)
        ),
        year_mask=(
            None
            if train_data.year_mask is None
            else np.asarray(train_data.year_mask)
        ),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=num_workers > 0,
    )


def _estimate_storage_bytes(array: np.ndarray | torch.Tensor | None) -> int:
    if array is None:
        return 0
    if isinstance(array, torch.Tensor):
        return int(array.numel() * array.element_size())
    return int(array.nbytes)


def _supports_gpu_resident_data(model_name: str, temporal_config: TemporalConfig) -> bool:
    return model_name in {"mlp", "resmlp"} and not temporal_config.enabled


def should_use_gpu_resident_data(
    *,
    mode: str,
    model_name: str,
    temporal_config: TemporalConfig,
    device: torch.device,
    train_data: PreparedTrainData,
    val_data: PreparedEvalData,
) -> bool:
    if mode == "never" or device.type != "cuda":
        return False

    if not _supports_gpu_resident_data(model_name, temporal_config):
        if mode == "always":
            raise ValueError("GPU-resident data is supported only for non-temporal mlp/resmlp training.")
        LOGGER.info("GPU-resident data disabled for model=%s temporal=%s", model_name, temporal_config.enabled)
        return False

    estimated_bytes = (
        _estimate_storage_bytes(train_data.features)
        + _estimate_storage_bytes(train_data.labels)
        + _estimate_storage_bytes(train_data.weights)
        + _estimate_storage_bytes(train_data.sequences)
        + _estimate_storage_bytes(train_data.year_classes)
        + _estimate_storage_bytes(train_data.year_mask)
        + _estimate_storage_bytes(val_data.features)
        + _estimate_storage_bytes(val_data.sequences)
    )
    total_vram = int(torch.cuda.get_device_properties(device).total_memory)
    LOGGER.info(
        "GPU-resident data estimate %.2f GiB / %.2f GiB (mode=%s)",
        estimated_bytes / 1024**3,
        total_vram / 1024**3,
        mode,
    )

    if mode == "always":
        if estimated_bytes > int(total_vram * 0.9):
            raise ValueError(
                f"GPU-resident data requested but estimated {estimated_bytes / 1024**3:.2f} GiB exceeds 90% of VRAM."
            )
        return True

    return estimated_bytes <= total_vram // 2


def _to_device_tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(array)
    if device.type == "cuda":
        tensor = tensor.pin_memory()
    return tensor.to(device=device, non_blocking=device.type == "cuda")


def move_train_data_to_device(train_data: PreparedTrainData, device: torch.device) -> PreparedTrainData:
    return PreparedTrainData(
        features=_to_device_tensor(np.asarray(train_data.features, dtype=np.float32), device),
        labels=_to_device_tensor(np.asarray(train_data.labels, dtype=np.float32), device),
        weights=_to_device_tensor(np.asarray(train_data.weights, dtype=np.float32), device),
        feature_names=train_data.feature_names,
        sequences=(
            None
            if train_data.sequences is None
            else _to_device_tensor(np.asarray(train_data.sequences, dtype=np.float32), device)
        ),
        year_classes=(
            None
            if train_data.year_classes is None
            else _to_device_tensor(np.asarray(train_data.year_classes, dtype=np.int64), device)
        ),
        year_mask=(
            None
            if train_data.year_mask is None
            else _to_device_tensor(np.asarray(train_data.year_mask, dtype=np.float32), device)
        ),
        device_resident=True,
    )


def move_eval_data_to_device(eval_data: PreparedEvalData, device: torch.device) -> PreparedEvalData:
    return PreparedEvalData(
        features=_to_device_tensor(np.asarray(eval_data.features, dtype=np.float32), device),
        labels=eval_data.labels,
        weights=eval_data.weights,
        sequences=(
            None
            if eval_data.sequences is None
            else _to_device_tensor(np.asarray(eval_data.sequences, dtype=np.float32), device)
        ),
        device_resident=True,
    )


def train_one_epoch_resident(
    *,
    model: torch.nn.Module,
    train_data: PreparedTrainData,
    batch_size: int,
    optimizer: torch.optim.Optimizer,
    runtime_config: RuntimeConfig,
    loss_config: LossConfig,
    scaler: torch.cuda.amp.GradScaler,
    temporal_config: TemporalConfig,
) -> float:
    if not isinstance(train_data.features, torch.Tensor):
        raise TypeError("GPU-resident training requires tensor features.")

    model.train()
    total_loss = torch.zeros((), device=runtime_config.device, dtype=torch.float32)
    total_weight = torch.zeros((), device=runtime_config.device, dtype=torch.float32)
    row_count = int(train_data.features.shape[0])
    order = torch.randperm(row_count, device=runtime_config.device)

    for start in range(0, row_count, batch_size):
        stop = min(start + batch_size, row_count)
        index = order[start:stop]
        features = train_data.features.index_select(0, index)
        labels = train_data.labels.index_select(0, index)
        weights = train_data.weights.index_select(0, index)
        sequence = None if train_data.sequences is None else train_data.sequences.index_select(0, index)
        year_class = None if train_data.year_classes is None else train_data.year_classes.index_select(0, index)
        year_mask = None if train_data.year_mask is None else train_data.year_mask.index_select(0, index)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(runtime_config):
            outputs = model(features, sequence=sequence)
            loss, weighted_loss_sum, weight_sum = compute_total_loss(
                outputs=outputs,
                labels=labels,
                weights=weights,
                loss_config=loss_config,
                temporal_config=temporal_config,
                year_class=year_class,
                year_mask=year_mask,
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

        total_loss += weighted_loss_sum.detach().float()
        total_weight += weight_sum.detach().float()

    return float((total_loss / total_weight.clamp_min(1e-6)).cpu())


def evaluate_prepared_data(
    *,
    model: torch.nn.Module,
    eval_data: PreparedEvalData,
    runtime_config: RuntimeConfig,
    batch_size: int,
) -> dict[str, np.ndarray]:
    probabilities = predict_probabilities(
        model=model,
        features=eval_data.features,
        sequences=eval_data.sequences,
        runtime_config=runtime_config,
        batch_size=batch_size,
    )
    return {
        "probabilities": probabilities,
        "labels": eval_data.labels,
        "weights": eval_data.weights,
    }


def train_one_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    runtime_config: RuntimeConfig,
    loss_config: LossConfig,
    scaler: torch.cuda.amp.GradScaler,
    temporal_config: TemporalConfig,
) -> float:
    model.train()
    total_loss = 0.0
    total_weight = 0.0

    for batch in loader:
        features = batch["features"].to(runtime_config.device, non_blocking=True)
        labels = batch["labels"].to(runtime_config.device, non_blocking=True)
        weights = batch["weights"].to(runtime_config.device, non_blocking=True)
        sequence = batch.get("sequence")
        if sequence is not None:
            sequence = sequence.to(runtime_config.device, non_blocking=True)
        year_class = batch.get("year_class")
        year_mask = batch.get("year_mask")
        if year_class is not None and year_mask is not None:
            year_class = year_class.to(runtime_config.device, non_blocking=True)
            year_mask = year_mask.to(runtime_config.device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(runtime_config):
            outputs = model(features, sequence=sequence)
            loss, weighted_loss_sum, weight_sum = compute_total_loss(
                outputs=outputs,
                labels=labels,
                weights=weights,
                loss_config=loss_config,
                temporal_config=temporal_config,
                year_class=year_class,
                year_mask=year_mask,
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


def compute_total_loss(
    *,
    outputs: ModelOutputs,
    labels: torch.Tensor,
    weights: torch.Tensor,
    loss_config: LossConfig,
    temporal_config: TemporalConfig,
    year_class: torch.Tensor | None,
    year_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    binary_loss, weighted_loss_sum, weight_sum = compute_weighted_loss(
        logits=outputs.logits,
        labels=labels,
        weights=weights,
        loss_config=loss_config,
    )
    loss = binary_loss

    if (
        temporal_config.aux_year_loss_weight > 0.0
        and outputs.year_logits is not None
        and year_class is not None
        and year_mask is not None
    ):
        per_pixel_year_loss = torch.nn.functional.cross_entropy(
            outputs.year_logits,
            year_class,
            reduction="none",
        )
        masked_year_sum = (per_pixel_year_loss * year_mask).sum()
        masked_year_weight = year_mask.sum().clamp_min(1e-6)
        year_loss = masked_year_sum / masked_year_weight
        loss = loss + temporal_config.aux_year_loss_weight * year_loss

    return loss, weighted_loss_sum, weight_sum


def compute_weighted_loss(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    loss_config: LossConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    smoothed_labels = labels
    if loss_config.label_smoothing > 0:
        eps = loss_config.label_smoothing
        smoothed_labels = labels * (1.0 - eps) + 0.5 * eps

    if loss_config.name == "gce":
        loss_per_item = _generalized_cross_entropy(logits, smoothed_labels, q=loss_config.gce_q)
        if loss_config.positive_class_weight is not None:
            multiplier = torch.where(
                labels > 0.5,
                torch.full_like(loss_per_item, loss_config.positive_class_weight),
                torch.ones_like(loss_per_item),
            )
            loss_per_item = loss_per_item * multiplier
    else:
        pos_weight = None
        if loss_config.positive_class_weight is not None:
            pos_weight = torch.tensor(
                loss_config.positive_class_weight,
                device=logits.device,
                dtype=logits.dtype,
            )
        loss_per_item = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            smoothed_labels,
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


def _generalized_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    q: float,
) -> torch.Tensor:
    """GCE loss for binary classification, robust to label noise.

    Reference: Zhang & Sabuncu (2018), "Generalized Cross Entropy Loss for
    Training Deep Neural Networks with Noisy Labels".
    """
    if not 0.0 < q <= 1.0:
        raise ValueError(f"GCE q must lie in (0, 1]; got {q}")
    probabilities = torch.sigmoid(logits)
    p_target = labels * probabilities + (1.0 - labels) * (1.0 - probabilities)
    p_target = p_target.clamp(min=1e-6, max=1.0)
    return (1.0 - p_target.pow(q)) / q


def evaluate_on_tiles(
    *,
    model: torch.nn.Module,
    metadata: list[TileMetadata],
    target_years: tuple[int, ...],
    normalizer: Normalizer,
    runtime_config: RuntimeConfig,
    batch_size: int,
    temporal_config: TemporalConfig,
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
        sequence_matrix = None
        if temporal_config.enabled:
            sequence = build_monthly_sequence(tile, temporal_config.target_months)
            sequence_matrix = gather_eval_sequences(tile_features, sequence)
        tile_probabilities = predict_probabilities(
            model=model,
            features=feature_matrix,
            sequences=sequence_matrix,
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
    features: np.ndarray | torch.Tensor,
    runtime_config: RuntimeConfig,
    batch_size: int,
    sequences: np.ndarray | torch.Tensor | None = None,
) -> np.ndarray:
    if features.shape[0] == 0:
        return np.empty((0,), dtype=np.float32)
    if sequences is not None and sequences.shape[0] != features.shape[0]:
        raise ValueError(
            f"sequences and features must share the first dim; got {sequences.shape[0]} vs {features.shape[0]}."
        )
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, features.shape[0], batch_size):
            stop = min(start + batch_size, features.shape[0])
            if isinstance(features, torch.Tensor):
                batch_features = features[start:stop]
                if batch_features.device != runtime_config.device:
                    batch_features = batch_features.to(runtime_config.device, non_blocking=True)
            else:
                batch_features = torch.from_numpy(features[start:stop]).to(runtime_config.device, non_blocking=True)
            batch_sequence = None
            if sequences is not None:
                if isinstance(sequences, torch.Tensor):
                    batch_sequence = sequences[start:stop]
                    if batch_sequence.device != runtime_config.device:
                        batch_sequence = batch_sequence.to(runtime_config.device, non_blocking=True)
                else:
                    batch_sequence = torch.from_numpy(sequences[start:stop]).to(runtime_config.device, non_blocking=True)
            with autocast_context(runtime_config):
                model_outputs = model(batch_features, sequence=batch_sequence)
                probabilities = torch.sigmoid(model_outputs.logits)
            outputs.append(probabilities.detach().float().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def predict_year_logits(
    *,
    model: torch.nn.Module,
    features: np.ndarray | torch.Tensor,
    sequences: np.ndarray | torch.Tensor | None,
    runtime_config: RuntimeConfig,
    batch_size: int,
) -> np.ndarray | None:
    if features.shape[0] == 0:
        return None
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, features.shape[0], batch_size):
            stop = min(start + batch_size, features.shape[0])
            if isinstance(features, torch.Tensor):
                batch_features = features[start:stop]
                if batch_features.device != runtime_config.device:
                    batch_features = batch_features.to(runtime_config.device, non_blocking=True)
            else:
                batch_features = torch.from_numpy(features[start:stop]).to(runtime_config.device, non_blocking=True)
            batch_sequence = None
            if sequences is not None:
                if isinstance(sequences, torch.Tensor):
                    batch_sequence = sequences[start:stop]
                    if batch_sequence.device != runtime_config.device:
                        batch_sequence = batch_sequence.to(runtime_config.device, non_blocking=True)
                else:
                    batch_sequence = torch.from_numpy(sequences[start:stop]).to(runtime_config.device, non_blocking=True)
            with autocast_context(runtime_config):
                model_outputs = model(batch_features, sequence=batch_sequence)
            if model_outputs.year_logits is None:
                return None
            outputs.append(model_outputs.year_logits.detach().cpu().float().numpy())
    if not outputs:
        return None
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
    temporal_config: TemporalConfig,
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
                "temporal": {
                    "enabled": temporal_config.enabled,
                    "channels": list(temporal_config.channels),
                    "sequence_length": len(temporal_config.target_months),
                    "year_min": temporal_config.year_min,
                    "num_year_classes": temporal_config.num_year_classes,
                    "aux_year_loss_weight": temporal_config.aux_year_loss_weight,
                },
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
    temporal_config: TemporalConfig,
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
            "sequence_channels": len(temporal_config.channels),
            "sequence_length": len(temporal_config.target_months),
            "target_months": [list(month) for month in temporal_config.target_months],
            "sequence_channel_names": list(temporal_config.channels),
            "year_min": temporal_config.year_min,
            "num_year_classes": temporal_config.num_year_classes,
            "aux_year_loss_weight": temporal_config.aux_year_loss_weight,
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
