"""Region-based K-Fold cross-validation training script.

Trains one model per region fold (leave-one-region-out), producing a checkpoint
per fold and a summary JSON with per-fold metrics.  Downstream ensembling is
done by passing multiple --checkpoint flags to ``src.infer`` or
``src.calibrate_thresholds``.

Usage::

    python -m src.train_kfold \
        --cache-root /workspace/cache/gump \
        --runs-dir /workspace/runs/gump \
        --epochs 10 \
        --batch-size 8192 \
        [--regions 48P 18N ...] \
        [--exclude-regions 20M] \
        [--model-name resmlp] \
        [--use-temporal]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .cache import (
    TileMetadata,
    discover_target_years,
    list_tile_paths,
    load_tile_data,
    load_tile_metadata,
)
from .dataset import PixelSampleDataset, sample_tile_training_data
from .feature_bank import (
    SEQUENCE_CHANNELS,
    build_monthly_sequence,
    build_target_months,
    build_tile_features,
)
from .metrics import find_best_threshold, weighted_average_precision
from .models import build_model
from .train import (
    LossConfig,
    Normalizer,
    RuntimeConfig,
    TemporalConfig,
    autocast_context,
    compute_total_loss,
    evaluate_on_tiles,
    fit_normalizer,
    load_hard_negatives,
    normalize_features,
    resolve_amp_dtype,
    resolve_positive_class_weight,
    save_checkpoint,
    set_seed,
    train_one_epoch,
    write_run_config,
)


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Region-based K-Fold cross-validation training."
    )
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--eval-batch-size", type=int, default=131072)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--negative-ratio", type=float, default=1.5)
    parser.add_argument("--max-samples-per-tile", type=int, default=500_000)
    parser.add_argument(
        "--regions",
        nargs="*",
        default=None,
        help="Specific regions to fold over.  If not set, uses all discovered regions.",
    )
    parser.add_argument(
        "--exclude-regions",
        nargs="*",
        default=[],
        help="Regions to exclude from all folds (never trained on, never validated on).",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--model-name", choices=["mlp", "resmlp", "tcnn"], default="tcnn"
    )
    parser.add_argument(
        "--hidden-dims", nargs="+", type=int, default=[512, 512, 256, 256, 128]
    )
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument(
        "--loss", choices=["bce", "focal", "gce"], default="focal"
    )
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--gce-q", type=float, default=0.7)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--positive-class-weight", type=float, default=0.0)
    parser.add_argument("--auto-positive-class-weight", action="store_true")
    parser.add_argument(
        "--amp-dtype", choices=["none", "bf16", "fp16"], default="bf16"
    )
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument(
        "--use-temporal",
        action="store_true",
        help="Build monthly sequences (auto when --model-name=tcnn).",
    )
    parser.add_argument(
        "--aux-year-loss-weight",
        type=float,
        default=0.3,
        help="Multi-task weight for year classification head.",
    )
    parser.add_argument(
        "--hard-negatives-json",
        type=Path,
        default=None,
        help="Optional JSON from src.mine_hard_negatives: injects FPs as preferred negatives.",
    )
    parser.add_argument(
        "--hard-negative-fraction",
        type=float,
        default=0.5,
        help="Fraction of the negative budget per tile filled from the hard-negatives cache.",
    )
    return parser.parse_args()


def _discover_regions(metadata: list[TileMetadata]) -> list[str]:
    return sorted({item.region for item in metadata})


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

    all_metadata = [load_tile_metadata(path) for path in train_tile_paths]
    exclude_set = set(args.exclude_regions or [])
    all_metadata = [item for item in all_metadata if item.region not in exclude_set]

    all_regions = _discover_regions(all_metadata)
    fold_regions = args.regions if args.regions else all_regions
    fold_regions = [region for region in fold_regions if region not in exclude_set]

    if not fold_regions:
        raise SystemExit("No regions to fold over after exclusions.")

    LOGGER.info("All regions: %s", all_regions)
    LOGGER.info("Fold-over regions: %s", fold_regions)
    LOGGER.info("Excluded regions: %s", list(exclude_set))

    target_years = discover_target_years(args.cache_root)
    LOGGER.info("Target years: %s", target_years)

    use_temporal = args.use_temporal or args.model_name == "tcnn"
    target_months = build_target_months(target_years) if use_temporal else ()
    sequence_length = len(target_months)
    sequence_channels = len(SEQUENCE_CHANNELS) if use_temporal else 0

    year_min = min(target_years) if target_years else None
    num_year_classes = (
        (max(target_years) - min(target_years) + 1)
        if (target_years and args.aux_year_loss_weight > 0)
        else 0
    )
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

    kfold_timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    kfold_root = args.runs_dir / f"{kfold_timestamp}-kfold"
    kfold_root.mkdir(parents=True, exist_ok=True)

    fold_summaries: list[dict[str, object]] = []

    for fold_index, val_region in enumerate(fold_regions):
        val_regions = (val_region,)
        train_metadata = [item for item in all_metadata if item.region not in val_regions]
        val_metadata = [item for item in all_metadata if item.region in val_regions]

        if not train_metadata:
            LOGGER.warning("Fold %s (val=%s): no training tiles, skipping.", fold_index, val_region)
            continue
        if not val_metadata:
            LOGGER.warning("Fold %s (val=%s): no validation tiles, skipping.", fold_index, val_region)
            continue

        LOGGER.info(
            "=== Fold %s/%s — val region: %s, train=%s tiles, val=%s tiles ===",
            fold_index + 1,
            len(fold_regions),
            val_region,
            len(train_metadata),
            len(val_metadata),
        )

        fold_dir = kfold_root / f"fold_{fold_index}_{val_region}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        # Build training data for this fold
        rng = np.random.default_rng(args.seed + fold_index)
        hard_negatives_map = load_hard_negatives(args.hard_negatives_json)
        train_feature_chunks: list[np.ndarray] = []
        train_label_chunks: list[np.ndarray] = []
        train_weight_chunks: list[np.ndarray] = []
        train_sequence_chunks: list[np.ndarray] = []
        train_year_class_chunks: list[np.ndarray] = []
        train_year_mask_chunks: list[np.ndarray] = []
        feature_names: tuple[str, ...] | None = None

        for item in train_metadata:
            LOGGER.info("Building train samples for %s (fold %s)", item.tile_id, fold_index)
            tile = load_tile_data(item.path)
            tile_features = build_tile_features(tile, target_years)
            feature_names = tile_features.feature_names
            sequence = build_monthly_sequence(tile, target_months) if use_temporal else None
            hard_negatives_for_tile = hard_negatives_map.get(item.tile_id) if hard_negatives_map else None
            try:
                sampled = sample_tile_training_data(
                    item,
                    tile_features,
                    rng,
                    negative_ratio=args.negative_ratio,
                    max_samples_per_tile=args.max_samples_per_tile,
                    sequence=sequence,
                    year_min=temporal_config.year_min,
                    num_year_classes=temporal_config.num_year_classes,
                    hard_negative_coords=hard_negatives_for_tile,
                    hard_negative_fraction=args.hard_negative_fraction,
                )
            except ValueError as exc:
                LOGGER.warning("Skipping train tile %s: %s", item.tile_id, exc)
                continue
            train_feature_chunks.append(sampled.features)
            train_label_chunks.append(sampled.labels)
            train_weight_chunks.append(sampled.weights)
            if sampled.sequences is not None:
                train_sequence_chunks.append(sampled.sequences)
            if sampled.year_classes is not None and sampled.year_mask is not None:
                train_year_class_chunks.append(sampled.year_classes)
                train_year_mask_chunks.append(sampled.year_mask)

        if feature_names is None or not train_feature_chunks:
            LOGGER.warning("Fold %s: no trainable samples, skipping.", fold_index)
            continue

        raw_train_features = np.concatenate(train_feature_chunks, axis=0)
        train_labels = np.concatenate(train_label_chunks, axis=0)
        train_weights = np.concatenate(train_weight_chunks, axis=0)
        train_sequences = (
            np.concatenate(train_sequence_chunks, axis=0) if train_sequence_chunks else None
        )
        train_year_classes = (
            np.concatenate(train_year_class_chunks, axis=0) if train_year_class_chunks else None
        )
        train_year_mask = (
            np.concatenate(train_year_mask_chunks, axis=0) if train_year_mask_chunks else None
        )

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
            gce_q=float(args.gce_q),
            label_smoothing=float(args.label_smoothing),
        )

        normalizer = fit_normalizer(raw_train_features)
        train_features = normalize_features(raw_train_features, normalizer)

        dataset = PixelSampleDataset(
            train_features,
            train_labels,
            train_weights,
            sequences=train_sequences,
            year_classes=train_year_classes,
            year_mask=train_year_mask,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
            persistent_workers=args.num_workers > 0,
        )

        model = build_model(
            args.model_name,
            input_dim=train_features.shape[1],
            hidden_dims=args.hidden_dims,
            dropout=args.dropout,
            sequence_channels=sequence_channels,
            sequence_length=sequence_length,
            num_year_classes=num_year_classes,
        ).to(device)
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
        scaler = torch.cuda.amp.GradScaler(
            enabled=device.type == "cuda" and amp_dtype == torch.float16
        )

        write_run_config(
            run_dir=fold_dir,
            args=args,
            val_regions=val_regions,
            target_years=target_years,
            train_metadata=train_metadata,
            val_metadata=val_metadata,
            feature_names=feature_names,
            positive_class_weight=positive_class_weight,
            temporal_config=temporal_config,
        )

        history: list[dict[str, float]] = []
        best_f1 = -1.0
        best_threshold = 0.5
        epochs_without_improvement = 0

        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(
                model=model,
                loader=loader,
                optimizer=optimizer,
                runtime_config=runtime_config,
                loss_config=loss_config,
                scaler=scaler,
                temporal_config=temporal_config,
            )
            scheduler.step()

            val_result = evaluate_on_tiles(
                model=model,
                metadata=val_metadata,
                target_years=target_years,
                normalizer=normalizer,
                runtime_config=runtime_config,
                batch_size=args.eval_batch_size,
                temporal_config=temporal_config,
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
                "fold": fold_index,
                "val_region": val_region,
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
                "Fold %s Epoch %s lr=%.6f train_loss=%.5f val_f1=%.4f val_ap=%.4f prec=%.4f rec=%.4f thr=%.2f",
                fold_index,
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
                path=fold_dir / "last.pt",
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
                temporal_config=temporal_config,
            )

            if improved:
                best_f1 = checkpoint_best_f1
                best_threshold = checkpoint_best_threshold
                epochs_without_improvement = 0
                save_checkpoint(
                    path=fold_dir / "best.pt",
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
                    temporal_config=temporal_config,
                )
            else:
                epochs_without_improvement += 1

            if (
                args.early_stopping_patience > 0
                and epochs_without_improvement >= args.early_stopping_patience
            ):
                LOGGER.info(
                    "Fold %s: early stopping after %s epochs without improvement.",
                    fold_index,
                    epochs_without_improvement,
                )
                break

        (fold_dir / "history.json").write_text(json.dumps(history, indent=2))
        fold_summaries.append({
            "fold": fold_index,
            "val_region": val_region,
            "num_train_tiles": len(train_metadata),
            "num_val_tiles": len(val_metadata),
            "num_samples": raw_train_features.shape[0],
            "best_f1": best_f1,
            "best_threshold": best_threshold,
            "epochs_trained": len(history),
            "checkpoint": str(fold_dir / "best.pt"),
        })
        LOGGER.info(
            "Fold %s complete — best_f1=%.4f  best_threshold=%.2f",
            fold_index,
            best_f1,
            best_threshold,
        )

    # Write K-Fold summary
    kfold_summary = {
        "timestamp": kfold_timestamp,
        "model_name": args.model_name,
        "fold_regions": fold_regions,
        "excluded_regions": list(exclude_set),
        "folds": fold_summaries,
        "mean_f1": float(np.mean([fold["best_f1"] for fold in fold_summaries])) if fold_summaries else 0.0,
        "std_f1": float(np.std([fold["best_f1"] for fold in fold_summaries])) if fold_summaries else 0.0,
        "checkpoints": [fold["checkpoint"] for fold in fold_summaries],
    }
    summary_path = kfold_root / "kfold_summary.json"
    summary_path.write_text(json.dumps(kfold_summary, indent=2))
    LOGGER.info(
        "K-Fold complete — %s folds, mean_f1=%.4f ± %.4f",
        len(fold_summaries),
        kfold_summary["mean_f1"],
        kfold_summary["std_f1"],
    )
    LOGGER.info("Summary: %s", summary_path)
    LOGGER.info("Checkpoints for ensemble: %s", " --checkpoint ".join(kfold_summary["checkpoints"]))


if __name__ == "__main__":
    main()
