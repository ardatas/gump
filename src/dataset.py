from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from .cache import TileMetadata
from .feature_bank import TileFeatures, TileSequence


@dataclass(frozen=True)
class TileTrainingSamples:
    tile_id: str
    region: str
    features: np.ndarray
    labels: np.ndarray
    weights: np.ndarray
    positive_count: int
    negative_count: int
    sequences: np.ndarray | None = None
    year_classes: np.ndarray | None = None
    year_mask: np.ndarray | None = None


class PixelSampleDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        weights: np.ndarray,
        sequences: np.ndarray | None = None,
        year_classes: np.ndarray | None = None,
        year_mask: np.ndarray | None = None,
    ) -> None:
        self.features = torch.from_numpy(features.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.float32))
        self.weights = torch.from_numpy(weights.astype(np.float32))
        self.sequences = (
            torch.from_numpy(sequences.astype(np.float32)) if sequences is not None else None
        )
        self.year_classes = (
            torch.from_numpy(year_classes.astype(np.int64)) if year_classes is not None else None
        )
        self.year_mask = (
            torch.from_numpy(year_mask.astype(np.float32)) if year_mask is not None else None
        )

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item: dict[str, torch.Tensor] = {
            "features": self.features[index],
            "labels": self.labels[index],
            "weights": self.weights[index],
        }
        if self.sequences is not None:
            item["sequence"] = self.sequences[index]
        if self.year_classes is not None and self.year_mask is not None:
            item["year_class"] = self.year_classes[index]
            item["year_mask"] = self.year_mask[index]
        return item


def sample_tile_training_data(
    metadata: TileMetadata,
    tile_features: TileFeatures,
    rng: np.random.Generator,
    *,
    negative_ratio: float,
    max_samples_per_tile: int | None = None,
    negatives_if_no_positive: int = 100_000,
    sequence: TileSequence | None = None,
    year_min: int | None = None,
    num_year_classes: int = 0,
) -> TileTrainingSamples:
    if tile_features.labels is None or tile_features.weights is None:
        raise ValueError(f"Tile {metadata.tile_id} does not contain training labels.")

    valid_mask = tile_features.forest_mask & (tile_features.weights > 0)
    positive_mask = valid_mask & (tile_features.labels == 1)
    negative_mask = valid_mask & (tile_features.labels == 0)

    pos_coords = np.column_stack(np.nonzero(positive_mask))
    neg_coords = np.column_stack(np.nonzero(negative_mask))

    if pos_coords.size == 0 and neg_coords.size == 0:
        raise ValueError(f"Tile {metadata.tile_id} does not contain trainable pixels.")

    if pos_coords.shape[0] > 0:
        target_negatives = int(round(pos_coords.shape[0] * negative_ratio))
    else:
        target_negatives = min(neg_coords.shape[0], negatives_if_no_positive)

    sampled_negatives = _sample_rows(neg_coords, target_negatives, rng)
    sampled_positives = pos_coords

    all_coords = np.concatenate([sampled_positives, sampled_negatives], axis=0)
    if all_coords.size == 0:
        raise ValueError(f"Tile {metadata.tile_id} produced no sampled pixels.")

    if max_samples_per_tile is not None and all_coords.shape[0] > max_samples_per_tile:
        if sampled_positives.shape[0] >= max_samples_per_tile:
            keep_pos = rng.choice(sampled_positives.shape[0], size=max_samples_per_tile, replace=False)
            all_coords = sampled_positives[keep_pos]
        else:
            remaining = max_samples_per_tile - sampled_positives.shape[0]
            sampled_negatives = _sample_rows(sampled_negatives, remaining, rng)
            all_coords = np.concatenate([sampled_positives, sampled_negatives], axis=0)

    rng.shuffle(all_coords)
    rows = all_coords[:, 0]
    cols = all_coords[:, 1]
    features = tile_features.feature_cube[:, rows, cols].T.astype(np.float32)
    labels = tile_features.labels[rows, cols].astype(np.float32)
    weights = tile_features.weights[rows, cols].astype(np.float32)

    sequences = None
    if sequence is not None:
        sequences = _gather_sequences(sequence, rows, cols)

    year_classes = None
    year_mask = None
    if (
        num_year_classes > 0
        and year_min is not None
        and tile_features.alert_year is not None
    ):
        year_classes, year_mask = _gather_year_classes(
            tile_features.alert_year,
            labels,
            rows,
            cols,
            year_min=year_min,
            num_year_classes=num_year_classes,
        )

    return TileTrainingSamples(
        tile_id=metadata.tile_id,
        region=metadata.region,
        features=features,
        labels=labels,
        weights=weights,
        positive_count=int(pos_coords.shape[0]),
        negative_count=int(neg_coords.shape[0]),
        sequences=sequences,
        year_classes=year_classes,
        year_mask=year_mask,
    )


def build_eval_matrix(tile_features: TileFeatures) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if tile_features.labels is None or tile_features.weights is None:
        raise ValueError("Evaluation requires labels and weights.")

    eval_mask = tile_features.forest_mask & (tile_features.weights > 0)
    rows, cols = np.nonzero(eval_mask)
    features = tile_features.feature_cube[:, rows, cols].T.astype(np.float32)
    labels = tile_features.labels[rows, cols].astype(np.float32)
    weights = tile_features.weights[rows, cols].astype(np.float32)
    return features, labels, weights, eval_mask


def gather_eval_sequences(
    tile_features: TileFeatures,
    sequence: TileSequence,
) -> np.ndarray:
    eval_mask = tile_features.forest_mask & (tile_features.weights > 0)
    rows, cols = np.nonzero(eval_mask)
    return _gather_sequences(sequence, rows, cols)


def build_prediction_matrix(
    tile_features: TileFeatures,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    prediction_mask = tile_features.forest_mask if mask is None else mask.astype(bool)
    rows, cols = np.nonzero(prediction_mask)
    features = tile_features.feature_cube[:, rows, cols].T.astype(np.float32)
    return features, prediction_mask


def gather_prediction_sequences(
    sequence: TileSequence,
    prediction_mask: np.ndarray,
) -> np.ndarray:
    rows, cols = np.nonzero(prediction_mask)
    return _gather_sequences(sequence, rows, cols)


def scatter_predictions(
    values: np.ndarray,
    mask: np.ndarray,
    *,
    fill_value: float = 0.0,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    output = np.full(mask.shape, fill_value, dtype=dtype)
    rows, cols = np.nonzero(mask)
    output[rows, cols] = values.astype(dtype, copy=False)
    return output


def _sample_rows(array: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    if count <= 0 or array.shape[0] == 0:
        return np.empty((0, array.shape[1] if array.ndim == 2 else 0), dtype=array.dtype)
    if array.shape[0] <= count:
        return array
    index = rng.choice(array.shape[0], size=count, replace=False)
    return array[index]


def _gather_sequences(sequence: TileSequence, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Return sequences in shape (N, T, C+1) with the validity mask appended as the last channel."""
    seq_pixels = sequence.sequence[:, :, rows, cols]  # (T, C, N)
    mask_pixels = sequence.sequence_mask[:, rows, cols].astype(np.float32)  # (T, N)
    seq_pixels = np.transpose(seq_pixels, (2, 0, 1)).astype(np.float32)  # (N, T, C)
    mask_pixels = np.transpose(mask_pixels, (1, 0))[..., None].astype(np.float32)  # (N, T, 1)
    return np.concatenate([seq_pixels, mask_pixels], axis=-1)


def _gather_year_classes(
    alert_year: np.ndarray,
    labels: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    *,
    year_min: int,
    num_year_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    sampled_year = alert_year[rows, cols].astype(np.int32)
    has_year = sampled_year > 0
    is_positive = labels.astype(bool)
    year_mask = (has_year & is_positive).astype(np.float32)
    classes = np.zeros(rows.shape[0], dtype=np.int64)
    if num_year_classes > 0:
        clipped = np.clip(sampled_year - year_min, 0, num_year_classes - 1)
        classes[has_year] = clipped[has_year]
    return classes, year_mask
