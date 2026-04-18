from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from .cache import TileMetadata
from .feature_bank import TileFeatures


@dataclass(frozen=True)
class TileTrainingSamples:
    tile_id: str
    region: str
    features: np.ndarray
    labels: np.ndarray
    weights: np.ndarray
    positive_count: int
    negative_count: int


class PixelSampleDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> None:
        self.features = torch.from_numpy(features.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.float32))
        self.weights = torch.from_numpy(weights.astype(np.float32))

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index], self.weights[index]


def sample_tile_training_data(
    metadata: TileMetadata,
    tile_features: TileFeatures,
    rng: np.random.Generator,
    *,
    negative_ratio: float,
    max_samples_per_tile: int | None = None,
    negatives_if_no_positive: int = 100_000,
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

    return TileTrainingSamples(
        tile_id=metadata.tile_id,
        region=metadata.region,
        features=features,
        labels=labels,
        weights=weights,
        positive_count=int(pos_coords.shape[0]),
        negative_count=int(neg_coords.shape[0]),
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


def _sample_rows(array: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    if count <= 0 or array.shape[0] == 0:
        return np.empty((0, array.shape[1] if array.ndim == 2 else 0), dtype=array.dtype)
    if array.shape[0] <= count:
        return array
    index = rng.choice(array.shape[0], size=count, replace=False)
    return array[index]
