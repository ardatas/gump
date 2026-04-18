from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np
import torch
from torch.utils.data import Dataset

from .cache import TileMetadata
from .feature_bank import TileFeatures, TileSequence


LOGGER = logging.getLogger(__name__)


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
    max_positives_per_tile: int | None = None,
    negatives_if_no_positive: int = 100_000,
    sequence: TileSequence | None = None,
    year_min: int | None = None,
    num_year_classes: int = 0,
) -> TileTrainingSamples:
    """Sample training pixels from a single tile.

    Args:
        max_positives_per_tile: Cap positive pixels per tile before applying
            ``negative_ratio``. This ensures that tiles with very high positive
            rates (e.g. 35%) do not dominate training over tiles with low positive
            rates (e.g. 0.08%). When set, ``max_samples_per_tile`` is ignored.
    """
    if tile_features.labels is None or tile_features.weights is None:
        raise ValueError(f"Tile {metadata.tile_id} does not contain training labels.")

    valid_mask = tile_features.forest_mask & (tile_features.weights > 0)
    positive_mask = valid_mask & (tile_features.labels == 1)
    negative_mask = valid_mask & (tile_features.labels == 0)

    pos_coords = np.column_stack(np.nonzero(positive_mask))
    neg_coords = np.column_stack(np.nonzero(negative_mask))

    if pos_coords.size == 0 and neg_coords.size == 0:
        raise ValueError(f"Tile {metadata.tile_id} does not contain trainable pixels.")

    if max_positives_per_tile is not None and pos_coords.shape[0] > max_positives_per_tile:
        keep = rng.choice(pos_coords.shape[0], size=max_positives_per_tile, replace=False)
        sampled_positives = pos_coords[keep]
    else:
        sampled_positives = pos_coords

    if sampled_positives.shape[0] > 0:
        target_negatives = int(round(sampled_positives.shape[0] * negative_ratio))
    else:
        target_negatives = min(neg_coords.shape[0], negatives_if_no_positive)

    sampled_negatives = _sample_rows(neg_coords, target_negatives, rng)
    all_coords = np.concatenate([sampled_positives, sampled_negatives], axis=0)
    if all_coords.size == 0:
        raise ValueError(f"Tile {metadata.tile_id} produced no sampled pixels.")

    if max_positives_per_tile is None and max_samples_per_tile is not None and all_coords.shape[0] > max_samples_per_tile:
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


class PatchDataset(Dataset):
    """Patch-based dataset for spatial models (e.g. U-Net).

    Samples ``patches_per_tile`` patches per tile, split by ``pos_fraction``
    into patches centred on a deforested pixel vs. patches centred on a
    forest-but-not-deforested pixel.  Patches that would extend outside the
    tile boundary are zero-padded (weight=0 on the padded margin).

    Output per sample: (features, labels, weights) with shapes
    ``(C, patch_size, patch_size)``, ``(patch_size, patch_size)``,
    ``(patch_size, patch_size)``.
    """

    def __init__(
        self,
        tile_features_list: list[TileFeatures],
        patch_size: int = 128,
        patches_per_tile: int = 200,
        pos_fraction: float = 0.5,
        augment: bool = True,
        seed: int = 42,
    ) -> None:
        self._tile_features = tile_features_list
        self._patch_size = patch_size
        self._augment = augment
        rng = np.random.default_rng(seed)
        # List of (tile_idx, row_center, col_center)
        self._patches: list[tuple[int, int, int]] = []

        for tile_idx, tf in enumerate(tile_features_list):
            if tf.labels is None or tf.weights is None:
                continue
            valid = tf.forest_mask & (tf.weights > 0)
            pos_mask = valid & (tf.labels == 1)
            neg_mask = valid & (tf.labels == 0)

            h, w = tf.forest_mask.shape
            half = patch_size // 2
            r_min, r_max = half, h - half
            c_min, c_max = half, w - half
            if r_min >= r_max or c_min >= c_max:
                LOGGER.warning("Tile %d too small for patch_size=%d, skipping.", tile_idx, patch_size)
                continue

            n_pos = int(patches_per_tile * pos_fraction)
            n_neg = patches_per_tile - n_pos

            def _sample_centers(mask: np.ndarray, n: int) -> np.ndarray:
                coords = np.column_stack(np.nonzero(mask))
                if coords.shape[0] == 0:
                    return coords
                in_range = (
                    (coords[:, 0] >= r_min) & (coords[:, 0] < r_max) &
                    (coords[:, 1] >= c_min) & (coords[:, 1] < c_max)
                )
                coords = coords[in_range]
                if coords.shape[0] == 0:
                    return coords
                size = min(n, coords.shape[0])
                return coords[rng.choice(coords.shape[0], size=size, replace=False)]

            for r, c in _sample_centers(pos_mask, n_pos):
                self._patches.append((tile_idx, int(r), int(c)))
            for r, c in _sample_centers(neg_mask, n_neg):
                self._patches.append((tile_idx, int(r), int(c)))

        rng.shuffle(self._patches)
        LOGGER.info("PatchDataset: %d patches from %d tiles.", len(self._patches), len(tile_features_list))

    def __len__(self) -> int:
        return len(self._patches)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tile_idx, r, c = self._patches[index]
        tf = self._tile_features[tile_idx]
        half = self._patch_size // 2
        r0, r1 = r - half, r + half
        c0, c1 = c - half, c + half

        features_patch = tf.feature_cube[:, r0:r1, c0:c1].astype(np.float32)
        labels_patch = (
            tf.labels[r0:r1, c0:c1].astype(np.float32)
            if tf.labels is not None
            else np.zeros((r1 - r0, c1 - c0), dtype=np.float32)
        )
        weights_patch = (
            tf.weights[r0:r1, c0:c1].astype(np.float32)
            if tf.weights is not None
            else np.zeros((r1 - r0, c1 - c0), dtype=np.float32)
        )

        # Pad if patch goes out of bounds (safety net)
        if features_patch.shape[1] != self._patch_size or features_patch.shape[2] != self._patch_size:
            features_patch, labels_patch, weights_patch = _pad_to_size(
                features_patch, labels_patch, weights_patch, self._patch_size
            )

        if self._augment:
            features_patch, labels_patch, weights_patch = _augment_patch(
                features_patch, labels_patch, weights_patch
            )

        return (
            torch.from_numpy(features_patch),
            torch.from_numpy(labels_patch),
            torch.from_numpy(weights_patch),
        )


def _pad_to_size(
    features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    target: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ph = max(0, target - features.shape[1])
    pw = max(0, target - features.shape[2])
    if ph > 0 or pw > 0:
        features = np.pad(features, ((0, 0), (0, ph), (0, pw)), constant_values=0.0)
        labels = np.pad(labels, ((0, ph), (0, pw)), constant_values=0.0)
        weights = np.pad(weights, ((0, ph), (0, pw)), constant_values=0.0)
    return features, labels, weights


def _augment_patch(
    features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Random horizontal flip
    if np.random.random() > 0.5:
        features = features[:, :, ::-1].copy()
        labels = labels[:, ::-1].copy()
        weights = weights[:, ::-1].copy()
    # Random vertical flip
    if np.random.random() > 0.5:
        features = features[:, ::-1, :].copy()
        labels = labels[::-1, :].copy()
        weights = weights[::-1, :].copy()
    # Random 90° rotation (0/90/180/270)
    k = np.random.randint(0, 4)
    if k > 0:
        features = np.rot90(features, k=k, axes=(1, 2)).copy()
        labels = np.rot90(labels, k=k).copy()
        weights = np.rot90(weights, k=k).copy()
    return features, labels, weights


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
