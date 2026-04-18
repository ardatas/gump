from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np

from .cache import TileData


S2_SCALE = 10_000.0
S2_RED_INDEX = 3
S2_NIR_INDEX = 7
S2_SWIR1_INDEX = 10
S2_SWIR2_INDEX = 11


@dataclass(frozen=True)
class TileFeatures:
    feature_cube: np.ndarray
    feature_names: tuple[str, ...]
    forest_mask: np.ndarray
    labels: np.ndarray | None
    weights: np.ndarray | None
    persistence_score: np.ndarray
    persistence_mask: np.ndarray


def build_tile_features(tile: TileData, target_years: tuple[int, ...]) -> TileFeatures:
    s2 = tile.s2_data / S2_SCALE
    s2_yearly = _yearly_reduce(s2, tile.s2_year, target_years, reducer="median")
    s2_std = _safe_nanstd(s2, axis=0)
    s2_delta = s2_yearly[-1] - s2_yearly[0]

    ndvi_monthly = _normalized_difference(s2[:, S2_NIR_INDEX], s2[:, S2_RED_INDEX])
    nbr_monthly = _normalized_difference(s2[:, S2_NIR_INDEX], s2[:, S2_SWIR2_INDEX])
    ndmi_monthly = _normalized_difference(s2[:, S2_NIR_INDEX], s2[:, S2_SWIR1_INDEX])

    ndvi_yearly = _yearly_reduce(ndvi_monthly[:, None], tile.s2_year, target_years, reducer="median")
    nbr_yearly = _yearly_reduce(nbr_monthly[:, None], tile.s2_year, target_years, reducer="median")
    ndmi_yearly = _yearly_reduce(ndmi_monthly[:, None], tile.s2_year, target_years, reducer="median")
    index_yearly = np.concatenate([ndvi_yearly, nbr_yearly, ndmi_yearly], axis=1)
    index_delta = index_yearly[-1] - index_yearly[0]
    index_std = np.concatenate(
        [
            _safe_nanstd(ndvi_monthly, axis=0, keepdims=True),
            _safe_nanstd(nbr_monthly, axis=0, keepdims=True),
            _safe_nanstd(ndmi_monthly, axis=0, keepdims=True),
        ],
        axis=0,
    )

    s1_asc = _yearly_reduce(tile.s1_ascending_data, tile.s1_ascending_year, target_years, reducer="median")
    s1_desc = _yearly_reduce(tile.s1_descending_data, tile.s1_descending_year, target_years, reducer="median")
    s1_delta = np.concatenate([s1_asc[-1] - s1_asc[0], s1_desc[-1] - s1_desc[0]], axis=0)
    s1_std = np.concatenate(
        [
            _reduce_or_zeros(tile.s1_ascending_data, reducer="std"),
            _reduce_or_zeros(tile.s1_descending_data, reducer="std"),
        ],
        axis=0,
    )

    aef_yearly = _align_yearly_observations(tile.aef_data, tile.aef_year, target_years)
    aef_mean = _safe_nanmean(aef_yearly, axis=0)
    aef_delta = aef_yearly[-1] - aef_yearly[0]

    persistence = compute_persistence_targets(
        forest_mask=tile.forest_mask,
        raw_labels=tile.label_consensus,
        raw_weights=tile.label_weight,
        raw_votes=tile.label_votes,
        ndvi_yearly=ndvi_yearly[:, 0],
        nbr_yearly=nbr_yearly[:, 0],
        ndmi_yearly=ndmi_yearly[:, 0],
    )

    feature_arrays = [
        s2_yearly.reshape(-1, *tile.forest_mask.shape),
        s2_delta,
        s2_std,
        index_yearly.reshape(-1, *tile.forest_mask.shape),
        index_delta,
        index_std,
        s1_asc.reshape(-1, *tile.forest_mask.shape),
        s1_desc.reshape(-1, *tile.forest_mask.shape),
        s1_delta,
        s1_std,
        aef_mean,
        aef_delta,
        persistence.feature_layers,
    ]
    feature_cube = np.concatenate(feature_arrays, axis=0).astype(np.float32)
    feature_cube = np.nan_to_num(feature_cube, nan=0.0, posinf=0.0, neginf=0.0)

    feature_names = (
        tuple(f"s2_median_y{year}_b{band:02d}" for year in target_years for band in range(1, 13))
        + tuple(f"s2_delta_b{band:02d}" for band in range(1, 13))
        + tuple(f"s2_std_b{band:02d}" for band in range(1, 13))
        + tuple(f"{index}_median_y{year}" for year in target_years for index in ("ndvi", "nbr", "ndmi"))
        + tuple(f"{index}_delta" for index in ("ndvi", "nbr", "ndmi"))
        + tuple(f"{index}_std" for index in ("ndvi", "nbr", "ndmi"))
        + tuple(f"s1_asc_median_y{year}" for year in target_years)
        + tuple(f"s1_desc_median_y{year}" for year in target_years)
        + ("s1_asc_delta", "s1_desc_delta", "s1_asc_std", "s1_desc_std")
        + tuple(f"aef_mean_{index:02d}" for index in range(64))
        + tuple(f"aef_delta_{index:02d}" for index in range(64))
        + (
            "recent_ndvi",
            "recent_nbr",
            "recent_ndmi",
            "ndvi_drop",
            "nbr_drop",
            "ndmi_drop",
            "persistence_score",
            "persistence_mask",
        )
    )

    return TileFeatures(
        feature_cube=feature_cube,
        feature_names=feature_names,
        forest_mask=tile.forest_mask,
        labels=persistence.training_labels,
        weights=persistence.training_weights,
        persistence_score=persistence.persistence_score,
        persistence_mask=persistence.persistence_mask,
    )


@dataclass(frozen=True)
class PersistenceTargets:
    training_labels: np.ndarray | None
    training_weights: np.ndarray | None
    persistence_score: np.ndarray
    persistence_mask: np.ndarray
    feature_layers: np.ndarray


def compute_persistence_targets(
    *,
    forest_mask: np.ndarray,
    raw_labels: np.ndarray | None,
    raw_weights: np.ndarray | None,
    raw_votes: np.ndarray | None,
    ndvi_yearly: np.ndarray,
    nbr_yearly: np.ndarray,
    ndmi_yearly: np.ndarray,
) -> PersistenceTargets:
    recent_years = min(2, ndvi_yearly.shape[0])
    recent_ndvi = _safe_nanmean(ndvi_yearly[-recent_years:], axis=0)
    recent_nbr = _safe_nanmean(nbr_yearly[-recent_years:], axis=0)
    recent_ndmi = _safe_nanmean(ndmi_yearly[-recent_years:], axis=0)

    baseline_ndvi = ndvi_yearly[0]
    baseline_nbr = nbr_yearly[0]
    baseline_ndmi = ndmi_yearly[0]

    ndvi_drop = baseline_ndvi - recent_ndvi
    nbr_drop = baseline_nbr - recent_nbr
    ndmi_drop = baseline_ndmi - recent_ndmi

    drop_score = (
        0.45 * _scaled_score(ndvi_drop, low=0.10, high=0.35)
        + 0.40 * _scaled_score(nbr_drop, low=0.05, high=0.30)
        + 0.15 * _scaled_score(ndmi_drop, low=0.05, high=0.25)
    )
    state_score = (
        0.5 * _scaled_score(0.60 - recent_ndvi, low=0.05, high=0.30)
        + 0.3 * _scaled_score(0.45 - recent_nbr, low=0.05, high=0.25)
        + 0.2 * _scaled_score(0.35 - recent_ndmi, low=0.02, high=0.18)
    )
    persistence_score = np.clip(0.6 * drop_score + 0.4 * state_score, 0.0, 1.0).astype(np.float32)
    persistence_mask = (
        forest_mask
        & (ndvi_drop >= 0.18)
        & (nbr_drop >= 0.10)
        & (recent_ndvi <= 0.60)
        & (persistence_score >= 0.45)
    )

    feature_layers = np.stack(
        [
            recent_ndvi.astype(np.float32),
            recent_nbr.astype(np.float32),
            recent_ndmi.astype(np.float32),
            ndvi_drop.astype(np.float32),
            nbr_drop.astype(np.float32),
            ndmi_drop.astype(np.float32),
            persistence_score.astype(np.float32),
            persistence_mask.astype(np.float32),
        ],
        axis=0,
    )

    if raw_labels is None or raw_weights is None or raw_votes is None:
        return PersistenceTargets(
            training_labels=None,
            training_weights=None,
            persistence_score=persistence_score,
            persistence_mask=persistence_mask,
            feature_layers=feature_layers,
        )

    votes = raw_votes.astype(np.uint8)
    labels = raw_labels.astype(np.uint8)
    weights = raw_weights.astype(np.float32).copy()

    strong_consensus = votes >= 2
    triple_consensus = votes == 3
    single_source = votes == 1

    refined_positive = (
        (strong_consensus & (persistence_score >= 0.30))
        | (triple_consensus & (persistence_score >= 0.15))
        | (single_source & (persistence_score >= 0.75))
    )
    refined_positive &= forest_mask

    weights[~forest_mask] = 0.0

    ambiguous_alert = (labels == 1) & ~refined_positive
    weights[ambiguous_alert] *= 0.15

    likely_unlabeled_positive = (votes == 0) & forest_mask & (persistence_score >= 0.75)
    weights[likely_unlabeled_positive] = 0.0

    medium_persistence_single_source = single_source & (persistence_score >= 0.45) & (persistence_score < 0.75)
    weights[medium_persistence_single_source] *= 0.25

    training_labels = refined_positive.astype(np.uint8)

    return PersistenceTargets(
        training_labels=training_labels,
        training_weights=weights,
        persistence_score=persistence_score,
        persistence_mask=persistence_mask,
        feature_layers=feature_layers,
    )


def _yearly_reduce(
    values: np.ndarray,
    years: np.ndarray,
    target_years: tuple[int, ...],
    *,
    reducer: str,
) -> np.ndarray:
    if values.shape[0] == 0:
        return np.zeros((len(target_years), *values.shape[1:]), dtype=np.float32)

    available: dict[int, np.ndarray] = {}
    for year in sorted(set(years.tolist())):
        year_values = values[years == year]
        if year_values.size == 0:
            continue
        if reducer == "median":
            available[int(year)] = _safe_nanmedian(year_values, axis=0)
        elif reducer == "mean":
            available[int(year)] = _safe_nanmean(year_values, axis=0)
        else:
            raise ValueError(f"Unsupported reducer: {reducer}")
    return _align_year_maps(available, target_years)


def _align_yearly_observations(
    values: np.ndarray,
    years: np.ndarray,
    target_years: tuple[int, ...],
) -> np.ndarray:
    if values.shape[0] == 0:
        return np.zeros((len(target_years), *values.shape[1:]), dtype=np.float32)
    available = {int(year): values[index] for index, year in enumerate(years.tolist())}
    return _align_year_maps(available, target_years)


def _align_year_maps(
    available: dict[int, np.ndarray],
    target_years: tuple[int, ...],
) -> np.ndarray:
    if not available:
        raise ValueError("No yearly data available to align.")

    available_years = sorted(available)
    aligned = []
    for year in target_years:
        nearest_year = min(available_years, key=lambda candidate: (abs(candidate - year), candidate))
        aligned.append(available[nearest_year])
    return np.stack(aligned, axis=0)


def _normalized_difference(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    denom = numerator + denominator
    result = np.full_like(numerator, np.nan, dtype=np.float32)
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (np.abs(denom) > 1e-6)
    result[valid] = (numerator[valid] - denominator[valid]) / denom[valid]
    return result


def _reduce_or_zeros(values: np.ndarray, *, reducer: str) -> np.ndarray:
    if values.shape[0] == 0:
        return np.zeros(values.shape[1:], dtype=np.float32)
    if reducer == "std":
        return _safe_nanstd(values, axis=0)
    raise ValueError(f"Unsupported reducer: {reducer}")


def _safe_nanmedian(values: np.ndarray, axis: int = 0) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(values, axis=axis)


def _safe_nanmean(values: np.ndarray, axis: int = 0) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(values, axis=axis)


def _safe_nanstd(values: np.ndarray, axis: int = 0, keepdims: bool = False) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanstd(values, axis=axis, keepdims=keepdims)


def _scaled_score(values: np.ndarray, *, low: float, high: float) -> np.ndarray:
    if high <= low:
        raise ValueError("high must be greater than low")
    return np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)
