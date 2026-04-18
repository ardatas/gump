from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import warnings

import numcodecs
import numpy as np
import zarr

from .cache import TileData


LOGGER = logging.getLogger(__name__)

S2_SCALE = 10_000.0
S2_RED_INDEX = 3
S2_NIR_INDEX = 7
S2_SWIR1_INDEX = 10
S2_SWIR2_INDEX = 11

_CACHE_COMPRESSOR = numcodecs.Blosc(cname="zstd", clevel=3, shuffle=numcodecs.Blosc.BITSHUFFLE)


@dataclass(frozen=True)
class TileFeatures:
    feature_cube: np.ndarray
    feature_names: tuple[str, ...]
    forest_mask: np.ndarray
    labels: np.ndarray | None
    weights: np.ndarray | None
    persistence_score: np.ndarray
    persistence_mask: np.ndarray
    alert_year: np.ndarray | None = None


@dataclass(frozen=True)
class TileSequence:
    sequence: np.ndarray
    sequence_mask: np.ndarray
    channel_names: tuple[str, ...]
    target_months: tuple[tuple[int, int], ...]


SEQUENCE_CHANNELS = (
    "ndvi",
    "nbr",
    "ndmi",
    "s2_red",
    "s2_nir",
    "s2_swir1",
    "s2_swir2",
    "s1_asc",
    "s1_desc",
)


def build_target_months(target_years: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    if not target_years:
        return ()
    start = min(target_years)
    stop = max(target_years)
    months: list[tuple[int, int]] = []
    for year in range(start, stop + 1):
        for month in range(1, 13):
            months.append((year, month))
    return tuple(months)


def build_monthly_sequence(
    tile: TileData,
    target_months: tuple[tuple[int, int], ...],
) -> TileSequence:
    height, width = tile.forest_mask.shape
    num_months = len(target_months)
    num_channels = len(SEQUENCE_CHANNELS)
    sequence = np.zeros((num_months, num_channels, height, width), dtype=np.float32)
    sequence_mask = np.zeros((num_months, height, width), dtype=bool)

    if num_months == 0:
        return TileSequence(
            sequence=sequence,
            sequence_mask=sequence_mask,
            channel_names=SEQUENCE_CHANNELS,
            target_months=target_months,
        )

    s2_scaled = tile.s2_data / S2_SCALE
    s2_index_map = _index_by_year_month(tile.s2_year, tile.s2_month)
    s1_asc_index_map = _index_by_year_month(tile.s1_ascending_year, tile.s1_ascending_month)
    s1_desc_index_map = _index_by_year_month(tile.s1_descending_year, tile.s1_descending_month)

    for time_idx, (year, month) in enumerate(target_months):
        s2_obs_index = s2_index_map.get((year, month))
        if s2_obs_index is not None:
            band_red = s2_scaled[s2_obs_index, S2_RED_INDEX]
            band_nir = s2_scaled[s2_obs_index, S2_NIR_INDEX]
            band_swir1 = s2_scaled[s2_obs_index, S2_SWIR1_INDEX]
            band_swir2 = s2_scaled[s2_obs_index, S2_SWIR2_INDEX]

            ndvi = _normalized_difference(band_nir, band_red)
            nbr = _normalized_difference(band_nir, band_swir2)
            ndmi = _normalized_difference(band_nir, band_swir1)

            sequence[time_idx, 0] = np.nan_to_num(ndvi, nan=0.0)
            sequence[time_idx, 1] = np.nan_to_num(nbr, nan=0.0)
            sequence[time_idx, 2] = np.nan_to_num(ndmi, nan=0.0)
            sequence[time_idx, 3] = np.nan_to_num(band_red, nan=0.0)
            sequence[time_idx, 4] = np.nan_to_num(band_nir, nan=0.0)
            sequence[time_idx, 5] = np.nan_to_num(band_swir1, nan=0.0)
            sequence[time_idx, 6] = np.nan_to_num(band_swir2, nan=0.0)

            valid_pixel = (
                np.isfinite(band_red)
                & np.isfinite(band_nir)
                & ((band_red + band_nir + band_swir1 + band_swir2) > 0)
            )
            sequence_mask[time_idx] |= valid_pixel

        s1_asc_obs_index = s1_asc_index_map.get((year, month))
        if s1_asc_obs_index is not None:
            asc_value = tile.s1_ascending_data[s1_asc_obs_index, 0]
            sequence[time_idx, 7] = np.nan_to_num(asc_value, nan=0.0)
            sequence_mask[time_idx] |= np.isfinite(asc_value) & (asc_value != 0)

        s1_desc_obs_index = s1_desc_index_map.get((year, month))
        if s1_desc_obs_index is not None:
            desc_value = tile.s1_descending_data[s1_desc_obs_index, 0]
            sequence[time_idx, 8] = np.nan_to_num(desc_value, nan=0.0)
            sequence_mask[time_idx] |= np.isfinite(desc_value) & (desc_value != 0)

    return TileSequence(
        sequence=sequence,
        sequence_mask=sequence_mask,
        channel_names=SEQUENCE_CHANNELS,
        target_months=target_months,
    )


def _index_by_year_month(years: np.ndarray, months: np.ndarray) -> dict[tuple[int, int], int]:
    mapping: dict[tuple[int, int], int] = {}
    for index in range(len(years)):
        key = (int(years[index]), int(months[index]))
        if key not in mapping:
            mapping[key] = index
    return mapping


def build_tile_features(
    tile: TileData,
    target_years: tuple[int, ...],
    *,
    use_cache: bool = True,
) -> TileFeatures:
    if use_cache:
        cached = _load_feature_cache(tile.metadata.path, target_years)
        if cached is not None:
            feature_cube, feature_names, persistence_score, persistence_mask = cached
            return TileFeatures(
                feature_cube=feature_cube,
                feature_names=feature_names,
                forest_mask=tile.forest_mask,
                labels=tile.label_consensus,
                weights=tile.label_weight,
                persistence_score=persistence_score,
                persistence_mask=persistence_mask,
                alert_year=tile.label_alert_year,
            )

    # --- S2 ---
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
    index_yoy_delta = _year_over_year_deltas(index_yearly)

    ndvi_min_overall = _safe_nanmin(ndvi_monthly, axis=0)
    nbr_min_overall = _safe_nanmin(nbr_monthly, axis=0)
    ndmi_min_overall = _safe_nanmin(ndmi_monthly, axis=0)
    ndvi_max_overall = _safe_nanmax(ndvi_monthly, axis=0)
    nbr_max_overall = _safe_nanmax(nbr_monthly, axis=0)
    ndmi_max_overall = _safe_nanmax(ndmi_monthly, axis=0)
    temporal_min_max = np.stack(
        [
            ndvi_min_overall.astype(np.float32),
            nbr_min_overall.astype(np.float32),
            ndmi_min_overall.astype(np.float32),
            ndvi_max_overall.astype(np.float32),
            nbr_max_overall.astype(np.float32),
            ndmi_max_overall.astype(np.float32),
        ],
        axis=0,
    )

    ndvi_slope = _temporal_slope(ndvi_monthly)
    nbr_slope = _temporal_slope(nbr_monthly)
    ndmi_slope = _temporal_slope(ndmi_monthly)
    temporal_slope = np.stack([ndvi_slope, nbr_slope, ndmi_slope], axis=0)

    # --- S1: impute NaN with per-pixel temporal median before aggregation ---
    s1_asc_imputed = _impute_nan_with_median(tile.s1_ascending_data)
    s1_desc_imputed = _impute_nan_with_median(tile.s1_descending_data)

    s1_asc = _yearly_reduce(s1_asc_imputed, tile.s1_ascending_year, target_years, reducer="median")
    s1_desc = _yearly_reduce(s1_desc_imputed, tile.s1_descending_year, target_years, reducer="median")
    s1_delta = np.concatenate([s1_asc[-1] - s1_asc[0], s1_desc[-1] - s1_desc[0]], axis=0)
    s1_std = np.concatenate(
        [
            _reduce_or_zeros(s1_asc_imputed, reducer="std"),
            _reduce_or_zeros(s1_desc_imputed, reducer="std"),
        ],
        axis=0,
    )

    # S1 data-availability flags: 1 where original data existed, 0 where it was all-NaN
    s1_asc_avail = (~np.all(np.isnan(tile.s1_ascending_data), axis=0)).astype(np.float32)
    s1_desc_avail = (~np.all(np.isnan(tile.s1_descending_data), axis=0)).astype(np.float32)

    # --- AEF ---
    aef_yearly = _align_yearly_observations(tile.aef_data, tile.aef_year, target_years)
    aef_mean = _safe_nanmean(aef_yearly, axis=0)
    aef_delta = aef_yearly[-1] - aef_yearly[0]
    aef_yearly_flat = aef_yearly.reshape(-1, *tile.forest_mask.shape)

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
        index_yoy_delta.reshape(-1, *tile.forest_mask.shape) if index_yoy_delta.size else np.zeros((0, *tile.forest_mask.shape), dtype=np.float32),
        temporal_min_max,
        temporal_slope,
        s1_asc.reshape(-1, *tile.forest_mask.shape),
        s1_desc.reshape(-1, *tile.forest_mask.shape),
        s1_delta,
        s1_std,
        s1_asc_avail,
        s1_desc_avail,
        aef_mean,
        aef_delta,
        aef_yearly_flat,
        persistence.feature_layers,
    ]
    feature_cube = np.concatenate(feature_arrays, axis=0).astype(np.float32)
    feature_cube = np.nan_to_num(feature_cube, nan=0.0, posinf=0.0, neginf=0.0)

    yoy_pairs = [
        f"{target_years[i]}_{target_years[i + 1]}"
        for i in range(len(target_years) - 1)
    ]
    feature_names = (
        tuple(f"s2_median_y{year}_b{band:02d}" for year in target_years for band in range(1, 13))
        + tuple(f"s2_delta_b{band:02d}" for band in range(1, 13))
        + tuple(f"s2_std_b{band:02d}" for band in range(1, 13))
        + tuple(f"{index}_median_y{year}" for year in target_years for index in ("ndvi", "nbr", "ndmi"))
        + tuple(f"{index}_delta" for index in ("ndvi", "nbr", "ndmi"))
        + tuple(f"{index}_std" for index in ("ndvi", "nbr", "ndmi"))
        + tuple(f"{index}_yoy_{pair}" for pair in yoy_pairs for index in ("ndvi", "nbr", "ndmi"))
        + (
            "ndvi_min",
            "nbr_min",
            "ndmi_min",
            "ndvi_max",
            "nbr_max",
            "ndmi_max",
            "ndvi_slope",
            "nbr_slope",
            "ndmi_slope",
        )
        + tuple(f"s1_asc_median_y{year}" for year in target_years)
        + tuple(f"s1_desc_median_y{year}" for year in target_years)
        + ("s1_asc_delta", "s1_desc_delta", "s1_asc_std", "s1_desc_std")
        + ("s1_asc_avail", "s1_desc_avail")
        + tuple(f"aef_mean_{index:02d}" for index in range(64))
        + tuple(f"aef_delta_{index:02d}" for index in range(64))
        + tuple(f"aef_y{year}_{index:02d}" for year in target_years for index in range(64))
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

    if use_cache:
        _save_feature_cache(
            tile.metadata.path,
            target_years,
            feature_cube,
            feature_names,
            persistence.persistence_score,
            persistence.persistence_mask,
        )

    return TileFeatures(
        feature_cube=feature_cube,
        feature_names=feature_names,
        forest_mask=tile.forest_mask,
        labels=persistence.training_labels,
        weights=persistence.training_weights,
        persistence_score=persistence.persistence_score,
        persistence_mask=persistence.persistence_mask,
        alert_year=tile.label_alert_year,
    )


def _cache_path(tile_path: Path) -> Path:
    return tile_path.parent / (tile_path.stem + ".features.zarr")


def _load_feature_cache(
    tile_path: Path,
    target_years: tuple[int, ...],
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray, np.ndarray] | None:
    path = _cache_path(tile_path)
    if not path.exists():
        return None
    try:
        z = zarr.open_group(str(path), mode="r")
        if list(z.attrs.get("target_years", [])) != list(target_years):
            return None
        feature_cube = z["cube"][:].astype(np.float32)
        feature_names = tuple(z.attrs["feature_names"])
        persistence_score = z["persistence_score"][:].astype(np.float32)
        persistence_mask = z["persistence_mask"][:].astype(bool)
        return feature_cube, feature_names, persistence_score, persistence_mask
    except Exception as exc:
        LOGGER.warning("Feature cache load failed for %s: %s", tile_path.name, exc)
        return None


def _save_feature_cache(
    tile_path: Path,
    target_years: tuple[int, ...],
    feature_cube: np.ndarray,
    feature_names: tuple[str, ...],
    persistence_score: np.ndarray,
    persistence_mask: np.ndarray,
) -> None:
    path = _cache_path(tile_path)
    try:
        z = zarr.open_group(str(path), mode="w")
        z.create_dataset(
            "cube",
            data=feature_cube,
            chunks=(1, 256, 256),
            compressor=_CACHE_COMPRESSOR,
            overwrite=True,
        )
        z.create_dataset(
            "persistence_score",
            data=persistence_score,
            chunks=(256, 256),
            compressor=_CACHE_COMPRESSOR,
            overwrite=True,
        )
        z.create_dataset(
            "persistence_mask",
            data=persistence_mask,
            chunks=(256, 256),
            compressor=_CACHE_COMPRESSOR,
            overwrite=True,
        )
        z.attrs["target_years"] = list(target_years)
        z.attrs["feature_names"] = list(feature_names)
    except Exception as exc:
        LOGGER.warning("Feature cache save failed for %s: %s", tile_path.name, exc)


def compute_feature_norm_stats(
    tile_paths: list[Path],
    target_years: tuple[int, ...],
    output_path: Path,
) -> None:
    """Compute per-feature mean/std from a list of tile zarr paths and save as JSON.

    Loads feature cubes (using cache if available) and computes running statistics
    over all valid (forest & weight>0) pixels across all tiles.
    """
    from .cache import load_tile_data

    sum_x = None
    sum_x2 = None
    count = 0

    for tile_path in tile_paths:
        LOGGER.info("Computing norm stats for %s", tile_path.name)
        tile = load_tile_data(tile_path)
        tf = build_tile_features(tile, target_years)
        valid = tf.forest_mask
        if tf.weights is not None:
            valid = valid & (tf.weights > 0)
        rows, cols = np.nonzero(valid)
        if rows.size == 0:
            continue
        samples = tf.feature_cube[:, rows, cols].T  # (N, C)
        if sum_x is None:
            n_features = samples.shape[1]
            sum_x = np.zeros(n_features, dtype=np.float64)
            sum_x2 = np.zeros(n_features, dtype=np.float64)
        sum_x += samples.sum(axis=0).astype(np.float64)
        sum_x2 += (samples ** 2).sum(axis=0).astype(np.float64)
        count += samples.shape[0]

    if sum_x is None or count == 0:
        raise ValueError("No valid pixels found across provided tiles.")

    mean = (sum_x / count).astype(np.float32)
    variance = (sum_x2 / count - (sum_x / count) ** 2).astype(np.float32)
    std = np.sqrt(np.maximum(variance, 0.0)).astype(np.float32)
    std[std < 1e-6] = 1.0

    # Get feature names from last tile
    tile = load_tile_data(tile_paths[-1])
    tf = build_tile_features(tile, target_years)
    feature_names = list(tf.feature_names)

    stats = {
        "n_pixels": int(count),
        "n_features": int(len(mean)),
        "feature_names": feature_names,
        "mean": mean.tolist(),
        "std": std.tolist(),
    }
    output_path.write_text(json.dumps(stats, indent=2))
    LOGGER.info("Saved feature norm stats (%d features, %d pixels) to %s", len(mean), count, output_path)


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

    weights[~forest_mask] = 0.0

    # Use raw consensus labels (≥2 sources agree) as training targets.
    # Gating by spectral thresholds here creates circular feature-label leakage
    # and causes the model to fail on non-tropical biomes where NDVI/NBR drop
    # patterns differ from the training region.  The persistence score is still
    # present as a feature, so the model can learn to use it as one signal.
    training_labels = labels.astype(np.uint8)

    # Suppress pixels flagged as positive by only one source AND showing no
    # spectral signal at all — these are likely labelling artefacts.
    single_source_no_signal = (votes == 1) & (labels == 0) & (persistence_score < 0.05)
    weights[single_source_no_signal] *= 0.5

    # Zero-weight likely-unlabelled positives (high spectral drop but no alert):
    # the model should not learn them as negatives since they may be real events.
    likely_unlabeled_positive = (votes == 0) & forest_mask & (persistence_score >= 0.75)
    weights[likely_unlabeled_positive] = 0.0

    return PersistenceTargets(
        training_labels=training_labels,
        training_weights=weights,
        persistence_score=persistence_score,
        persistence_mask=persistence_mask,
        feature_layers=feature_layers,
    )


def _impute_nan_with_median(data: np.ndarray) -> np.ndarray:
    """Replace NaN values with per-pixel temporal median; fall back to global median."""
    if not np.isnan(data).any():
        return data
    out = data.copy()
    # Per-pixel median across time (axis=0): shape (C, H, W) or (1, H, W)
    pixel_median = _safe_nanmedian(data, axis=0)
    # Global fallback for pixels that are always NaN
    finite_vals = data[np.isfinite(data)]
    global_median = float(np.median(finite_vals)) if finite_vals.size > 0 else 0.0
    pixel_median = np.where(np.isnan(pixel_median), global_median, pixel_median)
    # Broadcast pixel_median to (T, C, H, W) and fill NaN positions
    nan_mask = np.isnan(out)
    out[nan_mask] = np.broadcast_to(pixel_median[np.newaxis], out.shape)[nan_mask]
    return out


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
        elif reducer == "min":
            available[int(year)] = _safe_nanmin(year_values, axis=0)
        elif reducer == "max":
            available[int(year)] = _safe_nanmax(year_values, axis=0)
        else:
            raise ValueError(f"Unsupported reducer: {reducer}")
    return _align_year_maps(available, target_years)


def _year_over_year_deltas(yearly_values: np.ndarray) -> np.ndarray:
    if yearly_values.shape[0] < 2:
        return np.zeros((0, *yearly_values.shape[1:]), dtype=np.float32)
    return (yearly_values[1:] - yearly_values[:-1]).astype(np.float32)


def _temporal_slope(monthly_values: np.ndarray) -> np.ndarray:
    if monthly_values.ndim != 3:
        raise ValueError("temporal_slope expects a (T, H, W) array")
    height, width = monthly_values.shape[1], monthly_values.shape[2]
    if monthly_values.shape[0] < 2:
        return np.zeros((height, width), dtype=np.float32)

    valid = np.isfinite(monthly_values)
    n = valid.sum(axis=0).astype(np.float32)
    safe_values = np.where(valid, monthly_values.astype(np.float32), 0.0)
    time_index = np.arange(monthly_values.shape[0], dtype=np.float32).reshape(-1, 1, 1)
    safe_time = np.where(valid, time_index, 0.0)

    sum_t = safe_time.sum(axis=0)
    sum_v = safe_values.sum(axis=0)
    sum_tv = (safe_time * safe_values).sum(axis=0)
    sum_t2 = (safe_time * safe_time).sum(axis=0)

    denom = n * sum_t2 - sum_t * sum_t
    numer = n * sum_tv - sum_t * sum_v
    slope = np.zeros((height, width), dtype=np.float32)
    enough = (n >= 2) & (np.abs(denom) > 1e-6)
    slope[enough] = (numer[enough] / denom[enough]).astype(np.float32)
    return slope


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


def compute_year_heuristic_map(tile: "TileData", target_years: tuple[int, ...]) -> np.ndarray:
    """Per-pixel alert year from the largest year-over-year NDVI+NBR drop.

    Fallback for time_step prediction when no year-classification head is available.
    Returns float32 (H, W); pixels with no valid S2 data get the last target year.
    """
    height, width = tile.forest_mask.shape
    last_year = float(target_years[-1]) if target_years else 0.0
    if len(target_years) < 2:
        return np.full((height, width), last_year, dtype=np.float32)

    s2 = tile.s2_data / S2_SCALE
    ndvi = _normalized_difference(s2[:, S2_NIR_INDEX], s2[:, S2_RED_INDEX])
    nbr = _normalized_difference(s2[:, S2_NIR_INDEX], s2[:, S2_SWIR2_INDEX])
    ndvi_yearly = _yearly_reduce(ndvi[:, None], tile.s2_year, target_years, reducer="median")[:, 0]
    nbr_yearly = _yearly_reduce(nbr[:, None], tile.s2_year, target_years, reducer="median")[:, 0]

    yoy_drop = (ndvi_yearly[:-1] - ndvi_yearly[1:]) + (nbr_yearly[:-1] - nbr_yearly[1:])
    yoy_drop = np.nan_to_num(yoy_drop, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
    best_idx = yoy_drop.argmax(axis=0)
    candidate_years = np.asarray(target_years[1:], dtype=np.float32)
    year_map = candidate_years[best_idx]
    no_signal = np.all(~np.isfinite(yoy_drop) | (yoy_drop == -np.inf), axis=0)
    year_map[no_signal] = last_year
    return year_map.astype(np.float32)


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


def _safe_nanmin(values: np.ndarray, axis: int = 0) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        result = np.nanmin(values, axis=axis)
    return np.where(np.isfinite(result), result, 0.0).astype(np.float32)


def _safe_nanmax(values: np.ndarray, axis: int = 0) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        result = np.nanmax(values, axis=axis)
    return np.where(np.isfinite(result), result, 0.0).astype(np.float32)


def _scaled_score(values: np.ndarray, *, low: float, high: float) -> np.ndarray:
    if high <= low:
        raise ValueError("high must be greater than low")
    return np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)
