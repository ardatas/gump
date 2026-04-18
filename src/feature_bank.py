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


def build_tile_features(
    tile: TileData,
    target_years: tuple[int, ...],
    *,
    use_cache: bool = True,
) -> TileFeatures:
    if use_cache:
        cached = _load_feature_cache(tile.metadata.path, target_years)
        if cached is not None:
            feature_cube, feature_names = cached
            return TileFeatures(
                feature_cube=feature_cube,
                feature_names=feature_names,
                forest_mask=tile.forest_mask,
                labels=tile.label_consensus,
                weights=tile.label_weight,
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
        s1_asc_avail,
        s1_desc_avail,
        aef_mean,
        aef_delta,
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
        + ("s1_asc_avail", "s1_desc_avail")
        + tuple(f"aef_mean_{index:02d}" for index in range(64))
        + tuple(f"aef_delta_{index:02d}" for index in range(64))
    )

    if use_cache:
        _save_feature_cache(tile.metadata.path, target_years, feature_cube, feature_names)

    return TileFeatures(
        feature_cube=feature_cube,
        feature_names=feature_names,
        forest_mask=tile.forest_mask,
        labels=tile.label_consensus,
        weights=tile.label_weight,
    )


def _cache_path(tile_path: Path) -> Path:
    return tile_path.parent / (tile_path.stem + ".features.zarr")


def _load_feature_cache(
    tile_path: Path,
    target_years: tuple[int, ...],
) -> tuple[np.ndarray, tuple[str, ...]] | None:
    path = _cache_path(tile_path)
    if not path.exists():
        return None
    try:
        z = zarr.open_group(str(path), mode="r")
        if list(z.attrs.get("target_years", [])) != list(target_years):
            return None
        feature_cube = z["cube"][:].astype(np.float32)
        feature_names = tuple(z.attrs["feature_names"])
        return feature_cube, feature_names
    except Exception as exc:
        LOGGER.warning("Feature cache load failed for %s: %s", tile_path.name, exc)
        return None


def _save_feature_cache(
    tile_path: Path,
    target_years: tuple[int, ...],
    feature_cube: np.ndarray,
    feature_names: tuple[str, ...],
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
