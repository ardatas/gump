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
    )

    return TileFeatures(
        feature_cube=feature_cube,
        feature_names=feature_names,
        forest_mask=tile.forest_mask,
        labels=tile.label_consensus,
        weights=tile.label_weight,
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
