from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rasterio.enums import Resampling

from .index import GladlYearPaths, TileSources
from .raster_utils import RasterGrid, read_reprojected


RADD_BASE_DATE = np.datetime64("2014-12-31")
GLADS2_BASE_DATE = np.datetime64("2019-01-01")
TARGET_START_DATE = np.datetime64("2020-01-01")


@dataclass(frozen=True)
class LabelProducts:
    consensus: np.ndarray
    weight: np.ndarray
    votes: np.ndarray
    source_mask: np.ndarray
    alert_year: np.ndarray


def build_label_products(
    tile: TileSources,
    grid: RasterGrid,
    forest_mask: np.ndarray,
    *,
    positive_vote_threshold: int = 2,
    single_source_weight: float = 0.3,
    double_source_weight: float = 0.8,
    recent_only_weight: float = 0.2,
) -> LabelProducts:
    if tile.split != "train":
        zeros_uint8 = np.zeros(grid.shape, dtype=np.uint8)
        zeros_year = np.zeros(grid.shape, dtype=np.uint16)
        zeros_weight = np.zeros(grid.shape, dtype=np.float32)
        return LabelProducts(
            consensus=zeros_uint8,
            weight=zeros_weight,
            votes=zeros_uint8,
            source_mask=zeros_uint8,
            alert_year=zeros_year,
        )

    radd_positive, radd_year = decode_radd(tile.radd_path, grid)
    gladl_positive, gladl_year = decode_gladl(tile.gladl, grid)
    glads2_positive, glads2_recent_only, glads2_year = decode_glads2(
        tile.glads2_alert_path,
        tile.glads2_alert_date_path,
        grid,
    )

    votes = (
        radd_positive.astype(np.uint8)
        + gladl_positive.astype(np.uint8)
        + glads2_positive.astype(np.uint8)
    )
    consensus = (votes >= positive_vote_threshold).astype(np.uint8)

    weight = np.ones(grid.shape, dtype=np.float32)
    weight[votes == 1] = single_source_weight
    weight[votes == 2] = double_source_weight
    weight[glads2_recent_only] = np.minimum(weight[glads2_recent_only], recent_only_weight)
    weight[~forest_mask] = 0.0

    source_mask = np.zeros(grid.shape, dtype=np.uint8)
    source_mask |= radd_positive.astype(np.uint8) * 1
    source_mask |= gladl_positive.astype(np.uint8) * 2
    source_mask |= glads2_positive.astype(np.uint8) * 4
    source_mask |= glads2_recent_only.astype(np.uint8) * 8

    alert_year = earliest_year(radd_year, gladl_year, glads2_year)
    alert_year[~forest_mask] = 0

    return LabelProducts(
        consensus=consensus,
        weight=weight,
        votes=votes.astype(np.uint8),
        source_mask=source_mask,
        alert_year=alert_year,
    )


def decode_radd(path, grid: RasterGrid) -> tuple[np.ndarray, np.ndarray]:
    if path is None:
        return zeros_bool(grid), zeros_year(grid)

    encoded = read_reprojected(
        path,
        grid,
        resampling=Resampling.nearest,
        out_dtype=np.int32,
        fill_value=0,
    )[0]

    confidence = encoded // 10_000
    days_since_base = encoded % 10_000
    has_alert = np.isin(confidence, (2, 3)) & (days_since_base > 0)
    alert_dates = RADD_BASE_DATE + days_since_base.astype("timedelta64[D]")
    positive = has_alert & (alert_dates >= TARGET_START_DATE)

    years = np.zeros(grid.shape, dtype=np.uint16)
    years[positive] = _years_from_datetime64(alert_dates[positive])
    return positive, years


def decode_gladl(paths: list[GladlYearPaths], grid: RasterGrid) -> tuple[np.ndarray, np.ndarray]:
    positive = zeros_bool(grid)
    years = zeros_year(grid)

    for item in paths:
        alerts = read_reprojected(
            item.alert_path,
            grid,
            resampling=Resampling.nearest,
            out_dtype=np.uint8,
            fill_value=0,
        )[0]
        alert_dates = read_reprojected(
            item.alert_date_path,
            grid,
            resampling=Resampling.nearest,
            out_dtype=np.uint16,
            fill_value=0,
        )[0]
        current_positive = np.isin(alerts, (2, 3)) & (alert_dates > 0) & (item.year >= 2020)
        positive |= current_positive
        update_mask = current_positive & ((years == 0) | (item.year < years))
        years[update_mask] = item.year

    return positive, years


def decode_glads2(
    alert_path,
    alert_date_path,
    grid: RasterGrid,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if alert_path is None or alert_date_path is None:
        return zeros_bool(grid), zeros_bool(grid), zeros_year(grid)

    alerts = read_reprojected(
        alert_path,
        grid,
        resampling=Resampling.nearest,
        out_dtype=np.uint8,
        fill_value=0,
    )[0]
    alert_dates = read_reprojected(
        alert_date_path,
        grid,
        resampling=Resampling.nearest,
        out_dtype=np.uint16,
        fill_value=0,
    )[0]
    absolute_dates = GLADS2_BASE_DATE + alert_dates.astype("timedelta64[D]")

    valid_date = alert_dates > 0
    positive = np.isin(alerts, (2, 3, 4)) & valid_date & (absolute_dates >= TARGET_START_DATE)
    recent_only = (alerts == 1) & valid_date & (absolute_dates >= TARGET_START_DATE)

    years = np.zeros(grid.shape, dtype=np.uint16)
    years[positive] = _years_from_datetime64(absolute_dates[positive])
    return positive, recent_only, years


def earliest_year(*arrays: np.ndarray) -> np.ndarray:
    stacked = np.stack(arrays, axis=0).astype(np.uint16)
    masked = np.where(stacked == 0, np.iinfo(np.uint16).max, stacked)
    minimum = masked.min(axis=0)
    minimum[minimum == np.iinfo(np.uint16).max] = 0
    return minimum.astype(np.uint16)


def zeros_bool(grid: RasterGrid) -> np.ndarray:
    return np.zeros(grid.shape, dtype=bool)


def zeros_year(grid: RasterGrid) -> np.ndarray:
    return np.zeros(grid.shape, dtype=np.uint16)


def _years_from_datetime64(values: np.ndarray) -> np.ndarray:
    return (values.astype("datetime64[Y]").astype(np.int32) + 1970).astype(np.uint16)
