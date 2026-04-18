from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import zarr


@dataclass(frozen=True)
class TileMetadata:
    tile_id: str
    split: str
    path: Path
    region: str
    crs: str
    transform: tuple[float, float, float, float, float, float]
    width: int
    height: int
    available_years: tuple[int, ...]


@dataclass
class TileData:
    metadata: TileMetadata
    s2_data: np.ndarray
    s2_year: np.ndarray
    s2_month: np.ndarray
    s1_ascending_data: np.ndarray
    s1_ascending_year: np.ndarray
    s1_ascending_month: np.ndarray
    s1_descending_data: np.ndarray
    s1_descending_year: np.ndarray
    s1_descending_month: np.ndarray
    aef_data: np.ndarray
    aef_year: np.ndarray
    forest_mask: np.ndarray
    label_consensus: np.ndarray | None = None
    label_weight: np.ndarray | None = None
    label_votes: np.ndarray | None = None
    label_alert_year: np.ndarray | None = None
    label_source_mask: np.ndarray | None = None


def list_tile_paths(cache_root: Path, split: str) -> list[Path]:
    return sorted(
        path
        for path in (cache_root / split).glob("*.zarr")
        if path.is_dir()
    )


def load_tile_metadata(tile_path: Path) -> TileMetadata:
    root = zarr.open_group(tile_path, mode="r")
    tile_id = str(root.attrs["tile_id"])
    split = str(root.attrs["split"])
    mgrs_grid = tile_id.split("_")[0]
    available_years = sorted(
        set(root["s2/year"][:].astype(np.int16).tolist())
        | set(root["aef/year"][:].astype(np.int16).tolist())
    )
    return TileMetadata(
        tile_id=tile_id,
        split=split,
        path=tile_path,
        region=mgrs_grid[:3],
        crs=str(root.attrs["crs"]),
        transform=tuple(root.attrs["transform"]),
        width=int(root.attrs["width"]),
        height=int(root.attrs["height"]),
        available_years=tuple(int(year) for year in available_years),
    )


def load_tile_data(tile_path: Path) -> TileData:
    metadata = load_tile_metadata(tile_path)
    root = zarr.open_group(tile_path, mode="r")
    labels = root["labels"] if "labels" in root.group_keys() else None

    return TileData(
        metadata=metadata,
        s2_data=root["s2/data"][:].astype(np.float32),
        s2_year=root["s2/year"][:].astype(np.int16),
        s2_month=root["s2/month"][:].astype(np.int8),
        s1_ascending_data=root["s1_ascending/data"][:].astype(np.float32),
        s1_ascending_year=root["s1_ascending/year"][:].astype(np.int16),
        s1_ascending_month=root["s1_ascending/month"][:].astype(np.int8),
        s1_descending_data=root["s1_descending/data"][:].astype(np.float32),
        s1_descending_year=root["s1_descending/year"][:].astype(np.int16),
        s1_descending_month=root["s1_descending/month"][:].astype(np.int8),
        aef_data=root["aef/data"][:].astype(np.float32),
        aef_year=root["aef/year"][:].astype(np.int16),
        forest_mask=root["forest_2020"][:].astype(bool),
        label_consensus=None if labels is None else labels["consensus"][:].astype(np.uint8),
        label_weight=None if labels is None else labels["weight"][:].astype(np.float32),
        label_votes=None if labels is None else labels["votes"][:].astype(np.uint8),
        label_alert_year=None if labels is None else labels["alert_year"][:].astype(np.uint16),
        label_source_mask=None if labels is None else labels["source_mask"][:].astype(np.uint8),
    )


def discover_target_years(cache_root: Path, splits: tuple[str, ...] = ("train", "test")) -> tuple[int, ...]:
    years: set[int] = set()
    for split in splits:
        for path in list_tile_paths(cache_root, split):
            years.update(load_tile_metadata(path).available_years)
    return tuple(sorted(years))


def choose_validation_regions(
    metadata: list[TileMetadata],
    desired_fraction: float = 0.2,
) -> tuple[str, ...]:
    region_counts: dict[str, int] = {}
    for item in metadata:
        region_counts[item.region] = region_counts.get(item.region, 0) + 1

    target = max(1.0, len(metadata) * desired_fraction)
    best_region = min(
        region_counts,
        key=lambda region: (
            abs(region_counts[region] - target),
            -region_counts[region],
            region,
        ),
    )
    return (best_region,)
