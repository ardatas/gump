from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re


S2_FILENAME_RE = re.compile(
    r"(?P<tile>.+)__s2_l2a_(?P<year>\d{4})_(?P<month>\d{1,2})\.tif$"
)
S1_FILENAME_RE = re.compile(
    r"(?P<tile>.+)__s1_rtc_(?P<year>\d{4})_(?P<month>\d{1,2})_(?P<orbit>ascending|descending)\.tif$"
)
AEF_FILENAME_RE = re.compile(r"(?P<tile>.+)_(?P<year>\d{4})\.tiff?$")
GLADL_ALERT_RE = re.compile(r"gladl_(?P<tile>.+)_alert(?P<yy>\d{2})\.tif$")
GLADL_DATE_RE = re.compile(r"gladl_(?P<tile>.+)_alertDate(?P<yy>\d{2})\.tif$")


@dataclass(frozen=True)
class RasterTimePoint:
    path: Path
    year: int
    month: int | None = None
    orbit: str | None = None


@dataclass(frozen=True)
class GladlYearPaths:
    year: int
    alert_path: Path
    alert_date_path: Path


@dataclass
class TileSources:
    tile_id: str
    split: str
    s2: list[RasterTimePoint] = field(default_factory=list)
    s1_ascending: list[RasterTimePoint] = field(default_factory=list)
    s1_descending: list[RasterTimePoint] = field(default_factory=list)
    aef: list[RasterTimePoint] = field(default_factory=list)
    radd_path: Path | None = None
    gladl: list[GladlYearPaths] = field(default_factory=list)
    glads2_alert_path: Path | None = None
    glads2_alert_date_path: Path | None = None

    def sort_in_place(self) -> None:
        self.s2.sort(key=lambda item: (item.year, item.month or 0, item.path.name))
        self.s1_ascending.sort(key=lambda item: (item.year, item.month or 0, item.path.name))
        self.s1_descending.sort(key=lambda item: (item.year, item.month or 0, item.path.name))
        self.aef.sort(key=lambda item: (item.year, item.path.name))
        self.gladl.sort(key=lambda item: item.year)


def discover_tiles(data_root: Path, split: str = "train") -> list[TileSources]:
    split_names = ["train", "test"] if split == "all" else [split]
    discovered: list[TileSources] = []
    for split_name in split_names:
        discovered.extend(_discover_split_tiles(data_root, split_name))
    return sorted(discovered, key=lambda item: (item.split, item.tile_id))


def _discover_split_tiles(data_root: Path, split: str) -> list[TileSources]:
    s2_root = data_root / "sentinel-2" / split
    s1_root = data_root / "sentinel-1" / split
    aef_root = data_root / "aef-embeddings" / split
    labels_root = data_root / "labels" / "train"

    tiles: dict[str, TileSources] = {}

    for directory in sorted(s2_root.glob("*__s2_l2a")):
        if not directory.is_dir():
            continue
        tile_id = directory.name.removesuffix("__s2_l2a")
        tiles[tile_id] = TileSources(tile_id=tile_id, split=split)
        for path in sorted(directory.glob("*.tif")):
            match = S2_FILENAME_RE.match(path.name)
            if not match:
                continue
            tiles[tile_id].s2.append(
                RasterTimePoint(
                    path=path,
                    year=int(match.group("year")),
                    month=int(match.group("month")),
                )
            )

    for directory in sorted(s1_root.glob("*__s1_rtc")):
        if not directory.is_dir():
            continue
        tile_id = directory.name.removesuffix("__s1_rtc")
        tile = tiles.setdefault(tile_id, TileSources(tile_id=tile_id, split=split))
        for path in sorted(directory.glob("*.tif")):
            match = S1_FILENAME_RE.match(path.name)
            if not match:
                continue
            observation = RasterTimePoint(
                path=path,
                year=int(match.group("year")),
                month=int(match.group("month")),
                orbit=match.group("orbit"),
            )
            if observation.orbit == "ascending":
                tile.s1_ascending.append(observation)
            else:
                tile.s1_descending.append(observation)

    if aef_root.exists():
        for path in sorted(aef_root.glob("*.tif*")):
            match = AEF_FILENAME_RE.match(path.name)
            if not match:
                continue
            tile_id = match.group("tile")
            tile = tiles.setdefault(tile_id, TileSources(tile_id=tile_id, split=split))
            tile.aef.append(
                RasterTimePoint(path=path, year=int(match.group("year")))
            )

    if split == "train" and labels_root.exists():
        _attach_train_labels(labels_root, tiles)

    result = [tile for tile in tiles.values() if tile.s2]
    for tile in result:
        tile.sort_in_place()
    return result


def _attach_train_labels(labels_root: Path, tiles: dict[str, TileSources]) -> None:
    radd_root = labels_root / "radd"
    if radd_root.exists():
        for path in sorted(radd_root.glob("radd_*_labels.tif")):
            tile_id = path.name.removeprefix("radd_").removesuffix("_labels.tif")
            tile = tiles.setdefault(tile_id, TileSources(tile_id=tile_id, split="train"))
            tile.radd_path = path

    gladl_alerts: dict[tuple[str, int], Path] = {}
    gladl_dates: dict[tuple[str, int], Path] = {}
    gladl_root = labels_root / "gladl"
    if gladl_root.exists():
        for path in sorted(gladl_root.glob("*.tif")):
            alert_match = GLADL_ALERT_RE.match(path.name)
            if alert_match:
                year = 2000 + int(alert_match.group("yy"))
                gladl_alerts[(alert_match.group("tile"), year)] = path
                continue
            date_match = GLADL_DATE_RE.match(path.name)
            if date_match:
                year = 2000 + int(date_match.group("yy"))
                gladl_dates[(date_match.group("tile"), year)] = path

    for key, alert_path in gladl_alerts.items():
        tile_id, year = key
        date_path = gladl_dates.get(key)
        if date_path is None:
            continue
        tile = tiles.setdefault(tile_id, TileSources(tile_id=tile_id, split="train"))
        tile.gladl.append(
            GladlYearPaths(
                year=year,
                alert_path=alert_path,
                alert_date_path=date_path,
            )
        )

    glads2_root = labels_root / "glads2"
    if glads2_root.exists():
        for path in sorted(glads2_root.glob("glads2_*_alert.tif")):
            tile_id = path.name.removeprefix("glads2_").removesuffix("_alert.tif")
            tile = tiles.setdefault(tile_id, TileSources(tile_id=tile_id, split="train"))
            tile.glads2_alert_path = path
        for path in sorted(glads2_root.glob("glads2_*_alertDate.tif")):
            tile_id = path.name.removeprefix("glads2_").removesuffix("_alertDate.tif")
            tile = tiles.setdefault(tile_id, TileSources(tile_id=tile_id, split="train"))
            tile.glads2_alert_date_path = path
