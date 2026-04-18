"""Tile discovery and filesystem path parsing for the raw data tree.

Expected layout under ``data_root``::

    sentinel-2/{split}/{tile}__s2_l2a/{tile}__s2_l2a_{YYYY}_{M}.tif
    sentinel-1/{split}/{tile}__s1_rtc/{tile}__s1_rtc_{YYYY}_{M}_{ascending|descending}.tif
    aef-embeddings/{split}/{tile}_{YYYY}.tiff
"""

from __future__ import annotations

import re
from pathlib import Path

Split = str  # "train" | "test"

_S2_RE = re.compile(r"^(?P<tile>.+?)__s2_l2a_(?P<year>\d{4})_(?P<month>\d{1,2})\.tif$")
_S1_RE = re.compile(
    r"^(?P<tile>.+?)__s1_rtc_(?P<year>\d{4})_(?P<month>\d{1,2})_(?P<orbit>ascending|descending)\.tif$"
)
_AEF_RE = re.compile(r"^(?P<tile>.+?)_(?P<year>\d{4})\.tiff?$")


def list_tiles(data_root: Path, split: Split) -> list[str]:
    """Return sorted tile names present in ``sentinel-2/{split}/``."""
    s2_dir = Path(data_root) / "sentinel-2" / split
    suffix = "__s2_l2a"
    tiles = [p.name[: -len(suffix)] for p in s2_dir.iterdir() if p.is_dir() and p.name.endswith(suffix)]
    return sorted(tiles)


def s2_paths(data_root: Path, tile: str, split: Split) -> list[tuple[tuple[int, int], Path]]:
    """Return ``[((year, month), path), ...]`` sorted chronologically."""
    d = Path(data_root) / "sentinel-2" / split / f"{tile}__s2_l2a"
    out: list[tuple[tuple[int, int], Path]] = []
    for p in d.iterdir():
        m = _S2_RE.match(p.name)
        if m and m["tile"] == tile:
            out.append(((int(m["year"]), int(m["month"])), p))
    out.sort(key=lambda x: x[0])
    return out


def s1_paths(
    data_root: Path, tile: str, split: Split, orbit: str
) -> list[tuple[tuple[int, int], Path]]:
    """Return ``[((year, month), path), ...]`` for the given orbit, sorted."""
    assert orbit in ("ascending", "descending"), orbit
    d = Path(data_root) / "sentinel-1" / split / f"{tile}__s1_rtc"
    out: list[tuple[tuple[int, int], Path]] = []
    for p in d.iterdir():
        m = _S1_RE.match(p.name)
        if m and m["tile"] == tile and m["orbit"] == orbit:
            out.append(((int(m["year"]), int(m["month"])), p))
    out.sort(key=lambda x: x[0])
    return out


def aef_paths(data_root: Path, tile: str, split: Split) -> list[tuple[int, Path]]:
    """Return ``[(year, path), ...]`` sorted by year."""
    d = Path(data_root) / "aef-embeddings" / split
    out: list[tuple[int, Path]] = []
    for p in d.iterdir():
        m = _AEF_RE.match(p.name)
        if m and m["tile"] == tile:
            out.append((int(m["year"]), p))
    out.sort(key=lambda x: x[0])
    return out
