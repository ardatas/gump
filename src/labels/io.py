"""Filesystem discovery for RADD, GLAD-L, and GLAD-S2 label files."""

from __future__ import annotations

import re
from pathlib import Path


def radd_path(labels_root: Path, tile: str) -> Path | None:
    p = Path(labels_root) / "radd" / f"radd_{tile}_labels.tif"
    return p if p.exists() else None


def gladl_paths(labels_root: Path, tile: str) -> list[tuple[int, Path, Path]]:
    """Return [(yy, alert_path, alertDate_path), ...] sorted by yy."""
    d = Path(labels_root) / "gladl"
    out = []
    for p in d.glob(f"gladl_{tile}_alert[0-9][0-9].tif"):
        m = re.search(r"alert(\d{2})\.tif$", p.name)
        if not m:
            continue
        yy = int(m.group(1))
        date_p = d / f"gladl_{tile}_alertDate{yy:02d}.tif"
        if date_p.exists():
            out.append((yy, p, date_p))
    out.sort(key=lambda x: x[0])
    return out


def glads2_paths(labels_root: Path, tile: str) -> tuple[Path, Path] | None:
    d = Path(labels_root) / "glads2"
    alert_p = d / f"glads2_{tile}_alert.tif"
    date_p = d / f"glads2_{tile}_alertDate.tif"
    if alert_p.exists() and date_p.exists():
        return alert_p, date_p
    return None
