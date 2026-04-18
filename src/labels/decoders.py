"""Decode raw label rasters → (is_defo: bool[H,W], alert_year: uint16[H,W]).

All decoders receive arrays already warped onto the target grid.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np

# ── pre-computed day offsets ────────────────────────────────────────────────
# RADD epoch: 2014-12-31
_RADD_EPOCH = date(2014, 12, 31)
# First day >= 2020-01-01 expressed as RADD days
_RADD_2020 = (date(2020, 1, 1) - _RADD_EPOCH).days  # 1827

# GLAD-S2 epoch: 2019-01-01
_GLADS2_EPOCH = date(2019, 1, 1)
_GLADS2_2020 = (date(2020, 1, 1) - _GLADS2_EPOCH).days  # 365


def _days_to_year(days: np.ndarray, epoch: date) -> np.ndarray:
    """Vectorised: convert integer day-offsets from epoch to calendar year."""
    years = np.zeros_like(days, dtype=np.uint16)
    # Iterate year boundaries; covers 2020-2030 generously.
    for yr in range(2020, 2031):
        lo = (date(yr, 1, 1) - epoch).days
        hi = (date(yr + 1, 1, 1) - epoch).days
        years[(days >= lo) & (days < hi)] = yr
    return years


def decode_radd(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """RADD: uint16 encoded as conf*10000 + days_since_2014-12-31."""
    raw = raw.astype(np.int32)
    conf = raw // 10000   # 0, 2, or 3
    days = raw % 10000
    is_defo = (conf >= 2) & (days >= _RADD_2020)
    alert_year = np.where(is_defo, _days_to_year(days, _RADD_EPOCH), 0).astype(np.uint16)
    return is_defo.astype(bool), alert_year


def decode_gladl(alert: np.ndarray, alert_date: np.ndarray, yy: int) -> tuple[np.ndarray, np.ndarray]:
    """GLAD-L: alert in {0,2,3}, alertDate = DOY in year 2000+yy."""
    alert = alert.astype(np.int16)
    # All years >= 20 map to >= 2020; no date threshold needed.
    is_defo = (alert >= 2) & (alert_date > 0)
    year = 2000 + yy
    alert_year = np.where(is_defo, year, 0).astype(np.uint16)
    return is_defo.astype(bool), alert_year


def decode_glads2(alert: np.ndarray, alert_date: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """GLAD-S2: alert in {0-4} (1=last-obs-only treated as uncertain), days since 2019-01-01."""
    alert = alert.astype(np.int16)
    alert_date = alert_date.astype(np.int32)
    is_defo = (alert >= 2) & (alert_date >= _GLADS2_2020)
    alert_year = np.where(is_defo, _days_to_year(alert_date, _GLADS2_EPOCH), 0).astype(np.uint16)
    return is_defo.astype(bool), alert_year
