"""Tests for src/labels/decoders.py — RADD, GLAD-L, GLAD-S2 label decoding."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from src.labels.decoders import (
    _RADD_EPOCH,
    _GLADS2_EPOCH,
    _RADD_2020,
    _GLADS2_2020,
    decode_radd,
    decode_gladl,
    decode_glads2,
)


def _radd_encode(conf: int, event_date: date) -> int:
    days = (event_date - _RADD_EPOCH).days
    return conf * 10000 + days


def _glads2_days(event_date: date) -> int:
    return (event_date - _GLADS2_EPOCH).days


# ── RADD ─────────────────────────────────────────────────────────────────────

class TestDecodeRadd:
    def test_high_confidence_post2020_is_deforestation(self):
        """conf=3, date after 2020-01-01 → deforested."""
        val = _radd_encode(3, date(2021, 6, 15))
        raw = np.array([[val]], dtype=np.int32)
        is_defo, alert_year = decode_radd(raw)
        assert is_defo[0, 0] is np.bool_(True)
        assert alert_year[0, 0] == 2021

    def test_low_confidence_post2020_is_deforestation(self):
        """conf=2 (low confidence) still counts."""
        val = _radd_encode(2, date(2022, 3, 1))
        raw = np.array([[val]], dtype=np.int32)
        is_defo, _ = decode_radd(raw)
        assert is_defo[0, 0] is np.bool_(True)

    def test_pre2020_alert_not_deforestation(self):
        """Deforestation must be after 2020-01-01 per task definition."""
        val = _radd_encode(3, date(2019, 12, 31))
        raw = np.array([[val]], dtype=np.int32)
        is_defo, alert_year = decode_radd(raw)
        assert not is_defo[0, 0]
        assert alert_year[0, 0] == 0

    def test_exactly_2020_01_01_is_included(self):
        """Boundary: exactly 2020-01-01 must be considered deforestation."""
        val = _radd_encode(3, date(2020, 1, 1))
        raw = np.array([[val]], dtype=np.int32)
        is_defo, alert_year = decode_radd(raw)
        assert is_defo[0, 0]
        assert alert_year[0, 0] == 2020

    def test_conf_zero_not_deforestation(self):
        """conf=0 (no alert) → not deforestation even if date is valid."""
        val = _radd_encode(0, date(2021, 7, 1))
        raw = np.array([[val]], dtype=np.int32)
        is_defo, alert_year = decode_radd(raw)
        assert not is_defo[0, 0]
        assert alert_year[0, 0] == 0

    def test_all_zeros_input(self):
        raw = np.zeros((4, 4), dtype=np.int32)
        is_defo, alert_year = decode_radd(raw)
        assert not is_defo.any()
        assert (alert_year == 0).all()

    def test_output_shapes_match_input(self):
        raw = np.zeros((10, 20), dtype=np.int32)
        is_defo, alert_year = decode_radd(raw)
        assert is_defo.shape == (10, 20)
        assert alert_year.shape == (10, 20)

    def test_output_dtypes(self):
        raw = np.array([[_radd_encode(3, date(2021, 1, 1))]], dtype=np.int32)
        is_defo, alert_year = decode_radd(raw)
        assert is_defo.dtype == bool
        assert alert_year.dtype == np.uint16


# ── GLAD-L ───────────────────────────────────────────────────────────────────

class TestDecodeGladl:
    def test_confirmed_yy20_is_deforestation(self):
        """alert=3, YY=20 → deforested in year 2020."""
        alert = np.array([[3]], dtype=np.int16)
        alert_date = np.array([[180]], dtype=np.int16)  # DOY 180
        is_defo, alert_year = decode_gladl(alert, alert_date, yy=20)
        assert is_defo[0, 0]
        assert alert_year[0, 0] == 2020

    def test_probable_yy22_is_deforestation(self):
        """alert=2 (probable) also counts."""
        alert = np.array([[2]], dtype=np.int16)
        alert_date = np.array([[90]], dtype=np.int16)
        is_defo, alert_year = decode_gladl(alert, alert_date, yy=22)
        assert is_defo[0, 0]
        assert alert_year[0, 0] == 2022

    def test_no_alert_zero_date(self):
        """alert=0 or date=0 → not deforested."""
        alert = np.array([[0]], dtype=np.int16)
        alert_date = np.array([[0]], dtype=np.int16)
        is_defo, alert_year = decode_gladl(alert, alert_date, yy=21)
        assert not is_defo[0, 0]
        assert alert_year[0, 0] == 0

    def test_alert_date_zero_suppresses_deforestation(self):
        """Date == 0 means no valid alert date → treat as no detection."""
        alert = np.array([[3]], dtype=np.int16)
        alert_date = np.array([[0]], dtype=np.int16)
        is_defo, _ = decode_gladl(alert, alert_date, yy=21)
        assert not is_defo[0, 0]

    def test_output_shapes_match_input(self):
        alert = np.zeros((8, 8), dtype=np.int16)
        alert_date = np.zeros((8, 8), dtype=np.int16)
        is_defo, alert_year = decode_gladl(alert, alert_date, yy=21)
        assert is_defo.shape == (8, 8)
        assert alert_year.shape == (8, 8)

    def test_output_dtypes(self):
        alert = np.array([[3]], dtype=np.int16)
        alert_date = np.array([[100]], dtype=np.int16)
        is_defo, alert_year = decode_gladl(alert, alert_date, yy=21)
        assert is_defo.dtype == bool
        assert alert_year.dtype == np.uint16


# ── GLAD-S2 ──────────────────────────────────────────────────────────────────

class TestDecodeGlads2:
    def test_high_confidence_post2020(self):
        """alert=4 (max conf), date after 2020-01-01 → deforested."""
        days = _glads2_days(date(2021, 8, 1))
        alert = np.array([[4]], dtype=np.int16)
        alert_date = np.array([[days]], dtype=np.int32)
        is_defo, alert_year = decode_glads2(alert, alert_date)
        assert is_defo[0, 0]
        assert alert_year[0, 0] == 2021

    def test_alert_level_2_counts(self):
        days = _glads2_days(date(2022, 5, 10))
        alert = np.array([[2]], dtype=np.int16)
        alert_date = np.array([[days]], dtype=np.int32)
        is_defo, _ = decode_glads2(alert, alert_date)
        assert is_defo[0, 0]

    def test_alert_level_1_is_uncertain_not_counted(self):
        """alert=1 ('only in last observation') must NOT be treated as positive."""
        days = _glads2_days(date(2022, 5, 10))
        alert = np.array([[1]], dtype=np.int16)
        alert_date = np.array([[days]], dtype=np.int32)
        is_defo, alert_year = decode_glads2(alert, alert_date)
        assert not is_defo[0, 0], "alert=1 should be uncertain, not a positive detection"
        assert alert_year[0, 0] == 0

    def test_pre2020_not_deforestation(self):
        """Dates before 2020-01-01 must not be detected regardless of confidence."""
        days = _glads2_days(date(2019, 11, 30))
        alert = np.array([[4]], dtype=np.int16)
        alert_date = np.array([[days]], dtype=np.int32)
        is_defo, alert_year = decode_glads2(alert, alert_date)
        assert not is_defo[0, 0]
        assert alert_year[0, 0] == 0

    def test_exactly_boundary_2020_01_01(self):
        days = _glads2_days(date(2020, 1, 1))
        alert = np.array([[3]], dtype=np.int16)
        alert_date = np.array([[days]], dtype=np.int32)
        is_defo, alert_year = decode_glads2(alert, alert_date)
        assert is_defo[0, 0]
        assert alert_year[0, 0] == 2020

    def test_zero_alert_not_deforestation(self):
        days = _glads2_days(date(2021, 6, 1))
        alert = np.array([[0]], dtype=np.int16)
        alert_date = np.array([[days]], dtype=np.int32)
        is_defo, alert_year = decode_glads2(alert, alert_date)
        assert not is_defo[0, 0]

    def test_output_shapes_and_dtypes(self):
        alert = np.zeros((5, 6), dtype=np.int16)
        alert_date = np.zeros((5, 6), dtype=np.int32)
        is_defo, alert_year = decode_glads2(alert, alert_date)
        assert is_defo.shape == (5, 6)
        assert alert_year.shape == (5, 6)
        assert is_defo.dtype == bool
        assert alert_year.dtype == np.uint16

    def test_alert_year_range_valid(self):
        """Any non-zero alert_year must be in the expected range [2020, 2030]."""
        days_2021 = _glads2_days(date(2021, 3, 15))
        alert = np.array([[3]], dtype=np.int16)
        alert_date = np.array([[days_2021]], dtype=np.int32)
        _, alert_year = decode_glads2(alert, alert_date)
        nonzero = alert_year[alert_year > 0]
        assert all(2020 <= y <= 2030 for y in nonzero.tolist())
