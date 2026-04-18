"""Tests for src/labels/harmonize.py — label consensus and weight logic."""

from __future__ import annotations

import numpy as np
import pytest

from src.labels.harmonize import harmonize


H, W = 8, 8


def _all_forest() -> np.ndarray:
    return np.ones((H, W), dtype=bool)


def _no_forest() -> np.ndarray:
    return np.zeros((H, W), dtype=bool)


def _source(is_defo: bool, year: int = 2021) -> tuple[np.ndarray, np.ndarray]:
    defo = np.full((H, W), is_defo, dtype=bool)
    yr = np.where(defo, year, 0).astype(np.uint16)
    return defo, yr


class TestConsensusLogic:
    def test_three_sources_agree_positive(self):
        """All 3 sources say deforested → consensus=1."""
        sources = [_source(True)] * 3
        consensus, _, _ = harmonize(sources, _all_forest())
        assert (consensus == 1).all()

    def test_two_of_three_agree_positive(self):
        """2 out of 3 agree → consensus=1 (≥2 required)."""
        sources = [_source(True), _source(True), _source(False)]
        consensus, _, _ = harmonize(sources, _all_forest())
        assert (consensus == 1).all()

    def test_one_of_three_agree_not_consensus(self):
        """Only 1 source positive → consensus=0 (needs ≥2)."""
        sources = [_source(True), _source(False), _source(False)]
        consensus, _, _ = harmonize(sources, _all_forest())
        assert (consensus == 0).all()

    def test_zero_votes_is_negative(self):
        """No source positive → consensus=0."""
        sources = [_source(False)] * 3
        consensus, _, _ = harmonize(sources, _all_forest())
        assert (consensus == 0).all()

    def test_two_sources_both_agree(self):
        """With 2 available sources, both must agree for consensus=1."""
        sources = [_source(True), _source(True)]
        consensus, _, _ = harmonize(sources, _all_forest())
        assert (consensus == 1).all()

    def test_two_sources_one_disagrees(self):
        """With 2 available sources, 1 positive doesn't make consensus."""
        sources = [_source(True), _source(False)]
        consensus, _, _ = harmonize(sources, _all_forest())
        assert (consensus == 0).all()

    def test_single_source_never_makes_consensus(self):
        """Single source can never produce consensus (needs ≥2)."""
        sources = [_source(True)]
        consensus, _, _ = harmonize(sources, _all_forest())
        assert (consensus == 0).all()

    def test_no_sources_returns_zeros(self):
        consensus, weight, alert_year = harmonize([], _all_forest())
        assert consensus.shape == (H, W)
        assert (consensus == 0).all()
        assert (weight == 0).all()
        assert (alert_year == 0).all()


class TestWeightLogic:
    def test_unanimous_three_sources_weight_1(self):
        """3/3 votes → weight=1.0."""
        sources = [_source(True)] * 3
        _, weight, _ = harmonize(sources, _all_forest())
        assert np.allclose(weight, 1.0)

    def test_two_of_three_weight_0_8(self):
        """2/3 votes → weight=0.8."""
        sources = [_source(True), _source(True), _source(False)]
        _, weight, _ = harmonize(sources, _all_forest())
        assert np.allclose(weight, 0.8)

    def test_one_of_three_weight_w_single(self):
        """1/3 votes → weight=w_single (default 0.3)."""
        sources = [_source(True), _source(False), _source(False)]
        _, weight, _ = harmonize(sources, _all_forest(), w_single=0.3)
        assert np.allclose(weight, 0.3)

    def test_zero_votes_in_forest_weight_1(self):
        """0/3 votes (in-forest negatives) → weight=1.0 (confident negative)."""
        sources = [_source(False)] * 3
        _, weight, _ = harmonize(sources, _all_forest())
        assert np.allclose(weight, 1.0)

    def test_outside_forest_weight_is_zero(self):
        """Pixels outside forest_2020 must have weight=0 regardless of votes."""
        sources = [_source(True)] * 3
        forest = _no_forest()
        _, weight, _ = harmonize(sources, forest)
        assert (weight == 0.0).all()

    def test_mixed_forest_mask(self):
        """Only forest pixels keep non-zero weight."""
        forest = np.zeros((H, W), dtype=bool)
        forest[:4, :] = True  # top half is forest
        sources = [_source(True)] * 3
        _, weight, _ = harmonize(sources, forest)
        assert (weight[:4, :] > 0).all()
        assert (weight[4:, :] == 0).all()

    def test_w_single_ablation_zero(self):
        """w_single=0 makes single-source pixels ignored in loss."""
        sources = [_source(True), _source(False), _source(False)]
        _, weight, _ = harmonize(sources, _all_forest(), w_single=0.0)
        assert (weight == 0.0).all()

    def test_weight_values_are_discrete(self):
        """Weights must only take on known discrete values: 0, 0.3, 0.8, 1.0."""
        rng = np.random.default_rng(1)
        defo_a = rng.random((H, W)) > 0.5
        defo_b = rng.random((H, W)) > 0.5
        defo_c = rng.random((H, W)) > 0.5
        sources = [
            (defo_a, np.zeros((H, W), dtype=np.uint16)),
            (defo_b, np.zeros((H, W), dtype=np.uint16)),
            (defo_c, np.zeros((H, W), dtype=np.uint16)),
        ]
        _, weight, _ = harmonize(sources, _all_forest())
        unique_weights = np.unique(weight)
        allowed = np.array([0.0, 0.3, 0.8, 1.0], dtype=np.float32)
        for w in unique_weights:
            assert any(np.isclose(w, a, atol=1e-5) for a in allowed), (
                f"Weight {w} is not in allowed set {allowed}"
            )


class TestAlertYearLogic:
    def test_earliest_year_wins(self):
        """alert_year should be the minimum year across positive sources."""
        src_2021 = (np.ones((H, W), dtype=bool), np.full((H, W), 2021, dtype=np.uint16))
        src_2023 = (np.ones((H, W), dtype=bool), np.full((H, W), 2023, dtype=np.uint16))
        _, _, alert_year = harmonize([src_2021, src_2023], _all_forest())
        assert (alert_year == 2021).all()

    def test_no_alert_gives_zero(self):
        sources = [_source(False), _source(False)]
        _, _, alert_year = harmonize(sources, _all_forest())
        assert (alert_year == 0).all()

    def test_alert_year_only_from_positive_sources(self):
        """A source that votes negative should not contribute its year."""
        src_pos = (np.ones((H, W), dtype=bool), np.full((H, W), 2020, dtype=np.uint16))
        src_neg = (np.zeros((H, W), dtype=bool), np.full((H, W), 2019, dtype=np.uint16))
        _, _, alert_year = harmonize([src_pos, src_neg], _all_forest())
        # 2020 should appear (from positive source), not 2019
        assert not (alert_year == 2019).any()
