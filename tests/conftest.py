"""Shared fixtures for the test suite."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.cache import TileData, TileMetadata


H, W = 32, 32
N_S2 = 12   # 6 months × 2 years
N_S1 = 6    # 3 months × 2 years
N_AEF = 2
TARGET_YEARS = (2020, 2021)


def _make_metadata(tile_id: str = "18NWG_6_6", region: str = "18N") -> TileMetadata:
    return TileMetadata(
        tile_id=tile_id,
        split="train",
        path=Path("/tmp/fake.zarr"),
        region=region,
        crs="EPSG:32618",
        transform=(10.0, 0.0, 300000.0, 0.0, -10.0, 1000000.0),
        width=W,
        height=H,
        available_years=tuple(TARGET_YEARS),
    )


def _make_tile(
    rng: np.random.Generator,
    *,
    forest_fraction: float = 0.7,
    deforestation_fraction: float = 0.05,
) -> TileData:
    """Build a synthetic TileData with realistic shapes and value ranges."""
    # S2: 12 months spread across 2 years, 12 bands, raw reflectance scale
    s2_data = rng.uniform(500, 8000, size=(N_S2, 12, H, W)).astype(np.float32)
    s2_years = np.array([2020] * 6 + [2021] * 6, dtype=np.int16)
    s2_months = np.tile(np.arange(1, 7, dtype=np.int8), 2)

    # Elevate NIR (band index 7) and suppress red (band index 3) → high NDVI → forest
    s2_data[:, 7] = rng.uniform(5000, 8000, size=(N_S2, H, W)).astype(np.float32)
    s2_data[:, 3] = rng.uniform(300, 800, size=(N_S2, H, W)).astype(np.float32)

    # S1: linear backscatter (small positive values)
    s1_asc = rng.uniform(0.001, 0.1, size=(N_S1, 1, H, W)).astype(np.float32)
    s1_asc_years = np.array([2020] * 3 + [2021] * 3, dtype=np.int16)
    s1_asc_months = np.tile(np.array([1, 5, 9], dtype=np.int8), 2)

    s1_desc = rng.uniform(0.001, 0.1, size=(N_S1, 1, H, W)).astype(np.float32)
    s1_desc_years = s1_asc_years.copy()
    s1_desc_months = s1_asc_months.copy()

    # AEF: 64-dim yearly embeddings
    aef_data = rng.normal(0, 1, size=(N_AEF, 64, H, W)).astype(np.float32)
    aef_years = np.array(list(TARGET_YEARS), dtype=np.int16)

    # Forest mask
    forest_mask = rng.random(size=(H, W)) < forest_fraction

    # Labels: build consensus, weight, votes arrays.
    # `compute_persistence_targets` requires raw_votes to return training labels.
    n_forest = int(forest_mask.sum())
    n_pos = max(1, int(n_forest * deforestation_fraction))
    pos_flat = rng.choice(n_forest, size=n_pos, replace=False)
    forest_indices = np.argwhere(forest_mask)

    consensus = np.zeros((H, W), dtype=np.uint8)
    votes = np.zeros((H, W), dtype=np.uint8)    # votes per pixel (0–3)
    weight = np.ones((H, W), dtype=np.float32)
    weight[~forest_mask] = 0.0

    for flat_idx in pos_flat:
        r, c = forest_indices[flat_idx]
        consensus[r, c] = 1
        votes[r, c] = 3   # unanimous → high-confidence training positive
        weight[r, c] = 1.0

    return TileData(
        metadata=_make_metadata(),
        s2_data=s2_data,
        s2_year=s2_years,
        s2_month=s2_months,
        s1_ascending_data=s1_asc,
        s1_ascending_year=s1_asc_years,
        s1_ascending_month=s1_asc_months,
        s1_descending_data=s1_desc,
        s1_descending_year=s1_desc_years,
        s1_descending_month=s1_desc_months,
        aef_data=aef_data,
        aef_year=aef_years,
        forest_mask=forest_mask,
        label_consensus=consensus,
        label_weight=weight,
        label_votes=votes,
        label_alert_year=np.zeros((H, W), dtype=np.uint16),
        label_source_mask=np.zeros((H, W), dtype=np.uint8),
    )


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture()
def synthetic_tile(rng: np.random.Generator) -> TileData:
    return _make_tile(rng)


@pytest.fixture()
def tile_metadata() -> TileMetadata:
    return _make_metadata()
