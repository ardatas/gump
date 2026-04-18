from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

import src.train as train_module
from src.feature_bank import build_tile_features
from src.models import build_model
from src.train import RuntimeConfig, TemporalConfig, prepare_train_data, predict_probabilities
from tests.conftest import TARGET_YEARS, _make_metadata, _make_tile


def _make_temporal_config() -> TemporalConfig:
    return TemporalConfig(
        enabled=False,
        target_months=(),
        channels=(),
        year_min=min(TARGET_YEARS),
        num_year_classes=0,
        aux_year_loss_weight=0.0,
    )


def test_prepare_train_data_is_deterministic_across_worker_counts(monkeypatch, tmp_path):
    metadata = []
    tile_map = {}
    for index, tile_id in enumerate(("18NWG_6_6", "19NBD_4_4", "47QMB_0_8")):
        path = tmp_path / f"{tile_id}.zarr"
        meta = replace(_make_metadata(tile_id=tile_id, region=tile_id[:3]), path=path)
        tile = replace(_make_tile(np.random.default_rng(index)), metadata=meta)
        metadata.append(meta)
        tile_map[path] = tile

    monkeypatch.setattr(train_module, "load_tile_data", lambda path: tile_map[path])
    monkeypatch.setattr(
        train_module,
        "build_tile_features",
        lambda tile, target_years: build_tile_features(tile, target_years, use_cache=False),
    )

    serial = prepare_train_data(
        metadata=metadata,
        target_years=TARGET_YEARS,
        temporal_config=_make_temporal_config(),
        negative_ratio=2.0,
        max_samples_per_tile=64,
        seed=42,
        prep_workers=1,
    )
    parallel = prepare_train_data(
        metadata=metadata,
        target_years=TARGET_YEARS,
        temporal_config=_make_temporal_config(),
        negative_ratio=2.0,
        max_samples_per_tile=64,
        seed=42,
        prep_workers=3,
    )

    np.testing.assert_allclose(np.asarray(serial.features), np.asarray(parallel.features))
    np.testing.assert_allclose(np.asarray(serial.labels), np.asarray(parallel.labels))
    np.testing.assert_allclose(np.asarray(serial.weights), np.asarray(parallel.weights))


def test_predict_probabilities_accepts_tensor_features():
    model = build_model("mlp", input_dim=4, hidden_dims=(8, 4), dropout=0.0)
    runtime_config = RuntimeConfig(
        device=torch.device("cpu"),
        amp_dtype=None,
        grad_clip_norm=0.0,
    )
    features = torch.randn(10, 4)

    probabilities = predict_probabilities(
        model=model,
        features=features,
        runtime_config=runtime_config,
        batch_size=3,
    )

    assert probabilities.shape == (10,)
    assert np.isfinite(probabilities).all()
