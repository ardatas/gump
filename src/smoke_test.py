"""Sanity-check script: verifies imports, module wiring, and basic logic
with synthetic data — no GPU or real cache needed.

Run with:
    python3 -m src.smoke_test
"""

from __future__ import annotations

import sys
import traceback

PASS = 0
FAIL = 0


def check(name: str, fn) -> None:
    global PASS, FAIL
    try:
        fn()
        PASS += 1
        print(f"  ✓ {name}")
    except Exception as exc:
        FAIL += 1
        print(f"  ✗ {name}: {exc}")
        traceback.print_exc()


def main() -> None:
    global PASS, FAIL
    import numpy as np
    import torch

    # ── 1. Imports ──────────────────────────────────────────────────────
    print("\n1. Module imports")

    def _import_cache():
        from src.cache import TileMetadata, TileData, list_tile_paths, load_tile_metadata  # noqa: F401
    check("src.cache", _import_cache)

    def _import_dataset():
        from src.dataset import (  # noqa: F401
            PixelSampleDataset, build_eval_matrix, build_prediction_matrix,
            gather_eval_sequences, gather_prediction_sequences,
            sample_tile_training_data, scatter_predictions,
        )
    check("src.dataset", _import_dataset)

    def _import_feature_bank():
        from src.feature_bank import (  # noqa: F401
            SEQUENCE_CHANNELS, TileFeatures, TileSequence,
            build_monthly_sequence, build_target_months, build_tile_features,
        )
    check("src.feature_bank", _import_feature_bank)

    def _import_metrics():
        from src.metrics import (  # noqa: F401
            find_best_threshold, find_best_threshold_per_region,
            weighted_average_precision, weighted_precision_recall_f1,
        )
    check("src.metrics", _import_metrics)

    def _import_models():
        from src.models import (  # noqa: F401
            MLPBaseline, ModelOutputs, ResidualMLP, TemporalCNN,
            build_model, load_model_from_checkpoint,
        )
    check("src.models", _import_models)

    def _import_postprocess():
        from src.postprocess import PostprocessConfig, postprocess_probability_map  # noqa: F401
    check("src.postprocess", _import_postprocess)

    def _import_train():
        from src.train import (  # noqa: F401
            LossConfig, Normalizer, RuntimeConfig, TemporalConfig,
            autocast_context, compute_total_loss, compute_weighted_loss,
            evaluate_on_tiles, fit_normalizer, load_normalizer,
            normalize_features, predict_probabilities, predict_year_logits,
            resolve_amp_dtype, resolve_positive_class_weight,
            save_checkpoint, set_seed, train_one_epoch, write_run_config,
        )
    check("src.train", _import_train)

    def _import_infer():
        from src.infer import (  # noqa: F401
            CheckpointBundle, load_checkpoint_bundle, parse_args,
        )
    check("src.infer", _import_infer)

    def _import_evaluate():
        from src.evaluate import (  # noqa: F401
            evaluate_config, precompute_tile_predictions,
        )
    check("src.evaluate", _import_evaluate)

    def _import_calibrate():
        from src.calibrate_thresholds import main as cal_main  # noqa: F401
    check("src.calibrate_thresholds", _import_calibrate)

    def _import_train_kfold():
        from src.train_kfold import main as kf_main  # noqa: F401
    check("src.train_kfold", _import_train_kfold)

    # ── 2. Model construction ──────────────────────────────────────────
    print("\n2. Model construction")
    from src.models import build_model, ModelOutputs

    def _build_mlp():
        m = build_model("mlp", input_dim=128, hidden_dims=(64, 32), dropout=0.1)
        out = m(torch.randn(4, 128))
        assert isinstance(out, ModelOutputs)
        assert out.logits.shape == (4,)
        assert out.year_logits is None
    check("MLP build + forward", _build_mlp)

    def _build_resmlp():
        m = build_model("resmlp", input_dim=128, hidden_dims=(64, 64, 32), dropout=0.1)
        out = m(torch.randn(4, 128))
        assert out.logits.shape == (4,)
    check("ResidualMLP build + forward", _build_resmlp)

    def _build_tcnn():
        m = build_model(
            "tcnn", input_dim=128, hidden_dims=(64,), dropout=0.1,
            sequence_channels=9, sequence_length=36, num_year_classes=4,
        )
        seq = torch.randn(4, 36, 10)  # C+1 = 9+1
        out = m(torch.randn(4, 128), sequence=seq)
        assert out.logits.shape == (4,)
        assert out.year_logits is not None
        assert out.year_logits.shape == (4, 4)
    check("TemporalCNN build + forward", _build_tcnn)

    # ── 3. Loss functions ──────────────────────────────────────────────
    print("\n3. Loss functions")
    from src.train import LossConfig, TemporalConfig, compute_total_loss, compute_weighted_loss

    def _bce_loss():
        logits = torch.randn(16)
        labels = torch.randint(0, 2, (16,)).float()
        weights = torch.ones(16)
        cfg = LossConfig(name="bce", focal_gamma=2.0, positive_class_weight=None, gce_q=0.7, label_smoothing=0.0)
        loss, wls, ws = compute_weighted_loss(logits=logits, labels=labels, weights=weights, loss_config=cfg)
        assert loss.dim() == 0
        assert loss.item() > 0
    check("BCE loss", _bce_loss)

    def _focal_loss():
        logits = torch.randn(16)
        labels = torch.randint(0, 2, (16,)).float()
        weights = torch.ones(16)
        cfg = LossConfig(name="focal", focal_gamma=2.0, positive_class_weight=None, gce_q=0.7, label_smoothing=0.0)
        loss, _, _ = compute_weighted_loss(logits=logits, labels=labels, weights=weights, loss_config=cfg)
        assert loss.item() > 0
    check("Focal loss", _focal_loss)

    def _gce_loss():
        logits = torch.randn(16)
        labels = torch.randint(0, 2, (16,)).float()
        weights = torch.ones(16)
        cfg = LossConfig(name="gce", focal_gamma=2.0, positive_class_weight=2.0, gce_q=0.7, label_smoothing=0.0)
        loss, _, _ = compute_weighted_loss(logits=logits, labels=labels, weights=weights, loss_config=cfg)
        assert loss.item() > 0
    check("GCE loss", _gce_loss)

    def _total_loss_with_year():
        m = build_model("tcnn", input_dim=32, hidden_dims=(16,), dropout=0.0,
                        sequence_channels=3, sequence_length=12, num_year_classes=4)
        seq = torch.randn(8, 12, 4)
        out = m(torch.randn(8, 32), sequence=seq)
        labels = torch.randint(0, 2, (8,)).float()
        weights = torch.ones(8)
        year_class = torch.randint(0, 4, (8,))
        year_mask = torch.ones(8)
        cfg = LossConfig(name="focal", focal_gamma=2.0, positive_class_weight=None, gce_q=0.7, label_smoothing=0.0)
        tcfg = TemporalConfig(enabled=True, target_months=(), channels=("a", "b", "c"),
                              year_min=2020, num_year_classes=4, aux_year_loss_weight=0.5)
        loss, _, _ = compute_total_loss(
            outputs=out, labels=labels, weights=weights, loss_config=cfg,
            temporal_config=tcfg, year_class=year_class, year_mask=year_mask,
        )
        assert loss.item() > 0
    check("Total loss with year head", _total_loss_with_year)

    def _label_smoothing():
        logits = torch.full((16,), 10.0)  # confident predictions
        labels = torch.ones(16)
        weights = torch.ones(16)
        cfg_no = LossConfig(name="bce", focal_gamma=2.0, positive_class_weight=None, gce_q=0.7, label_smoothing=0.0)
        cfg_sm = LossConfig(name="bce", focal_gamma=2.0, positive_class_weight=None, gce_q=0.7, label_smoothing=0.1)
        loss_no, _, _ = compute_weighted_loss(logits=logits, labels=labels, weights=weights, loss_config=cfg_no)
        loss_sm, _, _ = compute_weighted_loss(logits=logits, labels=labels, weights=weights, loss_config=cfg_sm)
        # Label smoothing should increase loss for "perfect" labels
        assert loss_sm.item() > loss_no.item()
    check("Label smoothing increases loss", _label_smoothing)

    # ── 4. Metrics ─────────────────────────────────────────────────────
    print("\n4. Metrics")
    from src.metrics import (
        find_best_threshold, find_best_threshold_per_region,
        weighted_average_precision, weighted_precision_recall_f1,
    )

    def _precision_recall_f1():
        labels = np.array([1, 1, 0, 0, 1, 0], dtype=np.float32)
        preds = np.array([1, 0, 0, 0, 1, 1], dtype=np.float32)
        weights = np.ones(6, dtype=np.float32)
        m = weighted_precision_recall_f1(labels, preds, weights)
        assert 0 < m["f1"] < 1
        assert 0 < m["precision"] <= 1
        assert 0 < m["recall"] <= 1
    check("weighted_precision_recall_f1", _precision_recall_f1)

    def _best_threshold():
        probs = np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=np.float32)
        labels = np.array([0, 0, 1, 1, 1], dtype=np.float32)
        weights = np.ones(5, dtype=np.float32)
        thr, met = find_best_threshold(probs, labels, weights)
        assert 0 < thr < 1
        assert met["f1"] > 0
    check("find_best_threshold", _best_threshold)

    def _best_threshold_per_region():
        probs = np.array([0.1, 0.9, 0.2, 0.8], dtype=np.float32)
        labels = np.array([0, 1, 0, 1], dtype=np.float32)
        weights = np.ones(4, dtype=np.float32)
        regions = np.array(["A", "A", "B", "B"])
        result = find_best_threshold_per_region(probs, labels, weights, regions)
        assert "A" in result and "B" in result
    check("find_best_threshold_per_region", _best_threshold_per_region)

    def _average_precision():
        probs = np.array([0.9, 0.8, 0.3, 0.1], dtype=np.float32)
        labels = np.array([1, 1, 0, 0], dtype=np.float32)
        weights = np.ones(4, dtype=np.float32)
        ap = weighted_average_precision(labels, probs, weights)
        assert 0.5 < ap <= 1.0
    check("weighted_average_precision", _average_precision)

    # ── 5. Dataset utilities ───────────────────────────────────────────
    print("\n5. Dataset utilities")
    from src.dataset import PixelSampleDataset, scatter_predictions

    def _pixel_dataset():
        ds = PixelSampleDataset(
            features=np.random.randn(100, 32).astype(np.float32),
            labels=np.random.randint(0, 2, 100).astype(np.float32),
            weights=np.ones(100, dtype=np.float32),
        )
        assert len(ds) == 100
        item = ds[0]
        assert "features" in item and "labels" in item and "weights" in item
    check("PixelSampleDataset", _pixel_dataset)

    def _pixel_dataset_with_sequences():
        ds = PixelSampleDataset(
            features=np.random.randn(50, 32).astype(np.float32),
            labels=np.random.randint(0, 2, 50).astype(np.float32),
            weights=np.ones(50, dtype=np.float32),
            sequences=np.random.randn(50, 12, 10).astype(np.float32),
            year_classes=np.random.randint(0, 4, 50).astype(np.int64),
            year_mask=np.ones(50, dtype=np.float32),
        )
        item = ds[0]
        assert "sequence" in item
        assert "year_class" in item
        assert "year_mask" in item
    check("PixelSampleDataset with sequences", _pixel_dataset_with_sequences)

    def _scatter():
        values = np.array([0.5, 0.9, 0.1], dtype=np.float32)
        mask = np.zeros((3, 3), dtype=bool)
        mask[0, 0] = True
        mask[1, 2] = True
        mask[2, 1] = True
        out = scatter_predictions(values, mask, dtype=np.float32)
        assert out.shape == (3, 3)
        assert np.isclose(out[0, 0], 0.5)
        assert np.isclose(out[1, 2], 0.9)
        assert np.isclose(out[2, 1], 0.1)
        assert out[0, 1] == 0.0
    check("scatter_predictions", _scatter)

    def _scatter_with_fill():
        values = np.array([1.0], dtype=np.float32)
        mask = np.zeros((2, 2), dtype=bool)
        mask[0, 0] = True
        out = scatter_predictions(values, mask, fill_value=np.nan, dtype=np.float32)
        assert np.isnan(out[1, 1])
        assert out[0, 0] == 1.0
    check("scatter_predictions fill_value=nan", _scatter_with_fill)

    # ── 6. Normalizer ──────────────────────────────────────────────────
    print("\n6. Normalizer")
    from src.train import fit_normalizer, normalize_features, Normalizer

    def _normalizer():
        features = np.random.randn(1000, 32).astype(np.float32) * 5 + 10
        norm = fit_normalizer(features)
        normed = normalize_features(features, norm)
        assert abs(normed.mean()) < 0.5
        assert abs(normed.std() - 1.0) < 0.5
    check("fit_normalizer + normalize_features", _normalizer)

    # ── 7. Postprocess ─────────────────────────────────────────────────
    print("\n7. Postprocess")
    from src.postprocess import PostprocessConfig, postprocess_probability_map
    from rasterio import Affine

    def _postprocess():
        prob = np.random.rand(100, 100).astype(np.float32)
        forest = np.ones((100, 100), dtype=bool)
        transform = Affine(10.0, 0.0, 0.0, 0.0, -10.0, 0.0)
        cfg = PostprocessConfig(threshold=0.5, closing_radius=1, opening_radius=1, min_area_ha=0.01)
        pred = postprocess_probability_map(prob, forest_mask=forest, transform=transform, config=cfg)
        assert pred.dtype == bool
        assert pred.shape == (100, 100)
    check("postprocess_probability_map", _postprocess)

    # ── 8. Predict probabilities ───────────────────────────────────────
    print("\n8. Predict probabilities")
    from src.train import predict_probabilities, predict_year_logits, RuntimeConfig

    def _predict_probs():
        m = build_model("resmlp", input_dim=32, hidden_dims=(16, 16), dropout=0.0)
        rc = RuntimeConfig(device=torch.device("cpu"), amp_dtype=None, grad_clip_norm=0.0)
        features = np.random.randn(50, 32).astype(np.float32)
        probs = predict_probabilities(model=m, features=features, runtime_config=rc, batch_size=16)
        assert probs.shape == (50,)
        assert probs.min() >= 0 and probs.max() <= 1
    check("predict_probabilities", _predict_probs)

    def _predict_probs_with_sequence():
        m = build_model("tcnn", input_dim=32, hidden_dims=(16,), dropout=0.0,
                        sequence_channels=3, sequence_length=12)
        rc = RuntimeConfig(device=torch.device("cpu"), amp_dtype=None, grad_clip_norm=0.0)
        features = np.random.randn(50, 32).astype(np.float32)
        seqs = np.random.randn(50, 12, 4).astype(np.float32)  # C+1
        probs = predict_probabilities(model=m, features=features, runtime_config=rc,
                                      batch_size=16, sequences=seqs)
        assert probs.shape == (50,)
    check("predict_probabilities with sequences", _predict_probs_with_sequence)

    def _predict_year_logits():
        m = build_model("tcnn", input_dim=32, hidden_dims=(16,), dropout=0.0,
                        sequence_channels=3, sequence_length=12, num_year_classes=4)
        rc = RuntimeConfig(device=torch.device("cpu"), amp_dtype=None, grad_clip_norm=0.0)
        features = np.random.randn(20, 32).astype(np.float32)
        seqs = np.random.randn(20, 12, 4).astype(np.float32)
        yl = predict_year_logits(model=m, features=features, sequences=seqs,
                                 runtime_config=rc, batch_size=8)
        assert yl is not None
        assert yl.shape == (20, 4)
    check("predict_year_logits", _predict_year_logits)

    def _predict_year_logits_none():
        m = build_model("resmlp", input_dim=32, hidden_dims=(16,), dropout=0.0)
        rc = RuntimeConfig(device=torch.device("cpu"), amp_dtype=None, grad_clip_norm=0.0)
        features = np.random.randn(10, 32).astype(np.float32)
        yl = predict_year_logits(model=m, features=features, sequences=None,
                                 runtime_config=rc, batch_size=8)
        assert yl is None
    check("predict_year_logits returns None for non-year model", _predict_year_logits_none)

    # ── 9. Training loop (1 step) ──────────────────────────────────────
    print("\n9. Training loop (1 micro-step)")
    from src.train import train_one_epoch
    from torch.utils.data import DataLoader

    def _train_step():
        features = np.random.randn(64, 32).astype(np.float32)
        labels = np.random.randint(0, 2, 64).astype(np.float32)
        weights = np.ones(64, dtype=np.float32)
        ds = PixelSampleDataset(features, labels, weights)
        loader = DataLoader(ds, batch_size=32, shuffle=False)
        m = build_model("resmlp", input_dim=32, hidden_dims=(16,), dropout=0.0)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        rc = RuntimeConfig(device=torch.device("cpu"), amp_dtype=None, grad_clip_norm=1.0)
        lcfg = LossConfig(name="focal", focal_gamma=2.0, positive_class_weight=None,
                           gce_q=0.7, label_smoothing=0.0)
        tcfg = TemporalConfig(enabled=False, target_months=(), channels=(),
                              year_min=None, num_year_classes=0, aux_year_loss_weight=0.0)
        loss = train_one_epoch(model=m, loader=loader, optimizer=opt,
                               runtime_config=rc, loss_config=lcfg, scaler=scaler,
                               temporal_config=tcfg)
        assert isinstance(loss, float)
        assert loss > 0
    check("train_one_epoch", _train_step)

    # ── Summary ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Passed: {PASS}   Failed: {FAIL}")
    print(f"{'='*60}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
