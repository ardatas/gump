"""Tests for src/models.py — model architectures."""

from __future__ import annotations

import pytest
import torch

from src.models import (
    MLPBaseline,
    ModelOutputs,
    ResidualMLP,
    TemporalCNN,
    build_model,
    load_model_from_checkpoint,
)


class TestMLPBaseline:
    def test_forward_returns_model_outputs(self):
        model = MLPBaseline(input_dim=64)
        x = torch.randn(32, 64)
        out = model(x)
        assert isinstance(out, ModelOutputs)

    def test_logits_shape_batch(self):
        model = MLPBaseline(input_dim=64)
        x = torch.randn(32, 64)
        out = model(x)
        assert out.logits.shape == (32,), f"Expected (32,) got {out.logits.shape}"

    def test_logits_shape_single_sample(self):
        model = MLPBaseline(input_dim=64)
        x = torch.randn(1, 64)
        out = model(x)
        assert out.logits.shape == (1,)

    def test_no_year_logits(self):
        model = MLPBaseline(input_dim=32)
        x = torch.randn(8, 32)
        out = model(x)
        assert out.year_logits is None

    def test_sequence_argument_ignored(self):
        """MLPBaseline ignores the optional sequence tensor."""
        model = MLPBaseline(input_dim=32)
        x = torch.randn(8, 32)
        seq = torch.randn(8, 24, 10)
        out = model(x, sequence=seq)
        assert out.logits.shape == (8,)

    def test_default_architecture_linear_count(self):
        model = MLPBaseline(input_dim=100)
        linears = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
        assert len(linears) == 4  # 3 hidden + 1 output

    def test_output_dim_is_one(self):
        model = MLPBaseline(input_dim=64)
        last_linear = [m for m in model.modules() if isinstance(m, torch.nn.Linear)][-1]
        assert last_linear.out_features == 1

    def test_gradients_flow(self):
        model = MLPBaseline(input_dim=64)
        x = torch.randn(16, 64)
        y = torch.zeros(16)
        out = model(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(out.logits, y)
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

    def test_deterministic_in_eval_mode(self):
        model = MLPBaseline(input_dim=32)
        model.eval()
        x = torch.randn(8, 32)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.allclose(out1.logits, out2.logits)


class TestResidualMLP:
    def test_forward_returns_model_outputs(self):
        model = ResidualMLP(input_dim=64)
        x = torch.randn(16, 64)
        out = model(x)
        assert isinstance(out, ModelOutputs)

    def test_logits_shape(self):
        model = ResidualMLP(input_dim=64, hidden_dims=(128, 128, 64))
        x = torch.randn(20, 64)
        out = model(x)
        assert out.logits.shape == (20,)

    def test_residual_preserves_gradient_flow(self):
        model = ResidualMLP(input_dim=32, hidden_dims=(64, 64))
        x = torch.randn(8, 32)
        out = model(x)
        loss = out.logits.sum()
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient: {name}"

    def test_single_hidden_dim_works(self):
        model = ResidualMLP(input_dim=16, hidden_dims=(32,))
        x = torch.randn(4, 16)
        out = model(x)
        assert out.logits.shape == (4,)

    def test_empty_hidden_dims_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            ResidualMLP(input_dim=16, hidden_dims=())


class TestTemporalCNN:
    def test_forward_returns_model_outputs(self):
        model = TemporalCNN(static_dim=64, sequence_channels=9, sequence_length=24)
        features = torch.randn(8, 64)
        seq = torch.randn(8, 24, 10)  # (B, T, C+1)
        out = model(features, sequence=seq)
        assert isinstance(out, ModelOutputs)

    def test_logits_shape(self):
        model = TemporalCNN(static_dim=32, sequence_channels=9, sequence_length=12)
        features = torch.randn(16, 32)
        seq = torch.randn(16, 12, 10)  # C+1 = 10
        out = model(features, sequence=seq)
        assert out.logits.shape == (16,)

    def test_missing_sequence_raises(self):
        model = TemporalCNN(static_dim=32, sequence_channels=9, sequence_length=12)
        features = torch.randn(4, 32)
        with pytest.raises(ValueError, match="sequence"):
            model(features, sequence=None)

    def test_year_head_present_when_num_year_classes_positive(self):
        model = TemporalCNN(
            static_dim=32, sequence_channels=9, sequence_length=12,
            num_year_classes=4,
        )
        features = torch.randn(8, 32)
        seq = torch.randn(8, 12, 10)
        out = model(features, sequence=seq)
        assert out.year_logits is not None
        assert out.year_logits.shape == (8, 4)

    def test_year_head_absent_by_default(self):
        model = TemporalCNN(static_dim=32, sequence_channels=9, sequence_length=12)
        features = torch.randn(4, 32)
        seq = torch.randn(4, 12, 10)
        out = model(features, sequence=seq)
        assert out.year_logits is None

    def test_gradients_flow_through_both_branches(self):
        model = TemporalCNN(static_dim=32, sequence_channels=4, sequence_length=8)
        features = torch.randn(4, 32, requires_grad=False)
        seq = torch.randn(4, 8, 5)
        out = model(features, sequence=seq)
        out.logits.sum().backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient: {name}"


class TestBuildModel:
    def test_build_mlp(self):
        model = build_model("mlp", input_dim=64, hidden_dims=[256, 128], dropout=0.1)
        assert isinstance(model, MLPBaseline)

    def test_build_resmlp(self):
        model = build_model("resmlp", input_dim=64, hidden_dims=[256, 128], dropout=0.1)
        assert isinstance(model, ResidualMLP)

    def test_build_tcnn(self):
        model = build_model(
            "tcnn", input_dim=64, hidden_dims=[], dropout=0.1,
            sequence_channels=9, sequence_length=12,
        )
        assert isinstance(model, TemporalCNN)

    def test_build_tcnn_without_sequence_raises(self):
        with pytest.raises(ValueError):
            build_model(
                "tcnn", input_dim=64, hidden_dims=[], dropout=0.1,
                sequence_channels=0, sequence_length=0,
            )

    def test_unknown_model_name_raises(self):
        with pytest.raises(ValueError):
            build_model("unknown", input_dim=64, hidden_dims=[], dropout=0.1)

    def test_load_model_from_checkpoint(self):
        model = build_model("resmlp", input_dim=32, hidden_dims=[64, 32], dropout=0.1)
        checkpoint = {
            "model_name": "resmlp",
            "hidden_dims": [64, 32],
            "dropout": 0.1,
            "input_dim": 32,
            "model_state": model.state_dict(),
        }
        loaded = load_model_from_checkpoint(checkpoint)
        assert isinstance(loaded, ResidualMLP)
        x = torch.randn(4, 32)
        out = loaded(x)
        assert out.logits.shape == (4,)
