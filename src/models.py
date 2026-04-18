from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn


@dataclass(frozen=True)
class ModelOutputs:
    logits: torch.Tensor
    year_logits: torch.Tensor | None = None


class _BinaryHeadModel(nn.Module):
    """Compatibility shim that lets pure-static models share the ModelOutputs API."""

    def forward(self, features: torch.Tensor, sequence: torch.Tensor | None = None) -> ModelOutputs:  # noqa: ARG002
        raise NotImplementedError


class MLPBaseline(_BinaryHeadModel):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (512, 256, 128),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor, sequence: torch.Tensor | None = None) -> ModelOutputs:  # noqa: ARG002
        logits = self.network(features).squeeze(-1)
        return ModelOutputs(logits=logits)


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = features
        hidden = self.norm(features)
        hidden = self.fc1(hidden)
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        hidden = self.fc2(hidden)
        hidden = self.dropout(hidden)
        return residual + hidden


class ResidualMLP(_BinaryHeadModel):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (512, 512, 256, 256, 128),
        dropout: float = 0.2,
        num_year_classes: int = 0,
    ) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError("ResidualMLP requires at least one hidden dimension.")

        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        stages: list[nn.Module] = []
        current_dim = hidden_dims[0]
        for hidden_dim in hidden_dims:
            if hidden_dim != current_dim:
                stages.append(
                    nn.Sequential(
                        nn.LayerNorm(current_dim),
                        nn.Linear(current_dim, hidden_dim),
                        nn.GELU(),
                        nn.Dropout(dropout),
                    )
                )
                current_dim = hidden_dim
            stages.append(ResidualBlock(current_dim, dropout))
        self.stages = nn.Sequential(*stages)
        self.head = nn.Sequential(
            nn.LayerNorm(current_dim),
            nn.GELU(),
            nn.Linear(current_dim, 1),
        )
        self.year_head = nn.Linear(current_dim, num_year_classes) if num_year_classes > 0 else None
        self.num_year_classes = num_year_classes

    def forward(self, features: torch.Tensor, sequence: torch.Tensor | None = None) -> ModelOutputs:  # noqa: ARG002
        hidden = self.input_projection(features)
        hidden = self.stages(hidden)
        logits = self.head(hidden).squeeze(-1)
        year_logits = self.year_head(hidden) if self.year_head is not None else None
        return ModelOutputs(logits=logits, year_logits=year_logits)


class TemporalConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.norm = nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels) if out_channels >= 8 else nn.GroupNorm(num_groups=1, num_channels=out_channels)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.conv(features)
        hidden = self.norm(hidden)
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        return hidden


class TemporalCNN(nn.Module):
    def __init__(
        self,
        static_dim: int,
        sequence_channels: int,
        sequence_length: int,
        conv_channels: tuple[int, ...] = (64, 128, 128),
        kernel_size: int = 5,
        static_hidden_dims: tuple[int, ...] = (256, 128),
        fusion_hidden_dim: int = 256,
        dropout: float = 0.2,
        num_year_classes: int = 0,
    ) -> None:
        super().__init__()
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive for TemporalCNN.")
        if not conv_channels:
            raise ValueError("conv_channels must contain at least one entry.")

        # Sequence branch: input is (B, C+1, T) where +1 is the validity-mask channel.
        in_channels = sequence_channels + 1
        conv_layers: list[nn.Module] = []
        current_channels = in_channels
        for out_channels in conv_channels:
            conv_layers.append(TemporalConvBlock(current_channels, out_channels, kernel_size, dropout))
            current_channels = out_channels
        self.temporal_layers = nn.Sequential(*conv_layers)
        self.temporal_pool_dim = current_channels * 2  # mean + max pool

        # Static branch
        static_layers: list[nn.Module] = []
        current_dim = static_dim
        for hidden_dim in static_hidden_dims:
            static_layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            current_dim = hidden_dim
        self.static_branch = nn.Sequential(*static_layers) if static_layers else nn.Identity()
        self.static_dim_out = current_dim if static_layers else static_dim

        fused_dim = self.temporal_pool_dim + self.static_dim_out
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.binary_head = nn.Linear(fusion_hidden_dim, 1)
        self.year_head = nn.Linear(fusion_hidden_dim, num_year_classes) if num_year_classes > 0 else None
        self.sequence_channels = sequence_channels
        self.sequence_length = sequence_length
        self.num_year_classes = num_year_classes

    def forward(self, features: torch.Tensor, sequence: torch.Tensor | None = None) -> ModelOutputs:
        if sequence is None:
            raise ValueError("TemporalCNN requires a sequence input of shape (batch, T, C+1).")
        if sequence.dim() != 3:
            raise ValueError(f"Expected sequence with 3 dims (B, T, C+1), got {sequence.dim()}.")

        # (B, T, C+1) -> (B, C+1, T) for Conv1d
        sequence_btc = sequence.transpose(1, 2).contiguous()
        temporal = self.temporal_layers(sequence_btc)
        # Mean and max pool along time axis
        temporal_mean = temporal.mean(dim=-1)
        temporal_max = temporal.amax(dim=-1)
        temporal_pooled = torch.cat([temporal_mean, temporal_max], dim=-1)

        static = self.static_branch(features)
        fused = torch.cat([temporal_pooled, static], dim=-1)
        fused = self.fusion(fused)
        binary_logits = self.binary_head(fused).squeeze(-1)
        year_logits = self.year_head(fused) if self.year_head is not None else None
        return ModelOutputs(logits=binary_logits, year_logits=year_logits)


def build_model(
    model_name: str,
    *,
    input_dim: int,
    hidden_dims: Iterable[int],
    dropout: float,
    sequence_channels: int = 0,
    sequence_length: int = 0,
    num_year_classes: int = 0,
) -> nn.Module:
    hidden_dims = tuple(int(value) for value in hidden_dims)
    if model_name == "mlp":
        return MLPBaseline(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)
    if model_name == "resmlp":
        return ResidualMLP(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout, num_year_classes=num_year_classes)
    if model_name == "tcnn":
        if sequence_channels <= 0 or sequence_length <= 0:
            raise ValueError("TemporalCNN requires sequence_channels>0 and sequence_length>0.")
        return TemporalCNN(
            static_dim=input_dim,
            sequence_channels=sequence_channels,
            sequence_length=sequence_length,
            dropout=dropout,
            num_year_classes=num_year_classes,
        )
    raise ValueError(f"Unsupported model_name: {model_name}")


def load_model_from_checkpoint(checkpoint: dict[str, object]) -> nn.Module:
    model_name = str(checkpoint.get("model_name", "mlp"))
    hidden_dims = tuple(int(value) for value in checkpoint.get("hidden_dims", (512, 256, 128)))
    dropout = float(checkpoint.get("dropout", 0.2))
    input_dim = int(checkpoint["input_dim"])
    sequence_channels = int(checkpoint.get("sequence_channels", 0) or 0)
    sequence_length = int(checkpoint.get("sequence_length", 0) or 0)
    num_year_classes = int(checkpoint.get("num_year_classes", 0) or 0)
    model = build_model(
        model_name,
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        sequence_channels=sequence_channels,
        sequence_length=sequence_length,
        num_year_classes=num_year_classes,
    )
    model.load_state_dict(checkpoint["model_state"])
    return model
