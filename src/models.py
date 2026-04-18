from __future__ import annotations

from typing import Iterable

import torch
from torch import nn


class MLPBaseline(nn.Module):
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

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


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


class ResidualMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (512, 512, 256, 256, 128),
        dropout: float = 0.2,
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

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.input_projection(features)
        hidden = self.stages(hidden)
        return self.head(hidden).squeeze(-1)


def build_model(
    model_name: str,
    *,
    input_dim: int,
    hidden_dims: Iterable[int],
    dropout: float,
) -> nn.Module:
    hidden_dims = tuple(int(value) for value in hidden_dims)
    if model_name == "mlp":
        return MLPBaseline(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)
    if model_name == "resmlp":
        return ResidualMLP(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)
    raise ValueError(f"Unsupported model_name: {model_name}")


def load_model_from_checkpoint(checkpoint: dict[str, object]) -> nn.Module:
    model_name = str(checkpoint.get("model_name", "mlp"))
    hidden_dims = tuple(int(value) for value in checkpoint.get("hidden_dims", (512, 256, 128)))
    dropout = float(checkpoint.get("dropout", 0.2))
    input_dim = int(checkpoint["input_dim"])
    model = build_model(
        model_name,
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    model.load_state_dict(checkpoint["model_state"])
    return model
