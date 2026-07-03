# -*- coding: utf-8 -*-
"""PyTorch model definition for the Vietnamese sign-language word transformer."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn


FEATURE_DIM = 228
D_MODEL = 256
DIM_FEEDFORWARD = 512
NUM_LAYERS = 4
NHEAD = 8
MAX_SEQUENCE_LENGTH = 500
DROPOUT = 0.3


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding with checkpoint-compatible key `position.pe`."""

    def __init__(self, d_model: int = D_MODEL, max_len: int = MAX_SEQUENCE_LENGTH) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(1, max_len, d_model, dtype=torch.float32)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class VSLTransformer(nn.Module):
    """Transformer encoder for the current `best_model.pth` checkpoint."""

    def __init__(
        self,
        num_classes: int,
        feature_dim: int = FEATURE_DIM,
        d_model: int = D_MODEL,
        nhead: int = NHEAD,
        num_layers: int = NUM_LAYERS,
        dim_feedforward: int = DIM_FEEDFORWARD,
        max_len: int = MAX_SEQUENCE_LENGTH,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.position = PositionalEncoding(d_model=d_model, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.position(x)
        x = self.transformer(x)
        pooled = x.mean(dim=1)
        return self.classifier(pooled)


def load_vsl_transformer(
    state_path: str | Path,
    num_classes: int,
    device: torch.device | str = "cpu",
) -> VSLTransformer:
    state_dict = torch.load(Path(state_path), map_location=device)
    if "embedding.0.weight" in state_dict:
        feature_dim = int(state_dict["embedding.0.weight"].shape[1])
    elif "embedding.weight" in state_dict:
        feature_dim = int(state_dict["embedding.weight"].shape[1])
    else:
        feature_dim = FEATURE_DIM

    output_key = "classifier.3.bias" if "classifier.3.bias" in state_dict else "classifier.bias"
    inferred_classes = int(state_dict[output_key].shape[0]) if output_key in state_dict else num_classes

    model = VSLTransformer(num_classes=inferred_classes, feature_dim=feature_dim)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
