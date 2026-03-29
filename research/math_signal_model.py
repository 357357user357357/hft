#!/usr/bin/env python3
"""Transformer model for predicting mathematical trading signals from price sequences.

Architecture:
    Prices → Embedding → Transformer Encoder → Multi-task Heads
                                              ├─ Poincaré regime (3-class)
                                              ├─ Poincaré score (continuous)
                                              ├─ Whitehead signal (2-class)
                                              ├─ Torsion ratio (continuous)
                                              ├─ Hecke L-value (complex)
                                              ├─ Zeta signal (3-class)
                                              └─ Future return (continuous)

Usage:
    model = MathSignalTransformer()
    outputs = model(prices_tensor)  # [batch, seq_len, 1]
"""

import torch
import torch.nn as nn
import math
from typing import Dict, Tuple, Optional


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MathSignalTransformer(nn.Module):
    """
    Transformer that predicts mathematical signals from price sequences.

    Input:  [batch, seq_len, 1]  (normalized prices)
    Output: Dict of signal predictions
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 200,
    ):
        super().__init__()

        self.d_model = d_model

        # Input embedding: price → d_model
        self.price_embed = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Shared MLP before heads
        self.shared_mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # === Signal Prediction Heads ===

        # Poincaré: 3-class regime classification
        self.poincare_regime_head = nn.Sequential(
            nn.Linear(d_model // 2, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 3),  # mean-reversion, trending, neutral
        )

        # Poincaré score: continuous -1 to +1
        self.poincare_score_head = nn.Sequential(
            nn.Linear(d_model // 2, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Tanh(),  # Output in [-1, 1]
        )

        # Whitehead: binary regime_change vs same_regime
        self.whitehead_head = nn.Sequential(
            nn.Linear(d_model // 2, 64),
            nn.GELU(),
            nn.Linear(64, 2),
        )

        # Whitehead torsion ratio: continuous > 0
        self.torsion_head = nn.Sequential(
            nn.Linear(d_model // 2, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Softplus(),  # Output > 0
        )

        # Hecke L-value: complex (real + imag)
        self.hecke_head = nn.Sequential(
            nn.Linear(d_model // 2, 64),
            nn.GELU(),
            nn.Linear(64, 2),  # [real, imag]
        )

        # Zeta signal: 3-class
        self.zeta_head = nn.Sequential(
            nn.Linear(d_model // 2, 64),
            nn.GELU(),
            nn.Linear(64, 3),  # strong, moderate, weak
        )

        # Future return: continuous (for supervised learning)
        self.return_head = nn.Sequential(
            nn.Linear(d_model // 2, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        # Polar features: 4 continuous outputs
        self.polar_head = nn.Sequential(
            nn.Linear(d_model // 2, 64),
            nn.GELU(),
            nn.Linear(64, 4),  # [mean_r, mean_theta, mean_dr_dt, mean_dtheta_dt]
        )

        # Frenet-Serret: 2 continuous outputs
        self.frenet_head = nn.Sequential(
            nn.Linear(d_model // 2, 32),
            nn.GELU(),
            nn.Linear(32, 2),  # [curvature, torsion]
        )

    def forward(self, prices: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            prices: [batch, seq_len, 1] normalized prices

        Returns:
            Dict of signal predictions
        """
        # Embed prices
        x = self.price_embed(prices)  # [batch, seq_len, d_model]
        x = self.pos_encoder(x)

        # Transformer encoder
        x = self.transformer(x)  # [batch, seq_len, d_model]

        # Global pooling
        x = x.transpose(1, 2)  # [batch, d_model, seq_len]
        x = self.global_pool(x).squeeze(-1)  # [batch, d_model]

        # Shared MLP
        x = self.shared_mlp(x)  # [batch, d_model/2]

        # Predict all signals
        return {
            "poincare_regime": self.poincare_regime_head(x),  # [batch, 3]
            "poincare_score": self.poincare_score_head(x),  # [batch, 1]
            "whitehead_signal": self.whitehead_head(x),  # [batch, 2]
            "torsion_ratio": self.torsion_head(x),  # [batch, 1]
            "hecke_l_value": self.hecke_head(x),  # [batch, 2]
            "zeta_signal": self.zeta_head(x),  # [batch, 3]
            "future_return": self.return_head(x),  # [batch, 1]
            "polar_features": self.polar_head(x),  # [batch, 4]
            "frenet_features": self.frenet_head(x),  # [batch, 2]
        }


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_model(
    size: str = "small",
    pretrained: bool = False,
    checkpoint_path: Optional[str] = None,
) -> MathSignalTransformer:
    """Create model with predefined sizes.

    Args:
        size: "tiny" (~100k), "small" (~500k), "medium" (~2M), "large" (~8M)
        pretrained: Load pretrained weights
        checkpoint_path: Path to checkpoint

    Returns:
        MathSignalTransformer model
    """
    configs = {
        "tiny": {"d_model": 64, "nhead": 4, "num_layers": 3, "dim_feedforward": 256},
        "small": {"d_model": 128, "nhead": 8, "num_layers": 6, "dim_feedforward": 512},
        "medium": {"d_model": 256, "nhead": 8, "num_layers": 8, "dim_feedforward": 1024},
        "large": {"d_model": 512, "nhead": 16, "num_layers": 12, "dim_feedforward": 2048},
    }

    cfg = configs.get(size, configs["small"])
    model = MathSignalTransformer(**cfg)

    if pretrained and checkpoint_path:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)
        print(f"Loaded pretrained weights from {checkpoint_path}")

    return model


if __name__ == "__main__":
    # Test model creation
    model = create_model("small")
    print(f"Model parameters: {count_parameters(model):,}")

    # Test forward pass
    batch_size = 4
    seq_len = 100
    prices = torch.randn(batch_size, seq_len, 1)

    outputs = model(prices)
    print("\nOutput shapes:")
    for key, value in outputs.items():
        print(f"  {key}: {value.shape}")
