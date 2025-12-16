"""
model_components.py
Model components used by train.py:
  - Temporal encoders: GRU, LSTM, 1D-CNN, GRU+CNN
  - Trainable H3 embedding
  - Projection layers (map temporal/H3 → LLM hidden size)
  - Fusion function (simple additive fusion)

These are all end-to-end trainable with the LLM + LoRA setup.
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------
# 1. Temporal Encoders
# ---------------------------------------------------------

class GRUTemporalEncoder(nn.Module):
    """
    GRU-based temporal encoder.
    Input:
        sequence: (B, T, 1)
    Output:
        aggregated hidden vector: (B, out_dim)
    """

    def __init__(self, input_dim: int = 1, hidden_dim: int = 64,
                 num_layers: int = 1, bidirectional: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        self.out_dim = hidden_dim * (2 if bidirectional else 1)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """
        seq: (B, T, 1)
        Returns:
            h: (B, out_dim)
        """
        out, h_n = self.gru(seq)  # h_n: (num_layers * num_directions, B, H)

        if self.bidirectional:
            # Last forward + last backward
            h_fwd = h_n[-2, :, :]
            h_bwd = h_n[-1, :, :]
            h = torch.cat([h_fwd, h_bwd], dim=-1)
        else:
            h = h_n[-1, :, :]

        return h


class LSTMTemporalEncoder(nn.Module):
    """
    LSTM-based temporal encoder.
    Input:
        sequence: (B, T, 1)
    Output:
        aggregated hidden vector: (B, out_dim)
    """

    def __init__(self, input_dim: int = 1, hidden_dim: int = 64,
                 num_layers: int = 1, bidirectional: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        self.out_dim = hidden_dim * (2 if bidirectional else 1)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """
        seq: (B, T, 1)
        Returns:
            h: (B, out_dim)
        """
        out, (h_n, c_n) = self.lstm(seq)

        if self.bidirectional:
            h_fwd = h_n[-2, :, :]
            h_bwd = h_n[-1, :, :]
            h = torch.cat([h_fwd, h_bwd], dim=-1)
        else:
            h = h_n[-1, :, :]

        return h


class CNNTemporalEncoder1D(nn.Module):
    """
    1D CNN temporal encoder.
    Treats time as the length dimension and feature as channels.
    Input:
        sequence: (B, T, 1)
    Output:
        pooled feature: (B, out_dim)
    """

    def __init__(self, input_dim: int = 1, hidden_dim: int = 64,
                 kernel_size: int = 3, num_layers: int = 2):
        super().__init__()

        layers = []
        in_channels = input_dim
        out_channels = hidden_dim

        for i in range(num_layers):
            layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
            )
            layers.append(nn.ReLU())
            in_channels = out_channels

        self.conv = nn.Sequential(*layers)
        self.out_dim = hidden_dim

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """
        seq: (B, T, 1)
        Returns:
            (B, out_dim)
        """
        # (B, T, 1) -> (B, C=1, T)
        x = seq.transpose(1, 2)   # (B, 1, T)
        x = self.conv(x)          # (B, hidden_dim, T)
        # Global average pooling over time
        x = x.mean(dim=2)         # (B, hidden_dim)
        return x


class GRUPlusCNNEncoder(nn.Module):
    """
    Combination encoder: GRU + CNN over the same sequence, concatenated then projected.
    Input:
        sequence: (B, T, 1)
    Output:
        (B, hidden_dim)
    """

    def __init__(self, input_dim: int = 1, hidden_dim: int = 64):
        super().__init__()
        self.gru = GRUTemporalEncoder(input_dim=input_dim, hidden_dim=hidden_dim)
        self.cnn = CNNTemporalEncoder1D(input_dim=input_dim, hidden_dim=hidden_dim)
        self.proj = nn.Linear(self.gru.out_dim + self.cnn.out_dim, hidden_dim)
        self.out_dim = hidden_dim

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        h_gru = self.gru(seq)   # (B, H)
        h_cnn = self.cnn(seq)   # (B, H)
        h = torch.cat([h_gru, h_cnn], dim=-1)  # (B, 2H)
        h = self.proj(h)        # (B, H)
        return h


def build_temporal_encoder(mode: str, input_dim: int, hidden_dim: int) -> nn.Module:
    """
    Helper to build temporal encoder by name.

    mode ∈ {"gru", "lstm", "cnn", "gru_cnn"}
    Returns a module with attribute `.out_dim`.
    """
    mode = mode.lower()
    if mode == "gru":
        enc = GRUTemporalEncoder(input_dim=input_dim, hidden_dim=hidden_dim)
    elif mode == "lstm":
        enc = LSTMTemporalEncoder(input_dim=input_dim, hidden_dim=hidden_dim)
    elif mode == "cnn":
        enc = CNNTemporalEncoder1D(input_dim=input_dim, hidden_dim=hidden_dim)
    elif mode == "gru_cnn":
        enc = GRUPlusCNNEncoder(input_dim=input_dim, hidden_dim=hidden_dim)
    else:
        raise ValueError(f"Unknown temporal encoder mode: {mode}")
    return enc


# ---------------------------------------------------------
# 2. H3 Embedding
# ---------------------------------------------------------

class H3Embedding(nn.Module):
    """
    Trainable embedding for H3 indices (single chosen resolution).

    Input:
        h3_ids: (B,) long tensor (0..num_h3_cells-1)
    Output:
        embedding: (B, embed_dim)
    """

    def __init__(self, num_h3_cells: int, embed_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(num_h3_cells, embed_dim)

    def forward(self, h3_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(h3_ids)  # (B, embed_dim)


# ---------------------------------------------------------
# 3. Projection (map to LLM hidden size)
# ---------------------------------------------------------

class Projection(nn.Module):
    """
    Simple linear projection layer: input_dim → output_dim

    Typically:
      - temporal encoder dim → LLM hidden dim
      - H3 embed dim → LLM hidden dim
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


# ---------------------------------------------------------
# 4. Fusion function
# ---------------------------------------------------------

def fuse_additive(llm_vec: torch.Tensor,
                  temp_vec: torch.Tensor,
                  h3_vec: torch.Tensor) -> torch.Tensor:
    """
    Simple additive fusion:
        fused = llm_vec + temp_vec + h3_vec

    Inputs:
        llm_vec:  (B, d)
        temp_vec: (B, d)
        h3_vec:   (B, d)

    Output:
        fused: (B, d)
    """
    return llm_vec + temp_vec + h3_vec
