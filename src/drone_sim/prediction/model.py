"""AV-LSTM trajectory prediction model.

Implements the architecture from:
  Hybrid Neural Trajectory Forecasting with Uncertainty-Aware Safety Zone
  Estimation for MPC-Based Collision Avoidance (TETCI-2025-1601)

Architecture:
  1. Linear embedding: (batch, m, n) -> (batch, m, d)
  2. Sinusoidal positional encoding: adds PE to embedded sequence
  3. TransformerEncoder: L=2 layers, H_heads=1, d_model=64, batch_first=True
  4. LSTM init: h0 from mean-pool of encoder output, c0 from max-pool
  5. LSTM Decoder: input_size=n, hidden_size=128, autoregressive T=80 steps
  6. Output heads: mu = Linear(h_lstm, n), sigma via softplus + sigma_min floor
  7. Teacher forcing: per-step coin flip using torch.rand(1).item()
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding as in Vaswani et al. (2017).

    Adds a fixed positional signal to the input embedding. The buffer is
    registered so it moves to the correct device with the model.

    Args:
        d_model: Embedding dimension (must match model's ``d``).
        max_len: Maximum sequence length supported (default: 512).
    """

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.

        Args:
            x: Input tensor of shape ``(batch, seq_len, d_model)``.

        Returns:
            Tensor of same shape as ``x`` with positional signal added.
        """
        return x + self.pe[:, : x.size(1)]  # type: ignore[index]


class AVLSTMModel(nn.Module):
    """Attention-Velocity LSTM model for probabilistic trajectory prediction.

    Encodes a noisy history sequence with a Transformer Encoder, then
    autoregressively decodes ``T`` future steps using an LSTM, producing
    Gaussian mean (``mu``) and standard-deviation (``sigma``) outputs.

    The sigma output is guaranteed to be >= ``sigma_min`` everywhere via
    ``softplus(raw_sigma) + sigma_min``, preventing sigma collapse.

    Args:
        n:           State dimension (default: 6 for [px, py, pz, vx, vy, vz]).
        d:           Transformer embedding dimension (default: 64).
        L:           Number of Transformer Encoder layers (default: 2).
        H_heads:     Number of attention heads (default: 1).
        h_lstm:      LSTM hidden state dimension (default: 128).
        T:           Prediction horizon in steps (default: 80).
        sigma_min:   Minimum sigma value to prevent collapse (default: 0.01).
    """

    def __init__(
        self,
        n: int = 6,
        d: int = 64,
        L: int = 2,
        H_heads: int = 1,
        h_lstm: int = 128,
        T: int = 80,
        sigma_min: float = 0.01,
    ) -> None:
        super().__init__()
        self.T = T
        self.n = n
        self.sigma_min = sigma_min

        # Stage 1: Embedding + positional encoding + Transformer Encoder
        self.embedding = nn.Linear(n, d)
        self.pos_enc = SinusoidalPositionalEncoding(d_model=d)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=H_heads,
            batch_first=True,
            dropout=0.0,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=L)

        # Stage 2: LSTM state initialisation from encoder output
        #   h0 = tanh(W_h @ Mean(E_out))   — mean pool over time
        #   c0 = tanh(W_c @ Max(E_out))    — max pool over time
        self.h_init = nn.Linear(d, h_lstm)
        self.c_init = nn.Linear(d, h_lstm)

        # Stage 2: Autoregressive LSTM decoder
        # input_size=n because each step receives the raw state vector
        self.lstm = nn.LSTM(input_size=n, hidden_size=h_lstm, batch_first=True)

        # Output heads: mu and raw_sigma (sigma processed via softplus + floor)
        self.mu_head = nn.Linear(h_lstm, n)
        self.sigma_head = nn.Linear(h_lstm, n)

    def forward(
        self,
        X_hat: torch.Tensor,
        Y_gt: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the model forward pass.

        Args:
            X_hat: Noisy history tensor of shape ``(batch, m, n)``.
            Y_gt:  Optional ground-truth future tensor ``(batch, T, n)`` for
                   teacher forcing. If ``None``, model always uses its own
                   reparameterised samples as next-step input.
            teacher_forcing_ratio: Probability in [0, 1] of using ``Y_gt``
                   at each decoder step. ``1.0`` = fully teacher-forced
                   (training); ``0.0`` = fully free-running (inference).
                   Uses ``torch.rand(1).item()`` for the per-step coin flip.

        Returns:
            A tuple ``(mu, sigma)`` each of shape ``(batch, T, n)``.
            ``sigma >= sigma_min`` everywhere.
        """
        # ------------------------------------------------------------------
        # Stage 1: Embed, encode via Transformer
        # ------------------------------------------------------------------
        E = self.embedding(X_hat)   # (batch, m, d)
        E = self.pos_enc(E)         # (batch, m, d) — adds sinusoidal PE
        E_out = self.transformer(E) # (batch, m, d)

        # ------------------------------------------------------------------
        # LSTM hidden state initialisation
        #   mean-pool → h0,  max-pool → c0
        # nn.LSTM expects hidden shape: (num_layers=1, batch, h_lstm)
        # ------------------------------------------------------------------
        h0 = torch.tanh(self.h_init(E_out.mean(dim=1)))           # (batch, h_lstm)
        c0 = torch.tanh(self.c_init(E_out.max(dim=1).values))     # (batch, h_lstm)
        h = h0.unsqueeze(0)   # (1, batch, h_lstm)
        c = c0.unsqueeze(0)   # (1, batch, h_lstm)

        # ------------------------------------------------------------------
        # Stage 2: Autoregressive LSTM decoding
        # Start token = last observed state from history
        # ------------------------------------------------------------------
        x_prev = X_hat[:, -1, :]   # (batch, n)

        mus: list[torch.Tensor] = []
        sigmas: list[torch.Tensor] = []

        for t in range(self.T):
            lstm_in = x_prev.unsqueeze(1)             # (batch, 1, n)
            out, (h, c) = self.lstm(lstm_in, (h, c))  # out: (batch, 1, h_lstm)
            hidden = out.squeeze(1)                    # (batch, h_lstm)

            mu_t = self.mu_head(hidden)                # (batch, n)
            raw_sigma_t = self.sigma_head(hidden)      # (batch, n)
            # softplus ensures positivity; sigma_min provides the floor
            sigma_t = F.softplus(raw_sigma_t) + self.sigma_min

            mus.append(mu_t)
            sigmas.append(sigma_t)

            # Select next input: teacher forcing or reparameterised sample
            if Y_gt is not None and torch.rand(1).item() < teacher_forcing_ratio:
                x_prev = Y_gt[:, t, :]
            else:
                # Reparameterization: x ~ N(mu, sigma)
                eps = torch.randn_like(mu_t)
                x_prev = mu_t + sigma_t * eps

        mu = torch.stack(mus, dim=1)      # (batch, T, n)
        sigma = torch.stack(sigmas, dim=1)  # (batch, T, n)
        return mu, sigma
