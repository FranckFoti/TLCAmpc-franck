"""Unit tests for AVLSTMModel and SinusoidalPositionalEncoding.

RED phase: Written before model.py implementation exists.
All tests must pass after the GREEN phase implementation.
"""
from __future__ import annotations

import torch
import pytest

from drone_sim.prediction.model import AVLSTMModel, SinusoidalPositionalEncoding


# ---------------------------------------------------------------------------
# Constants matching architecture spec (lstm.md + 21-01-PLAN.md)
# ---------------------------------------------------------------------------
BATCH = 4
M = 20   # history steps
T = 80   # prediction horizon
N = 6    # state dimension (px, py, pz, vx, vy, vz)
SIGMA_MIN = 0.01
D_Z = 32  # default latent dimension


@pytest.fixture
def model() -> AVLSTMModel:
    """Default model with architecture hyperparameters from spec."""
    return AVLSTMModel(n=N, d=64, L=2, H_heads=1, h_lstm=128, T=T, sigma_min=SIGMA_MIN)


@pytest.fixture
def x_hat() -> torch.Tensor:
    """Synthetic noisy history: (batch, m, n) float32."""
    torch.manual_seed(42)
    return torch.randn(BATCH, M, N)


@pytest.fixture
def y_gt() -> torch.Tensor:
    """Synthetic ground-truth future: (batch, T, n) float32."""
    torch.manual_seed(43)
    return torch.randn(BATCH, T, N)


# ---------------------------------------------------------------------------
# TestAVLSTMModel
# ---------------------------------------------------------------------------
class TestAVLSTMModel:
    def test_output_shapes(self, model: AVLSTMModel, x_hat: torch.Tensor) -> None:
        """Forward pass without teacher forcing produces correct shapes."""
        model.eval()
        with torch.no_grad():
            mu, sigma, mu_z, logvar_z = model(x_hat)
        assert mu.shape == (BATCH, T, N), f"Expected mu shape ({BATCH},{T},{N}), got {mu.shape}"
        assert sigma.shape == (BATCH, T, N), f"Expected sigma shape ({BATCH},{T},{N}), got {sigma.shape}"
        assert mu_z.shape == (BATCH, D_Z), f"Expected mu_z shape ({BATCH},{D_Z}), got {mu_z.shape}"
        assert logvar_z.shape == (BATCH, D_Z), f"Expected logvar_z shape ({BATCH},{D_Z}), got {logvar_z.shape}"

    def test_sigma_floor(self, model: AVLSTMModel, x_hat: torch.Tensor) -> None:
        """Sigma must be >= sigma_min everywhere — no sigma collapse possible."""
        model.eval()
        with torch.no_grad():
            _, sigma, _, _ = model(x_hat)
        assert (sigma >= SIGMA_MIN).all(), (
            f"sigma below sigma_min={SIGMA_MIN}: min={sigma.min().item():.6f}"
        )

    def test_free_running_mode(self, model: AVLSTMModel, x_hat: torch.Tensor) -> None:
        """Free-running inference: no Y_gt, teacher_forcing_ratio=0.0."""
        model.eval()
        with torch.no_grad():
            mu, sigma, mu_z, logvar_z = model(x_hat, Y_gt=None, teacher_forcing_ratio=0.0)
        assert mu.shape == (BATCH, T, N)
        assert sigma.shape == (BATCH, T, N)
        assert mu_z.shape == (BATCH, D_Z)
        assert logvar_z.shape == (BATCH, D_Z)

    def test_teacher_forced_mode(
        self, model: AVLSTMModel, x_hat: torch.Tensor, y_gt: torch.Tensor
    ) -> None:
        """Teacher-forced training mode: Y_gt provided, ratio=1.0."""
        model.train()
        mu, sigma, mu_z, logvar_z = model(x_hat, Y_gt=y_gt, teacher_forcing_ratio=1.0)
        assert mu.shape == (BATCH, T, N)
        assert sigma.shape == (BATCH, T, N)
        assert mu_z.shape == (BATCH, D_Z)
        assert logvar_z.shape == (BATCH, D_Z)

    def test_float32_dtype(self, model: AVLSTMModel, x_hat: torch.Tensor) -> None:
        """All output tensors must be float32 — no dtype mismatch."""
        model.eval()
        with torch.no_grad():
            mu, sigma, mu_z, logvar_z = model(x_hat)
        assert mu.dtype == torch.float32, f"mu dtype: {mu.dtype}"
        assert sigma.dtype == torch.float32, f"sigma dtype: {sigma.dtype}"
        assert mu_z.dtype == torch.float32, f"mu_z dtype: {mu_z.dtype}"
        assert logvar_z.dtype == torch.float32, f"logvar_z dtype: {logvar_z.dtype}"

    def test_batch_size_one(self, model: AVLSTMModel) -> None:
        """Single-sample batch (batch_size=1) must work correctly."""
        x = torch.randn(1, M, N)
        model.eval()
        with torch.no_grad():
            mu, sigma, mu_z, logvar_z = model(x)
        assert mu.shape == (1, T, N)
        assert sigma.shape == (1, T, N)
        assert mu_z.shape == (1, D_Z)
        assert logvar_z.shape == (1, D_Z)
        assert (sigma >= SIGMA_MIN).all()

    def test_partial_teacher_forcing(
        self, model: AVLSTMModel, x_hat: torch.Tensor, y_gt: torch.Tensor
    ) -> None:
        """Mixed teacher forcing ratio (0 < ratio < 1) produces correct shapes."""
        model.train()
        torch.manual_seed(0)
        mu, sigma, _, _ = model(x_hat, Y_gt=y_gt, teacher_forcing_ratio=0.5)
        assert mu.shape == (BATCH, T, N)
        assert sigma.shape == (BATCH, T, N)
        assert (sigma >= SIGMA_MIN).all()

    def test_sigma_positivity_all_modes(
        self, model: AVLSTMModel, x_hat: torch.Tensor, y_gt: torch.Tensor
    ) -> None:
        """Sigma floor holds in all forward modes (free-running and teacher-forced)."""
        model.eval()
        with torch.no_grad():
            _, s_free, _, _ = model(x_hat, teacher_forcing_ratio=0.0)
        model.train()
        _, s_forced, _, _ = model(x_hat, y_gt, teacher_forcing_ratio=1.0)
        assert (s_free >= SIGMA_MIN).all(), "Free-running: sigma below floor"
        assert (s_forced >= SIGMA_MIN).all(), "Teacher-forced: sigma below floor"

    def test_kl_divergence_computable(self, model: AVLSTMModel, x_hat: torch.Tensor) -> None:
        """mu_z and logvar_z can be used to compute a finite, non-negative KL divergence."""
        model.train()
        _, _, mu_z, logvar_z = model(x_hat)
        kl = -0.5 * torch.mean(1 + logvar_z - mu_z.pow(2) - logvar_z.exp())
        assert kl.shape == (), "KL should be a scalar"
        assert torch.isfinite(kl), f"KL is not finite: {kl.item()}"

    def test_inference_deterministic(self, model: AVLSTMModel, x_hat: torch.Tensor) -> None:
        """In eval mode, z = mu_z (no sampling), so the latent is deterministic.

        Note: the decoder loop still uses reparameterized sampling for x_prev
        when Y_gt is None, so mu/sigma differ across calls. But mu_z/logvar_z
        must be identical since z = mu_z in eval mode.
        """
        model.eval()
        with torch.no_grad():
            _, _, mu_z1, logvar_z1 = model(x_hat)
            _, _, mu_z2, logvar_z2 = model(x_hat)
        assert torch.allclose(mu_z1, mu_z2), "mu_z should be deterministic in eval mode"
        assert torch.allclose(logvar_z1, logvar_z2), "logvar_z should be deterministic in eval mode"


# ---------------------------------------------------------------------------
# TestSinusoidalPositionalEncoding
# ---------------------------------------------------------------------------
class TestSinusoidalPositionalEncoding:
    def test_positional_encoding_shape(self) -> None:
        """PE output shape matches input shape (batch, seq_len, d_model)."""
        d_model = 64
        seq_len = 20
        batch = 3
        pe = SinusoidalPositionalEncoding(d_model=d_model)
        x = torch.randn(batch, seq_len, d_model)
        out = pe(x)
        assert out.shape == x.shape, f"PE output shape {out.shape} != input shape {x.shape}"

    def test_positional_encoding_dtype(self) -> None:
        """PE output dtype is float32."""
        pe = SinusoidalPositionalEncoding(d_model=64)
        x = torch.randn(2, 10, 64)
        out = pe(x)
        assert out.dtype == torch.float32, f"PE output dtype: {out.dtype}"

    def test_positional_encoding_adds_to_input(self) -> None:
        """PE adds a non-zero signal to the input (output != input)."""
        pe = SinusoidalPositionalEncoding(d_model=64)
        torch.manual_seed(1)
        x = torch.randn(2, 20, 64)
        out = pe(x)
        assert not torch.allclose(out, x), "PE output identical to input — no PE added"
