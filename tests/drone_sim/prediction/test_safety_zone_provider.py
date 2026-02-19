"""Unit tests for LSTMSafetyZoneProvider.

Uses a real AVLSTMModel (small: n=6, d=16, L=1, H_heads=1, h_lstm=32, T=80)
with random weights saved to a tmp .pt file, loaded via LSTMModelLoader.
Uses a real UncertaintyPropagator and TrajectoryHistoryBuffer.
This avoids fragile mocking and tests the actual integration.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from drone_sim.prediction.history_buffer import TrajectoryHistoryBuffer
from drone_sim.prediction.model import AVLSTMModel
from drone_sim.prediction.model_loader import LSTMModelLoader
from drone_sim.prediction.safety_zone_provider import LSTMSafetyZoneProvider
from drone_sim.prediction.uncertainty import UncertaintyPropagator


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def provider_fixture():
   """Create a small model, save to tmp, load, and assemble LSTMSafetyZoneProvider."""
   # Small model with T=80 (model default)
   model_kwargs = dict(n=6, d=16, L=1, H_heads=1, h_lstm=32, T=80)
   model = AVLSTMModel(**model_kwargs)

   with tempfile.TemporaryDirectory() as tmpdir:
      ckpt_path = Path(tmpdir) / "test_model.pt"
      torch.save(model.state_dict(), ckpt_path)

      loader = LSTMModelLoader(ckpt_path, model_kwargs=model_kwargs)

   propagator = UncertaintyPropagator(k_alpha=1.96, r_ego=0.2, r_safety_max=5.0)
   buffer = TrajectoryHistoryBuffer(m=20)
   provider = LSTMSafetyZoneProvider(loader, propagator, buffer, horizon=5)

   return provider, buffer


def _fill_buffer(buffer: TrajectoryHistoryBuffer, drone_id: str, n_steps: int) -> None:
   """Helper: fill buffer with n_steps random 6D states for drone_id."""
   for _ in range(n_steps):
      state = np.random.randn(6).astype(float)
      buffer.update(drone_id, state)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLSTMSafetyZoneProvider:

   def test_buffer_not_full_fallback(self, provider_fixture):
      """Buffer has <20 steps for drone_0. Should fall back to np.full(H, r_floor)."""
      provider, buffer = provider_fixture
      buffer.reset()
      # Add only 5 steps (< 20 required)
      _fill_buffer(buffer, "drone_fallback", 5)

      result = provider.compute_neighbor_safety_radii(
         ["drone_fallback"], {"drone_fallback": 1.0}
      )

      expected = np.full(5, 1.0)
      np.testing.assert_array_almost_equal(result["drone_fallback"], expected)
      assert result["drone_fallback"].shape == (5,)

   def test_single_drone_full_buffer(self, provider_fixture):
      """Buffer full for drone_0. Should get LSTM-based radii with correct shape."""
      provider, buffer = provider_fixture
      buffer.reset()
      _fill_buffer(buffer, "drone_0", 20)

      result = provider.compute_neighbor_safety_radii(
         ["drone_0"], {"drone_0": 1.0}
      )

      assert "drone_0" in result
      assert result["drone_0"].shape == (5,)
      # All values >= r_floor=1.0
      assert np.all(result["drone_0"] >= 1.0)
      # All values <= r_safety_max=5.0
      assert np.all(result["drone_0"] <= 5.0)

   def test_multiple_drones(self, provider_fixture):
      """Both drone_0 and drone_1 have full buffers. Both keys present in result."""
      provider, buffer = provider_fixture
      buffer.reset()
      _fill_buffer(buffer, "drone_0", 20)
      _fill_buffer(buffer, "drone_1", 20)

      result = provider.compute_neighbor_safety_radii(
         ["drone_0", "drone_1"],
         {"drone_0": 0.5, "drone_1": 0.5},
      )

      assert "drone_0" in result
      assert "drone_1" in result
      assert result["drone_0"].shape == (5,)
      assert result["drone_1"].shape == (5,)

   def test_mixed_ready_and_not_ready(self, provider_fixture):
      """drone_0 full (20 steps), drone_1 partial (5 steps). Mixed output."""
      provider, buffer = provider_fixture
      buffer.reset()
      _fill_buffer(buffer, "drone_0", 20)
      _fill_buffer(buffer, "drone_1", 5)

      result = provider.compute_neighbor_safety_radii(
         ["drone_0", "drone_1"],
         {"drone_0": 1.0, "drone_1": 2.0},
      )

      # drone_0: LSTM radii (>= floor)
      assert result["drone_0"].shape == (5,)
      assert np.all(result["drone_0"] >= 1.0)

      # drone_1: fallback to r_floor=2.0
      np.testing.assert_array_almost_equal(result["drone_1"], np.full(5, 2.0))

   def test_output_shape_matches_horizon(self, provider_fixture):
      """horizon=5. Output shape must be (5,), not (80,)."""
      provider, buffer = provider_fixture
      buffer.reset()
      _fill_buffer(buffer, "drone_shape", 20)

      result = provider.compute_neighbor_safety_radii(
         ["drone_shape"], {"drone_shape": 0.5}
      )

      assert result["drone_shape"].shape == (5,), (
         f"Expected (5,), got {result['drone_shape'].shape}"
      )

   def test_floor_never_violated(self, provider_fixture):
      """r_floor=2.0. All radii must be >= 2.0."""
      provider, buffer = provider_fixture
      buffer.reset()
      _fill_buffer(buffer, "drone_floor", 20)

      result = provider.compute_neighbor_safety_radii(
         ["drone_floor"], {"drone_floor": 2.0}
      )

      assert np.all(result["drone_floor"] >= 2.0), (
         f"Floor violated: min={result['drone_floor'].min()}"
      )

   def test_cap_never_violated(self, provider_fixture):
      """r_safety_max=5.0 (set in propagator). All radii must be <= 5.0."""
      provider, buffer = provider_fixture
      buffer.reset()
      _fill_buffer(buffer, "drone_cap", 20)

      result = provider.compute_neighbor_safety_radii(
         ["drone_cap"], {"drone_cap": 0.1}
      )

      assert np.all(result["drone_cap"] <= 5.0), (
         f"Cap violated: max={result['drone_cap'].max()}"
      )

   def test_unknown_neighbor_id_in_r_floor(self, provider_fixture):
      """neighbor_id not in r_floor_by_id. Should use default r_floor=1.0."""
      provider, buffer = provider_fixture
      buffer.reset()
      # No states for "unknown_drone" — buffer not full, triggers fallback
      # with default r_floor from get(neighbor_id, 1.0)

      result = provider.compute_neighbor_safety_radii(
         ["unknown_drone"], {}  # empty r_floor_by_id
      )

      # Should fall back to np.full(5, 1.0) with default r_floor=1.0
      np.testing.assert_array_almost_equal(result["unknown_drone"], np.full(5, 1.0))
      assert result["unknown_drone"].shape == (5,)
