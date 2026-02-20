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


class TestLSTMSafetyZoneProviderLookAhead:
   """Verify that look_ahead selects the correct sigma step.

   Uses a mock model that returns sigma growing linearly with step index:
     sigma[t] = sigma_min + t * step_size
   This makes it easy to verify which step index is being used.
   """

   def _make_provider_with_mock_model(
      self,
      horizon: int,
      look_ahead: int | None,
      T: int = 80,
      sigma_min: float = 0.01,
      sigma_step: float = 0.01,
   ):
      """Create a provider with a mock LSTM model returning growing sigma."""
      import numpy as np
      import torch
      from unittest.mock import MagicMock
      from drone_sim.prediction.safety_zone_provider import LSTMSafetyZoneProvider
      from drone_sim.prediction.uncertainty import UncertaintyPropagator
      from drone_sim.prediction.history_buffer import TrajectoryHistoryBuffer

      # Build a sigma profile that grows linearly: sigma[t] = sigma_min + t * sigma_step
      # At step 19 (look_ahead=20): sigma_pos = 0.01 + 19 * 0.01 = 0.20
      # At step 0 (look_ahead=None, uses first 4): sigma_pos ≈ 0.01–0.04
      sigma_profile = np.zeros((T, 6))
      for t in range(T):
         sigma_profile[t, :] = sigma_min + t * sigma_step

      # Mock the model: returns (mu, sigma) as tensors
      mock_model = MagicMock()
      mu_out = torch.zeros(1, T, 6)
      sigma_out = torch.tensor(sigma_profile, dtype=torch.float32).unsqueeze(0)  # (1, T, 6)
      mock_model.return_value = (mu_out, sigma_out)

      # Mock loader
      mock_loader = MagicMock()
      mock_loader.model = mock_model

      propagator = UncertaintyPropagator(k_alpha=1.96, r_ego=0.2, r_safety_max=5.0)

      # Buffer with m=1 to always return a window (avoid buffer_not_full)
      buffer = TrajectoryHistoryBuffer(m=1)
      buffer.update("drone-1", np.zeros(6))

      return LSTMSafetyZoneProvider(
         mock_loader, propagator, buffer, horizon=horizon, look_ahead=look_ahead
      )

   def test_default_look_ahead_none_uses_first_steps(self):
      """look_ahead=None uses sigma[:horizon] — first H steps, small sigma."""
      import numpy as np

      H = 4
      provider = self._make_provider_with_mock_model(horizon=H, look_ahead=None)
      # sigma[0:4, :3] = 0.01, 0.02, 0.03, 0.04 → r_lstm ≈ 0.21..0.28 < r_floor=0.6
      result = provider.compute_neighbor_safety_radii(["drone-1"], {"drone-1": 0.6})
      r = result["drone-1"]
      assert r.shape == (H,)
      # With sigma_pos[0:4] = 0.01..0.04, all r_lstm < 0.6 → floor dominates
      np.testing.assert_allclose(r, 0.6, rtol=1e-4)

   def test_look_ahead_20_uses_step_19_sigma(self):
      """look_ahead=20 uses sigma[19] — larger sigma that may exceed floor."""
      import numpy as np

      H = 4
      # sigma[19, :3] = 0.01 + 19 * 0.01 = 0.20
      # r_lstm = 1.96 * 0.20 + 0.2 = 0.592 — still just below r_floor=0.6
      # Use look_ahead=21 (step 20): sigma[20, :3] = 0.21 → r_lstm = 1.96*0.21+0.2 = 0.612 > 0.6
      provider = self._make_provider_with_mock_model(
         horizon=H, look_ahead=21, sigma_step=0.01
      )
      result = provider.compute_neighbor_safety_radii(["drone-1"], {"drone-1": 0.6})
      r = result["drone-1"]
      assert r.shape == (H,)
      # sigma[20] = 0.21, r_lstm = 0.612 > r_floor=0.6 → NOT floor dominated
      assert np.all(r > 0.6), (
         f"Expected look_ahead=21 to produce r > r_floor=0.6 (r_lstm~0.612), got {r}"
      )
      # All H steps should be equal (tiled from the same step)
      np.testing.assert_allclose(r, r[0], rtol=1e-5,
         err_msg="look_ahead tiles the same sigma across all H steps — radii must be equal")

   def test_look_ahead_clamps_to_T(self):
      """look_ahead > T clamps to last step without error."""
      import numpy as np

      T = 80
      provider = self._make_provider_with_mock_model(
         horizon=4, look_ahead=T + 100, T=T  # way beyond T
      )
      # Should not raise; clamps to sigma[T-1]
      result = provider.compute_neighbor_safety_radii(["drone-1"], {"drone-1": 0.6})
      r = result["drone-1"]
      assert r.shape == (4,)
      # sigma[79] = 0.01 + 79 * 0.01 = 0.80 → r_lstm = 1.96*0.80+0.2 = 1.768 >> 0.6
      assert np.all(r > 1.0), f"Expected clamped step 79 to give large r, got {r}"
