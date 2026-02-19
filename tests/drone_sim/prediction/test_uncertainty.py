"""Unit tests for UncertaintyPropagator.

Tests the position-only sigma -> safety radius conversion formula:
  lambda_max(t) = max(sigma[t, :3])^2
  r_safety(t) = k_alpha * sqrt(lambda_max(t)) + r_ego
with floor at r_floor and cap at r_safety_max.
"""
import numpy as np
import pytest

from drone_sim.prediction.uncertainty import UncertaintyPropagator


class TestUncertaintyPropagator:
   """Unit tests for UncertaintyPropagator.compute_safety_radii()."""

   def test_basic_formula(self):
      """sigma all 0.1 => r_lstm = k_alpha * 0.1 + r_ego, floor not triggered."""
      k_alpha = 1.96
      r_ego = 0.2
      r_safety_max = 5.0
      propagator = UncertaintyPropagator(k_alpha=k_alpha, r_ego=r_ego, r_safety_max=r_safety_max)

      # sigma (5, 6): position dims all 0.1, velocity dims also 0.1
      sigma = np.full((5, 6), 0.1)
      r_floor = 0.0  # no floor so we can see the raw formula result

      result = propagator.compute_safety_radii(sigma, r_floor=r_floor)

      expected = k_alpha * 0.1 + r_ego  # 1.96 * 0.1 + 0.2 = 0.396
      assert result.shape == (5,)
      np.testing.assert_allclose(result, expected, rtol=1e-6)

   def test_position_only_3d(self):
      """Velocity dims (3:6) are large but must NOT affect the result — only :3 used."""
      k_alpha = 1.96
      r_ego = 0.2
      r_safety_max = 5.0
      propagator = UncertaintyPropagator(k_alpha=k_alpha, r_ego=r_ego, r_safety_max=r_safety_max)

      T = 4
      sigma = np.zeros((T, 6))
      sigma[:, :3] = 0.05   # position dims: small
      sigma[:, 3:6] = 10.0  # velocity dims: large — should be ignored

      r_floor = 0.0

      result = propagator.compute_safety_radii(sigma, r_floor=r_floor)

      # Result must use only :3 dims; if velocity dims were included, result would be ~20
      epsilon = 1e-6
      upper_bound = k_alpha * 0.05 + r_ego + epsilon
      assert result.shape == (T,)
      assert np.all(result < upper_bound), (
         f"Result {result} should be < {upper_bound} — velocity dims must be excluded"
      )

   def test_floor_applied(self):
      """sigma very small so r_lstm < r_floor — floor must dominate."""
      propagator = UncertaintyPropagator(k_alpha=1.96, r_ego=0.2, r_safety_max=5.0)

      # sigma tiny: r_lstm = 1.96 * 0.001 + 0.2 = 0.2020 < r_floor=1.0
      sigma = np.full((3, 6), 0.001)
      r_floor = 1.0

      result = propagator.compute_safety_radii(sigma, r_floor=r_floor)

      assert result.shape == (3,)
      assert np.all(result >= r_floor), f"All r_safety must be >= r_floor={r_floor}, got {result}"

   def test_cap_applied(self):
      """sigma very large so r_lstm > r_safety_max — cap must dominate."""
      r_safety_max = 5.0
      propagator = UncertaintyPropagator(k_alpha=1.96, r_ego=0.2, r_safety_max=r_safety_max)

      # sigma=100: r_lstm = 1.96 * 100 + 0.2 = 196.2 >> 5.0
      sigma = np.full((4, 6), 100.0)
      r_floor = 0.0

      result = propagator.compute_safety_radii(sigma, r_floor=r_floor)

      assert result.shape == (4,)
      assert np.all(result <= r_safety_max), (
         f"All r_safety must be <= r_safety_max={r_safety_max}, got {result}"
      )

   def test_floor_and_cap_together(self):
      """Mixed sigma array: floor triggered on small, cap triggered on large, middle unaffected."""
      k_alpha = 1.96
      r_ego = 0.2
      r_floor = 0.5
      r_safety_max = 3.0
      propagator = UncertaintyPropagator(k_alpha=k_alpha, r_ego=r_ego, r_safety_max=r_safety_max)

      # Step 0: sigma_pos=0.001 => r_lstm=0.2020 < floor => result=0.5
      # Step 1: sigma_pos=0.5   => r_lstm=1.96*0.5+0.2=1.18 in (0.5,3.0) => result~1.18
      # Step 2: sigma_pos=50.0  => r_lstm=1.96*50+0.2=98.2 > cap => result=3.0
      sigma = np.zeros((3, 6))
      sigma[0, :3] = 0.001
      sigma[1, :3] = 0.5
      sigma[2, :3] = 50.0

      result = propagator.compute_safety_radii(sigma, r_floor=r_floor)

      assert result.shape == (3,)
      # Step 0: floor
      assert result[0] == pytest.approx(r_floor, rel=1e-6)
      # Step 1: in-between (not floored, not capped)
      expected_mid = k_alpha * 0.5 + r_ego  # 1.96*0.5+0.2 = 1.18
      assert result[1] == pytest.approx(expected_mid, rel=1e-4)
      # Step 2: cap
      assert result[2] == pytest.approx(r_safety_max, rel=1e-6)

   def test_output_shape(self):
      """Input (H, 6) must produce output (H,)."""
      propagator = UncertaintyPropagator()
      for H in [1, 5, 10, 80]:
         sigma = np.full((H, 6), 0.1)
         result = propagator.compute_safety_radii(sigma, r_floor=0.5)
         assert result.shape == (H,), f"Expected shape ({H},), got {result.shape}"

   def test_heterogeneous_sigma(self):
      """Different sigma per step — each step computed independently."""
      k_alpha = 1.96
      r_ego = 0.2
      propagator = UncertaintyPropagator(k_alpha=k_alpha, r_ego=r_ego, r_safety_max=10.0)

      sigma_values = [0.1, 0.2, 0.3, 0.4, 0.5]
      T = len(sigma_values)
      sigma = np.zeros((T, 6))
      for i, v in enumerate(sigma_values):
         sigma[i, :3] = v
         sigma[i, 3:6] = 0.0  # velocity dims zero

      r_floor = 0.0
      result = propagator.compute_safety_radii(sigma, r_floor=r_floor)

      assert result.shape == (T,)
      for i, v in enumerate(sigma_values):
         expected = k_alpha * v + r_ego
         assert result[i] == pytest.approx(expected, rel=1e-5), (
            f"Step {i}: expected {expected}, got {result[i]}"
         )
