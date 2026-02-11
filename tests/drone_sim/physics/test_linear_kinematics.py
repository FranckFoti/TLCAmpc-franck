"""Tests for drone_sim.physics.linear_kinematics module.

Tests for:
- LinearKinematicsPhysics.__post_init__ (A, B matrix generation)
- LinearKinematicsPhysics.step (state update)
"""

from __future__ import annotations

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal

from drone_sim.physics.linear_kinematics import LinearKinematicsPhysics


class TestLinearKinematicsPhysicsInit:
   """Tests for LinearKinematicsPhysics initialization."""

   def test_init_matrix(self):
      """Test __post_init__ creates correct A and B matrices."""
      dt = 0.09
      physics = LinearKinematicsPhysics(dt=dt)

      assert physics.dt == 0.09
      assert physics.A.shape == (6, 6)
      assert physics.B.shape == (6, 3)

      expected_A = np.array([
         [1.0, 0.0, 0.0, dt, 0.0, 0.0],
         [0.0, 1.0, 0.0, 0.0, dt, 0.0],
         [0.0, 0.0, 1.0, 0.0, 0.0, dt],
         [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
         [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
      ])
      assert_array_almost_equal(physics.A, expected_A)

      half_dt_sq = 0.5 * dt ** 2
      expected_B = np.array([
         [half_dt_sq, 0.0, 0.0],
         [0.0, half_dt_sq, 0.0],
         [0.0, 0.0, half_dt_sq],
         [dt, 0.0, 0.0],
         [0.0, dt, 0.0],
         [0.0, 0.0, dt],
      ])
      assert_array_almost_equal(physics.B, expected_B)

   def test_init_different_dt_values(self):
      """Test A and B matrices scale correctly with different dt values."""
      dt1 = 0.1
      dt2 = 0.2
      physics1 = LinearKinematicsPhysics(dt=dt1)
      physics2 = LinearKinematicsPhysics(dt=dt2)

      # Position update coefficients in A should scale linearly with dt
      assert physics2.A[0, 3] == 2 * physics1.A[0, 3]

      # Velocity update coefficients in B should scale linearly with dt
      assert physics2.B[3, 0] == 2 * physics1.B[3, 0]


class TestLinearKinematicsPhysicsStep:
   """Tests for LinearKinematicsPhysics.step method."""

   def test_step_zero_control(self, sample_physics: LinearKinematicsPhysics):
      """Test step with zero control input maintains velocity and updates position."""
      x0 = np.array([1.0, 2.0, 3.0, 1.0, 0.0, 0.0])  # Moving in x direction
      u = np.array([0.0, 0.0, 0.0])

      x1 = sample_physics.step(x0, u)

      # Position should update by velocity * dt
      dt = sample_physics.dt
      expected_pos = x0[:3] + dt * x0[3:]
      assert_array_almost_equal(x1[:3], expected_pos)

      # Velocity should remain unchanged
      assert_array_almost_equal(x1[3:], x0[3:])

   def test_step_nonzero_control(self, sample_physics: LinearKinematicsPhysics):
      """Test step with nonzero control input accelerates the drone."""
      x0 = np.array([1.0, 2.0, 3.0, 0.5, -0.5, 0.2])  # At rest
      u = np.array([0.3, -0.2, 0.1]) # Accelerate in x

      x1 = sample_physics.step(x0, u)

      # has correct shape
      assert x1.shape == (6,)
      assert x1.dtype == float

      expected = sample_physics.A @ x0 + sample_physics.B @ u
      assert_array_almost_equal(x1, expected)

   @pytest.mark.skip(reason="v_max is clipped, so the test is invalid, maybe change or remove it")
   def test_step_large_control_values(self, sample_physics: LinearKinematicsPhysics):
      """Test step with large control values."""
      x0 = np.zeros(6)
      u = np.array([1000.0, -1000.0, 500.0])

      x1 = sample_physics.step(x0, u)

      # Should handle large values without overflow
      assert np.all(np.isfinite(x1))
      dt = sample_physics.dt
      assert x1[3] == pytest.approx(dt * 1000.0)
      assert x1[4] == pytest.approx(dt * -1000.0)

   def test_step_negative_control_values(self, sample_physics: LinearKinematicsPhysics):
      """Test step with negative control values (deceleration)."""
      x0 = np.array([0.0, 0.0, 0.0, 10.0, 10.0, 10.0])  # Moving fast
      u = np.array([-5.0, -5.0, -5.0])  # Decelerate

      x1 = sample_physics.step(x0, u)

      # Velocity should decrease
      assert x1[3] < x0[3]
      assert x1[4] < x0[4]
      assert x1[5] < x0[5]

   def test_step_very_small_dt(self):
      """Test step with very small dt value."""
      physics = LinearKinematicsPhysics(dt=0.0001)
      x0 = np.array([1.0, 2.0, 3.0, 1.0, 1.0, 1.0])
      u = np.array([1.0, 1.0, 1.0])

      x1 = physics.step(x0, u)

      # Position should barely change with small dt
      assert np.allclose(x1[:3], x0[:3], atol=1e-3)


class TestLinearKinematicsPhysicsEdgeCases:
   """Edge case tests for LinearKinematicsPhysics."""

   def test_step_extreme_positions(self):
      """Test step with extreme position values."""
      physics = LinearKinematicsPhysics(dt=0.1)
      x0 = np.array([1e9, 1e9, 1e9, 0.0, 0.0, 0.0])
      u = np.array([0.0, 0.0, 0.0])

      x1 = physics.step(x0, u)

      assert np.all(np.isfinite(x1))
      assert_array_almost_equal(x1[:3], x0[:3])

   def test_step_extreme_velocities(self):
      """Test step with extreme velocity values."""
      physics = LinearKinematicsPhysics(dt=0.1)
      x0 = np.array([0.0, 0.0, 0.0, 1e6, 1e6, 1e6])
      u = np.array([0.0, 0.0, 0.0])

      x1 = physics.step(x0, u)

      assert np.all(np.isfinite(x1))
      # Position should update significantly with high velocity
      assert x1[0] > 0

   def test_step_mixed_positive_negative(self):
      """Test step with mixed positive and negative values."""
      physics = LinearKinematicsPhysics(dt=0.1)
      x0 = np.array([-10.0, 20.0, -30.0, 5.0, -5.0, 0.0])
      u = np.array([-1.0, 1.0, -1.0])

      x1 = physics.step(x0, u)


      expected = physics.A @ x0 + physics.B @ u
      assert_array_almost_equal(x1[:3], expected[:3]) # position should be equal
      # remind the clipping
      clipped_vel = [3.53516583, -3.53516583, -0.07214624]
      assert_array_almost_equal(x1[3:], clipped_vel) # velocity is clipped to v_max
