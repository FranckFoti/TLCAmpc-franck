"""Unit tests for BoFSafetyZoneProvider.

Exercises the orchestration the provider owns: pulling history from
``TrajectoryHistoryBuffer``, falling back to ``r_floor`` when the buffer is
not yet full, post-processing radii (floor + cap), and exposing the most
recent predicted trajectories. The actual BoF call is replaced with a
``FakeAdapter`` — we don't depend on Brian's predictor in unit tests.
"""
from __future__ import annotations

import numpy as np
import pytest

from drone_sim.prediction.bof_safety_zone_provider import BoFSafetyZoneProvider
from drone_sim.prediction.history_buffer import TrajectoryHistoryBuffer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeAdapter:
   """Returns fixed (trajectory, radii) and records predict() invocations."""

   def __init__(self, horizon: int, traj_value: float, radius_value: float):
      self.horizon = horizon
      self.traj_value = traj_value
      self.radius_value = radius_value
      self.calls: list[np.ndarray] = []
      self.drone_ids: list[int | None] = []

   def predict(self, history: np.ndarray, drone_id: int | None = None):
      self.calls.append(history)
      self.drone_ids.append(drone_id)
      traj = np.full((self.horizon, 3), self.traj_value, dtype=float)
      radii = np.full(self.horizon, self.radius_value, dtype=float)
      return traj, radii


class RaisingAdapter:
   def predict(self, history, drone_id: int | None = None):
      raise RuntimeError("boom")


def _fill(buffer: TrajectoryHistoryBuffer, drone_id: str, n: int) -> None:
   for _ in range(n):
      buffer.update(drone_id, np.random.randn(6))


# ---------------------------------------------------------------------------
# Public-API tests
# ---------------------------------------------------------------------------

class TestBoFSafetyZoneProviderFallback:

   def test_returns_floor_when_buffer_empty(self):
      adapter = FakeAdapter(horizon=4, traj_value=0.0, radius_value=99.0)
      buffer = TrajectoryHistoryBuffer(m=10)
      provider = BoFSafetyZoneProvider(adapter=adapter, buffer=buffer, horizon=4)

      result = provider.compute_neighbor_safety_radii(["d0"], {"d0": 1.7})

      np.testing.assert_array_almost_equal(result["d0"], np.full(4, 1.7))
      assert adapter.calls == []  # adapter must NOT be called

   def test_returns_floor_when_buffer_not_yet_full(self):
      adapter = FakeAdapter(horizon=3, traj_value=0.0, radius_value=99.0)
      buffer = TrajectoryHistoryBuffer(m=10)
      _fill(buffer, "d0", 5)  # only half full
      provider = BoFSafetyZoneProvider(adapter=adapter, buffer=buffer, horizon=3)

      result = provider.compute_neighbor_safety_radii(["d0"], {"d0": 2.0})

      np.testing.assert_array_almost_equal(result["d0"], np.full(3, 2.0))
      assert adapter.calls == []

   def test_unknown_neighbor_id_uses_default_floor_1_0(self):
      adapter = FakeAdapter(horizon=4, traj_value=0.0, radius_value=0.0)
      buffer = TrajectoryHistoryBuffer(m=10)
      provider = BoFSafetyZoneProvider(adapter=adapter, buffer=buffer, horizon=4)

      result = provider.compute_neighbor_safety_radii(["unknown"], {})

      np.testing.assert_array_almost_equal(result["unknown"], np.full(4, 1.0))


class TestBoFSafetyZoneProviderActive:

   def test_radii_floored_at_r_floor(self):
      """Adapter returns 0.5; r_floor=2.0 → output is exactly 2.0 everywhere."""
      adapter = FakeAdapter(horizon=4, traj_value=0.0, radius_value=0.5)
      buffer = TrajectoryHistoryBuffer(m=5)
      _fill(buffer, "d0", 5)
      provider = BoFSafetyZoneProvider(adapter=adapter, buffer=buffer, horizon=4)

      result = provider.compute_neighbor_safety_radii(["d0"], {"d0": 2.0})

      np.testing.assert_array_almost_equal(result["d0"], np.full(4, 2.0))
      assert len(adapter.calls) == 1

   def test_radii_capped_at_r_safety_max(self):
      adapter = FakeAdapter(horizon=4, traj_value=0.0, radius_value=99.0)
      buffer = TrajectoryHistoryBuffer(m=5)
      _fill(buffer, "d0", 5)
      provider = BoFSafetyZoneProvider(
         adapter=adapter, buffer=buffer, horizon=4, r_safety_max=5.0,
      )

      result = provider.compute_neighbor_safety_radii(["d0"], {"d0": 1.0})

      np.testing.assert_array_almost_equal(result["d0"], np.full(4, 5.0))

   def test_radii_pass_through_when_within_bounds(self):
      adapter = FakeAdapter(horizon=3, traj_value=0.0, radius_value=3.0)
      buffer = TrajectoryHistoryBuffer(m=5)
      _fill(buffer, "d0", 5)
      provider = BoFSafetyZoneProvider(adapter=adapter, buffer=buffer, horizon=3)

      result = provider.compute_neighbor_safety_radii(["d0"], {"d0": 1.0})

      np.testing.assert_array_almost_equal(result["d0"], np.full(3, 3.0))

   def test_window_passed_to_adapter_has_expected_shape(self):
      adapter = FakeAdapter(horizon=3, traj_value=0.0, radius_value=1.0)
      buffer = TrajectoryHistoryBuffer(m=8)
      _fill(buffer, "d0", 8)
      provider = BoFSafetyZoneProvider(adapter=adapter, buffer=buffer, horizon=3)

      provider.compute_neighbor_safety_radii(["d0"], {"d0": 1.0})

      assert len(adapter.calls) == 1
      assert adapter.calls[0].shape == (8, 6)


class TestPredictedTrajectories:

   def test_holds_last_compute_result(self):
      adapter = FakeAdapter(horizon=3, traj_value=7.0, radius_value=1.0)
      buffer = TrajectoryHistoryBuffer(m=4)
      _fill(buffer, "d0", 4)
      _fill(buffer, "d1", 4)
      provider = BoFSafetyZoneProvider(adapter=adapter, buffer=buffer, horizon=3)

      provider.compute_neighbor_safety_radii(["d0", "d1"], {"d0": 1.0, "d1": 1.0})
      trajs = provider.predicted_trajectories

      assert set(trajs) == {"d0", "d1"}
      assert trajs["d0"].shape == (3, 3)
      np.testing.assert_array_almost_equal(trajs["d0"], np.full((3, 3), 7.0))

   def test_accumulates_across_calls(self):
      """DMPC calls compute_neighbor_safety_radii once per ego drone within a
      single timestep; the cache must accumulate so the GUI can see tubes for
      every neighbor predicted that step. Resetting is the simulator's job
      (clear_step_cache, called once per step)."""
      adapter = FakeAdapter(horizon=3, traj_value=1.0, radius_value=1.0)
      buffer = TrajectoryHistoryBuffer(m=4)
      _fill(buffer, "d0", 4)
      _fill(buffer, "d1", 4)
      _fill(buffer, "d2", 4)
      provider = BoFSafetyZoneProvider(adapter=adapter, buffer=buffer, horizon=3)

      provider.compute_neighbor_safety_radii(["d0", "d1"], {"d0": 1.0, "d1": 1.0})
      provider.compute_neighbor_safety_radii(["d2"], {"d2": 1.0})

      assert set(provider.predicted_trajectories) == {"d0", "d1", "d2"}

   def test_clear_step_cache_drops_entries(self):
      adapter = FakeAdapter(horizon=3, traj_value=1.0, radius_value=1.0)
      buffer = TrajectoryHistoryBuffer(m=4)
      _fill(buffer, "d0", 4)
      _fill(buffer, "d1", 4)
      provider = BoFSafetyZoneProvider(adapter=adapter, buffer=buffer, horizon=3)

      provider.compute_neighbor_safety_radii(["d0", "d1"], {"d0": 1.0, "d1": 1.0})
      assert set(provider.predicted_trajectories) == {"d0", "d1"}

      provider.clear_step_cache()
      assert provider.predicted_trajectories == {}
      assert provider.last_radii == {}

   def test_skips_fallback_neighbors(self):
      adapter = FakeAdapter(horizon=3, traj_value=1.0, radius_value=1.0)
      buffer = TrajectoryHistoryBuffer(m=10)  # never filled → fallback
      provider = BoFSafetyZoneProvider(adapter=adapter, buffer=buffer, horizon=3)

      provider.compute_neighbor_safety_radii(["d0"], {"d0": 1.0})

      assert provider.predicted_trajectories == {}

   def test_returns_fresh_copy(self):
      adapter = FakeAdapter(horizon=3, traj_value=1.0, radius_value=1.0)
      buffer = TrajectoryHistoryBuffer(m=4)
      _fill(buffer, "d0", 4)
      provider = BoFSafetyZoneProvider(adapter=adapter, buffer=buffer, horizon=3)
      provider.compute_neighbor_safety_radii(["d0"], {"d0": 1.0})

      first = provider.predicted_trajectories
      first.clear()
      assert "d0" in provider.predicted_trajectories  # internal cache untouched


class TestErrorPaths:

   def test_adapter_exception_propagates(self):
      buffer = TrajectoryHistoryBuffer(m=4)
      _fill(buffer, "d0", 4)
      provider = BoFSafetyZoneProvider(
         adapter=RaisingAdapter(), buffer=buffer, horizon=3,
      )

      with pytest.raises(RuntimeError, match="boom"):
         provider.compute_neighbor_safety_radii(["d0"], {"d0": 1.0})

   def test_short_radii_array_raises(self):
      class _ShortAdapter:
         def predict(self, history, drone_id=None):
            return np.zeros((5, 3)), np.zeros(4)  # one too short for mpc_horizon=5

      buffer = TrajectoryHistoryBuffer(m=4)
      _fill(buffer, "d0", 4)
      provider = BoFSafetyZoneProvider(
         adapter=_ShortAdapter(), buffer=buffer, horizon=5,
      )

      with pytest.raises(ValueError, match="mpc_horizon=5"):
         provider.compute_neighbor_safety_radii(["d0"], {"d0": 1.0})


class TestDroneIdMapping:
   """Provider assigns insertion-order ints to neighbor strings."""

   def test_first_seen_neighbor_gets_zero(self):
      adapter = FakeAdapter(horizon=3, traj_value=0.0, radius_value=1.0)
      buffer = TrajectoryHistoryBuffer(m=4)
      _fill(buffer, "drone-A", 4)
      provider = BoFSafetyZoneProvider(adapter=adapter, buffer=buffer, horizon=3)

      provider.compute_neighbor_safety_radii(["drone-A"], {"drone-A": 1.0})

      assert adapter.drone_ids == [0]

   def test_distinct_neighbors_get_distinct_ids(self):
      adapter = FakeAdapter(horizon=3, traj_value=0.0, radius_value=1.0)
      buffer = TrajectoryHistoryBuffer(m=4)
      _fill(buffer, "alpha", 4)
      _fill(buffer, "beta", 4)
      _fill(buffer, "gamma", 4)
      provider = BoFSafetyZoneProvider(adapter=adapter, buffer=buffer, horizon=3)

      provider.compute_neighbor_safety_radii(
         ["alpha", "beta", "gamma"],
         {"alpha": 1.0, "beta": 1.0, "gamma": 1.0},
      )

      assert adapter.drone_ids == [0, 1, 2]

   def test_id_stable_across_calls(self):
      """Same neighbor must always get the same drone_id — BoF's σ EMA depends on it."""
      adapter = FakeAdapter(horizon=3, traj_value=0.0, radius_value=1.0)
      buffer = TrajectoryHistoryBuffer(m=4)
      _fill(buffer, "alpha", 4)
      _fill(buffer, "beta", 4)
      provider = BoFSafetyZoneProvider(adapter=adapter, buffer=buffer, horizon=3)

      provider.compute_neighbor_safety_radii(["alpha", "beta"], {"alpha": 1.0, "beta": 1.0})
      provider.compute_neighbor_safety_radii(["beta", "alpha"], {"alpha": 1.0, "beta": 1.0})

      # Call 1: alpha=0, beta=1.  Call 2 reverses query order: beta still 1, alpha still 0.
      assert adapter.drone_ids == [0, 1, 1, 0]

   def test_fallback_neighbor_does_not_consume_id(self):
      """Neighbor with empty buffer falls back without invoking adapter — must not bump counter."""
      adapter = FakeAdapter(horizon=3, traj_value=0.0, radius_value=1.0)
      buffer = TrajectoryHistoryBuffer(m=4)
      _fill(buffer, "alpha", 4)  # ready
      # 'beta' never filled -> fallback
      provider = BoFSafetyZoneProvider(adapter=adapter, buffer=buffer, horizon=3)

      provider.compute_neighbor_safety_radii(
         ["beta", "alpha"], {"alpha": 1.0, "beta": 1.0},
      )

      # alpha is the only one that hit the adapter — must have id 0,
      # not 1 (which would imply beta consumed an id during fallback).
      assert adapter.drone_ids == [0]


class TestDecoupledHorizons:
   """Adapter horizon (BoF lookahead) is independent of provider horizon (MPC)."""

   def test_planner_slice_shorter_than_bof_horizon(self):
      adapter = FakeAdapter(horizon=10, traj_value=2.0, radius_value=0.7)
      buffer = TrajectoryHistoryBuffer(m=4)
      _fill(buffer, "d0", 4)
      provider = BoFSafetyZoneProvider(adapter=adapter, buffer=buffer, horizon=3)

      result = provider.compute_neighbor_safety_radii(["d0"], {"d0": 0.5})

      # Planner sees the first mpc_horizon=3 entries.
      assert result["d0"].shape == (3,)
      np.testing.assert_array_almost_equal(result["d0"], np.full(3, 0.7))

   def test_full_bof_trajectory_and_radii_cached_for_viz(self):
      adapter = FakeAdapter(horizon=10, traj_value=2.0, radius_value=0.7)
      buffer = TrajectoryHistoryBuffer(m=4)
      _fill(buffer, "d0", 4)
      provider = BoFSafetyZoneProvider(adapter=adapter, buffer=buffer, horizon=3)

      provider.compute_neighbor_safety_radii(["d0"], {"d0": 0.5})

      # GUI tube reads full bof_horizon=10 length, not the planner slice.
      assert provider.predicted_trajectories["d0"].shape == (10, 3)
      assert provider.last_radii["d0"].shape == (10,)
      np.testing.assert_array_almost_equal(provider.last_radii["d0"], np.full(10, 0.7))

   def test_floor_and_cap_apply_to_full_length(self):
      adapter = FakeAdapter(horizon=10, traj_value=0.0, radius_value=0.05)
      buffer = TrajectoryHistoryBuffer(m=4)
      _fill(buffer, "d0", 4)
      provider = BoFSafetyZoneProvider(adapter=adapter, buffer=buffer, horizon=3, r_safety_max=5.0)

      provider.compute_neighbor_safety_radii(["d0"], {"d0": 0.6})

      # Floor (0.6) clamps the entire 10-element radii array — not just the
      # planner slice — so the GUI tube reflects what the planner used.
      np.testing.assert_array_almost_equal(provider.last_radii["d0"], np.full(10, 0.6))
