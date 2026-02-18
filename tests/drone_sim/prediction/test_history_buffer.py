"""Tests for drone_sim.prediction.history_buffer module.

Tests for:
- TrajectoryHistoryBuffer — thread-safe ring buffer of 6D drone states
- update, get_window, is_ready, drone_ids, reset public API
"""

from __future__ import annotations

import threading

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from drone_sim.prediction.history_buffer import TrajectoryHistoryBuffer


class TestTrajectoryHistoryBufferBasic:
   """Tests for basic buffer behaviour."""

   def test_get_window_before_full_returns_none(self):
      """Buffer returns None when fewer than m states recorded."""
      buf = TrajectoryHistoryBuffer(m=20)
      state = np.zeros(6)
      buf.update("drone_0", state)
      assert buf.get_window("drone_0") is None

   def test_get_window_when_full_returns_array(self):
      """Buffer returns (m, 6) array once m states recorded."""
      buf = TrajectoryHistoryBuffer(m=20)
      for i in range(20):
         buf.update("drone_0", np.full(6, float(i)))
      window = buf.get_window("drone_0")
      assert window is not None
      assert window.shape == (20, 6)

   def test_get_window_contains_correct_values(self):
      """Window values match the states that were inserted."""
      buf = TrajectoryHistoryBuffer(m=5)
      for i in range(5):
         buf.update("drone_0", np.full(6, float(i)))
      window = buf.get_window("drone_0")
      for i in range(5):
         assert_array_equal(window[i], np.full(6, float(i)))

   def test_ring_eviction_keeps_last_m_states(self):
      """After m+5 updates, get_window returns the last m states."""
      buf = TrajectoryHistoryBuffer(m=20)
      for i in range(25):
         buf.update("drone_0", np.full(6, float(i)))
      window = buf.get_window("drone_0")
      assert window is not None
      assert window.shape == (20, 6)
      # Last 20 states: indices 5..24
      for row_idx, state_idx in enumerate(range(5, 25)):
         assert_array_equal(window[row_idx], np.full(6, float(state_idx)))

   def test_get_window_unknown_drone_returns_none(self):
      """get_window on unknown drone ID returns None."""
      buf = TrajectoryHistoryBuffer(m=20)
      assert buf.get_window("ghost_drone") is None


class TestTrajectoryHistoryBufferIsReady:
   """Tests for is_ready()."""

   def test_not_ready_after_19_updates(self):
      """is_ready returns False after 19 updates (m=20)."""
      buf = TrajectoryHistoryBuffer(m=20)
      for _ in range(19):
         buf.update("drone_0", np.zeros(6))
      assert buf.is_ready("drone_0") is False

   def test_ready_after_20_updates(self):
      """is_ready returns True after exactly m updates."""
      buf = TrajectoryHistoryBuffer(m=20)
      for _ in range(20):
         buf.update("drone_0", np.zeros(6))
      assert buf.is_ready("drone_0") is True

   def test_not_ready_for_unknown_drone(self):
      """is_ready returns False for unknown drone."""
      buf = TrajectoryHistoryBuffer(m=20)
      assert buf.is_ready("ghost") is False


class TestTrajectoryHistoryBufferDroneIds:
   """Tests for drone_ids()."""

   def test_drone_ids_empty_initially(self):
      """drone_ids returns empty list on fresh buffer."""
      buf = TrajectoryHistoryBuffer()
      assert buf.drone_ids() == []

   def test_drone_ids_contains_updated_drones(self):
      """drone_ids returns all drones that received at least one update."""
      buf = TrajectoryHistoryBuffer()
      buf.update("drone_0", np.zeros(6))
      buf.update("drone_1", np.zeros(6))
      ids = buf.drone_ids()
      assert sorted(ids) == ["drone_0", "drone_1"]


class TestTrajectoryHistoryBufferReset:
   """Tests for reset()."""

   def test_reset_clears_all_buffers(self):
      """reset() empties all drone histories."""
      buf = TrajectoryHistoryBuffer(m=5)
      for _ in range(5):
         buf.update("drone_0", np.zeros(6))
      buf.reset()
      assert buf.drone_ids() == []
      assert buf.get_window("drone_0") is None

   def test_can_update_after_reset(self):
      """Buffer accepts new states after reset."""
      buf = TrajectoryHistoryBuffer(m=3)
      for _ in range(3):
         buf.update("drone_0", np.ones(6))
      buf.reset()
      for _ in range(3):
         buf.update("drone_0", np.full(6, 2.0))
      window = buf.get_window("drone_0")
      assert window is not None
      assert_array_equal(window, np.full((3, 6), 2.0))


class TestTrajectoryHistoryBufferCustomM:
   """Tests for custom window size."""

   def test_custom_m_fills_after_m_updates(self):
      """TrajectoryHistoryBuffer(m=5) becomes ready after exactly 5 updates."""
      buf = TrajectoryHistoryBuffer(m=5)
      for i in range(4):
         buf.update("d", np.zeros(6))
      assert buf.is_ready("d") is False
      buf.update("d", np.zeros(6))
      assert buf.is_ready("d") is True
      window = buf.get_window("d")
      assert window is not None
      assert window.shape == (5, 6)


class TestTrajectoryHistoryBufferStateCopy:
   """Tests that stored state is a defensive copy."""

   def test_mutating_input_does_not_affect_buffer(self):
      """Mutating the array passed to update() does not change stored state."""
      buf = TrajectoryHistoryBuffer(m=1)
      state = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
      buf.update("drone_0", state)
      state[:] = 0.0  # mutate original
      window = buf.get_window("drone_0")
      assert window is not None
      assert_array_equal(window[0], np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))


class TestTrajectoryHistoryBufferThreadSafety:
   """Thread-safety stress test."""

   def test_concurrent_read_write_no_corruption(self):
      """Concurrent updates and reads from multiple threads do not corrupt data."""
      buf = TrajectoryHistoryBuffer(m=20)
      errors: list[str] = []

      def writer(drone_id: str, n: int) -> None:
         for i in range(n):
            buf.update(drone_id, np.full(6, float(i)))

      def reader(drone_id: str, n: int) -> None:
         for _ in range(n):
            window = buf.get_window(drone_id)
            if window is not None:
               if window.shape != (20, 6):
                  errors.append(f"Bad shape: {window.shape}")

      n_iters = 200
      threads = []
      for i in range(4):
         drone_id = f"drone_{i}"
         t_w = threading.Thread(target=writer, args=(drone_id, n_iters))
         t_r = threading.Thread(target=reader, args=(drone_id, n_iters))
         threads.extend([t_w, t_r])

      for t in threads:
         t.start()
      for t in threads:
         t.join()

      assert errors == [], f"Thread-safety violations: {errors}"
