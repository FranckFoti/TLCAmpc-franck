"""Thread-safe ring buffer of 6D drone state vectors.

State vector convention: [px, py, pz, vx, vy, vz]
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Optional

import numpy as np


class TrajectoryHistoryBuffer:
   """Stores the last *m* timesteps of 6D state per drone in a ring buffer.

   State vector layout: [px, py, pz, vx, vy, vz]

   Each drone's history is kept in a ``collections.deque(maxlen=m)`` so that
   once the buffer is full the oldest entry is automatically evicted on each
   new :meth:`update` call.

   Thread safety is provided by a single :class:`threading.RLock` which is
   acquired on every mutating or reading operation.

   Args:
      m: Window size — the number of timesteps to retain per drone.
         Defaults to 20.
   """

   def __init__(self, m: int = 20) -> None:
      self._m = m
      self._lock: threading.RLock = threading.RLock()
      self._buffers: dict[str, deque[np.ndarray]] = {}

   # ------------------------------------------------------------------
   # Public API
   # ------------------------------------------------------------------

   def update(self, drone_id: str, state: np.ndarray) -> None:
      """Record a new 6D state for *drone_id*.

      Creates a fresh deque for *drone_id* on first call.  The state is
      stored as a **copy** so that subsequent mutations of the caller's
      array do not affect the buffer contents.

      Args:
         drone_id: Identifier for the drone.
         state:    1-D array of shape (6,) — [px, py, pz, vx, vy, vz].
      """
      state = np.asarray(state, dtype=float).copy()
      with self._lock:
         if drone_id not in self._buffers:
            self._buffers[drone_id] = deque(maxlen=self._m)
         self._buffers[drone_id].append(state)

   def get_window(self, drone_id: str) -> Optional[np.ndarray]:
      """Return the current window for *drone_id* as a NumPy array.

      Returns:
         ``ndarray`` of shape ``(m, 6)`` with the last *m* states in
         chronological order when the buffer is full, or ``None`` when
         fewer than *m* states have been recorded or *drone_id* is
         unknown.
      """
      with self._lock:
         buf = self._buffers.get(drone_id)
         if buf is None or len(buf) < self._m:
            return None
         return np.stack(list(buf))

   def is_ready(self, drone_id: str) -> bool:
      """Return ``True`` iff *drone_id*'s buffer contains at least *m* states.

      Args:
         drone_id: Identifier for the drone.

      Returns:
         ``True`` when a full window is available, ``False`` otherwise.
      """
      with self._lock:
         buf = self._buffers.get(drone_id)
         return buf is not None and len(buf) >= self._m

   def drone_ids(self) -> list[str]:
      """Return the list of drone IDs that have at least one recorded state.

      Returns:
         A list of drone identifier strings in insertion order.
      """
      with self._lock:
         return list(self._buffers.keys())

   def reset(self) -> None:
      """Clear all drone histories.

      After this call :meth:`drone_ids` returns an empty list and
      :meth:`get_window` returns ``None`` for every drone.
      """
      with self._lock:
         self._buffers.clear()
