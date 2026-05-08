"""BoF (Backoff-Function) per-step safety-zone provider for DMPC neighbors.

Parallel to LSTMSafetyZoneProvider, but instead of LSTM inference + sigma propagation it delegates to a BoF adapter that returns both a predicted trajectory and per-step safety radii for each neighbor.

The class owns the orchestration: pulling each neighbor's history window from a TrajectoryHistoryBuffer, falling back to a constant ``r_floor`` when the buffer is not yet full, post-processing radii, and exposing the most
recent predicted trajectories. The actual BoF call lives in the adapter (``BoFLibraryAdapter`` / ``BoFRestAdapter``); see ``bof_adapter.py``.

Usage (matches LSTMSafetyZoneProvider — coordinators are duck-typed):

    adapter = BoFLibraryAdapter(horizon=H)            # or BoFRestAdapter(...)
    buffer = TrajectoryHistoryBuffer(m=100)
    provider = BoFSafetyZoneProvider(adapter=adapter, buffer=buffer, horizon=H)
    radii = provider.compute_neighbor_safety_radii(neighbor_ids, r_floor_by_id)
        # radii: {neighbor_id: np.ndarray of shape (H,)}
        # provider.predicted_trajectories: {neighbor_id: (H, 3)} for non-fallback hits
"""
from __future__ import annotations

import logging
import numpy as np

from drone_sim.prediction.bof_adapter import BoFAdapter
from drone_sim.prediction.history_buffer import TrajectoryHistoryBuffer

_log = logging.getLogger(__name__)


class BoFSafetyZoneProvider:
   """Provides BoF-based per-step safety radii for DMPC neighbors.

   Drop-in replacement for LSTMSafetyZoneProvider at the coordinator
   interface: same ``compute_neighbor_safety_radii(neighbor_ids, r_floor_by_id)`` signature returning a dict of (H,) arrays.

   :param adapter: BoF predictor adapter (library or REST). The adapter has its own (independent) horizon — typically much longer than the MPC horizon to support visualization and/or future longer-horizon planning.
   :param buffer: Shared TrajectoryHistoryBuffer; the simulator updates it once per step alongside the LSTM buffer.
   :param horizon: MPC horizon ``H_mpc``. The planner-facing return value of :meth:`compute_neighbor_safety_radii` is sliced to length ``H_mpc``. Adapter output is allowed (and expected) to be longer.
   :param r_safety_max: Hard cap on per-step radius (matches UncertaintyPropagator).
   """

   def __init__(self, adapter: BoFAdapter, buffer: TrajectoryHistoryBuffer, horizon: int, r_safety_max: float = 5.0) -> None:
      self._adapter = adapter
      self._buffer = buffer
      self._horizon = horizon
      self._r_safety_max = r_safety_max
      self._last_trajectories: dict[str, np.ndarray] = {}
      self._last_radii: dict[str, np.ndarray] = {}
      # FIX For BoF Drone-Id is an integer and not a string. !!!
      # Stable str -> int map used to key BoF's per-drone σ EMA. Insertion
      # order: first-seen neighbor gets 0, next gets 1, etc. Reset each
      # provider lifetime (i.e., each Simulator.from_config build).
      self._drone_id_map: dict[str, int] = {}

   def compute_neighbor_safety_radii(self, neighbor_ids: list[str], r_floor_by_id: dict[str, float]) -> dict[str, np.ndarray]:
      """Run the BoF adapter for each neighbor and return per-step safety radii.

      For neighbors whose history buffer is not yet full, falls back to a
      constant array ``np.full(H, r_floor)``.

      :param neighbor_ids: List of neighbor drone IDs.
      :param r_floor_by_id: Dict of neighbor_id -> drone.safety_zone (static floor). If a neighbor_id is missing, defaults to 1.0.
      :return: Dict of neighbor_id -> r_safety array (H,).
      """
      result: dict[str, np.ndarray] = {}
      # NOTE: don't reset _last_trajectories / _last_radii here. In DMPC, this method is called once per ego drone within a single timestep.
      # Resetting would let only the last ego drone's neighbors survive, hiding tubes for the remaining drones in the GUI.
      # The simulator clears the cache once per step via clear_step_cache() before the coordinator runs.

      for neighbor_id in neighbor_ids:
         r_floor = r_floor_by_id.get(neighbor_id, 1.0)

         payload = self._build_tool_payload(neighbor_id)
         if payload is None:
            _log.debug("BoF provider: neighbor=%s payload_unavailable -> fallback r_floor=%.3f", neighbor_id, r_floor)
            result[neighbor_id] = np.full(self._horizon, r_floor)
            continue

         trajectory, radii = self._call_bof_tool(payload)
         self._last_trajectories[neighbor_id] = np.asarray(trajectory, dtype=float)
         full_radii = self._postprocess_radii(radii, r_floor)
         if full_radii.shape[0] < self._horizon:
            raise ValueError(f"BoF tool returned {full_radii.shape[0]} radii, "
                             f"need >= mpc_horizon={self._horizon}")
         self._last_radii[neighbor_id] = full_radii
         result[neighbor_id] = full_radii[: self._horizon].copy()

         traj_arr = self._last_trajectories[neighbor_id]
         first_pt, last_pt = traj_arr[0, :3], traj_arr[-1, :3]
         _log.info("BoF provider: neighbor=%s traj %+0.3f,%+0.3f,%+0.3f -> %+0.3f,%+0.3f,%+0.3f (span=%.3fm, bof_H=%d) | mpc radii[:%d] mean=%.3f max=%.3f min=%.3f r_floor=%.3f",
                   neighbor_id, first_pt[0], first_pt[1], first_pt[2], last_pt[0], last_pt[1], last_pt[2], float(np.linalg.norm(last_pt - first_pt)), traj_arr.shape[0], 
                   self._horizon, float(result[neighbor_id].mean()), float(result[neighbor_id].max()), float(result[neighbor_id].min()), r_floor)

      return result

   def clear_step_cache(self) -> None:
      """Drop predictions/radii cached during the previous simulator step.

      Called by the simulator at the top of each step so the cache only holds entries from the current step.
      Resetting per-call would clobber earlier ego drones' neighbor predictions within the same step (DMPC calls this provider once per ego drone),
      leaving the GUI with only the last call's data.
      """
      self._last_trajectories = {}
      self._last_radii = {}

   @property
   def predicted_trajectories(self) -> dict[str, np.ndarray]:
      """Predicted trajectories from the most recent compute call.

      Returns a fresh dict copy so callers cannot mutate internal state.
      Each value is ``(H, 3)`` for neighbors that produced a payload.
      Missing keys mean the neighbor fell back to constant r_floor.
      """
      return dict(self._last_trajectories)

   @property
   def last_radii(self) -> dict[str, np.ndarray]:
      """Post-processed safety radii from the most recent compute call.

      Aligned with ``predicted_trajectories``: each value is ``(H,)`` and corresponds row-by-row to the ``(H, 3)`` trajectory. Missing keys mean the neighbor fell back to constant r_floor.
      """
      return dict(self._last_radii)

   def _build_tool_payload(self, neighbor_id: str) -> dict | None:
      """Pull this neighbor's history window from the shared buffer.

      Returns ``None`` when the buffer does not yet hold a full window, which makes ``compute_neighbor_safety_radii`` fall back to a constant ``r_floor`` array (mirrors the LSTM "buffer not full" branch).

      The payload also carries a stable int ``drone_id`` derived from ``neighbor_id`` (insertion-order counter) so BoF's per-drone σ EMA stays cleanly separated across neighbors.
      """
      window = self._buffer.get_window(neighbor_id)
      if window is None:
         return None
      return {"window": window, "drone_id": self._resolve_drone_id(neighbor_id)}

   def _resolve_drone_id(self, neighbor_id: str) -> int:
      """Return a stable int id for this neighbor, allocating on first sight."""
      if neighbor_id not in self._drone_id_map:
         self._drone_id_map[neighbor_id] = len(self._drone_id_map)
      return self._drone_id_map[neighbor_id]

   def _call_bof_tool(self, payload: dict) -> tuple[np.ndarray, np.ndarray]:
      """Delegate to the configured adapter.

      Adapter exceptions are logged with traceback and re-raised. Failures are intentionally visible to the caller rather than silently masked (consistent with the LSTM path's behavior on inference errors).

      :param payload: ``{"window": np.ndarray (m, 6), "drone_id": int}`` produced by ``_build_tool_payload``.
      :return: ``(trajectory (H, 3), radii (H,))``.
      """
      try:
         return self._adapter.predict(payload["window"], drone_id=payload.get("drone_id"))
      except Exception:
         _log.exception("BoF adapter call failed")
         raise

   def _postprocess_radii(self, radii: np.ndarray, r_floor: float) -> np.ndarray:
      """Floor at ``r_floor``, cap at ``r_safety_max``. Returns full length.

      Length check (>= mpc_horizon) is performed by the caller after the floor/cap so the cached ``_last_radii`` keeps the full bof-horizon array for visualization, not just the planner slice.

      Same floor/cap semantics as ``UncertaintyPropagator.compute_safety_radii`` so BoF and LSTM providers behave consistently downstream.
      """
      arr = np.asarray(radii, dtype=float).reshape(-1)
      arr = np.maximum(arr, r_floor)
      arr = np.minimum(arr, self._r_safety_max)
      return arr
