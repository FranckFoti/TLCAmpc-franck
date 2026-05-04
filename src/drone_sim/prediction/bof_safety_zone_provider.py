"""BoF (Backoff-Function) per-step safety-zone provider for DMPC neighbors.

Parallel to LSTMSafetyZoneProvider, but instead of LSTM inference + sigma
propagation it delegates to an external Backoff-Function (BoF) tool that
returns both a predicted trajectory and per-step safety radii for each neighbor.

The class owns the orchestration (per-neighbor loop, fallback handling, post-processing radii, exposing the most recent predicted trajectories).
The BoF tool integration itself is a stub — fill in `_call_bof_tool`.

Usage (matches LSTMSafetyZoneProvider — coordinators are duck-typed):
    provider = BoFSafetyZoneProvider(horizon=H)
         # provider.predicted_trajectories holds the most recent (H, 3) trajectories
    radii = provider.compute_neighbor_safety_radii(neighbor_ids, r_floor_by_id)
         # radii: {neighbor_id: np.ndarray of shape (H,)}


"""
from __future__ import annotations

import logging
import numpy as np

_log = logging.getLogger(__name__)


class BoFSafetyZoneProvider:
   """Provides BoF-based per-step safety radii for DMPC neighbors.

@ To Brian:
   Drop-in replacement for LSTMSafetyZoneProvider at the coordinator
   interface: same `compute_neighbor_safety_radii(neighbor_ids, r_floor_by_id)`
   signature returning a dict of (H,) arrays.

   The actual Backoff-Function computation is delegated from rest api:
     - `_build_tool_payload(neighbor_id)`: assemble whatever inputs the tool needs.
            Default packs `{neighbor_id, horizon}`.
            Return `None` to signal "not enough data" — the class falls back to a constant `r_floor` array.
     - `_call_bof_tool(payload)`: the tool call itself.
            Stub raises NotImplementedError; fill it in to return (trajectory: (H, 3), radii: (H,)).

   Once `_call_bof_tool` is implemented, the rest of the simulation pipeline works without further changes.

   :param horizon: DMPC horizon H. Output radii arrays are (H,).
   :param r_safety_max: Hard cap on per-step radius (matches UncertaintyPropagator).
   """

   def __init__(self, horizon: int, r_safety_max: float = 5.0) -> None:
      self._horizon = horizon
      self._r_safety_max = r_safety_max
      self._last_trajectories: dict[str, np.ndarray] = {}

   def compute_neighbor_safety_radii(
      self,
      neighbor_ids: list[str],
      r_floor_by_id: dict[str, float],
   ) -> dict[str, np.ndarray]:
      """Run the BoF tool for each neighbor and return per-step safety radii.

      For neighbors whose payload-builder returns None (e.g. insufficient data), falls back to a constant array `np.full(H, r_floor)`.

      :param neighbor_ids: List of neighbor drone IDs.
      :param r_floor_by_id: Dict of neighbor_id -> drone.safety_zone (static floor). If a neighbor_id is missing, defaults to 1.0.
      :return: Dict of neighbor_id -> r_safety array (H,).
      """
      result: dict[str, np.ndarray] = {}
      self._last_trajectories = {}

      for neighbor_id in neighbor_ids:
         r_floor = r_floor_by_id.get(neighbor_id, 1.0)

         payload = self._build_tool_payload(neighbor_id)
         if payload is None:
            _log.debug(
                "BoF provider: neighbor=%s payload_unavailable -> fallback r_floor=%.3f",
                neighbor_id, r_floor,
            )
            result[neighbor_id] = np.full(self._horizon, r_floor)
            continue

         trajectory, radii = self._call_bof_tool(payload)
         self._last_trajectories[neighbor_id] = np.asarray(trajectory, dtype=float)
         result[neighbor_id] = self._postprocess_radii(radii, r_floor)

         _log.debug(
             "BoF provider: neighbor=%s radii mean=%.3f max=%.3f min=%.3f r_floor=%.3f",
             neighbor_id,
             float(result[neighbor_id].mean()),
             float(result[neighbor_id].max()),
             float(result[neighbor_id].min()),
             r_floor,
         )

      return result

   @property
   def predicted_trajectories(self) -> dict[str, np.ndarray]:
      """Predicted trajectories from the most recent compute call.

      Returns a fresh dict copy so callers cannot mutate internal state.
      Each value is (H, 3) for neighbors that produced a payload. A missing keys mean the neighbor fell back to constant r_floor.
      """
      return dict(self._last_trajectories)

   def _build_tool_payload(self, neighbor_id: str) -> dict | None:
      """Assemble the inputs the BoF tool needs for one neighbor.

      Default impl packs `{neighbor_id, horizon}`. Extend this to add whatever your BoF tool requires
      Return `None` to signal that data is not yet available,
      which makes the class fall back to a constant `r_floor` array (mirrors the LSTM "buffer not full" branch).

      :param neighbor_id: ID of the neighbor drone the tool will be called for.
      :return: Dict passed straight to `_call_bof_tool`, or None for fallback.
      """
      return {"neighbor_id": neighbor_id, "horizon": self._horizon}

   def _call_bof_tool(self, payload: dict) -> tuple[np.ndarray, np.ndarray]:
      """To Brian: implement the Backoff-Function tool call here.

      The class will call this once per neighbor per `solve_controls`.
      The payload dict is whatever `_build_tool_payload` produced, extend that helper to pass the inputs your tool needs.

      Return a tuple of:
        trajectory: shape (H, 3) # predicted neighbor positions over the horizon.
        radii:      shape (H,)   # per-step safety radii (raw; the class floors at r_floor and caps at r_safety_max).

      :param payload: Dict assembled by `_build_tool_payload`.
      :return: (trajectory, radii) numpy arrays.
      """
      raise NotImplementedError(
         "BoFSafetyZoneProvider._call_bof_tool is a stub. "
         "Implement the Backoff-Function tool call here. "
         "Return (trajectory: (H,3), radii: (H,))."
      )

   # ---------------------------------------------------------------------
   # Internal helpers.
   # ---------------------------------------------------------------------

   def _postprocess_radii(self, radii: np.ndarray, r_floor: float) -> np.ndarray:
      """Floor at `r_floor`, cap at `r_safety_max`, truncate/validate length.

      Same semantics as UncertaintyPropagator.compute_safety_radii so BoF and LSTM providers behave consistently downstream.
      """
      arr = np.asarray(radii, dtype=float).reshape(-1)
      if arr.shape[0] < self._horizon:
         raise ValueError(
            f"BoF tool returned {arr.shape[0]} radii, need >= horizon={self._horizon}"
         )
      arr = arr[: self._horizon]
      arr = np.maximum(arr, r_floor)
      arr = np.minimum(arr, self._r_safety_max)
      return arr
