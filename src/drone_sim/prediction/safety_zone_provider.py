"""LSTM-based per-step safety radius provider for DMPC neighbors.

Composes TrajectoryHistoryBuffer, LSTMModelLoader, and UncertaintyPropagator
into a single entry point that DMPC coordinators call once per solve_controls()
invocation.

Usage:
   provider = LSTMSafetyZoneProvider(loader, propagator, buffer, horizon=H)
   radii = provider.compute_neighbor_safety_radii(neighbor_ids, r_floor_by_id)
   # radii: {neighbor_id: np.ndarray of shape (H,)}
"""
from __future__ import annotations

import logging
import numpy as np
import torch

from drone_sim.prediction.history_buffer import TrajectoryHistoryBuffer
from drone_sim.prediction.model_loader import LSTMModelLoader
from drone_sim.prediction.uncertainty import UncertaintyPropagator

_log = logging.getLogger(__name__)


class LSTMSafetyZoneProvider:
   """Provides LSTM-based per-step safety radii for DMPC neighbors.

   Runs AV-LSTM inference ONCE per compute call (outside SLSQP).
   Returns a dict mapping neighbor_id to r_safety array of shape (H,).

   :param loader: LSTMModelLoader with the loaded model.
   :param propagator: UncertaintyPropagator for sigma-to-radius conversion.
   :param buffer: TrajectoryHistoryBuffer shared with simulator state updates.
   :param horizon: DMPC horizon H. First H of model's T steps are used.
   :param look_ahead: Optional 1-indexed step to use for sigma extraction.
       When None (default), uses sigma[:horizon] — the first H steps, which
       have small near-term sigma (backward compatible).
       When set to N, uses sigma[N-1] (0-indexed) tiled across all H steps.
       sigma grows with autoregressive step count: look_ahead=20 gives ~T/4
       accumulated uncertainty, typically 3-5x larger than first-step sigma.
   """

   def __init__(
      self,
      loader: LSTMModelLoader,
      propagator: UncertaintyPropagator,
      buffer: TrajectoryHistoryBuffer,
      horizon: int,
      look_ahead: int | None = None,
   ) -> None:
      self._model = loader.model
      self._propagator = propagator
      self._buffer = buffer
      self._horizon = horizon
      self._look_ahead = look_ahead  # None = use sigma[:horizon] (backward compat)

   def compute_neighbor_safety_radii(
      self,
      neighbor_ids: list[str],
      r_floor_by_id: dict[str, float],
   ) -> dict[str, np.ndarray]:
      """Run inference for each neighbor and return per-step safety radii.

      For neighbors whose buffer is not yet full (< m steps), falls back to
      a constant array np.full(H, r_floor) using the static safety_zone value.

      :param neighbor_ids: List of neighbor drone IDs.
      :param r_floor_by_id: Dict of neighbor_id -> drone.safety_zone (static floor).
                            Caller's responsibility to pass drone.safety_zone per
                            neighbor. If a neighbor_id is missing, defaults to 1.0.
      :return: Dict of neighbor_id -> r_safety array (H,).
      """
      result = {}
      for neighbor_id in neighbor_ids:
         r_floor = r_floor_by_id.get(neighbor_id, 1.0)
         _log.debug(
             "LSTM provider: neighbor=%s r_floor=%.3f",
             neighbor_id, r_floor,
         )
         window = self._buffer.get_window(neighbor_id)
         if window is None:
            _log.debug(
                "LSTM provider: neighbor=%s buffer_not_full -> fallback r_floor=%.3f",
                neighbor_id, r_floor,
            )
            result[neighbor_id] = np.full(self._horizon, r_floor)
            continue
         sigma = self._infer_sigma(window)
         _log.debug(
             "LSTM provider: neighbor=%s sigma_pos_mean=%.4f sigma_pos_max=%.4f sigma_pos_min=%.4f (shape=%s)",
             neighbor_id,
             float(sigma[:, :3].mean()),
             float(sigma[:, :3].max()),
             float(sigma[:, :3].min()),
             sigma.shape,
         )
         if self._look_ahead is None:
            sigma_h = sigma[: self._horizon]
         else:
            # Use sigma at look_ahead-1 (0-indexed) for all H constraint steps.
            # Rationale: LSTM sigma grows with autoregressive step count; early steps
            # have near-zero sigma (model is confident about immediate future).
            # Using a later step gives accumulated uncertainty as a conservative proxy.
            idx = min(self._look_ahead - 1, sigma.shape[0] - 1)
            sigma_h = np.tile(sigma[idx], (self._horizon, 1))
         _log.debug(
             "LSTM provider: neighbor=%s look_ahead=%s using sigma[idx=%d] (T=%d)",
             neighbor_id,
             self._look_ahead,
             (min(self._look_ahead - 1, sigma.shape[0] - 1) if self._look_ahead is not None else 0),
             sigma.shape[0],
         )
         result[neighbor_id] = self._propagator.compute_safety_radii(sigma_h, r_floor)
         r_arr = result[neighbor_id]
         _log.debug(
             "LSTM provider: neighbor=%s final_radii mean=%.3f max=%.3f min=%.3f r_floor=%.3f",
             neighbor_id,
             float(r_arr.mean()),
             float(r_arr.max()),
             float(r_arr.min()),
             r_floor,
         )
      return result

   def _infer_sigma(self, window: np.ndarray) -> np.ndarray:
      """Run model inference, return sigma (T, n) as numpy.

      Runs inside torch.inference_mode() to disable gradient tracking.

      :param window: History window (m, 6) numpy array.
      :return: sigma (T, n) numpy array.
      """
      X = torch.from_numpy(window).float().unsqueeze(0)  # (1, m, n)
      with torch.inference_mode():
         _, sigma_t = self._model(X, Y_gt=None, teacher_forcing_ratio=0.0)
      return sigma_t.squeeze(0).numpy()  # (T, n)
