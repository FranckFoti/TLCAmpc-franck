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

import numpy as np
import torch

from drone_sim.prediction.history_buffer import TrajectoryHistoryBuffer
from drone_sim.prediction.model_loader import LSTMModelLoader
from drone_sim.prediction.uncertainty import UncertaintyPropagator


class LSTMSafetyZoneProvider:
   """Provides LSTM-based per-step safety radii for DMPC neighbors.

   Runs AV-LSTM inference ONCE per compute call (outside SLSQP).
   Returns a dict mapping neighbor_id to r_safety array of shape (H,).

   :param loader: LSTMModelLoader with the loaded model.
   :param propagator: UncertaintyPropagator for sigma-to-radius conversion.
   :param buffer: TrajectoryHistoryBuffer shared with simulator state updates.
   :param horizon: DMPC horizon H. First H of model's T steps are used.
   """

   def __init__(
      self,
      loader: LSTMModelLoader,
      propagator: UncertaintyPropagator,
      buffer: TrajectoryHistoryBuffer,
      horizon: int,
   ) -> None:
      self._model = loader.model
      self._propagator = propagator
      self._buffer = buffer
      self._horizon = horizon

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
         window = self._buffer.get_window(neighbor_id)
         if window is None:
            result[neighbor_id] = np.full(self._horizon, r_floor)
            continue
         sigma = self._infer_sigma(window)
         sigma_h = sigma[: self._horizon]
         result[neighbor_id] = self._propagator.compute_safety_radii(sigma_h, r_floor)
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
