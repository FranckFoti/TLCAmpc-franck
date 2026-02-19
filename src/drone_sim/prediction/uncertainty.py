"""Uncertainty propagation: AV-LSTM sigma outputs to per-step safety radii.

Formula (from lstm.md, position-only approximation):
  Sigma_p(t) = diag(sigma[t, :3]^2)   -- position-only 3D covariance
  lambda_max(t) = max(sigma[t, :3])^2  -- largest positional eigenvalue
  r_safety(t) = k_alpha * sqrt(lambda_max(t)) + r_ego

Full Jacobian propagation (J * Sigma_input * J^T) is intentionally skipped
per Phase 22 specification (Pitfall #7).
"""
from __future__ import annotations

import numpy as np


class UncertaintyPropagator:
   """Converts AV-LSTM sigma outputs to per-step safety radii.

   Formula (from lstm.md):
     Sigma_p(t) = diag(sigma[t, :3]^2)   -- position-only 3D covariance
     lambda_max(t) = max(sigma[t, 0:3])^2 -- largest positional eigenvalue
     r_safety(t) = k_alpha * sqrt(lambda_max(t)) + r_ego

   Full Jacobian propagation (J * Sigma_input * J^T) is intentionally skipped.

   :param k_alpha: Confidence scale factor (1.96 for 95% coverage).
   :param r_ego: Ego radius, typically drone.radius.
   :param r_safety_max: Hard cap preventing geometric infeasibility.
   """

   def __init__(
      self,
      k_alpha: float = 1.96,
      r_ego: float = 0.2,
      r_safety_max: float = 5.0,
   ) -> None:
      self._k_alpha = k_alpha
      self._r_ego = r_ego
      self._r_safety_max = r_safety_max

   def compute_safety_radii(self, sigma: np.ndarray, r_floor: float) -> np.ndarray:
      """Convert sigma (T, n) to safety radii (T,).

      Uses position-only dimensions sigma[:, :3] (px, py, pz).
      Velocity dimensions (3:6) are intentionally excluded — velocity
      uncertainty does not directly describe position spread in this
      approximation.

      :param sigma: Per-step std dev from model, shape (T, n), n>=3.
      :param r_floor: Minimum safety radius — caller passes drone.safety_zone.
      :return: Safety radii (T,), floored at r_floor, capped at r_safety_max.
      """
      sigma_pos = sigma[:, :3]                         # (T, 3) position only
      lambda_max = np.max(sigma_pos, axis=1) ** 2      # (T,) largest eigenvalue of diag cov
      r_lstm = self._k_alpha * np.sqrt(lambda_max) + self._r_ego  # (T,)
      r_safety = np.maximum(r_lstm, r_floor)           # floor: Pitfall #9
      r_safety = np.minimum(r_safety, self._r_safety_max)  # cap: Pitfall #8
      return r_safety
