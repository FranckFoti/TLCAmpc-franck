"""Data generation pipeline for LSTM trajectory prediction.

Runs existing MPC scenarios programmatically, extracts sliding-window
(past_history, future_trajectory) pairs, applies observation noise to the
history, and writes the result as compressed NPZ files.

NPZ contract (Phase 21):
  X — noisy history windows,  shape (num_windows, m, 6)
  Y — clean future windows,   shape (num_windows, T, 6)

State vector convention: [px, py, pz, vx, vy, vz]
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from drone_sim.domain.config import ScenarioConfig
from drone_sim.prediction.history_buffer import TrajectoryHistoryBuffer
from drone_sim.simulation.simulator import Simulator

_log = logging.getLogger(__name__)

_DEFAULT_M: int = 20   # history window length
_DEFAULT_T: int = 80   # prediction horizon


class DataGenerator:
   """Generates training data for the LSTM trajectory predictor.

   Runs a :class:`~drone_sim.simulation.simulator.Simulator` scenario for a
   fixed number of steps, extracts sliding (history, future) window pairs
   from each drone's recorded 6D states, adds Gaussian observation noise to
   the history windows, and writes everything as a compressed NPZ file.

   Args:
      m:         History window length (past steps).  Default: 20.
      T:         Prediction horizon (future steps).   Default: 80.
      sigma_obs: Observation noise standard deviation — scalar or (6,) array.
                 Noise is added to history windows (X) only; future windows
                 (Y) are kept clean.  Default: 0.01.
      seed:      Random seed for reproducible noise.  Default: 42.
   """

   def __init__(
      self,
      m: int = _DEFAULT_M,
      T: int = _DEFAULT_T,
      sigma_obs: float | np.ndarray = 0.01,
      seed: int = 42,
   ) -> None:
      self._m = m
      self._T = T
      self._sigma_obs = np.broadcast_to(
         np.asarray(sigma_obs, dtype=float), (6,)
      ).copy()
      self._rng = np.random.default_rng(seed)

   # ------------------------------------------------------------------
   # Public API
   # ------------------------------------------------------------------

   def run_scenario(
      self,
      cfg: ScenarioConfig,
      num_steps: int,
      output_path: Path,
      scenario_name: str = "scenario",
   ) -> int:
      """Run one scenario and write a compressed NPZ file with windows.

      Args:
         cfg:           Scenario configuration.
         num_steps:     Number of simulation steps to run.
         output_path:   Directory where the NPZ file will be written.
         scenario_name: Stem of the output file (``{scenario_name}.npz``).

      Returns:
         Number of windows written.  Returns 0 when the scenario produces
         fewer than ``m + T`` feasible steps (no file is written in that
         case).
      """
      sim = Simulator.from_config(cfg)
      buf = TrajectoryHistoryBuffer(m=self._m)

      # full_history[drone_id] accumulates every *feasible* state vector.
      full_history: dict[str, list[np.ndarray]] = {
         d.drone_id: [] for d in sim.drones
      }

      for _ in range(num_steps):
         sim.step()
         if sim.infeasible:
            _log.debug("Infeasible step, skipping recording")
            continue
         for d in sim.drones:
            state = d.x.copy()
            full_history[d.drone_id].append(state)
            buf.update(d.drone_id, state)

      # ------------------------------------------------------------------
      # Extract sliding windows across all drones
      # ------------------------------------------------------------------
      X_list: list[np.ndarray] = []
      Y_list: list[np.ndarray] = []

      for drone_id, states_list in full_history.items():
         if not states_list:
            continue
         states = np.stack(states_list, axis=0)   # (num_recorded, 6)
         total = states.shape[0]

         for t in range(self._m, total - self._T + 1):
            X_list.append(states[t - self._m : t])        # (m, 6)
            Y_list.append(states[t : t + self._T])        # (T, 6)

      if not X_list:
         _log.warning(
            "Scenario '%s': no windows collected "
            "(num_steps=%d, m=%d, T=%d). File not written.",
            scenario_name,
            num_steps,
            self._m,
            self._T,
         )
         return 0

      X_arr = np.stack(X_list, axis=0)   # (num_windows, m, 6)
      Y_arr = np.stack(Y_list, axis=0)   # (num_windows, T, 6)

      # Apply observation noise to history only (vectorised, reproducible).
      noise = self._rng.normal(0.0, self._sigma_obs, size=X_arr.shape)
      X_hat = X_arr + noise

      # ------------------------------------------------------------------
      # Write NPZ
      # ------------------------------------------------------------------
      output_path = Path(output_path)
      output_path.mkdir(parents=True, exist_ok=True)
      out_file = output_path / f"{scenario_name}.npz"
      np.savez_compressed(str(out_file), X=X_hat, Y=Y_arr)

      num_windows = X_arr.shape[0]
      _log.info(
         "Scenario '%s': wrote %d windows to %s",
         scenario_name,
         num_windows,
         out_file,
      )
      return num_windows
