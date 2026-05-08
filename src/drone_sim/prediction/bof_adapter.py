"""Adapter layer between BoFSafetyZoneProvider and the BoF predictor.

Two implementations satisfy the same Protocol:

- ``BoFLibraryAdapter`` calls Brian's ``bof_tpt.engine.PredictionEngine`` directly, in-process via the ``from_config(horizon=H)``
      constructor and the TLCAmpc-style ``predict(history, has_velocity, drone_id)`` method.
- ``BoFRestAdapter`` POSTs to a FastAPI endpoint exposed by Brian's BoF service (``POST /api/predict``).
      Useful when BoF runs as a separate process/host.

Both adapters return ``(trajectory: (H, 3), radii: (H,))`` from a 6D history window. The provider that owns them treats the call as opaque.

The ``has_velocity`` flag controls whether velocity columns are forwarded to BoF. With ``has_velocity=False`` (default),
   we slice the buffer's 6D state to positions before sending. With ``has_velocity=True``, we send the full 6 columns and slice the returned trajectory
   back to ``(H, 3)`` for the provider's downstream contract.

The ``drone_id`` argument (passed by the provider via ``BoFSafetyZoneProvider``) keys BoF's per-drone σ EMA. Two distinct neighbors must therefore receive
   distinct ``drone_id`` ints, otherwise BoF would smooth their uncertainties together. The provider owns the str -> int mapping.
"""
from __future__ import annotations

import logging
import time
from typing import Protocol, runtime_checkable

import numpy as np

_log = logging.getLogger(__name__)


@runtime_checkable
class BoFAdapter(Protocol):
   """Per-neighbor BoF predictor.

   Implementations take the last ``m`` 6D states of one neighbor and return the predicted trajectory plus per-step safety radii over the horizon.
   ``drone_id`` keys BoF's per-drone σ EMA — must be unique per neighbor.
   """

   def predict(self, history: np.ndarray, drone_id: int | None = None) -> tuple[np.ndarray, np.ndarray]:
      """Run BoF prediction.

      :param history: ``(m, 6)`` array of past states ``[px, py, pz, vx, vy, vz]``.
      :param drone_id: Stable int identifier for this neighbor. ``None`` lets the adapter use its own default (typically 0 — collides EMA across all
      neighbors, so prefer passing an explicit id).
      :return: ``(trajectory (H, 3), radii (H,))``.
      """
      ...


def _format_xyz(p: np.ndarray) -> str:
   return f"({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f})"


def _log_prediction(kind: str, history_sliced: np.ndarray, trajectory: np.ndarray, radii: np.ndarray, *, extra: str = "") -> None:
   """Emit one INFO summary + one DEBUG full-dump per BoF call.

   The summary is grep-able and shows the most useful sanity check:
   ``forward`` is the Euclidean distance from the last observed history position to the last predicted point — small values mean the prediction barely
   extends beyond the current position, which combined with r_floor-dominated radii produces a "ring around the drone" tube look.
   """
   if not _log.isEnabledFor(logging.INFO):
      return
   last_hist = history_sliced[-1, :3]
   first_pt = trajectory[0, :3]
   last_pt = trajectory[-1, :3]
   forward = float(np.linalg.norm(last_pt - last_hist))
   step_lens = np.linalg.norm(np.diff(trajectory, axis=0), axis=1) if trajectory.shape[0] > 1 else np.zeros(1)
   _log.info("BoF %s call: hist=%s -> traj=%s | from %s -> %s "
             "| forward=%.3fm step_mean=%.3fm | radii mean=%.3f max=%.3f min=%.3f%s", kind, history_sliced.shape, trajectory.shape, _format_xyz(last_hist),
             _format_xyz(last_pt), forward, float(step_lens.mean()), float(radii.mean()), float(radii.max()), float(radii.min()),
             f" | {extra}" if extra else "")
   if _log.isEnabledFor(logging.DEBUG):
      with np.printoptions(precision=3, suppress=True, linewidth=200):
         _log.debug("BoF %s traj points (%dx3):\n%s\nradii: %s", kind, trajectory.shape[0], trajectory, np.array2string(radii, precision=3))


def _slice_history(history: np.ndarray, has_velocity: bool) -> np.ndarray:
   """Slice the 6D state buffer to the columns BoF expects."""
   arr = np.asarray(history, dtype=float)
   if arr.ndim != 2 or arr.shape[1] < 3:
      raise ValueError(f"history must be (m, >=3); got shape {arr.shape}")
   want = 6 if has_velocity else 3
   if arr.shape[1] < want:
      raise ValueError(f"history has {arr.shape[1]} columns but has_velocity={has_velocity} needs {want}")
   return arr[:, :want]


class BoFLibraryAdapter:
   """In-process adapter: calls Brian's ``PredictionEngine`` directly.

   The ``bof_tpt.engine`` module is imported lazily so TLCAmpc tests and imports keep working when the ``bof`` package is not installed yet.

   :param horizon: BoF prediction horizon ``H``. Fixed at adapter construction; the inner ``PredictionEngine`` is built once and reused across calls.
   :param has_velocity: When False (default) only the first 3 history columns are forwarded. When True, all 6 columns are sent and the returned trajectory is
   sliced back to ``(H, 3)``.
   :param growth_tau: Time-decay constant for BoF's σ growth. ``None`` leaves σ flat over the horizon; positive float makes BoF scale radii by ``sqrt(1 + k/tau)``. Forwarded as-is to ``predict``.
   """

   def __init__(self, horizon: int, has_velocity: bool = False, growth_tau: float | None = None) -> None:
      from bof_tpt.engine import PredictionEngine  # lazy

      self._horizon = horizon
      self._has_velocity = has_velocity
      self._growth_tau = growth_tau
      self._predictor = PredictionEngine.from_config(horizon=horizon)
      _log.info("BoFLibraryAdapter ready: horizon=%d has_velocity=%s growth_tau=%s engine=%s", horizon, has_velocity, growth_tau,
                type(self._predictor).__name__)

   def predict(self, history: np.ndarray, drone_id: int | None = None) -> tuple[np.ndarray, np.ndarray]:
      sliced = _slice_history(history, self._has_velocity)
      out = self._predictor.predict(sliced, has_velocity=self._has_velocity, drone_id=int(drone_id) if drone_id is not None else 0, growth_tau=self._growth_tau)
      trajectory = np.asarray(out["trajectory"], dtype=float)[:, :3]
      radii = np.asarray(out["radii"], dtype=float)
      _log_prediction("library", sliced, trajectory, radii, extra=f"drone_id={drone_id} growth_tau={self._growth_tau}")
      return trajectory, radii


class BoFRestAdapter:
   """HTTP adapter: POSTs the history window to a remote BoF FastAPI server.

   Endpoint contract::

       POST  {url}/api/predict
       body  {"history": [[..3 or 6..], ...], "horizon": int, "has_velocity": bool,
              "drone_id": int, "growth_tau": float | null}
       resp  {"status": "OK", "active_model": str, "trajectory": [[x,y,z(,vx,vy,vz)], ...], "radii": [..]}

   The server returns proper HTTP error codes (BAD_REQUEST → 400, FIT_FAILED → 422, …), so :class:`requests.HTTPError` is the error signal.
   Network errors propagate via :class:`requests.Timeout`
   :class:`requests.ConnectionError`.

   :param url: Base URL, e.g. ``http://localhost:5003``. Trailing slash optional.
   :param horizon: BoF prediction horizon ``H``.
   :param has_velocity: Same semantics as ``BoFLibraryAdapter``.
   :param growth_tau: Same semantics as ``BoFLibraryAdapter``. Sent in payload.
   :param timeout_s: Per-call HTTP timeout in seconds.
   """

   def __init__(self, url: str, horizon: int, has_velocity: bool = False, growth_tau: float | None = None, timeout_s: float = 2.0) -> None:
      import requests  # lazy: avoids hard dep when only the lib adapter is used

      self._endpoint = url.rstrip("/") + "/api/predict"
      self._horizon = horizon
      self._has_velocity = has_velocity
      self._growth_tau = growth_tau
      self._timeout = timeout_s
      self._session = requests.Session()
      _log.info("BoFRestAdapter ready: endpoint=%s horizon=%d has_velocity=%s growth_tau=%s timeout=%.2fs", self._endpoint, horizon, has_velocity, growth_tau, timeout_s)

   def predict(self, history: np.ndarray, drone_id: int | None = None) -> tuple[np.ndarray, np.ndarray]:
      sliced = _slice_history(history, self._has_velocity)
      payload = {"history": sliced.tolist(), "horizon": self._horizon, "has_velocity": self._has_velocity,
                 "drone_id": int(drone_id) if drone_id is not None else 0, "growth_tau": self._growth_tau}
      t0 = time.perf_counter()
      resp = self._session.post(self._endpoint, json=payload, timeout=self._timeout)
      resp.raise_for_status()
      body = resp.json()
      trajectory = np.asarray(body["trajectory"], dtype=float)[:, :3]
      radii = np.asarray(body["radii"], dtype=float)
      _log_prediction("rest", sliced, trajectory, radii, 
                      extra=f"endpoint={self._endpoint} status={resp.status_code} latency={(time.perf_counter() - t0) * 1000.0:.1f}ms model={body.get('active_model', '?')} drone_id={drone_id} growth_tau={self._growth_tau}")
      return trajectory, radii
