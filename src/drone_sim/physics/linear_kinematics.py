from __future__ import annotations

import numpy as np

from drone_sim.domain.registry import register_physics
from drone_sim.physics.base import PhysicsModel


@register_physics("linear_kinematics")
class LinearKinematicsPhysics(PhysicsModel):
   """Discrete-time constant-acceleration model.

      State:
          x = [x, y, z, v_x, v_y, v_z]^T
      Control:
          u = [a_x, a_y, a_z]^T

      Update:
          x_{k+1} = A x_k + B u_k

      A = [[I, dt*I],
           [0,   I]]

      B = [[0.5*dt^2*I],
           [  dt*I  ]]
   """

   def __init__(self, dt: float, v_max: float = 5.0, u_min=None, u_max=None):
      super().__init__(dt=dt, v_max=v_max, u_min=u_min, u_max=u_max)
      self.A = np.block([[np.eye(3), self.dt * np.eye(3)], [np.zeros((3, 3)), np.eye(3)]])
      self.B = np.block([[0.5 * self.dt ** 2 * np.eye(3)], [self.dt * np.eye(3)]])

   def step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
      if not isinstance(x, np.ndarray) or x.shape != (6,):
         x = np.asarray(x, dtype=float).reshape(6)
      if not isinstance(u, np.ndarray) or u.shape != (3,):
         u = np.asarray(u, dtype=float).reshape(3)
      return self.A @ x + self.B @ u

   def clip_velocity(self, vel: np.ndarray) -> np.ndarray:
      vel_mag = np.linalg.norm(vel)
      if vel_mag > self._v_max:
         vel = vel * (self._v_max / vel_mag)
      return vel

   def predict_trajectory(self, x0: np.ndarray, u_seq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
      """Fully vectorized trajectory prediction without Python loop.

      :param x0: Initial state (6,).
      :param u_seq: Control sequence (H, 3).
      :return: (positions (H, 3), velocities (H, 3)).
      """
      x0 = np.asarray(x0, dtype=float).reshape(6)
      u_seq = np.asarray(u_seq, dtype=float)
      if u_seq.ndim == 1:
         u_seq = u_seq.reshape((-1, 3))

      dt = self.dt
      pos_0 = x0[:3]
      vel_0 = x0[3:6]

      # vel[k] = vel_0 + dt * cumsum(u[0:k])
      velocities = vel_0 + dt * np.cumsum(u_seq, axis=0)

      # pos[k] = pos[k-1] + dt * vel[k-1] + 0.5 * dt^2 * u[k-1]
      vel_before = np.empty_like(u_seq)
      vel_before[0] = vel_0
      vel_before[1:] = velocities[:-1]
      delta_pos = dt * vel_before + 0.5 * dt ** 2 * u_seq
      positions = pos_0 + np.cumsum(delta_pos, axis=0)

      return positions, velocities
