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
      x = np.asarray(x, dtype=float).reshape(6)
      u = np.asarray(u, dtype=float).reshape(3)
      new_state = self.A @ x + self.B @ u
      new_state[3:6] = self.clip_velocity(new_state[3:6])
      return new_state

   def clip_velocity(self, vel: np.ndarray) -> np.ndarray:
      vel = np.asarray(vel, dtype=float).reshape(3)
      vel_mag = np.linalg.norm(vel)
      if vel_mag > self.v_max():
         vel = vel * (self.v_max() / vel_mag)
      return vel
