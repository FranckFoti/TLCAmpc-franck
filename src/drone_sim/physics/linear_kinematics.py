from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from drone_sim.domain.registry import register_physics
from drone_sim.physics.base import PhysicsModel


@register_physics("linear_kinematics")
@dataclass
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

   def __post_init__(self) -> None:
      self.A = np.block([[np.eye(3), self.dt * np.eye(3)], [np.zeros((3, 3)), np.eye(3)]])
      self.B = np.block([[0.5 * self.dt ** 2 * np.eye(3)], [self.dt * np.eye(3)]])

   def step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
      x = np.asarray(x, dtype=float).reshape(6)
      u = np.asarray(u, dtype=float).reshape(3)
      return self.A @ x + self.B @ u
