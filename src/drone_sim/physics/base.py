from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PhysicsModel:
   dt: float

   def step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
      raise NotImplementedError
