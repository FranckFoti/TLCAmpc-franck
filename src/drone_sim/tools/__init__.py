from __future__ import annotations

import math
from typing import Sequence

from drone_sim.domain.config import (ControllerSpec, DroneConfig, ObstacleConfig, PhysicsSpec, RoomConfig, ScenarioConfig)

# Colors taken from the paper-style JSON configs in configs/*.json, indexed by drone index (1-based).
_COLOR_BY_DRONE_INDEX = {1: "#00FFFF",  # cyan / tab:cyan
      2: "#800080",  # purple
      3: "#DC143C",  # crimson
      4: "#FFD700",  # gold
      5: "#9400D3",  # dark violet
      6: "#008B8B",  # dark cyan
      7: "#FF69B4",  # hot pink
}

# Predefined start/target patterns inspired by the paper configs in `configs/`.
# Keys are the number of drones; values are lists of (start, target) pairs.
_PREDEFINED_PATTERNS: dict[int, list[tuple[Sequence[float], Sequence[float]]]] = {
      2: [([0.0, -2.4, 0.0], [0.0, 2.4, 0.0]), ([0.0, 2.4, 0.0], [0.0, -2.4, 0.0])],
      3: [([-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]), ([2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]),
          ([0.0, -2.0, 0.0], [0.0, 2.0, 0.0])],
      4: [([-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]), ([2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]), ([0.0, -2.0, 0.0], [0.0, 2.0, 0.0]),
          ([0.0, 2.0, 0.0], [0.0, -2.0, 0.0])],
      5: [([-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]), ([2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]), ([0.0, -2.0, 0.0], [0.0, 2.0, 0.0]),
          ([0.0, 2.0, 0.0], [0.0, -2.0, 0.0]), ([0.0, 0.0, -2.0], [0.0, 0.0, 2.0]), ],
      6: [([-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]), ([2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]), ([0.0, -2.0, 0.0], [0.0, 2.0, 0.0]),
          ([0.0, 2.0, 0.0], [0.0, -2.0, 0.0]), ([0.0, 0.0, -2.0], [0.0, 0.0, 2.0]),
          ([0.0, 0.0, 2.0], [0.0, 0.0, -2.0])],
      7: [([-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]), ([2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]), ([0.0, -2.0, 0.0], [0.0, 2.0, 0.0]),
          ([0.0, 2.0, 0.0], [0.0, -2.0, 0.0]), ([0.0, 0.0, -2.0], [0.0, 0.0, 2.0]), ([0.0, 0.0, 2.0], [0.0, 0.0, -2.0]),
          ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])]}


def _build_scenario(num_drones: int, horizon: int) -> ScenarioConfig:
   """Construct a ScenarioConfig using paper-style start/target patterns.

   For N=2..7 we use fixed patterns that mirror the JSON configs in `configs/`.
   If num_drones is outside this range, we fall back to a generic circle.
   """

   dt = 0.1

   room = RoomConfig(min=[-2.5, -2.5, -2.5], max=[2.5, 2.5, 2.5])

   physics = PhysicsSpec(type="linear_kinematics", params={})

   controller = ControllerSpec(type="mpc_agent",
                               params={"horizon": horizon, "q_pos": [3.0, 3.0, 3.0], "r_u": [0.1, 0.1, 0.1],
                                       "u_min": [-3.0, -3.0, -3.0], "u_max": [3.0, 3.0, 3.0]})

   coordinator = ControllerSpec(type="mpc_central", params={"horizon": horizon})

   drones: list[DroneConfig] = []

   pattern = _PREDEFINED_PATTERNS.get(num_drones)

   if pattern is not None:
      # Use the first `num_drones` pairs from the predefined pattern.
      for i in range(num_drones):
         start, target = pattern[i]
         drones.append(
               DroneConfig(drone_id=f"drone-{i + 1}", start=list(start), waypoints=[], target=list(target), radius=0.2,
                           safety_zone=1.0, drone_color="tab:blue"))
   else:
      # Fallback: place drones on a circle of radius r_start well inside the room bounds.
      r_start = 1.5
      for i in range(num_drones):
         angle = 2.0 * math.pi * float(i) / float(num_drones)
         x = r_start * math.cos(angle)
         y = r_start * math.sin(angle)
         start = [x, y, 0.0]
         target = [-x, -y, 0.0]

         drones.append(DroneConfig(drone_id=f"drone-{i + 1}", start=start, waypoints=[], target=target, radius=0.2,
                                   safety_zone=1.0, drone_color="tab:blue"))

   obstacles: list[ObstacleConfig] = []

   return ScenarioConfig(dt=dt, physics=physics, controller=controller, coordinator=coordinator, drones=drones,
                         obstacles=obstacles, room=room)
