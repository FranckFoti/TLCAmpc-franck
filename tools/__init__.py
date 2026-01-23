from __future__ import annotations

import math
from typing import Sequence

from drone_sim.domain.config import (ControllerSpec, DroneConfig, ObstacleConfig, PhysicsSpec, RoomConfig, ScenarioConfig)
from drone_sim.domain.drone import Drone

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
      2: [([0.964, 0.722, -0.222], [-0.964, -0.722, 0.222]), ([-0.699, 1.147, 1.352], [0.699, -1.147, -1.352])],
      3: [([0.869, 1.126, -0.532], [-0.869, -1.126, 0.532]), ([-0.078, -1.118, -0.184], [0.078, 1.118, 0.184]),
          ([-0.732, 1.309, 0.849], [0.732, -1.309, -0.849])],
      4: [([-0.146, -1.175, -0.504], [0.146, 1.175, 0.504]), ([0.022, 1.212, -1.095], [-0.022, -1.212, 1.095]),
          ([0.881, 0.113, 1.299], [-0.881, -0.113, -1.299]), ([-1.148, 0.721, 1.055], [1.148, -0.721, -1.055])],
      5: [([1.185, 0.959, 1.115], [-1.185, -0.959, -1.115]), ([0.821, -1.169, 0.316], [-0.821, 1.169, -0.316]),
          ([-0.72, 0.648, -1.072], [0.72, -0.648, 1.072]), ([-1.29, -1.117, 1.367], [1.29, 1.117, -1.367]),
          ([-1.048, 1.202, 1.256], [1.048, -1.202, -1.256])],
      6: [([-0.055, 1.251, 0.891], [0.055, -1.251, -0.891]), ([-0.913, -0.924, 0.445], [0.913, 0.924, -0.445]),
          ([0.965, 0.7, -0.965], [-0.965, -0.7, 0.965]), ([1.282, -0.939, 1.186], [-1.282, 0.939, -1.186]),
          ([-1.264, 0.598, -1.325], [1.264, -0.598, 1.325]), ([0.492, -1.341, -1.115], [-0.492, 1.341, 1.115])],
      7: [([-0.39, -1.417, 0.312], [0.39, 1.417, -0.312]), ([1.07, -0.933, -1.155], [-1.07, 0.933, 1.155]),
          ([-0.464, 1.368, -1.102], [0.464, -1.368, 1.102]), ([-0.618, 1.303, 1.365], [0.618, -1.303, -1.365]),
          ([1.489, 1.112, 0.595], [-1.489, -1.112, -0.595]), ([-1.487, -1.071, -1.36], [1.487, 1.071, 1.36]),
          ([1.365, -1.0, 1.262], [-1.365, 1.0, -1.262])]}

def _all_drones_reached_destination(drones: list[Drone]) -> bool:
   reached = [drone.route.target_reached(position=drone.position(), thresh=0.1) for drone in drones]
   return all(reached)

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

   for i in range(num_drones):
      if pattern is not None:
         start, target = list(pattern[i][0]), list(pattern[i][1])
      else:
         # Fallback: place drones on a circle of radius 1.5 well inside the room bounds.
         angle = 2.0 * math.pi * float(i) / float(num_drones)
         x = 1.5 * math.cos(angle)
         y = 1.5 * math.sin(angle)
         start, target = [x, y, 0.0], [-x, -y, 0.0]

      drones.append(DroneConfig(drone_id=f"drone-{i + 1}", start=start, waypoints=[], target=target,
                                radius=0.2, safety_zone=1.0, drone_color=_COLOR_BY_DRONE_INDEX[i+1]))

   obstacles: list[ObstacleConfig] = []

   return ScenarioConfig(dt=dt, physics=physics, controller=controller, coordinator=coordinator, drones=drones,
                         obstacles=obstacles, room=room)
