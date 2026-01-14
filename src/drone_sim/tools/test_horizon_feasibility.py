from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from drone_sim.domain.config import (ControllerSpec, DroneConfig, ObstacleConfig, PhysicsSpec, RoomConfig,
                                     ScenarioConfig, )
from drone_sim.simulation.simulator import Simulator


@dataclass
class ScenarioResult:
   num_drones: int
   horizon: int
   status: str  # "success", "infeasible", "timeout", "error"
   steps: int
   wall_time_s: float
   detail: str | None = None


# Predefined start/target patterns inspired by the paper configs in `configs/`.
# Keys are the number of drones; values are lists of (start, target) pairs.
_PREDEFINED_PATTERNS: dict[int, list[tuple[Sequence[float], Sequence[float]]]] = {
      2: [([0.0, -2.4, 0.0], [0.0, 2.4, 0.0]), ([0.0, 2.4, 0.0], [0.0, -2.4, 0.0]), ],
      3: [([-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]), ([2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]),
            ([0.0, -2.0, 0.0], [0.0, 2.0, 0.0]), ],
      4: [([-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]), ([2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]), ([0.0, -2.0, 0.0], [0.0, 2.0, 0.0]),
            ([0.0, 2.0, 0.0], [0.0, -2.0, 0.0]), ],
      5: [([-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]), ([2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]), ([0.0, -2.0, 0.0], [0.0, 2.0, 0.0]),
            ([0.0, 2.0, 0.0], [0.0, -2.0, 0.0]), ([0.0, 0.0, -2.0], [0.0, 0.0, 2.0]), ],
      6: [([-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]), ([2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]), ([0.0, -2.0, 0.0], [0.0, 2.0, 0.0]),
            ([0.0, 2.0, 0.0], [0.0, -2.0, 0.0]), ([0.0, 0.0, -2.0], [0.0, 0.0, 2.0]),
            ([0.0, 0.0, 2.0], [0.0, 0.0, -2.0]), ],
      7: [([-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]), ([2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]), ([0.0, -2.0, 0.0], [0.0, 2.0, 0.0]),
            ([0.0, 2.0, 0.0], [0.0, -2.0, 0.0]), ([0.0, 0.0, -2.0], [0.0, 0.0, 2.0]),
            ([0.0, 0.0, 2.0], [0.0, 0.0, -2.0]), ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]), ], }


def _build_scenario(num_drones: int, horizon: int) -> ScenarioConfig:
   """Construct a ScenarioConfig using paper-style start/target patterns.

   For N=2..7 we use fixed patterns that mirror the JSON configs in `configs/`.
   If num_drones is outside this range, we fall back to a generic circle.
   """

   dt = 0.1

   room = RoomConfig(min=[-2.5, -2.5, -2.5], max=[2.5, 2.5, 2.5])

   physics = PhysicsSpec(type="linear_kinematics", params={})

   controller = ControllerSpec(type="mpc_agent",
         params={"horizon": horizon, "q_pos": [3.0, 3.0, 3.0], "r_u": [0.1, 0.1, 0.1], "u_min": [-3.0, -3.0, -3.0],
               "u_max": [3.0, 3.0, 3.0], }, )

   coordinator = ControllerSpec(type="mpc_central", params={"horizon": horizon, "safety_buffer": 1.0, }, )

   drones: list[DroneConfig] = []

   pattern = _PREDEFINED_PATTERNS.get(num_drones)

   if pattern is not None:
      # Use the first `num_drones` pairs from the predefined pattern.
      for i in range(num_drones):
         start, target = pattern[i]
         drones.append(
               DroneConfig(drone_id=f"drone-{i + 1}", start=list(start), waypoints=[], target=list(target), radius=0.2,
                     safety_zone=1.0, drone_color="tab:blue", ))
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
               safety_zone=1.0, drone_color="tab:blue", ))

   obstacles: list[ObstacleConfig] = []

   return ScenarioConfig(dt=dt, physics=physics, controller=controller, coordinator=coordinator, drones=drones,
         obstacles=obstacles, room=room, )


def _all_routes_finished(sim: Simulator, pos_tol: float = 0.2, vel_tol: float = 0.1) -> bool:
   """Heuristic: consider scenario finished if all drones are near their targets and slow."""

   for d in sim.drones:
      p = d.position()
      v = d.velocity()
      target = d.route.target
      if float(np.linalg.norm(p - target)) > pos_tol:
         return False
      if float(np.linalg.norm(v)) > vel_tol:
         return False
   return True


def run_single_scenario(*, num_drones: int, horizon: int, max_wall_time_s: float = 120.0,
      max_steps: int = 10_000, ) -> ScenarioResult:
   scenario = _build_scenario(num_drones=num_drones, horizon=horizon)

   t0 = time.perf_counter()
   try:
      sim = Simulator.from_config(scenario)
   except Exception as exc:  # pragma: no cover - diagnostic path
      t1 = time.perf_counter()
      return ScenarioResult(num_drones=num_drones, horizon=horizon, status="error", steps=0, wall_time_s=t1 - t0,
            detail=f"failed to construct simulator: {exc}", )

   steps = 0

   while True:
      now = time.perf_counter()
      if now - t0 > max_wall_time_s:
         return ScenarioResult(num_drones=num_drones, horizon=horizon, status="timeout", steps=steps,
               wall_time_s=now - t0, detail=f"exceeded {max_wall_time_s:.1f}s wall time", )

      if steps >= max_steps:
         return ScenarioResult(num_drones=num_drones, horizon=horizon, status="timeout", steps=steps,
               wall_time_s=now - t0, detail=f"exceeded {max_steps} simulation steps", )

      # Check completion before taking another step.
      if _all_routes_finished(sim):
         return ScenarioResult(num_drones=num_drones, horizon=horizon, status="success", steps=steps,
               wall_time_s=now - t0, detail=None, )

      sim.step()
      steps += 1

      if getattr(sim, "infeasible", False):
         # Centralized MPC reported an infeasible optimization (no route found).
         return ScenarioResult(num_drones=num_drones, horizon=horizon, status="infeasible", steps=steps,
               wall_time_s=time.perf_counter() - t0, detail=getattr(sim, "infeasible_reason", None), )


def run_grid(*, drone_counts: Iterable[int] = range(2, 8), horizons: Iterable[int] = range(1, 16),
      max_wall_time_s: float = 120.0, ) -> list[ScenarioResult]:
   results: list[ScenarioResult] = []

   for n in drone_counts:
      for H in horizons:
         print(f"=== Running scenario: N={n}, H={H} ===")
         res = run_single_scenario(num_drones=n, horizon=H, max_wall_time_s=max_wall_time_s, )
         results.append(res)
         detail = f" ({res.detail})" if res.detail else ""
         print(f"N={res.num_drones}, H={res.horizon}: status={res.status}, "
               f"steps={res.steps}, wall_time={res.wall_time_s:.2f}s{detail}")
         print()

   return results


def main(argv: list[str] | None = None) -> None:
   # Keep the CLI intentionally simple: the default grid is exactly what the paper-like experiments need (2-7 drones, H=1..15).
   _ = argv  # currently unused
   run_grid()


if __name__ == "__main__":
   main()
