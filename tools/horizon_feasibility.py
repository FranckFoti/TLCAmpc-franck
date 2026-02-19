from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from drone_sim.simulation.simulator import Simulator
from tools.utility.scenario_creator import create_scenario


@dataclass
class ScenarioResult:
   num_drones: int
   horizon: int
   status: str  # "success", "infeasible", "timeout", "error"
   steps: int
   wall_time_s: float
   detail: str | None = None


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
                        max_steps: int = 10_000) -> ScenarioResult:
   scenario = create_scenario(n_drones=num_drones, horizon=horizon)

   t0 = time.perf_counter()
   try:
      sim = Simulator.from_config(scenario)
   except Exception as exc:  # pragma: no cover - diagnostic path
      t1 = time.perf_counter()
      return ScenarioResult(num_drones=num_drones, horizon=horizon, status="error", steps=0, wall_time_s=t1 - t0,
                            detail=f"failed to construct simulator: {exc}")

   steps = 0

   while True:
      now = time.perf_counter()
      if now - t0 > max_wall_time_s:
         return ScenarioResult(num_drones=num_drones, horizon=horizon, status="timeout", steps=steps,
                               wall_time_s=now - t0, detail=f"exceeded {max_wall_time_s:.1f}s wall time")

      if steps >= max_steps:
         return ScenarioResult(num_drones=num_drones, horizon=horizon, status="timeout", steps=steps,
                               wall_time_s=now - t0, detail=f"exceeded {max_steps} simulation steps")

      # Check completion before taking another step.
      if _all_routes_finished(sim):
         return ScenarioResult(num_drones=num_drones, horizon=horizon, status="success", steps=steps,
                               wall_time_s=now - t0, detail=None)

      sim.step()
      steps += 1

      if getattr(sim, "infeasible", False):
         # Centralized MPC reported an infeasible optimization (no route found).
         return ScenarioResult(num_drones=num_drones, horizon=horizon, status="infeasible", steps=steps,
                               wall_time_s=time.perf_counter() - t0, detail=getattr(sim, "infeasible_reason", None))


def run_grid(*, drone_counts: Iterable[int] = range(2, 8), horizons: Iterable[int] = range(1, 16),
             max_wall_time_s: float = 120.0) -> list[ScenarioResult]:
   results: list[ScenarioResult] = []

   for n in drone_counts:
      for H in horizons:
         print(f"=== Running scenario: N={n}, H={H} ===")
         res = run_single_scenario(num_drones=n, horizon=H, max_wall_time_s=max_wall_time_s)
         results.append(res)
         detail = f" ({res.detail})" if res.detail else ""
         print(f"N={res.num_drones}, H={res.horizon}: status={res.status}, steps={res.steps}, wall_time={res.wall_time_s:.2f}s{detail}\n")

   return results


def main(argv: list[str] | None = None) -> None:
   # The default grid is what the paper-like experiments need (2-7 drones, H=1..15).
   _ = argv
   run_grid()


if __name__ == "__main__":
   main()
