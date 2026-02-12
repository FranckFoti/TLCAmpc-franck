from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from drone_sim.domain.config import ScenarioConfig
from drone_sim.simulation.simulator import Simulator
from tools.live_view import load_parametrized_json


def _load_scenario(path: Path) -> ScenarioConfig:
   """Load a JSON scenario config (supports ${var} placeholders)."""

   cfg: Any = load_parametrized_json(path, params=None)
   if isinstance(cfg, str):
      cfg = json.loads(cfg)
   if not isinstance(cfg, dict):
      raise TypeError(f"Config must decode to a JSON object/dict, got {type(cfg).__name__}")
   return ScenarioConfig.model_validate(cfg)


def _adaptive_radius_for(drone) -> float:
   """Compute adaptive safety radius for a drone using its current velocity."""
   return drone.compute_adaptive_radius(drone.velocity())


def _step_margins(sim: Simulator) -> tuple[float, float, float]:
   """Compute constraint margins for the current state of the simulator using adaptive safety radii.

   Returns (min_pairwise_margin, max_speed, min_room_margin).

   - Pairwise margin: dist(i,j) - (r_safe_i + r_safe_j), using the drone's adaptive radius model.
   - Speed: max over drones of ||v||_2.
   - Room margin: min over all drones/directions of how far the adaptive safety sphere stays inside the room.
   """

   positions = [d.position() for d in sim.drones]
   velocities = [d.velocity() for d in sim.drones]

   # Use the drone's compute_adaptive_radius so that verification matches
   # the runtime and MPC semantics for adaptive spheres.
   r_safe = [_adaptive_radius_for(d) for d in sim.drones]

   n = len(sim.drones)

   # Pairwise distance margins (symmetric adaptive radii)
   pair_margins: list[float] = []
   for i in range(n):
      for j in range(i + 1, n):
         dist = float(np.linalg.norm(positions[i] - positions[j]))
         thresh = float(r_safe[i] + r_safe[j])
         pair_margins.append(dist - thresh)

   min_pair_margin = min(pair_margins) if pair_margins else math.inf

   # Speed
   speeds = [float(np.linalg.norm(v)) for v in velocities]
   max_speed = max(speeds) if speeds else 0.0

   # Room margins: ensure B_{r_safe}(p) subset [room_min, room_max]
   room_min = sim.room_min
   room_max = sim.room_max
   room_margins: list[float] = []
   for i in range(n):
      p = positions[i]
      r = float(r_safe[i])
      # Lower bounds: p - r >= room_min
      room_margins.extend([float(p[d] - r - room_min[d]) for d in range(3)])
      # Upper bounds: p + r <= room_max
      room_margins.extend([float(room_max[d] - p[d] - r) for d in range(3)])

   min_room_margin = min(room_margins) if room_margins else math.inf

   return min_pair_margin, max_speed, min_room_margin


@dataclass
class RunResult:
   """Collected metrics from a single simulation run."""
   config_name: str
   min_pair_margin: float = math.inf
   max_speed: float = 0.0
   min_room_margin: float = math.inf
   collision_count: int = 0
   total_solve_time: float = 0.0
   step_times: list[float] = field(default_factory=list)
   avg_speeds: list[float] = field(default_factory=list)
   min_pair_margins_per_step: list[float] = field(default_factory=list)
   max_adaptive_radii_per_step: list[float] = field(default_factory=list)

   @property
   def avg_step_time(self) -> float:
      return self.total_solve_time / len(self.step_times) if self.step_times else 0.0

   @property
   def max_step_time(self) -> float:
      return max(self.step_times) if self.step_times else 0.0

   @property
   def avg_speed_over_time(self) -> float:
      return float(np.mean(self.avg_speeds)) if self.avg_speeds else 0.0


def _run_simulation(path: Path, steps: int, *, enhanced: bool = False) -> RunResult:
   """Run a simulation and collect metrics.

   If enhanced=True, also collect per-step metrics for comparison mode.
   """
   scenario = _load_scenario(path)
   sim = Simulator.from_config(scenario)
   result = RunResult(config_name=path.name)

   for _ in range(steps):
      # Evaluate margins at current state
      pm, spd, rm = _step_margins(sim)
      result.min_pair_margin = min(result.min_pair_margin, pm)
      result.max_speed = max(result.max_speed, spd)
      result.min_room_margin = min(result.min_room_margin, rm)

      if enhanced:
         # Per-step average speed across all drones
         speeds = [float(np.linalg.norm(d.velocity())) for d in sim.drones]
         result.avg_speeds.append(float(np.mean(speeds)))
         result.min_pair_margins_per_step.append(pm)

         # Max adaptive radius at this step (only meaningful for adaptive configs)
         radii = [_adaptive_radius_for(d) for d in sim.drones]
         result.max_adaptive_radii_per_step.append(max(radii))

      t0 = time.perf_counter()
      sim.step()
      dt = time.perf_counter() - t0
      result.step_times.append(dt)
      result.total_solve_time += dt

   result.collision_count = len(sim.last_collisions)
   return result


def _print_result(result: RunResult) -> None:
   """Print per-config verification output."""
   print(f"{result.config_name}:")
   print(f"  min pairwise margin  : {result.min_pair_margin: .4f}  (>= 0 means no adaptive-sphere overlap)")
   print(f"  max speed            : {result.max_speed: .4f}")
   print(f"  min room margin      : {result.min_room_margin: .4f}  (>= 0 means adaptive spheres inside room)")
   if result.collision_count > 0:
      print(f"  collisions reported  : {result.collision_count} events")
   else:
      print("  collisions reported  : none")
   print(f"  total solve time     : {result.total_solve_time: .2f}s"
         f"  (avg {result.avg_step_time: .4f}s/step, max {result.max_step_time: .4f}s)")
   print()


def verify_adaptive_paper_configs(base_dir: Path, *, steps: int = 200) -> None:
   """Run all configs/*.json and report constraint margins for adaptive-sphere runs.

   For each config, we simulate `steps` steps and track:
     - minimal pairwise distance margin over time,
     - maximal speed over time,
     - minimal room-margin over time,
     - solve timing (total, average, max per step).
   """

   cfg_dir = base_dir / "configs"
   paths = sorted(cfg_dir.glob("*.json"))
   if not paths:
      print(f"No configs found in {cfg_dir}")
      return

   print("Verifying adaptive-spheres paper configs in", cfg_dir)
   print("(steps per config:", steps, ")")
   print()

   for path in paths:
      try:
         result = _run_simulation(path, steps)
      except Exception as exc:
         print(f"{path.name}:")
         print(f"  SKIPPED: {exc}")
         print()
         continue
      _print_result(result)


# Comparison pairs: (adaptive_config, fixed_baseline_config)
COMPARISON_PAIRS = [
   ("4DronesAdaptive.json", "4DronesFixedBaseline.json"),
   ("4DronesAdaptiveDMPC.json", "4DronesFixedBaselineDMPC.json"),
]


def _compare_configs(adaptive_path: Path, fixed_path: Path, steps: int) -> None:
   """Run both adaptive and fixed simulations, then print side-by-side comparison."""

   print(f"  Running adaptive: {adaptive_path.name} ...")
   adaptive = _run_simulation(adaptive_path, steps, enhanced=True)
   print(f"  Running fixed:    {fixed_path.name} ...")
   fixed = _run_simulation(fixed_path, steps, enhanced=True)

   print()
   print(f"=== Comparison: {adaptive_path.name} vs {fixed_path.name} ===")
   print()

   # Header
   hdr = f"{'Metric':<30s} {'Adaptive':>12s} {'Fixed':>12s} {'Delta':>16s}"
   print(hdr)
   print("-" * len(hdr))

   def row(label: str, a_val: float, f_val: float, fmt: str = ".4f", unit: str = "") -> None:
      delta = a_val - f_val
      sign = "+" if delta >= 0 else ""
      print(f"{label:<30s} {a_val:>12{fmt}}{unit} {f_val:>12{fmt}}{unit} {sign}{delta:>12{fmt}}{unit}")

   def row_time(label: str, a_val: float, f_val: float) -> None:
      delta = a_val - f_val
      sign = "+" if delta >= 0 else ""
      pct = (a_val / f_val - 1.0) * 100.0 if f_val > 0 else 0.0
      pct_str = f" ({pct:+.0f}%)" if f_val > 0 else ""
      print(f"{label:<30s} {a_val:>11.3f}s {f_val:>11.3f}s {sign}{delta:>10.3f}s{pct_str}")

   row("Min pairwise margin", adaptive.min_pair_margin, fixed.min_pair_margin)
   row("Max speed", adaptive.max_speed, fixed.max_speed)
   row("Avg speed (mean over time)", adaptive.avg_speed_over_time, fixed.avg_speed_over_time)
   row("Min room margin", adaptive.min_room_margin, fixed.min_room_margin)

   # Collisions (integer)
   print(f"{'Collisions':<30s} {adaptive.collision_count:>12d} {fixed.collision_count:>12d}")

   row_time("Total solve time", adaptive.total_solve_time, fixed.total_solve_time)
   row_time("Avg step time", adaptive.avg_step_time, fixed.avg_step_time)
   row_time("Max step time", adaptive.max_step_time, fixed.max_step_time)

   print()

   # Behavioral observations
   print("Behavioral observations:")

   # Speed comparison
   if fixed.max_speed > 0:
      speed_diff_pct = (1.0 - adaptive.max_speed / fixed.max_speed) * 100.0
      if speed_diff_pct > 1.0:
         print(f"  - Adaptive drones achieved {speed_diff_pct:.1f}% lower max speed")
      elif speed_diff_pct < -1.0:
         print(f"  - Adaptive drones achieved {-speed_diff_pct:.1f}% HIGHER max speed")
      else:
         print(f"  - Max speed roughly equal (within 1%)")

   # Margin comparison
   margin_diff = adaptive.min_pair_margin - fixed.min_pair_margin
   if margin_diff > 0.01:
      print(f"  - Adaptive maintained {margin_diff:.4f} larger minimum pairwise margin")
   elif margin_diff < -0.01:
      print(f"  - Fixed maintained {-margin_diff:.4f} larger minimum pairwise margin")
   else:
      print(f"  - Minimum pairwise margins roughly equal")

   # Timing comparison
   if fixed.total_solve_time > 0:
      overhead_pct = (adaptive.total_solve_time / fixed.total_solve_time - 1.0) * 100.0
      if overhead_pct > 0:
         print(f"  - Adaptive solving {overhead_pct:.0f}% slower than fixed")
      else:
         print(f"  - Adaptive solving {-overhead_pct:.0f}% FASTER than fixed")

   print()


def compare_adaptive_vs_fixed(base_dir: Path, *, steps: int = 200) -> None:
   """Run comparison pairs and print side-by-side tables."""
   cfg_dir = base_dir / "configs"

   print()
   print("=" * 70)
   print("ADAPTIVE vs FIXED COMPARISON")
   print("=" * 70)
   print()

   for adaptive_name, fixed_name in COMPARISON_PAIRS:
      adaptive_path = cfg_dir / adaptive_name
      fixed_path = cfg_dir / fixed_name
      if not adaptive_path.exists():
         print(f"  SKIP: {adaptive_name} not found")
         continue
      if not fixed_path.exists():
         print(f"  SKIP: {fixed_name} not found")
         continue
      _compare_configs(adaptive_path, fixed_path, steps)


def main(argv: list[str] | None = None) -> None:
   parser = argparse.ArgumentParser(
         description=("Verify adaptive-spheres paper configs by simulating them and reporting constraint margins "
                      "(distance, speed, room)."))
   parser.add_argument("--steps", type=int, default=200,
         help="Number of Simulator.step() calls per config (default: 200)")
   parser.add_argument("--compare", action="store_true",
         help="Run adaptive-vs-fixed comparison pairs after per-config verification")

   args = parser.parse_args(argv)

   verify_adaptive_paper_configs(Path(""), steps=args.steps)

   if args.compare:
      compare_adaptive_vs_fixed(Path(""), steps=args.steps)


if __name__ == "__main__":
   main()
