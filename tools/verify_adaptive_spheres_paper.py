from __future__ import annotations

import argparse
import json
import math
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


def _step_margins(sim: Simulator) -> tuple[float, float, float]:
   """Compute constraint margins for the current state of the simulator using adaptive safety radii.

   Returns (min_pairwise_margin, max_speed, min_room_margin).

   - Pairwise margin: dist(i,j) - (r_safe_i + r_safe_j), using the simulator's adaptive radius model.
   - Speed: max over drones of ||v||_2.
   - Room margin: min over all drones/directions of how far the adaptive safety sphere stays inside the room.
   """

   positions = [d.position() for d in sim.drones]
   velocities = [d.velocity() for d in sim.drones]

   # Use the simulator's adaptive radius helper so that verification matches
   # the runtime and MPC semantics for adaptive spheres.
   r_safe = [sim._adaptive_radius_for(d) for d in sim.drones]

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


def verify_adaptive_paper_configs(base_dir: Path, *, steps: int = 200) -> None:
   """Run all configs/*.json and report constraint margins for adaptive-sphere runs.

   For each config, we simulate `steps` steps (or until the routes are essentially finished
   if you later add such a stopping condition) and track:
     - minimal pairwise distance margin over time,
     - maximal speed over time,
     - minimal room-margin over time.
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
      scenario = _load_scenario(path)
      sim = Simulator.from_config(scenario)

      min_pair_margin = math.inf
      max_speed = 0.0
      min_room_margin = math.inf

      for _ in range(steps):
         # Evaluate margins at current state
         pm, spd, rm = _step_margins(sim)
         min_pair_margin = min(min_pair_margin, pm)
         max_speed = max(max_speed, spd)
         min_room_margin = min(min_room_margin, rm)

         sim.step()

      print(f"{path.name}:")
      print(f"  min pairwise margin  : {min_pair_margin: .4f}  (>= 0 means no adaptive-sphere overlap)")
      print(f"  max speed            : {max_speed: .4f}")
      print(f"  min room margin      : {min_room_margin: .4f}  (>= 0 means adaptive spheres inside room)")
      if sim.last_collisions:
         print(f"  collisions reported  : {len(sim.last_collisions)} events")
      else:
         print("  collisions reported  : none")
      print()


def main(argv: list[str] | None = None) -> None:
   parser = argparse.ArgumentParser(
         description=("Verify adaptive-spheres paper configs by simulating them and reporting constraint margins "
                      "(distance, speed, room)."))
   parser.add_argument("--steps", type=int, default=200,
         help="Number of Simulator.step() calls per config (default: 200)")

   args = parser.parse_args(argv)

   verify_adaptive_paper_configs(Path(""), steps=args.steps)


if __name__ == "__main__":
   main()
