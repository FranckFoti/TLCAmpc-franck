"""CLI entry point for batch LSTM training-data generation.

Walks a directory tree of scenario JSON configs, runs each scenario through
the simulator, and writes compressed NPZ files (one per scenario) to an
output directory.

Usage::

   python tools/generate_data.py \\
     --config-dir configs/ \\
     --output-dir dataset/ \\
     --steps 500 \\
     --sigma-obs 0.01 \\
     --seed 42

NPZ file format (Phase 21 contract):
  X — noisy history windows,  shape (num_windows, 20, 6)
  Y — clean future windows,   shape (num_windows, 80, 6)

State vector convention: [px, py, pz, vx, vy, vz]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from drone_sim.domain.config import ControllerSpec, PhysicsSpec, RoomConfig, ScenarioConfig, DroneConfig
from tools.utility.constants import COLOR_BY_DRONE_INDEX
from tools.utility.generate_sphere_positions import generate_positions


def _build_parser() -> argparse.ArgumentParser:
   p = argparse.ArgumentParser(description="Generate LSTM training data from scenario configs.", formatter_class=argparse.ArgumentDefaultsHelpFormatter, )
   p.add_argument("--config-dir", type=Path, metavar="DIR", help="Root directory to search recursively for *.json scenario configs.", )
   p.add_argument("--default",  action='store_true')
   p.add_argument("--output-dir", type=Path, required=True, metavar="DIR", help="Directory where NPZ files will be written.", )
   p.add_argument("--steps", type=int, default=500, metavar="N", help="Number of simulation steps to run per scenario.", )
   p.add_argument("--sigma-obs", type=float, default=0.01, metavar="SIGMA", help="Observation noise standard deviation applied to history windows.", )
   p.add_argument("--seed", type=int, default=42, metavar="SEED", help="Random seed for reproducible noise generation.", )
   p.add_argument("--exclude-dirs", nargs="*", default=["bof_tests", "fixed_baselin_bug", "romm_constraint_bug"], metavar="DIR_NAME",
         help="Subdirectory names to skip when scanning for configs.", )
   return p

def create_scenario(dt: float = 0.1, physics_v: float = 3.0, physics_u: float = 3.0, horizon: int = 4, coordinator_type: str = "dmpc_admm",
                    room_size: float = 5.0, n_drones: int = 4, drones_radius: float = 0.2, safety_zone: float = 1.5) -> ScenarioConfig:
   physics = PhysicsSpec(type="linear_kinematics", id="default", params = {"v_max": physics_v, "u_min": [-physics_u, -physics_u, -physics_u], "u_max": [physics_u, physics_u, physics_u]})
   controller = ControllerSpec(type="mpc_agent", params={"horizon": horizon,"q_pos": [3.0, 3.0, 3.0],"r_u": [0.1, 0.1, 0.1]})
   if coordinator_type == "dmpc_admm":
      coordinator_spec = ControllerSpec(type="dmpc_admm", params={"horizon": horizon, "rho": 1.0, "max_admm_iter": 20, "primal_tol": 1e-2, "dual_tol": 1e-2})
   elif coordinator_type == "mpc_central":
      coordinator_spec = ControllerSpec(type="mpc_central", params={"horizon": horizon})
   else:
      raise ValueError("Unknown coordinator type: %s", coordinator_type)

   room = RoomConfig(min=[-room_size/2, -room_size/2, -room_size/2], max=[room_size/2, room_size/2, room_size/2])
   waypoints = generate_positions(n_drones=n_drones, room_min=room.min, room_max=room.max, min_pair_distance=safety_zone*2, wall_margin=safety_zone)
   drones = [DroneConfig(drone_id=f"drone-{i}", start=waypoints[i][0], target=waypoints[i][1], controller=controller, physics="default",
                         radius=drones_radius, safety_zone=safety_zone, cons_stop=0.0, alpha=None, drone_color=COLOR_BY_DRONE_INDEX[i + 1]) for i in range(n_drones)]

   cfg = ScenarioConfig(dt=dt, physics=physics, controller=controller, coordinator=coordinator_spec, comm_radius=None, obstacles=[], room=room, drones=drones)
   return cfg

def generate_date(args, configs_to_run, log):
   from drone_sim.domain.config import ScenarioConfig
   from drone_sim.prediction.data_generation import DataGenerator
   # ------------------------------------------------------------------
   # Generate data
   # ------------------------------------------------------------------
   generator = DataGenerator(sigma_obs=args.sigma_obs, seed=args.seed)
   total_windows = 0

   for cfg_path in configs_to_run:
      scenario_name = cfg_path.stem
      log.info("Processing %s …", cfg_path)
      try:
         cfg = ScenarioConfig.model_validate_json(cfg_path.read_text())
         n = generator.run_scenario(cfg, args.steps, args.output_dir, scenario_name)
         total_windows += n
      except Exception as exc:  # noqa: BLE001
         log.warning("Skipping %s — error: %s", cfg_path, exc)
         continue

def run_self_generated_config(args, log):
   from drone_sim.prediction.data_generation import DataGenerator
   generator = DataGenerator(sigma_obs=args.sigma_obs, seed=args.seed)

   total_windows = 0
   configs_to_run = 0

   for n_drones in range(2, 6):
      for room_size in [5.0, 6.0, 7.0, 8.0, 9.0]:
         for u in [1.0, 2.0, 2.5, 3.0, 3.5, 5.0]:
            for v in [2.5, 3.0, 3.5, 4.0]:
               for horizon in [4, 5, 6]:
                  for safety in [1.0, 1.5, 2.0, 2.5]:
                     scenario_name = f"room{room_size}_u{u}_v{v}_h{horizon}_r{safety}"
                     try:
                        log.info("Processing %s …", scenario_name)
                        cfg = create_scenario(physics_v=v, physics_u=u, horizon=horizon, room_size=room_size, n_drones=n_drones, safety_zone=safety)
                        n = generator.run_scenario(cfg, args.steps, args.output_dir, scenario_name)
                        total_windows += n
                        configs_to_run += 1
                     except RuntimeError:
                        pass
                     except Exception as exc:  # noqa: BLE001
                        log.warning("Skipping %s — error: %s", scenario_name, exc)
                        continue

   log.info("Done. %d windows written across %d scenario(s).", total_windows, configs_to_run)


def main() -> None:
   # ------------------------------------------------------------------
   # Parse arguments
   # ------------------------------------------------------------------
   parser = _build_parser()
   args = parser.parse_args()

   logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout, )
   log = logging.getLogger(__name__)

   if args.default:
      run_self_generated_config(args, log)
      if not args.config_dir:
         return

   # ------------------------------------------------------------------
   # Deferred imports (keep startup fast for argparse --help)
   # ------------------------------------------------------------------
   from drone_sim.domain.config import ScenarioConfig
   from drone_sim.prediction.data_generation import DataGenerator

   # ------------------------------------------------------------------
   # Discover configs
   # ------------------------------------------------------------------
   config_dir: Path = args.config_dir
   if not config_dir.exists():
      log.error("Config directory does not exist: %s", config_dir)
      sys.exit(1)

   exclude_dirs: set[str] = set(args.exclude_dirs or [])
   all_configs = sorted(config_dir.rglob("*.json"))

   # Filter out paths that pass through any excluded directory name.
   configs_to_run = [p for p in all_configs if not any(part in exclude_dirs for part in p.parts)]

   if not configs_to_run:
      log.warning("No scenario configs found under %s", config_dir)
      return

   log.info("Found %d config(s) to process (excluded %d via --exclude-dirs).", len(configs_to_run), len(all_configs) - len(configs_to_run), )

   # ------------------------------------------------------------------
   # Generate data
   # ------------------------------------------------------------------
   generator = DataGenerator(sigma_obs=args.sigma_obs, seed=args.seed)
   total_windows = 0

   for cfg_path in configs_to_run:
      scenario_name = cfg_path.stem
      log.info("Processing %s …", cfg_path)
      try:
         cfg = ScenarioConfig.model_validate_json(cfg_path.read_text())
         n = generator.run_scenario(cfg, args.steps, args.output_dir, scenario_name)
         total_windows += n
      except Exception as exc:  # noqa: BLE001
         log.warning("Skipping %s — error: %s", cfg_path, exc)
         continue

   log.info("Done. %d windows written across %d scenario(s).", total_windows, len(configs_to_run), )


if __name__ == "__main__":
   main()
