"""Scenario factory functions for all 7 empirical paper scenarios.

Each function returns a ScenarioConfig that can be used directly with the
Simulator, or passed to sweep scripts. Parameters like n_drones, alpha, and
comm_radius are exposed so calling code can vary them systematically.

Standard physics defaults (used unless a scenario overrides them):
  r_min        = 0.4   (drone physical radius)
  v_max        = 3.0
  u_max        = 3.0
  safety_zone  = 1.2   (= 3 * r_min)
  dt           = 0.1
  horizon      = 4
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from pathlib import Path
import logging
import csv
import threading
import argparse

import tools.utility.scenario_creator
from drone_sim.domain.config import ScenarioConfig
from tools.horizon_live_view_grid import run_single_scenario, print_results_prep
from tools.utility.printer import Status, print_gif
from paper2_tools.plot_scenario_1_2 import main as plot_scenario_1_2

# adaptive -> mpc_agent_adaptive
# static- > mpc_agent
# central -> mpc_central
# distributed -> dmpc_admm
COORDINATORS = ['mpc_central', 'dmpc_admm']  # dmpc_threaded
CONTROLLERS = ['mpc_agent', 'mpc_agent_adaptive']
_HORIZON: int = 4
_DT: float = 0.1

# Lock for thread-safe CSV writing
_csv_lock = threading.Lock()


def _print_results(all_pair_dists: list[float], horizon: int, jerk_3d_value: float, num_drones: int, out_dir: Path, status: Status, step_durations: list[float],
                   step_mean_pair_dists: list[float], wall_time: float, coordinator_type: str, controller_type: str) -> None:
   fieldnames, row = print_results_prep(all_pair_dists=all_pair_dists, horizon=horizon, jerk_3d_value=jerk_3d_value, num_drones=num_drones, status=status,
                                        step_durations=step_durations, step_mean_pair_dists=step_mean_pair_dists, wall_time=wall_time,
                                        coordinator_type=coordinator_type, controller_type=controller_type)

   # Write metrics to CSV (one row per scenario) in the same output directory. Use lock for thread-safe writing.
   csv_path = out_dir / 'metrics.csv'

   out_dir.mkdir(parents=True, exist_ok=True)
   with _csv_lock:
      write_header = not csv_path.exists()
      with csv_path.open('a', newline='', encoding='utf-8') as f:
         writer = csv.DictWriter(f, fieldnames=fieldnames)
         if write_header:
            writer.writeheader()
         writer.writerow(row)


def _run_scenario_wrapper_and_print(num_drones: int, horizon: int, cfg: ScenarioConfig, max_steps: int, trace_len: int, out_dir: Path) -> tuple[int, int, str]:
   """Wrapper for thread pool execution. Returns (num_drones, horizon, status) for progress tracking."""
   try:
      status, wall_time, jerk_3d_value, step_durations, step_mean_pair_dists, all_pair_dists, frames = run_single_scenario(cfg, max_steps, trace_len)
      _print_results(all_pair_dists=all_pair_dists, horizon=horizon, jerk_3d_value=jerk_3d_value, num_drones=num_drones, out_dir=out_dir, status=status,
                     step_durations=step_durations, step_mean_pair_dists=step_mean_pair_dists, wall_time=wall_time, coordinator_type=cfg.coordinator.type,
                     controller_type=cfg.controller.type)
      return (num_drones, horizon, 'ok')
   except Exception as e:
      return (num_drones, horizon, f'exception: {e}')


def _run_scenario_wrapper(scenario: ScenarioConfig, out_dir: Path, max_steps: int, trace_len: int, timeout: int, should_print_gif: bool = False) -> tuple[int, int, str]:
   """Wrapper for thread pool execution. Returns (num_drones, horizon, status) for progress tracking."""
   num_drones = len(scenario.drones)
   horizon = scenario.controller.params.get('horizon', -1)
   try:
      print(f'  Starting {num_drones} drones ({scenario.controller.type}, {scenario.coordinator.type})')
      status, wall_time, jerk_3d_value, step_durations, step_mean_pair_dists, all_pair_dists, frames = run_single_scenario(
            scenario=scenario, max_steps=max_steps, trace_len=trace_len, timeout=timeout)
      _print_results(all_pair_dists=all_pair_dists, horizon=horizon, jerk_3d_value=jerk_3d_value, num_drones=num_drones, out_dir=out_dir, status=status,
                     step_durations=step_durations, step_mean_pair_dists=step_mean_pair_dists, wall_time=wall_time, coordinator_type=scenario.coordinator.type,
                     controller_type=scenario.controller.type)
      if should_print_gif:
         out_dir.mkdir(parents=True, exist_ok=True)
         coord_suffix = "dmpc" if scenario.coordinator.type == "dmpc_admm" else "central"
         gif_path = out_dir / f"blocked_N{num_drones}_{coord_suffix}.gif"
         print(f"status={status}, frames={len(frames)}, wall_time={wall_time:.2f}s -> GIF: {gif_path}")
         print_gif(frames=frames, gif_path=gif_path, gif_fps=20.0)

      print(f'  Completed {num_drones} drones ({scenario.controller.type}, {scenario.coordinator.type})')
      return num_drones, horizon, 'ok'
   except Exception as e:
      return num_drones, horizon, f'exception: {e}'


def process_scenarios(scenarios: list[tuple[ScenarioConfig, int]], result_path: Path, num_threads: int, should_print_gif: bool = False):
   completed = 0
   total_scenarios = len(scenarios)
   # coord_type = 'mpc_central' # 'dmpc_admm'

   with ThreadPoolExecutor(max_workers=num_threads) as executor:
      futures = {
            executor.submit(_run_scenario_wrapper, scenario=scenario, out_dir=Path(result_path), max_steps=5000, trace_len=5000, timeout=600, should_print_gif=should_print_gif):
               (scenario, i) for scenario, i in scenarios
            }

      for future in as_completed(futures):
         scenario, i = futures[future]
         completed += 1
         try:
            result_n, result_h, status = future.result()
            print(f'[{completed}/{total_scenarios}] Completed N={result_n}, H={result_h}: {status}')
         except Exception as e:
            print(f'[{completed}/{total_scenarios}] Failed N={len(scenario.drones)}, run={i}: {e}')


def run_scenario_1_3(result_path: str, room_size: float, v_max: float, u_max: float, alpha: float, static_safety_zone: float, adaptive_safety_zone: float, r_min: float):
   csv_path = Path(__file__).parent / 'paper2_results' / result_path
   # room_size = 10.0
   # v_max = 2.5
   # u_max = 3.0
   # alpha = 0.5
   # static_safety_zone = 1.52
   # adaptive_safety_zone = 1.0
   # r_min = 0.4
   obstacles = [{"center": [2.0, 3.1, 0.0], "half_extents": [0.25, 1.9, 5.0]}, {"center": [2.0, -3.1, 0.0], "half_extents": [0.25, 1.9, 5.0]},
                {"center": [2.0, 0.0, 3.1], "half_extents": [0.25, 1.2, 1.9]}, {"center": [2.0, 0.0, -3.1], "half_extents": [0.25, 1.2, 1.9]}]
   waypoints = [[[-3.4, -3.4, -3.4], [3.4, 1.5, 0.0]], [[-3.4, 3.4, -3.4], [3.4, -1.5, 0.0]]]

   scenarios = []
   for coord in COORDINATORS:
      for controller in CONTROLLERS:
         cfg = tools.utility.scenario_creator.create_scenario(horizon=_HORIZON, dt=_DT, controller_type=controller, coordinator_type=coord, physics_u=u_max,
                                                              physics_v=v_max, room_size=room_size, n_drones=2, drones_radius=r_min, obstacles=obstacles,
                                                              alpha=alpha if controller == 'mpc_agent_adaptive' else None, waypoints=waypoints,
                                                              safety_zone=adaptive_safety_zone if controller == 'mpc_agent_adaptive' else static_safety_zone)
         scenarios.append((cfg, 1))

   process_scenarios(scenarios=scenarios, result_path=csv_path, num_threads=1, should_print_gif=True)


def run_scenario_1_2(result_path: str, n_threads: int, v_max: float, u_max: float, horizon: int, room_size: float, n_runs: int, alpha: float, n_crit: int,
                     static_safety_zone: float, adaptive_safety_zone: float, r_min: float):
   csv_path = Path(__file__).parent / 'paper2_results' / result_path
   runs = range(n_runs)

   drones_range = range(max(2, int(n_crit * 0.5)), n_crit + 2)

   sum_runs = len(runs) * len(drones_range) * len(COORDINATORS) * len(COORDINATORS)

   scenarios = []
   for i in runs:
      for n in drones_range:
         for coord in COORDINATORS:
            for controller in CONTROLLERS:
               try:
                  print(f'Building scenario run {i}: {n} drones, {coord}, {controller}, {coord}')
                  cfg = tools.utility.scenario_creator.create_scenario(horizon=horizon, dt=_DT, controller_type=controller, coordinator_type=coord,
                                                                       physics_u=u_max, physics_v=v_max, room_size=room_size, n_drones=n,
                                                                       alpha=alpha if controller == 'mpc_agent_adaptive' else None, drones_radius=r_min,
                                                                       safety_zone=adaptive_safety_zone if controller == 'mpc_agent_adaptive' else
                                                                       static_safety_zone)
                  if cfg is not None:
                     scenarios.append((cfg, i))
               except Exception as e:
                  print(f'Building scenario run {i}/{sum_runs}: {n} drones, {coord}, {controller}, '
                        f'{adaptive_safety_zone if controller == 'mpc_agent_adaptive' else static_safety_zone}: {e}')
   process_scenarios(scenarios, csv_path, n_threads, should_print_gif=False)


def main(argv: list[str] | None = None):
   parser = argparse.ArgumentParser(description='Run testscenarios')

   parser.add_argument('--result_path', type=Path, default=Path('results_8_4'), help='Path to the result directory (relative to this <script>/paper2_results/)')
   parser.add_argument('--num_threads', type=int, default=1, help='Number of threads to use')
   parser.add_argument('--runs', type=int, default=10, help='Number of runs')
   parser.add_argument('--scenario', required=True, type=str, choices=['1_1', '1_2', '1_3', '2_1', '2_2', '2_3', '2_4'], help='Scenario to run')
   parser.add_argument('--horizon', type=int, default=4, help='Mpc Horizon')
   parser.add_argument('--n_crit', type=int, default=13, help='Critical number of drones')
   parser.add_argument('--u_max', type=float, default=2.5, help='Maximum velocity')
   parser.add_argument('--v_max', type=float, default=2.0, help='Maximum acceleration')
   parser.add_argument('--r_min', type=float, default=0.4, help='Radius of the drone')
   parser.add_argument('--static_safety_zone', type=float, default=1.3, help='Radius of the safety zone for static case')
   parser.add_argument('--adaptive_safety_zone', type=float, default=1.1, help='Radius of the safety zone for adaptive case')
   parser.add_argument('--room_size', type=float, default=8.0, help='Room size')
   parser.add_argument('--alpha', type=float, default=0.5, help='Alpha for adaptive case')
   parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")
   # ArgParser fails if required is not set, so we can be suer there is at least a scenario knowledge
   args = parser.parse_args(argv)
   logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(name)s %(levelname)s %(message)s", datefmt="%d.%m %H:%M:%S")

   match args.scenario:
      case '1_1':
         # run_scenario_1_1()
         pass
      case '1_2':
         # python -m paper2_tools.scenarios --scenario 1_2 --num_threads 1 --runs 9 --result_path results_s_1_2 --n_crit 11 --u_max 3.0 --v_max 2.5 --static_safety_zone 1.52 --adaptive_safety_zone 1.0 --room_size 8.0 --log-level INFO v_max=2.5, u_max=3.0, alpha=0.5, adaptive safety_zone=1.0; static safety_zone=1.52 -> n_crit=11
         run_scenario_1_2(result_path=args.result_path, n_threads=args.num_threads, v_max=args.v_max, u_max=args.u_max, horizon=args.horizon,
                          room_size=args.room_size, n_runs=args.runs, n_crit=args.n_crit, alpha=1.5, static_safety_zone=args.static_safety_zone,
                          adaptive_safety_zone=args.adaptive_safety_zone, r_min=args.r_min)
         plot_scenario_1_2(args.result_path)
      case '1_3':
         # python -m paper2_tools.scenarios --scenario 1_3 --result_path results_s_1_3 --u_max 3.0 --v_max 2.5 --static_safety_zone 1.52  --adaptive_safety_zone 1.0 --room_size 10.0 --alpha 0.5 --log-level INFO
         run_scenario_1_3(result_path=args.result_path, room_size=args.room_size, v_max=args.v_max, u_max=args.u_max, alpha=args.alpha, static_safety_zone=args.static_safety_zone,
                          adaptive_safety_zone=args.adaptive_safety_zone, r_min=args.r_min)
         pass
      case '2_1':
         # run_scenario_2_!()
         pass
      case '2_2':
         # run_scenario_2_2()
         pass
      case '2_3':
         # run_scenario_2_3()
         pass
      case '2_4':
         # run_scenario_2_4()
         pass


if __name__ == '__main__':
   main()
