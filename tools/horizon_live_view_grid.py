from __future__ import annotations

import argparse
import csv
import io
import time
from enum import StrEnum
from pathlib import Path

import numpy as np
from PIL import Image

from drone_sim.api.render import render_png
from drone_sim.simulation.simulator import Simulator
from drone_sim.simulation.distributed_coordinator import DistributedMPCCoordinator
from tools import _build_scenario, _COLOR_BY_DRONE_INDEX, _all_drones_reached_destination


class Status(StrEnum):
   RUNNING = "running"
   FINISHED = "finished"
   MAX_STEPS = "max_steps"
   INFEASIBLE = "infeasible"
   ERROR = "error"


def _apply_colors(scenario) -> None:
   for i, drone in enumerate(scenario.drones):
      idx = i + 1
      color = _COLOR_BY_DRONE_INDEX.get(idx)
      if color is None:
         continue
      drone.drone_color = color  # Let safety_color and trace_color default from drone_color inside the simulator.


def _pairwise_distances(positions: list[np.ndarray]) -> np.ndarray:
   n = len(positions)
   if n <= 1:
      return np.asarray([], dtype=float)

   dists: list[float] = []
   for i in range(n):
      pi = positions[i]
      for j in range(i + 1, n):
         pj = positions[j]
         dists.append(float(np.linalg.norm(pi - pj)))
   return np.asarray(dists, dtype=float)


def _print_results(all_pair_dists: list[float], frames: list[Image.Image], gif_fps: float,
                   gif_path: Path, horizon: int, jerk_3d_value: float, num_drones: int, out_dir: Path,
                   status: Status, step_durations: list[float], step_mean_pair_dists: list[float],
                   wall_time: float) -> None:
   if frames:
      duration_ms = int(round(1000.0 / max(0.1, float(gif_fps))))
      frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0, optimize=False)
   print(f"status={status}, frames={len(frames)}, wall_time={wall_time:.2f}s -> GIF: {gif_path}")

   # Print metrics summary for this scenario.
   num_steps = len(step_durations)

   if num_steps > 0:
      step_durations_arr = np.asarray(step_durations, dtype=float)
      min_step_time = float(step_durations_arr.min())
      max_step_time = float(step_durations_arr.max())
      mean_step_time = float(step_durations_arr.mean())
   else:
      min_step_time = max_step_time = mean_step_time = 0.0

   if all_pair_dists:
      all_pair_dists_arr = np.asarray(all_pair_dists, dtype=float)
      min_dist = float(all_pair_dists_arr.min())
      max_dist = float(all_pair_dists_arr.max())
      mean_dist = float(all_pair_dists_arr.mean())
   else:
      min_dist = max_dist = mean_dist = 0.0

   if step_mean_pair_dists:
      mean_step_mean_dist = float(np.asarray(step_mean_pair_dists, dtype=float).mean())
   else:
      mean_step_mean_dist = 0.0

   print(f"  distances: min={min_dist:.3f}, max={max_dist:.3f}, "
         f"mean(all pairs)={mean_dist:.3f}, mean(step mean)={mean_step_mean_dist:.3f}")
   print(f"  jerk_3d_value (piecewise linear loss over 3D trajectories)={jerk_3d_value:.3f}")
   print(f"  timing: steps={num_steps}, wall_time={wall_time:.3f}s, "
         f"min_step={min_step_time:.4f}s, max_step={max_step_time:.4f}s, mean_step={mean_step_time:.4f}s")

   # Write metrics to CSV (one row per scenario) in the same output directory.
   csv_path = out_dir / "metrics.csv"
   fieldnames = ["num_drones", "horizon", "status", "frames", "steps", "wall_time_s", "min_step_time_s",
                 "max_step_time_s", "mean_step_time_s", "min_distance", "max_distance", "mean_distance_all_pairs",
                 "mean_distance_step_mean", "jerk_3d_value"]
   row = {"num_drones": num_drones, "horizon": horizon, "status": status, "frames": len(frames), "steps": num_steps,
          "wall_time_s": wall_time, "min_step_time_s": min_step_time, "max_step_time_s": max_step_time,
          "mean_step_time_s": mean_step_time, "min_distance": min_dist, "max_distance": max_dist,
          "mean_distance_all_pairs": mean_dist, "mean_distance_step_mean": mean_step_mean_dist,
          "jerk_3d_value": jerk_3d_value}
   # Append with header creation if needed.
   write_header = not csv_path.exists()
   with csv_path.open("a", newline="", encoding="utf-8") as f:
      writer = csv.DictWriter(f, fieldnames=fieldnames)
      if write_header:
         writer.writeheader()
      writer.writerow(row)


def piecewise_linear_loss_3d(points, penalty=1.0, eps_step=1e-6, angle_threshold_deg=90.0):
   """
   Piecewise-linear fit of a 3D trajectory with penalties for direction turnarounds.

   Args:
       points: array-like of shape (n, 3)
           Sequence of 3D points [x_i, y_i, z_i].
       penalty: float
           Cost added for each turnaround (i.e. each break between line segments).
       eps_step: float
           Threshold to treat very small step vectors as zero (noise).
       angle_threshold_deg: float
           Angle (in degrees) at or above which the change in direction is considered a "turnaround" and causes a new segment to start.
           Default 90°, i.e. direction flips from "mostly one way" to "mostly the opposite way".

   Returns:
       loss: float
           Total loss = sum of squared Euclidean errors to piecewise-linear fit
           + penalty * (# of breaks).
       fitted: ndarray of shape (n, 3)
           Smoothed/fitted 3D points.
       segment_starts: list[int]
           Indices where each segment starts (0 is always included).
   """
   pts = np.asarray(points, dtype=float)
   if pts.ndim != 2 or pts.shape[1] != 3:
      raise ValueError("points must have shape (n, 3)")

   n = pts.shape[0]
   if n < 2:
      # Nothing to fit
      return 0.0, pts.copy(), [0]

   # Parameter along the path (could be time or just index)
   t = np.arange(n, dtype=float)

   # 1. Detect turnarounds based on 3D direction changes
   steps = np.diff(pts, axis=0)  # (n-1, 3)
   # Zero out tiny step components to reduce numerical noise
   steps[np.abs(steps) < eps_step] = 0.0

   # Compute unit direction vectors for non-zero steps
   step_norms = np.linalg.norm(steps, axis=1)
   dirs = np.zeros_like(steps)
   nonzero_mask = step_norms > eps_step
   dirs[nonzero_mask] = steps[nonzero_mask] / step_norms[nonzero_mask, None]

   cos_threshold = np.cos(np.deg2rad(angle_threshold_deg))

   breaks = [0]  # segment start indices

   for i in range(1, len(dirs)):
      d_prev = dirs[i - 1]
      d_cur = dirs[i]

      # Skip if either direction is basically undefined (zero step)
      if (np.linalg.norm(d_prev) < eps_step or np.linalg.norm(d_cur) < eps_step):
         continue

      # cos(theta) = d_prev · d_cur (both unit vectors)
      cos_angle = float(np.dot(d_prev, d_cur))

      # Turnaround if angle >= angle_threshold_deg
      if cos_angle < cos_threshold:
         # New segment starts at index i
         breaks.append(i)

   breaks.append(n)  # sentinel for the last segment end

   # 2. Fit line (3D) on each segment and accumulate squared error
   fitted = np.zeros_like(pts)
   sq_err = 0.0

   for s in range(len(breaks) - 1):
      start = breaks[s]
      end = breaks[s + 1]

      t_seg = t[start:end]  # shape (m,)
      pts_seg = pts[start:end]  # shape (m, 3)

      if end - start == 1:
         # Single point segment: nothing to fit
         fitted[start:end] = pts_seg
         continue

      # Design matrix for linear model: pts ≈ a * t + b
      # A has shape (m, 2)
      A = np.vstack([t_seg, np.ones_like(t_seg)]).T

      # Solve for a and b for all 3 dims at once: shape coeffs = (2, 3)
      coeffs, *_ = np.linalg.lstsq(A, pts_seg, rcond=None)
      a = coeffs[0]  # (3,)
      b = coeffs[1]  # (3,)

      # Fitted points on this segment
      fit_seg = (a[None, :] * t_seg[:, None]) + b[None, :]  # (m, 3)
      fitted[start:end] = fit_seg

      # Squared Euclidean error
      sq_err += np.sum((pts_seg - fit_seg) ** 2)

   num_segments = len(breaks) - 1
   num_breaks = max(0, num_segments - 1)
   loss = sq_err + penalty * num_breaks

   segment_starts = breaks[:-1]

   return loss, fitted, segment_starts


def run_single_scenario_live_view(*, num_drones: int, horizon: int, out_dir: Path, max_steps: int = 500,
                                  gif_fps: float = 20.0, width: int = 900, height: int = 700, dpi: int = 120,
                                  elev: float = 20.0, azim: float = -60.0, trace_len: int = 50) -> None:
   """
   Run one (N, H) pair using direct simulator execution and create a GIF.

   Returns (status, num_frames, wall_time_seconds), where status is one of:
     - "finished"           all drones reached their targets
     - "max_steps"          hit max_steps
     - "infeasible"         MPC reported infeasible
     - "error"              unexpected error
   """

   # Build the same ScenarioConfig as test_horizon_feasibility, then patch colors.
   scenario = _build_scenario(num_drones=num_drones, horizon=horizon)
   _apply_colors(scenario)

   out_dir.mkdir(parents=True, exist_ok=True)
   gif_path = out_dir / f"N{num_drones}_H{horizon}.gif"

   status = Status.RUNNING

   t0 = time.perf_counter()

   # Metrics accumulators
   step_durations: list[float] = []
   step_mean_pair_dists: list[float] = []
   all_pair_dists: list[float] = []

   frames: list[Image.Image] = []

   # For 3D jerk metric: store full position history per drone.
   positions_by_drone: dict[str, list[np.ndarray]] = {}

   def _compute_jerk_3d_value() -> float:
      jerk_3d_total = 0.0
      for traj in positions_by_drone.values():
         if len(traj) < 2:
            continue
         pts = np.stack(traj, axis=0)
         loss, _, _ = piecewise_linear_loss_3d(pts)
         jerk_3d_total += float(loss)
      return jerk_3d_total

   try:
      sim = Simulator.from_config(scenario)
   except Exception as e:
      print(f"  Error creating simulator: {e}")
      status = Status.ERROR
      wall_time = time.perf_counter() - t0
      jerk_3d_value = _compute_jerk_3d_value()
      _print_results(all_pair_dists, frames, gif_fps, gif_path, horizon, jerk_3d_value, num_drones,
                     out_dir, status, step_durations, step_mean_pair_dists, wall_time)
      return

   for step_idx in range(max_steps):
      t_step0 = time.perf_counter()

      # Render current frame
      traces = [[] if trace_len == 0 else sim.traces.get(d.drone_id, [])[-trace_len:] for d in sim.drones]
      safety_zones = [float(d.safety_zone) for d in sim.drones]

      is_distributed = isinstance(sim.coordinator, DistributedMPCCoordinator)
      neighbor_links = None
      admm_iteration_count = sim.coordinator.get_last_iteration_count() if is_distributed else None
      admm_converged = sim.coordinator.get_last_converged() if is_distributed else None

      try:
         png_bytes = render_png(room_min=sim.room_min, room_max=sim.room_max,
            drone_positions=[d.position() for d in sim.drones], drone_radii=[d.radius for d in sim.drones], drone_safety_zones=safety_zones,
            drone_colors=[d.color for d in sim.drones], safety_colors=[d.safety_color for d in sim.drones], trace_colors=[d.trace_color for d in sim.drones],
            drone_traces=traces, obstacles=sim.obstacles,
            step_count=sim.step_count, compute_time_s=sim.compute_time_s,
            neighbor_links=neighbor_links, admm_iteration_count=admm_iteration_count, admm_converged=admm_converged,
            width=width, height=height, dpi=dpi, elev=elev, azim=azim
         )
         img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
         frames.append(img)
      except Exception as e:
         print(f"  Render error at step {step_idx}: {e}")
         status = Status.ERROR
         break

      # Advance simulation by one step
      try:
         sim.step()
      except Exception as e:
         print(f"  Step error at step {step_idx}: {e}")
         status = Status.ERROR
         break

      if sim.infeasible:
         detail = sim.infeasible_reason or "MPC reported infeasible controls."
         print(f"  Infeasible at step {step_idx}: {detail}")
         status = Status.INFEASIBLE
         break

      # Collect metrics from current state
      drones_sorted = sorted(sim.drones, key=lambda d: d.drone_id)

      positions: list[np.ndarray] = []
      for d in drones_sorted:
         p = d.position()
         positions.append(p)

         # Store full position history per drone for 3D jerk metric.
         traj = positions_by_drone.setdefault(d.drone_id, [])
         traj.append(p)

      # Pairwise distance metrics.
      dists = _pairwise_distances(positions)
      if dists.size > 0:
         all_pair_dists.extend(dists.tolist())
         step_mean_pair_dists.append(float(dists.mean()))

      # Check if all drones reached destination
      if _all_drones_reached_destination(sim.drones):
         status = Status.FINISHED
         t_step1 = time.perf_counter()
         step_durations.append(t_step1 - t_step0)
         break

      t_step1 = time.perf_counter()
      step_durations.append(t_step1 - t_step0)

   else:
      # Loop completed without break - hit max_steps
      status = Status.MAX_STEPS

   wall_time = time.perf_counter() - t0
   jerk_3d_value = _compute_jerk_3d_value()
   _print_results(all_pair_dists, frames, gif_fps, gif_path, horizon, jerk_3d_value, num_drones,
                  out_dir, status, step_durations, step_mean_pair_dists, wall_time)


def main(argv: list[str] | None = None) -> None:
   p = argparse.ArgumentParser(
      description="Run the predefined horizon-feasibility scenarios and create live-view GIFs for N=2..7, H=1..20.")

   p.add_argument("--out-dir", type=Path, default=Path("live_view_horizon_grid/results"), help="Directory where GIFs will be written (one per N,H pair)")
   p.add_argument("--max-steps", type=int, default=500, help="Per-scenario maximum number of simulation steps")
   p.add_argument("--gif-fps", type=float, default=20.0, help="FPS for generated GIFs")
   p.add_argument("--trace-len", type=int, default=50, help="Number of trace points to render")

   args = p.parse_args(argv)

   drone_counts = range(2, 8)
   horizons = range(1, 21)

   print(f"Running live-view GIF grid for N in {list(drone_counts)}, H in {list(horizons)}")
   print(f"Output directory: {args.out_dir}")

   for n in drone_counts:
      for H in horizons:
         print(f"=== Scenario N={n}, H={H} ===")
         run_single_scenario_live_view(num_drones=n, horizon=H, out_dir=args.out_dir, max_steps=args.max_steps, gif_fps=args.gif_fps, trace_len=args.trace_len)


if __name__ == "__main__":
   main()
