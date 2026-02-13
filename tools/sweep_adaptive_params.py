"""Adaptive parameter sweep tool for 6-dimensional grid tuning.

Runs a grid of (horizon, q_pos, q_vel, r_u, alpha, lambda_vel) combinations on a
configurable drone scenario and collects simulation metrics to CSV. Optionally
generates heatmap slice visualizations.

Usage:
    # 2D sweep (backward compatible):
    python tools/sweep_adaptive_params.py \
        --alphas 0.1 0.2 0.3 0.4 0.5 0.6 0.8 1.0 \
        --lambdas 0.1 0.3 0.5 0.8 1.0 1.5 2.0 \
        --steps 200 \
        --output results/adaptive_sweep.csv

    # Full 6D sweep with box pattern:
    python tools/sweep_adaptive_params.py \
        --alphas 0.3 0.4 0.5 \
        --lambdas 0.3 0.4 0.5 0.6 \
        --horizon 3 4 5 \
        --q-pos 2.0 3.0 4.0 5.0 \
        --q-vel 0.1 0.2 0.3 \
        --r-u 0.1 0.5 0.8 \
        --v-max 2.5 \
        --num-drones 3 \
        --pattern box_8_1_5 \
        --room-size 8.0 \
        --safety-zone 0.6 \
        --steps 200 \
        --timeout 300 \
        --output results/full_adaptive_sweep.csv \
        --heatmap results/full_adaptive_heatmap.png \
        --heatmap-x lambda_vel \
        --heatmap-y alpha \
        --workers 10 \
        --heatmap-fix "horizon=4,q_pos=2.0,q_vel=0.2,r_u=0.5"

    # Generate heatmap from existing CSV:
    python tools/sweep_adaptive_params.py \
        --heatmap-from-csv results/adaptive_sweep.csv \
        --heatmap results/heatmap.png
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Ensure project root is on the path so that drone_sim is importable.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from drone_sim.domain.config import ScenarioConfig  # noqa: E402
from drone_sim.simulation.simulator import Simulator  # noqa: E402


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

_DRONE_COLORS = [
    "#00FFFF", "#800080", "#DC143C", "#008B8B",
    "#FF6347", "#4682B4", "#32CD32", "#FF69B4",
    "#FFD700", "#7B68EE", "#00CED1", "#FF4500",
    "#9ACD32", "#DA70D6", "#20B2AA", "#CD853F",
    "#6A5ACD", "#3CB371", "#FF1493", "#B22222",
    "#228B22", "#4169E1", "#C71585", "#2E8B57",
    "#D2691E", "#8B0000", "#556B2F", "#8A2BE2",
    "#5F9EA0", "#DC143C",
]


def _build_config(
    alpha: float,
    lambda_vel: float,
    q_pos: float,
    q_vel: float,
    r_u: float,
    v_max: float,
    horizon: int,
    num_drones: int,
    coordinator: str,
    pattern: str = "head_on",
    room_size: float = 5.0,
    safety_zone: float = 1.0,
) -> dict[str, Any]:
    """Build a full scenario config dict for a given parameter combination.

    Parameters
    ----------
    pattern : str
        "head_on" (default, X/Y axis pairs), "box_8_1_5", or "box_8".
    room_size : float
        Room dimension; produces [-room_size/2, room_size/2]^3.
    safety_zone : float
        Safety zone / r_min for each adaptive drone.
    """
    controller_params: dict[str, Any] = {
        "horizon": horizon,
        "q_pos": [q_pos] * 3,
        "q_vel": [q_vel] * 3,
        "r_u": [r_u] * 3,
        "lambda_vel": lambda_vel,
    }

    if pattern in ("box_8_1_5", "box_8"):
        # Import predefined patterns from constants
        if pattern == "box_8_1_5":
            from tools.utility.constants import PREDEFINED_PATTERNS_IN_BOX_8_1_5 as patterns
        else:
            from tools.utility.constants import PREDEFINED_PATTERNS_IN_BOX_8 as patterns

        if num_drones not in patterns:
            available = sorted(patterns.keys())
            raise ValueError(
                f"Pattern '{pattern}' does not have {num_drones} drones. "
                f"Available: {available}"
            )

        positions = patterns[num_drones]
        drones: list[dict[str, Any]] = []
        for idx, (start, target) in enumerate(positions):
            color = _DRONE_COLORS[idx % len(_DRONE_COLORS)]
            drones.append({
                "drone_id": f"drone-{idx + 1}",
                "physics": "standard",
                "start": list(start),
                "target": list(target),
                "radius": 0.2,
                "safety_zone": safety_zone,
                "cons_stop": 0.0,
                "alpha": alpha,
                "drone_color": color,
            })
    else:
        # head_on pattern: X-axis + optional Y-axis pairs
        drones = [
            {
                "drone_id": "drone-1",
                "physics": "standard",
                "start": [-2.0, 0.0, 0.0],
                "target": [2.0, 0.0, 0.0],
                "radius": 0.2,
                "safety_zone": safety_zone,
                "cons_stop": 0.0,
                "alpha": alpha,
                "drone_color": _DRONE_COLORS[0],
            },
            {
                "drone_id": "drone-2",
                "physics": "standard",
                "start": [2.0, 0.0, 0.0],
                "target": [-2.0, 0.0, 0.0],
                "radius": 0.2,
                "safety_zone": safety_zone,
                "cons_stop": 0.0,
                "alpha": alpha,
                "drone_color": _DRONE_COLORS[1],
            },
        ]

        if num_drones >= 4:
            # Y-axis head-on pair
            drones.extend([
                {
                    "drone_id": "drone-3",
                    "physics": "standard",
                    "start": [0.0, -2.0, 0.0],
                    "target": [0.0, 2.0, 0.0],
                    "radius": 0.2,
                    "safety_zone": safety_zone,
                    "cons_stop": 0.0,
                    "alpha": alpha,
                    "drone_color": _DRONE_COLORS[2],
                },
                {
                    "drone_id": "drone-4",
                    "physics": "standard",
                    "start": [0.0, 2.0, 0.0],
                    "target": [0.0, -2.0, 0.0],
                    "radius": 0.2,
                    "safety_zone": safety_zone,
                    "cons_stop": 0.0,
                    "alpha": alpha,
                    "drone_color": _DRONE_COLORS[3],
                },
            ])

    half = room_size / 2.0
    coordinator_params: dict[str, Any] = {"horizon": horizon}
    if coordinator == "mpc_central":
        coordinator_params["room_wall_tolerance"] = 0.5

    return {
        "dt": 0.1,
        "room": {"min": [-half, -half, -half], "max": [half, half, half]},
        "physics": [
            {
                "id": "standard",
                "type": "linear_kinematics",
                "params": {
                    "v_max": v_max,
                    "u_min": [-3.0, -3.0, -3.0],
                    "u_max": [3.0, 3.0, 3.0],
                },
            }
        ],
        "controller": {
            "type": "mpc_agent_adaptive",
            "params": controller_params,
        },
        "coordinator": {
            "type": coordinator,
            "params": coordinator_params,
        },
        "drones": drones,
    }


# ---------------------------------------------------------------------------
# Run result
# ---------------------------------------------------------------------------

@dataclass
class SweepResult:
    """Collected metrics from a single simulation run across the 6D parameter grid."""

    horizon: int
    q_pos: float
    q_vel: float
    r_u: float
    alpha: float
    lambda_vel: float
    status: str = "ok"
    min_pair_margin: float = float("inf")
    max_speed: float = 0.0
    avg_speed: float = 0.0
    collision_count: int = 0
    total_solve_time: float = 0.0
    avg_step_time: float = 0.0
    reached_target: bool = False


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------

def _run_single(
    alpha: float,
    lambda_vel: float,
    steps: int,
    q_pos: float,
    q_vel: float,
    r_u: float,
    v_max: float,
    horizon: int,
    num_drones: int,
    coordinator: str,
    pattern: str = "head_on",
    room_size: float = 5.0,
    safety_zone: float = 1.0,
    timeout: int = 300,
) -> SweepResult:
    """Run one simulation for a given parameter combination and return metrics."""
    result = SweepResult(
        horizon=horizon,
        q_pos=q_pos,
        q_vel=q_vel,
        r_u=r_u,
        alpha=alpha,
        lambda_vel=lambda_vel,
    )

    try:
        cfg_dict = _build_config(
            alpha=alpha,
            lambda_vel=lambda_vel,
            q_pos=q_pos,
            q_vel=q_vel,
            r_u=r_u,
            v_max=v_max,
            horizon=horizon,
            num_drones=num_drones,
            coordinator=coordinator,
            pattern=pattern,
            room_size=room_size,
            safety_zone=safety_zone,
        )
        scenario = ScenarioConfig.model_validate(cfg_dict)
        sim = Simulator.from_config(scenario)

        step_times: list[float] = []
        all_speeds: list[float] = []
        t_start = time.perf_counter()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for _ in range(steps):
                # Check per-simulation timeout
                if time.perf_counter() - t_start > timeout:
                    result.status = "timeout"
                    break

                # Metrics before step
                positions = [d.position() for d in sim.drones]
                velocities = [d.velocity() for d in sim.drones]
                r_safe = [d.compute_adaptive_radius(d.velocity()) for d in sim.drones]

                # Pairwise margin
                n = len(sim.drones)
                for i in range(n):
                    for j in range(i + 1, n):
                        dist = float(np.linalg.norm(positions[i] - positions[j]))
                        thresh = float(r_safe[i] + r_safe[j])
                        margin = dist - thresh
                        result.min_pair_margin = min(result.min_pair_margin, margin)

                # Speed
                speeds = [float(np.linalg.norm(v)) for v in velocities]
                max_spd = max(speeds) if speeds else 0.0
                result.max_speed = max(result.max_speed, max_spd)
                all_speeds.append(float(np.mean(speeds)) if speeds else 0.0)

                # Step with timing
                t0 = time.perf_counter()
                sim.step()
                dt = time.perf_counter() - t0
                step_times.append(dt)

        result.total_solve_time = sum(step_times)
        result.avg_step_time = result.total_solve_time / len(step_times) if step_times else 0.0
        result.avg_speed = float(np.mean(all_speeds)) if all_speeds else 0.0

        # Collision check
        result.collision_count = len([c for c in sim.last_collisions if c["kind"] == "drone_drone"])

        # Check if drones reached their targets (all within 0.5 of target at final step)
        all_reached = True
        for d in sim.drones:
            dist_to_target = float(np.linalg.norm(d.position() - d.route.target))
            if dist_to_target > 0.5:
                all_reached = False
                break
        result.reached_target = all_reached

        if result.status == "ok" and result.collision_count > 0:
            result.status = "collision"

    except Exception as exc:
        result.status = "failed"
        print(f"    FAILED: {exc}", file=sys.stderr)

    return result


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------

_CSV_FIELDS = [
    "horizon", "q_pos", "q_vel", "r_u", "alpha", "lambda_vel",
    "status", "min_pair_margin", "max_speed", "avg_speed",
    "collision_count", "total_solve_time", "avg_step_time", "reached_target",
]


def _write_csv(results: list[SweepResult], path: Path) -> None:
    """Write sweep results to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for r in results:
            not_failed = r.status != "failed"
            writer.writerow({
                "horizon": r.horizon,
                "q_pos": f"{r.q_pos:.4f}",
                "q_vel": f"{r.q_vel:.4f}",
                "r_u": f"{r.r_u:.4f}",
                "alpha": f"{r.alpha:.4f}",
                "lambda_vel": f"{r.lambda_vel:.4f}",
                "status": r.status,
                "min_pair_margin": f"{r.min_pair_margin:.4f}" if not_failed else "",
                "max_speed": f"{r.max_speed:.4f}" if not_failed else "",
                "avg_speed": f"{r.avg_speed:.4f}" if not_failed else "",
                "collision_count": r.collision_count if not_failed else "",
                "total_solve_time": f"{r.total_solve_time:.4f}" if not_failed else "",
                "avg_step_time": f"{r.avg_step_time:.4f}" if not_failed else "",
                "reached_target": str(r.reached_target).lower() if not_failed else "",
            })


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def _print_summary(results: list[SweepResult]) -> None:
    """Print summary table to stdout."""
    ok = [r for r in results if r.status == "ok"]
    collisions = [r for r in results if r.status == "collision"]
    failed = [r for r in results if r.status == "failed"]
    timeouts = [r for r in results if r.status == "timeout"]

    print()
    print("=" * 70)
    print("SWEEP SUMMARY")
    print("=" * 70)
    print(f"  Total combinations: {len(results)}")
    print(f"  Successful (ok):    {len(ok)}")
    print(f"  Collisions:         {len(collisions)}")
    print(f"  Timeouts:           {len(timeouts)}")
    print(f"  Failed:             {len(failed)}")
    print()

    if ok:
        def _param_str(r: SweepResult) -> str:
            return (f"H={r.horizon} q_pos={r.q_pos:.2f} q_vel={r.q_vel:.2f} "
                    f"r_u={r.r_u:.2f} alpha={r.alpha:.2f} lam={r.lambda_vel:.2f}")

        # Best by margin (safest)
        best_margin = max(ok, key=lambda r: r.min_pair_margin)
        print(f"  Best margin (safest):      {_param_str(best_margin)}  "
              f"margin={best_margin.min_pair_margin:.4f}")

        # Best by solve time (fastest)
        best_time = min(ok, key=lambda r: r.avg_step_time)
        print(f"  Best solve time (fastest): {_param_str(best_time)}  "
              f"avg_step={best_time.avg_step_time:.4f}s")

        # Best collision-free with highest margin
        collision_free = [r for r in ok if r.collision_count == 0]
        if collision_free:
            best_safe = max(collision_free, key=lambda r: r.min_pair_margin)
            print(f"  Best safe + high margin:   {_param_str(best_safe)}  "
                  f"margin={best_safe.min_pair_margin:.4f}")
        else:
            print("  No collision-free runs found!")

    print()


# ---------------------------------------------------------------------------
# Heatmap visualization (slice views for multi-dim data)
# ---------------------------------------------------------------------------

# The 6 sweepable parameter names (used for heatmap axis selection)
_PARAM_NAMES = ("horizon", "q_pos", "q_vel", "r_u", "alpha", "lambda_vel")


def _parse_heatmap_fix(fix_str: str | None) -> dict[str, float]:
    """Parse --heatmap-fix string like 'horizon=4,q_pos=2.0,q_vel=0.2,r_u=0.5'."""
    if not fix_str:
        return {}
    result: dict[str, float] = {}
    for part in fix_str.split(","):
        part = part.strip()
        if "=" not in part:
            continue
        key, val = part.split("=", 1)
        key = key.strip()
        val = val.strip()
        if key in _PARAM_NAMES:
            result[key] = float(val)
    return result


def _generate_heatmaps(
    csv_path: Path,
    output_path: Path,
    params: dict[str, Any],
    heatmap_x: str = "lambda_vel",
    heatmap_y: str = "alpha",
    heatmap_fix: dict[str, float] | None = None,
) -> None:
    """Generate 2x2 heatmap grid from sweep CSV results.

    For multi-dimensional sweeps, this produces a slice view: rows are filtered
    to match the fixed parameter values, then pivoted on the two chosen axes.

    Subplots:
    1. Min pairwise margin (higher = safer)
    2. Max speed (lower = more deceleration)
    3. Avg solve time per step (lower = faster)
    4. Collision count (0=good, >0=bad)
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available -- skipping heatmap generation.", file=sys.stderr)
        return

    # Read CSV (use csv module for compatibility -- no pandas dependency)
    rows: list[dict[str, str]] = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        print("CSV is empty -- nothing to plot.", file=sys.stderr)
        return

    # Determine which parameters are fixed vs. axis parameters
    if heatmap_fix is None:
        heatmap_fix = {}

    # For non-axis parameters that are not explicitly fixed, use the first value
    # found in the CSV
    fixed_params: dict[str, float] = dict(heatmap_fix)
    for param in _PARAM_NAMES:
        if param == heatmap_x or param == heatmap_y:
            continue
        if param not in fixed_params:
            # Use first value from CSV
            if param in rows[0]:
                fixed_params[param] = float(rows[0][param])

    # Filter rows to match fixed params
    def _matches_fixed(row: dict[str, str]) -> bool:
        for param, val in fixed_params.items():
            if param == heatmap_x or param == heatmap_y:
                continue
            if param not in row:
                continue
            # Use approximate comparison for floats
            if abs(float(row[param]) - val) > 1e-6:
                return False
        return True

    filtered = [row for row in rows if _matches_fixed(row)]

    if not filtered:
        print("No rows match the heatmap filter criteria. Check --heatmap-fix values.",
              file=sys.stderr)
        return

    # Extract unique sorted axis values
    x_set: set[float] = set()
    y_set: set[float] = set()
    for row in filtered:
        x_set.add(float(row[heatmap_x]))
        y_set.add(float(row[heatmap_y]))

    x_sorted = sorted(x_set)
    y_sorted = sorted(y_set)

    n_x = len(x_sorted)
    n_y = len(y_sorted)

    x_idx = {v: i for i, v in enumerate(x_sorted)}
    y_idx = {v: i for i, v in enumerate(y_sorted)}

    # Build metric grids (y_axis x x_axis)
    margin_grid = np.full((n_y, n_x), np.nan)
    speed_grid = np.full((n_y, n_x), np.nan)
    time_grid = np.full((n_y, n_x), np.nan)
    collision_grid = np.full((n_y, n_x), np.nan)
    status_grid = np.full((n_y, n_x), "", dtype=object)

    for row in filtered:
        xi = x_idx[float(row[heatmap_x])]
        yi = y_idx[float(row[heatmap_y])]
        status_grid[yi, xi] = row["status"]
        if row["status"] not in ("failed",):
            margin_grid[yi, xi] = float(row["min_pair_margin"]) if row["min_pair_margin"] else np.nan
            speed_grid[yi, xi] = float(row["max_speed"]) if row["max_speed"] else np.nan
            time_grid[yi, xi] = float(row["avg_step_time"]) if row["avg_step_time"] else np.nan
            collision_grid[yi, xi] = int(row["collision_count"]) if row["collision_count"] else np.nan

    # Build figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Title with fixed params
    title_parts = []
    for key in sorted(fixed_params.keys()):
        if key != heatmap_x and key != heatmap_y:
            val = fixed_params[key]
            if val == int(val):
                title_parts.append(f"{key}={int(val)}")
            else:
                title_parts.append(f"{key}={val}")
    for key in ("v_max", "num_drones", "steps", "pattern"):
        if key in params and key not in fixed_params:
            title_parts.append(f"{key}={params[key]}")
    fig.suptitle("Adaptive Param Sweep (" + ", ".join(title_parts) + ")", fontsize=13)

    # Helpers
    x_labels = [f"{v:.2f}" if isinstance(v, float) and v != int(v) else str(int(v))
                for v in x_sorted]
    y_labels = [f"{v:.2f}" if isinstance(v, float) and v != int(v) else str(int(v))
                for v in y_sorted]

    def _plot_heatmap(ax, data, title, cmap, fmt=".3f"):
        im = ax.imshow(data, aspect="auto", origin="lower", cmap=cmap)
        ax.set_xticks(range(n_x))
        ax.set_xticklabels(x_labels, fontsize=8)
        ax.set_yticks(range(n_y))
        ax.set_yticklabels(y_labels, fontsize=8)
        ax.set_xlabel(heatmap_x)
        ax.set_ylabel(heatmap_y)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.8)

        # Annotate cells
        for i in range(n_y):
            for j in range(n_x):
                status = status_grid[i, j]
                if status == "failed":
                    ax.text(j, i, "X", ha="center", va="center", color="red",
                            fontsize=10, fontweight="bold")
                elif status == "timeout":
                    ax.text(j, i, "T", ha="center", va="center", color="orange",
                            fontsize=10, fontweight="bold")
                elif not np.isnan(data[i, j]):
                    val = data[i, j]
                    txt = f"{val:{fmt}}"
                    ax.text(j, i, txt, ha="center", va="center", color="black",
                            fontsize=7)

    # 1. Min pairwise margin (green=safe=high, red=low)
    _plot_heatmap(axes[0, 0], margin_grid, "Min Pairwise Margin", "RdYlGn")

    # 2. Max speed
    _plot_heatmap(axes[0, 1], speed_grid, "Max Speed", "coolwarm")

    # 3. Avg solve time per step
    _plot_heatmap(axes[1, 0], time_grid, "Avg Step Solve Time (s)", "YlOrRd", fmt=".4f")

    # 4. Collision count (binary: green=0, red=>0)
    coll_display = collision_grid.copy()
    _plot_heatmap(axes[1, 1], coll_display, "Collision Count", "RdYlGn_r", fmt=".0f")

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Heatmap saved to {output_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Sweep adaptive safety zone parameters and collect metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--alphas", type=float, nargs="+",
                        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0],
                        help="Alpha values to sweep (default: 0.1 0.2 0.3 0.4 0.5 0.6 0.8 1.0)")
    parser.add_argument("--lambdas", type=float, nargs="+",
                        default=[0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0],
                        help="Lambda_vel values to sweep (default: 0.1 0.3 0.5 0.8 1.0 1.5 2.0)")
    parser.add_argument("--horizon", type=int, nargs="+", default=[4],
                        help="MPC horizon values to sweep (default: 4)")
    parser.add_argument("--q-pos", type=float, nargs="+", default=[2.0],
                        help="Position tracking weight values to sweep (default: 2.0)")
    parser.add_argument("--q-vel", type=float, nargs="+", default=[0.2],
                        help="Velocity tracking weight values to sweep (default: 0.2)")
    parser.add_argument("--r-u", type=float, nargs="+", default=[0.5],
                        help="Control effort weight values to sweep (default: 0.5)")
    parser.add_argument("--steps", type=int, default=200,
                        help="Simulation steps per combination (default: 200)")
    parser.add_argument("--coordinator", type=str, default="mpc_central",
                        choices=["mpc_central", "dmpc_admm"],
                        help="Coordinator type (default: mpc_central)")
    parser.add_argument("--num-drones", type=int, default=4,
                        help="Number of drones (default: 4)")
    parser.add_argument("--v-max", type=float, default=3.0,
                        help="Maximum velocity (default: 3.0)")
    parser.add_argument("--pattern", type=str, default="head_on",
                        choices=["head_on", "box_8_1_5", "box_8"],
                        help="Drone position pattern (default: head_on)")
    parser.add_argument("--room-size", type=float, default=5.0,
                        help="Room dimension, produces [-size/2, size/2]^3 (default: 5.0)")
    parser.add_argument("--safety-zone", type=float, default=1.0,
                        help="Safety zone / r_min for adaptive drones (default: 1.0)")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Per-simulation timeout in seconds (default: 300)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: CPU count)")
    parser.add_argument("--output", type=str, default="results/adaptive_sweep.csv",
                        help="CSV output path (default: results/adaptive_sweep.csv)")
    parser.add_argument("--heatmap", type=str, default=None,
                        help="If provided, generate heatmap PNG at this path")
    parser.add_argument("--heatmap-from-csv", type=str, default=None,
                        help="Generate heatmap from an existing CSV (no simulation)")
    parser.add_argument("--heatmap-x", type=str, default="lambda_vel",
                        choices=list(_PARAM_NAMES),
                        help="Parameter for heatmap X-axis (default: lambda_vel)")
    parser.add_argument("--heatmap-y", type=str, default="alpha",
                        choices=list(_PARAM_NAMES),
                        help="Parameter for heatmap Y-axis (default: alpha)")
    parser.add_argument("--heatmap-fix", type=str, default=None,
                        help="Fix other params for heatmap slice, e.g. "
                             "'horizon=4,q_pos=2.0,q_vel=0.2,r_u=0.5'")

    args = parser.parse_args(argv)

    heatmap_fix = _parse_heatmap_fix(args.heatmap_fix)

    # Mode: re-generate heatmap from existing CSV
    if args.heatmap_from_csv is not None:
        csv_path = Path(args.heatmap_from_csv)
        if not csv_path.exists():
            print(f"CSV file not found: {csv_path}", file=sys.stderr)
            sys.exit(1)
        if args.heatmap is None:
            print("--heatmap path required when using --heatmap-from-csv", file=sys.stderr)
            sys.exit(1)
        _generate_heatmaps(
            csv_path,
            Path(args.heatmap),
            params={
                "v_max": args.v_max, "num_drones": args.num_drones,
                "steps": args.steps, "pattern": args.pattern,
            },
            heatmap_x=args.heatmap_x,
            heatmap_y=args.heatmap_y,
            heatmap_fix=heatmap_fix,
        )
        return

    # Mode: run sweep
    alphas = sorted(set(args.alphas))
    lambdas = sorted(set(args.lambdas))
    horizons = sorted(set(args.horizon))
    q_pos_values = sorted(set(args.q_pos))
    q_vel_values = sorted(set(args.q_vel))
    r_u_values = sorted(set(args.r_u))

    total = (len(horizons) * len(q_pos_values) * len(q_vel_values)
             * len(r_u_values) * len(alphas) * len(lambdas))

    num_workers = args.workers if args.workers is not None else os.cpu_count()
    num_workers = max(1, min(num_workers, total))

    print(f"Adaptive parameter sweep: {len(horizons)} H x {len(q_pos_values)} q_pos x "
          f"{len(q_vel_values)} q_vel x {len(r_u_values)} r_u x "
          f"{len(alphas)} alpha x {len(lambdas)} lambda = {total} combinations",
          file=sys.stderr)
    print(f"Steps per combination: {args.steps}  |  Timeout: {args.timeout}s  |  "
          f"Workers: {num_workers}", file=sys.stderr)
    print(f"Coordinator: {args.coordinator}  |  Drones: {args.num_drones}  |  "
          f"Pattern: {args.pattern}", file=sys.stderr)
    print(f"Room: {args.room_size}  |  Safety zone: {args.safety_zone}  |  "
          f"v_max: {args.v_max}", file=sys.stderr)
    print(file=sys.stderr)

    # Build list of all parameter combos
    combos: list[dict[str, Any]] = []
    for horizon in horizons:
        for q_pos in q_pos_values:
            for q_vel in q_vel_values:
                for r_u in r_u_values:
                    for alpha in alphas:
                        for lam in lambdas:
                            combos.append(dict(
                                alpha=alpha, lambda_vel=lam,
                                steps=args.steps, q_pos=q_pos, q_vel=q_vel,
                                r_u=r_u, v_max=args.v_max, horizon=horizon,
                                num_drones=args.num_drones,
                                coordinator=args.coordinator,
                                pattern=args.pattern, room_size=args.room_size,
                                safety_zone=args.safety_zone, timeout=args.timeout,
                            ))

    def _format_result(idx: int, r: SweepResult) -> str:
        tag = (f"[{idx}/{total}] H={r.horizon} q_pos={r.q_pos:.2f} "
               f"q_vel={r.q_vel:.2f} r_u={r.r_u:.2f} "
               f"alpha={r.alpha:.2f} lam={r.lambda_vel:.2f}")
        if r.status == "ok":
            return (f"{tag} ... ok   (margin={r.min_pair_margin:.4f}, "
                    f"speed={r.max_speed:.2f}, time={r.total_solve_time:.1f}s)")
        elif r.status == "collision":
            return (f"{tag} ... COLLISION  (collisions={r.collision_count}, "
                    f"margin={r.min_pair_margin:.4f}, time={r.total_solve_time:.1f}s)")
        elif r.status == "timeout":
            return f"{tag} ... TIMEOUT  (time={r.total_solve_time:.1f}s)"
        else:
            return f"{tag} ... FAILED"

    # Run sweep in parallel
    results: list[SweepResult] = [None] * total  # type: ignore[list-item]
    done_count = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as pool:
        future_to_idx = {
            pool.submit(_run_single, **combo): i
            for i, combo in enumerate(combos)
        }
        for future in concurrent.futures.as_completed(future_to_idx):
            i = future_to_idx[future]
            r = future.result()
            results[i] = r
            done_count += 1
            print(_format_result(done_count, r), file=sys.stderr)

    # Write CSV
    output_path = Path(args.output)
    _write_csv(results, output_path)
    print(f"\nCSV written to {output_path}", file=sys.stderr)

    # Print summary to stdout
    _print_summary(results)

    # Generate heatmap if requested
    if args.heatmap is not None:
        _generate_heatmaps(
            output_path,
            Path(args.heatmap),
            params={
                "v_max": args.v_max, "num_drones": args.num_drones,
                "steps": args.steps, "pattern": args.pattern,
            },
            heatmap_x=args.heatmap_x,
            heatmap_y=args.heatmap_y,
            heatmap_fix=heatmap_fix,
        )


if __name__ == "__main__":
    main()
