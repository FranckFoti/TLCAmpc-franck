"""3-mode safety zone comparison across configs/lstm/ scenario configs.

Runs each config under fixed / adaptive / lstm safety zone modes and writes
per-scenario per-mode metrics to .planning/comparison.csv.

Usage:
    python tools/compare_modes.py [--steps N] [--output PATH]

Steps defaults to 150. Output defaults to .planning/comparison.csv.
"""
from __future__ import annotations

import argparse
import copy
import csv
import glob
import json
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from drone_sim.domain.config import ScenarioConfig  # noqa: E402
from drone_sim.simulation.simulator import Simulator  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WARMUP_STEPS = 20    # History buffer warmup — LSTM falls back to fixed for first 20 steps
HARD_COLLISION_DIST = 0.4  # d1.radius + d2.radius (0.2 + 0.2)
ADAPTIVE_ALPHA = 0.3  # Standard alpha for adaptive mode override


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ModeResult:
    scenario: str
    mode: str
    min_safety_margin: float = float("inf")
    goal_reaching_rate: float = 0.0
    step_time_ms: float = 0.0
    collision_count: int = 0
    lstm_active_rate: float = float("nan")  # only set for lstm mode
    status: str = "ok"


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

def build_mode_config(cfg_dict: dict[str, Any], mode: str) -> dict[str, Any]:
    """Return a deep copy of cfg_dict modified for the specified mode.

    Parameters
    ----------
    cfg_dict : dict
        The canonical LSTM config dict (already has lstm fields).
    mode : str
        One of "fixed", "adaptive", or "lstm".

    Returns
    -------
    dict
        Modified config dict. Original is never mutated.
    """
    d = copy.deepcopy(cfg_dict)

    if mode == "lstm":
        # Return unchanged — already configured for lstm mode
        return d

    # For fixed and adaptive: disable LSTM model
    d["lstm_model_path"] = None

    for drone in d.get("drones", []):
        if mode == "fixed":
            drone["safety_zone_mode"] = "fixed"
            # Remove alpha if present (no velocity-dependent scaling)
            drone.pop("alpha", None)
        elif mode == "adaptive":
            drone["safety_zone_mode"] = "adaptive"
            drone["alpha"] = ADAPTIVE_ALPHA

    return d


# ---------------------------------------------------------------------------
# Hard collision check
# ---------------------------------------------------------------------------

def count_hard_collisions(sim: Simulator) -> int:
    """Count current pairwise hard collisions (physical body overlap).

    Uses HARD_COLLISION_DIST = d1.radius + d2.radius (0.4m).
    """
    count = 0
    drones = sim.drones
    for i in range(len(drones)):
        for j in range(i + 1, len(drones)):
            dist = float(np.linalg.norm(drones[i].position() - drones[j].position()))
            if dist < drones[i].radius + drones[j].radius:
                count += 1
    return count


# ---------------------------------------------------------------------------
# LSTM active step detection
# ---------------------------------------------------------------------------

def compute_lstm_active_step(sim: Simulator, r_floor_by_id: dict[str, float]) -> bool:
    """Return True if at least one drone's LSTM safety radius exceeds its static floor.

    Calls compute_neighbor_safety_radii() on the sim's LSTM provider. Returns
    False if the provider is absent (non-LSTM mode).

    Parameters
    ----------
    sim : Simulator
        Running simulator instance.
    r_floor_by_id : dict
        Mapping of drone_id -> static safety_zone (floor radius).

    Returns
    -------
    bool
        True if any LSTM-computed radius exceeds the floor for that drone.
    """
    provider = sim._lstm_provider
    if provider is None:
        return False
    neighbor_ids = [d.drone_id for d in sim.drones]
    radii_by_drone = provider.compute_neighbor_safety_radii(neighbor_ids, r_floor_by_id)
    for drone_id, radii in radii_by_drone.items():
        if np.any(radii > r_floor_by_id[drone_id]):
            return True
    return False


# ---------------------------------------------------------------------------
# Per-mode simulation runner
# ---------------------------------------------------------------------------

def run_one_mode(config_path: str, mode: str, steps: int) -> ModeResult:
    """Run a single config under a single mode and return collected metrics.

    Parameters
    ----------
    config_path : str
        Path to the LSTM JSON config file.
    mode : str
        One of "fixed", "adaptive", or "lstm".
    steps : int
        Number of metric-collection steps (after warmup).

    Returns
    -------
    ModeResult
        Populated result object. On exception, status="failed" and metrics
        are left at their default values.
    """
    result = ModeResult(scenario=Path(config_path).stem, mode=mode)

    try:
        # 1. Load config
        with open(config_path) as f:
            raw = json.load(f)

        # 2. Build mode-specific config dict
        cfg_dict = build_mode_config(raw, mode)

        # 3. Validate and build Simulator
        scenario = ScenarioConfig.model_validate(cfg_dict)
        sim = Simulator.from_config(scenario)

        # 4. Warmup — suppress warnings, no metric collection
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(WARMUP_STEPS):
                sim.step()

        # 5. Pre-compute static floor for LSTM active detection
        r_floor_by_id: dict[str, float] = {d.drone_id: d.safety_zone for d in sim.drones}

        # 6. Metric loop
        step_times: list[float] = []
        min_margin = float("inf")
        total_hard_collisions = 0
        active_steps = 0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(steps):
                # Compute per-drone effective radii for safety margin
                positions = [d.position() for d in sim.drones]

                if mode == "adaptive":
                    radii = [d.compute_adaptive_radius(d.velocity()) for d in sim.drones]
                else:
                    # fixed and lstm: use static safety_zone as the observable floor
                    radii = [d.safety_zone for d in sim.drones]

                # Pairwise safety margin: dist - (ri + rj)
                n = len(sim.drones)
                for i in range(n):
                    for j in range(i + 1, n):
                        dist = float(np.linalg.norm(positions[i] - positions[j]))
                        margin = dist - (radii[i] + radii[j])
                        min_margin = min(min_margin, margin)

                # Step timing
                t0 = time.perf_counter()
                sim.step()
                dt_s = time.perf_counter() - t0
                step_times.append(dt_s * 1000.0)  # convert to ms

                # Hard collision count at this step
                total_hard_collisions += count_hard_collisions(sim)

                # LSTM active check (only for lstm mode)
                if mode == "lstm":
                    if compute_lstm_active_step(sim, r_floor_by_id):
                        active_steps += 1

        # 7. Aggregate metrics
        result.min_safety_margin = min_margin
        result.collision_count = total_hard_collisions
        result.step_time_ms = float(np.mean(step_times)) if step_times else 0.0

        # Goal-reaching rate: fraction of drones within 0.5m of their target
        drones_reached = sum(
            1 for d in sim.drones
            if float(np.linalg.norm(d.position() - d.route.target)) <= 0.5
        )
        result.goal_reaching_rate = drones_reached / max(1, len(sim.drones))

        # LSTM active rate
        if mode == "lstm":
            result.lstm_active_rate = active_steps / max(1, steps)
            # Flag that min_safety_margin used the static floor, not actual LSTM radius
            result.status = "lstm_static_floor"

    except Exception as exc:
        result.status = "failed"
        print(f"[FAILED] {Path(config_path).stem} / {mode}: {exc}", file=sys.stderr)

    return result


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

_CSV_FIELDS = [
    "scenario",
    "mode",
    "status",
    "min_safety_margin",
    "goal_reaching_rate",
    "step_time_ms",
    "collision_count",
    "lstm_active_rate",
]


def write_csv(results: list[ModeResult], output_path: str | Path) -> None:
    """Write comparison results to a CSV file.

    Parameters
    ----------
    results : list[ModeResult]
        Collected results from all scenario/mode combinations.
    output_path : str or Path
        Destination CSV path. Parent directories are created if needed.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            not_failed = r.status != "failed"
            writer.writerow({
                "scenario": r.scenario,
                "mode": r.mode,
                "status": r.status,
                "min_safety_margin": f"{r.min_safety_margin:.4f}" if not_failed else "",
                "goal_reaching_rate": f"{r.goal_reaching_rate:.4f}" if not_failed else "",
                "step_time_ms": f"{r.step_time_ms:.4f}" if not_failed else "",
                "collision_count": r.collision_count if not_failed else "",
                "lstm_active_rate": (
                    f"{r.lstm_active_rate:.4f}"
                    if not_failed and not (isinstance(r.lstm_active_rate, float) and
                                           r.lstm_active_rate != r.lstm_active_rate)
                    else ("" if not_failed else "")
                ),
            })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compare fixed / adaptive / lstm safety zone modes across configs/lstm/ scenarios.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=150,
        help="Number of metric-collection steps per scenario/mode (default: 150). "
             "A 20-step warmup is always run before metric collection.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=".planning/comparison.csv",
        help="Output CSV path (default: .planning/comparison.csv)",
    )

    args = parser.parse_args(argv)

    # Discover configs — must be run from project root
    configs = sorted(glob.glob("configs/lstm/*.json"))
    if not configs:
        print(
            "No configs found in configs/lstm/. "
            "Run this script from the project root after creating scenario configs.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        f"Found {len(configs)} config(s): {[Path(c).stem for c in configs]}",
        file=sys.stderr,
    )
    print(
        f"Running 3 modes x {len(configs)} configs = {3 * len(configs)} combinations "
        f"({WARMUP_STEPS} warmup + {args.steps} metric steps each)",
        file=sys.stderr,
    )

    results: list[ModeResult] = []
    for config in configs:
        for mode in ["fixed", "adaptive", "lstm"]:
            print(f"  Running {Path(config).stem} / {mode} ...", file=sys.stderr, end=" ", flush=True)
            r = run_one_mode(config, mode, args.steps)
            results.append(r)
            print(
                f"status={r.status} margin={r.min_safety_margin:.3f} "
                f"goal_rate={r.goal_reaching_rate:.2f} "
                f"collisions={r.collision_count} "
                f"step_ms={r.step_time_ms:.1f}"
                + (f" lstm_active={r.lstm_active_rate:.2f}" if mode == "lstm" else ""),
                file=sys.stderr,
            )

    write_csv(results, args.output)

    n_rows = len(results)
    print(
        f"\nWrote {n_rows} row(s) to {args.output}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
