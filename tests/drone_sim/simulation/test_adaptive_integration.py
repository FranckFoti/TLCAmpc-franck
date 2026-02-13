"""Integration tests for adaptive safety zone behavior.

Tests run real simulations (no mocks) with short step counts to verify:
- No collisions with adaptive safety zones
- Adaptive radius grows with speed
- Deceleration near conflict points
- Fixed vs adaptive parity (both collision-free)
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from drone_sim.domain.config import ScenarioConfig
from drone_sim.simulation.simulator import Simulator


def _make_config(
    adaptive: bool,
    num_drones: int = 4,
    coordinator: str = "mpc_central",
    alpha: float = 0.4,
    horizon: int = 4,
    lambda_vel: float = 0.5,
) -> dict:
    """Build a config dict for head-on drone scenarios.

    :param adaptive: Whether to use adaptive safety zones (alpha) and adaptive controller.
    :param num_drones: 2 or 4 drones.
    :param coordinator: Coordinator type string.
    :param alpha: Adaptive alpha parameter (only used when adaptive=True).
    :param horizon: MPC horizon.
    :param lambda_vel: Velocity penalty weight for adaptive controller.
    :return: Config dict suitable for ScenarioConfig.model_validate().
    """
    controller_type = "mpc_agent_adaptive" if adaptive else "mpc_agent"
    controller_params = {"horizon": horizon, "q_pos": [3.0, 3.0, 3.0], "q_vel": [1.0, 1.0, 1.0], "r_u": [0.5, 0.5, 0.5]}
    if adaptive:
        controller_params["lambda_vel"] = lambda_vel

    drone_alpha = alpha if adaptive else None

    # X-axis head-on pair
    drones = [
        {
            "drone_id": "drone-1",
            "physics": "standard",
            "start": [-2.0, 0.0, 0.0],
            "target": [2.0, 0.0, 0.0],
            "radius": 0.2,
            "safety_zone": 1.0,
            "cons_stop": 0.0,
            "alpha": drone_alpha,
        },
        {
            "drone_id": "drone-2",
            "physics": "standard",
            "start": [2.0, 0.0, 0.0],
            "target": [-2.0, 0.0, 0.0],
            "radius": 0.2,
            "safety_zone": 1.0,
            "cons_stop": 0.0,
            "alpha": drone_alpha,
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
                "safety_zone": 1.0,
                "cons_stop": 0.0,
                "alpha": drone_alpha,
            },
            {
                "drone_id": "drone-4",
                "physics": "standard",
                "start": [0.0, 2.0, 0.0],
                "target": [0.0, -2.0, 0.0],
                "radius": 0.2,
                "safety_zone": 1.0,
                "cons_stop": 0.0,
                "alpha": drone_alpha,
            },
        ])

    coordinator_params = {"horizon": horizon}
    if coordinator == "mpc_central":
        coordinator_params["room_wall_tolerance"] = 0.5

    return {
        "dt": 0.1,
        "room": {"min": [-3.5, -3.5, -3.5], "max": [3.5, 3.5, 3.5]},
        "physics": [
            {
                "id": "standard",
                "type": "linear_kinematics",
                "params": {
                    "v_max": 5.0,
                    "u_min": [-3.0, -3.0, -3.0],
                    "u_max": [3.0, 3.0, 3.0],
                },
            }
        ],
        "controller": {"type": controller_type, "params": controller_params},
        "coordinator": {"type": coordinator, "params": coordinator_params},
        "drones": drones,
    }


@pytest.mark.slow
class TestAdaptiveIntegration:
    """Integration tests for adaptive safety zone behavior via real simulations."""

    def test_adaptive_no_collisions_central(self):
        """Run 4-drone adaptive scenario with central coordinator for 100 steps.

        Assert no safety-zone collisions occur at any step.
        """
        config = _make_config(adaptive=True, num_drones=4, coordinator="mpc_central", alpha=0.4, horizon=4)
        cfg = ScenarioConfig.model_validate(config)
        sim = Simulator.from_config(cfg)

        total_collisions = 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for _ in range(100):
                sim.step()
                drone_collisions = [c for c in sim.last_collisions if c["kind"] == "drone_drone"]
                total_collisions += len(drone_collisions)

        assert total_collisions == 0, (
            f"Expected no drone-drone collisions, got {total_collisions}"
        )

    def test_adaptive_radius_grows_with_speed(self):
        """Run 2-drone adaptive scenario for 20 steps with high alpha.

        After drones accelerate, verify adaptive radius exceeds safety_zone floor.
        Uses alpha=2.0 and wider separation to allow enough acceleration room
        for the formula to exceed the safety_zone floor.
        """
        config = _make_config(adaptive=True, num_drones=2, coordinator="mpc_central",
                              alpha=2.0, horizon=4)
        # Widen the separation so drones have room to accelerate
        config["drones"][0]["start"] = [-4.0, 0.0, 0.0]
        config["drones"][0]["target"] = [4.0, 0.0, 0.0]
        config["drones"][1]["start"] = [4.0, 0.0, 0.0]
        config["drones"][1]["target"] = [-4.0, 0.0, 0.0]
        config["room"] = {"min": [-6.0, -6.0, -6.0], "max": [6.0, 6.0, 6.0]}

        cfg = ScenarioConfig.model_validate(config)
        sim = Simulator.from_config(cfg)

        # Track whether adaptive radius ever exceeds safety_zone floor
        any_above_floor = False
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for _ in range(20):
                sim.step()
                for d in sim.drones:
                    r_adaptive = d.compute_adaptive_radius(d.velocity())
                    if r_adaptive > d.safety_zone:
                        any_above_floor = True

        assert any_above_floor, (
            "Expected at least one drone's adaptive radius to exceed safety_zone "
            "during 20 steps of acceleration"
        )

    def test_adaptive_drones_decelerate_near_conflict(self):
        """Run 2-drone head-on adaptive scenario for 50 steps.

        Record speed at step 5 (cruising) and around step 25 (near conflict).
        Assert speed near conflict < speed while cruising, verifying the
        lambda_vel penalty encourages deceleration.
        """
        config = _make_config(adaptive=True, num_drones=2, coordinator="mpc_central",
                              alpha=0.5, horizon=4, lambda_vel=0.5)
        cfg = ScenarioConfig.model_validate(config)
        sim = Simulator.from_config(cfg)

        speed_at_5 = None
        speed_near_conflict = None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for step in range(50):
                sim.step()

                if step == 5:
                    # Record max speed at step 5 (cruising toward each other)
                    speed_at_5 = max(
                        float(np.linalg.norm(d.velocity())) for d in sim.drones
                    )
                elif step == 25:
                    # Record max speed around step 25 (near conflict point)
                    speed_near_conflict = max(
                        float(np.linalg.norm(d.velocity())) for d in sim.drones
                    )

        assert speed_at_5 is not None and speed_near_conflict is not None
        assert speed_near_conflict < speed_at_5, (
            f"Expected deceleration near conflict: speed@5={speed_at_5:.3f} "
            f"> speed@25={speed_near_conflict:.3f}"
        )

    def test_fixed_vs_adaptive_both_collision_free(self):
        """Run same 4-drone scenario with and without adaptive.

        Both should complete 100 steps with no drone-drone collisions.
        """
        config_adaptive = _make_config(adaptive=True, num_drones=4, coordinator="mpc_central", horizon=4)
        config_fixed = _make_config(adaptive=False, num_drones=4, coordinator="mpc_central", horizon=4)

        cfg_a = ScenarioConfig.model_validate(config_adaptive)
        cfg_f = ScenarioConfig.model_validate(config_fixed)

        sim_adaptive = Simulator.from_config(cfg_a)
        sim_fixed = Simulator.from_config(cfg_f)

        adaptive_collisions = 0
        fixed_collisions = 0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for _ in range(100):
                sim_adaptive.step()
                sim_fixed.step()

                adaptive_collisions += len([c for c in sim_adaptive.last_collisions if c["kind"] == "drone_drone"])
                fixed_collisions += len([c for c in sim_fixed.last_collisions if c["kind"] == "drone_drone"])

        assert adaptive_collisions == 0, (f"Adaptive mode had {adaptive_collisions} collisions")
        assert fixed_collisions == 0, (f"Fixed mode had {fixed_collisions} collisions")
