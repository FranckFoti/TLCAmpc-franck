"""Integration tests for threaded DMPC simulator integration.

Tests for:
- Loading threaded DMPC config via Simulator.from_config
- Running simulation with ThreadedMPCCoordinator
- Visualization method exposure
- Backward compatibility with ADMM DMPC
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pytest

from drone_sim.domain.config import (
    ControllerSpec,
    DroneConfig,
    PhysicsSpec,
    RoomConfig,
    ScenarioConfig,
)
from drone_sim.simulation.distributed.threaded_coordinator import ThreadedMPCCoordinator
from drone_sim.simulation.distributed.distributed_coordinator import DistributedMPCCoordinator
from drone_sim.simulation.simulator import Simulator


class TestThreadedDMPCIntegration:
    """Tests for loading and running threaded DMPC configs."""

    def test_simulator_loads_threaded_config_from_file(self):
        """Test Simulator.from_config loads 2DronesThreadedDMPC.json correctly."""
        with open("configs/threaded/2DronesThreadedDMPC.json") as f:
            cfg_dict = json.load(f)

        cfg = ScenarioConfig(**cfg_dict)
        sim = Simulator.from_config(cfg)

        assert isinstance(sim, Simulator)
        assert isinstance(sim.coordinator, ThreadedMPCCoordinator)
        assert len(sim.drones) == 2

    def test_simulator_threaded_coordinator_params_set(self):
        """Test threaded DMPC coordinator receives correct parameters."""
        cfg = ScenarioConfig(
            dt=0.1,
            physics=PhysicsSpec(type="linear_kinematics"),
            controller=ControllerSpec(type="mpc_agent", params={"horizon": 4}),
            coordinator=ControllerSpec(
                type="dmpc_threaded",
                params={
                    "horizon": 4,
                    "convergence_threshold": 1e-3,
                    "convergence_timeout_sec": 10.0,
                    "gauss_seidel": False,
                },
            ),
            drones=[
                DroneConfig(drone_id="d1", start=[0, 0, 5], target=[5, 0, 5]),
                DroneConfig(drone_id="d2", start=[5, 0, 5], target=[0, 0, 5]),
            ],
            room=RoomConfig(min=[-10, -10, 0], max=[10, 10, 10]),
        )

        sim = Simulator.from_config(cfg)
        coord = sim.coordinator

        assert isinstance(coord, ThreadedMPCCoordinator)
        assert coord.dt == 0.1
        assert coord.horizon == 4
        assert coord.convergence_threshold == 1e-3
        assert coord.convergence_timeout_sec == 10.0
        assert coord.gauss_seidel is False

    @pytest.mark.slow
    def test_threaded_sim_runs_one_step(self):
        """Test simulation step with threaded coordinator runs and drones move."""
        cfg = ScenarioConfig(
            dt=0.1,
            physics=PhysicsSpec(type="linear_kinematics"),
            controller=ControllerSpec(type="mpc_agent", params={"horizon": 4}),
            coordinator=ControllerSpec(
                type="dmpc_threaded",
                params={
                    "horizon": 4,
                    "convergence_threshold": 1e-3,
                    "convergence_timeout_sec": 10.0,
                },
            ),
            drones=[
                DroneConfig(
                    drone_id="d1",
                    start=[-2.0, 0.0, 5.0],
                    target=[2.0, 0.0, 5.0],
                    safety_zone=0.5,
                ),
                DroneConfig(
                    drone_id="d2",
                    start=[2.0, 0.0, 5.0],
                    target=[-2.0, 0.0, 5.0],
                    safety_zone=0.5,
                ),
            ],
            room=RoomConfig(min=[-5, -5, 0], max=[5, 5, 10]),
        )

        sim = Simulator.from_config(cfg)
        initial_positions = [d.position().copy() for d in sim.drones]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            sim.step()

        # Verify drones moved toward targets
        final_positions = [d.position().copy() for d in sim.drones]

        for i, (init, final) in enumerate(zip(initial_positions, final_positions)):
            dist = np.linalg.norm(final - init)
            assert dist > 0.01, f"Drone {i} should have moved"

    @pytest.mark.slow
    def test_threaded_coordinator_exposes_viz_methods(self):
        """Test visualization methods return valid data after a solve."""
        cfg = ScenarioConfig(
            dt=0.1,
            physics=PhysicsSpec(type="linear_kinematics"),
            controller=ControllerSpec(type="mpc_agent", params={"horizon": 4}),
            coordinator=ControllerSpec(
                type="dmpc_threaded",
                params={
                    "horizon": 4,
                    "convergence_threshold": 1e-3,
                    "convergence_timeout_sec": 10.0,
                },
            ),
            drones=[
                DroneConfig(
                    drone_id="d1",
                    start=[-2.0, 0.0, 5.0],
                    target=[2.0, 0.0, 5.0],
                    safety_zone=0.5,
                ),
                DroneConfig(
                    drone_id="d2",
                    start=[2.0, 0.0, 5.0],
                    target=[-2.0, 0.0, 5.0],
                    safety_zone=0.5,
                ),
            ],
            room=RoomConfig(min=[-5, -5, 0], max=[5, 5, 10]),
        )

        sim = Simulator.from_config(cfg)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            sim.step()

        coord = sim.coordinator
        assert isinstance(coord, ThreadedMPCCoordinator)

        # Verify visualization methods exist and return valid data
        neighbor_pairs = coord.get_neighbor_pairs()
        assert isinstance(neighbor_pairs, list)
        for pair in neighbor_pairs:
            assert isinstance(pair, tuple)
            assert len(pair) == 2

        iteration_count = coord.get_last_iteration_count()
        assert isinstance(iteration_count, int)
        assert iteration_count >= 0

        converged = coord.get_last_converged()
        assert isinstance(converged, bool)

    def test_existing_dmpc_config_still_loads(self):
        """Test existing 4DronesDMPC.json still loads with DistributedMPCCoordinator."""
        with open("configs/dmpc/4DronesDMPC.json") as f:
            cfg_dict = json.load(f)

        cfg = ScenarioConfig(**cfg_dict)
        sim = Simulator.from_config(cfg)

        assert isinstance(sim, Simulator)
        assert isinstance(sim.coordinator, DistributedMPCCoordinator)
        assert not isinstance(sim.coordinator, ThreadedMPCCoordinator)
        assert len(sim.drones) == 4

    @pytest.mark.slow
    def test_threaded_zero_axis_drones_warm_start(self):
        """Test threaded coordinator handles collinear drones (x=0, z=0) with warm-start.

        Regression test for the bug where drones with x=0 and z=0 in both start
        and destination failed to converge because:
        1. No cross-timestep warm-start (restarted from central_initial_guess each time)
        2. Static bootstrap trajectories (showed neighbors as stationary)

        This test verifies that drones with exactly x=0, z=0 positions can:
        - Complete multiple timesteps without convergence timeout
        - Make forward progress toward targets
        - Avoid collisions
        """
        cfg = ScenarioConfig(
            dt=0.1,
            physics=PhysicsSpec(type="linear_kinematics"),
            controller=ControllerSpec(type="mpc_agent", params={"horizon": 5}),
            coordinator=ControllerSpec(
                type="dmpc_threaded",
                params={
                    "horizon": 5,
                    "convergence_threshold": 1e-3,
                    "convergence_timeout_sec": 15.0,
                    "max_iterations": 100,
                },
            ),
            drones=[
                DroneConfig(
                    drone_id="d1",
                    start=[0.0, -1.49, 0.0],  # Exactly x=0, z=0
                    target=[0.0, 1.49, 0.0],
                    safety_zone=0.5,
                ),
                DroneConfig(
                    drone_id="d2",
                    start=[0.0, 1.49, 0.0],  # Exactly x=0, z=0
                    target=[0.0, -1.49, 0.0],
                    safety_zone=0.5,
                ),
            ],
            room=RoomConfig(min=[-3, -3, -1], max=[3, 3, 3]),
        )

        sim = Simulator.from_config(cfg)
        initial_positions = [d.position().copy() for d in sim.drones]

        # Run 3 timesteps to verify warm-start works across multiple solves
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for step in range(3):
                sim.step()

                # Verify drones moved toward targets
                current_positions = [d.position().copy() for d in sim.drones]
                for i, (init, current) in enumerate(zip(initial_positions, current_positions)):
                    dist = np.linalg.norm(current - init)
                    assert dist > 0.01, f"Drone {i} should have moved after {step+1} steps"

                # Verify no collisions
                d1_pos, d2_pos = current_positions
                separation = np.linalg.norm(d1_pos - d2_pos)
                min_separation = sim.drones[0].safety_zone + sim.drones[1].safety_zone
                assert separation >= min_separation * 0.95, (
                    f"Collision at step {step+1}: separation={separation:.3f}, "
                    f"min_separation={min_separation:.3f}"
                )
