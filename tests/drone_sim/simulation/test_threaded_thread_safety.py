"""Thread safety stress tests for threaded DMPC coordinator.

Tests for:
- Repeated solves without NaN, deadlock, or exceptions
- Back-to-back solve batches (stateless coordinator validation)
- No thread leaks after multiple solve calls
- Gauss-Seidel mode thread safety
- Concurrent simulator instances don't interfere

All tests marked @pytest.mark.slow for CI separation.

Usage with pytest-repeat for thorough stress testing:
    pytest --count=100 -x test_threaded_thread_safety.py
"""

from __future__ import annotations

import threading
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
from drone_sim.simulation.simulator import Simulator


@pytest.mark.slow
class TestThreadSafetyStress:
    """Thread safety stress tests for threaded DMPC coordinator."""

    def test_thread_safety_repeated_solves(self):
        """Stress test repeated solves without NaN/deadlock/exceptions.

        Pattern: 3-drone scenario with 20 steps. Check for NaN after each step.
        Designed to be run with pytest-repeat --count=100 for stress testing.
        """
        cfg = ScenarioConfig(
            dt=0.1,
            physics=PhysicsSpec(type="linear_kinematics"),
            controller=ControllerSpec(type="mpc_agent", params={"horizon": 4}),
            coordinator=ControllerSpec(
                type="dmpc_threaded",
                params={
                    "horizon": 4,
                    "convergence_timeout_sec": 10.0,
                    "convergence_threshold": 1e-3,
                },
            ),
            drones=[
                DroneConfig(drone_id="d1", start=[0.0, -1.0, 0.0], target=[0.0, 1.0, 0.0]),
                DroneConfig(drone_id="d2", start=[0.0, 1.0, 0.0], target=[0.0, -1.0, 0.0]),
                DroneConfig(drone_id="d3", start=[1.0, 0.0, 0.0], target=[-1.0, 0.0, 0.0]),
            ],
            room=RoomConfig(min=[-2.0, -2.0, -2.0], max=[2.0, 2.0, 2.0]),
        )

        sim = Simulator.from_config(cfg)

        # Run 20 steps — should not deadlock, raise exceptions, or produce NaN
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for step in range(20):
                sim.step()

                # Check for NaN in drone states
                for drone in sim.drones:
                    assert not np.any(np.isnan(drone.x)), (
                        f"Step {step}: NaN in drone {drone.drone_id} state"
                    )

    def test_thread_safety_back_to_back_solves(self):
        """Test back-to-back solve batches validate stateless coordinator.

        Pattern: Run 3 separate batches of 10 steps each. After each batch,
        verify step count incremented correctly and no NaN in states.
        Tests that stateless coordinator handles repeated solve_controls() calls
        without state leak.
        """
        cfg = ScenarioConfig(
            dt=0.1,
            physics=PhysicsSpec(type="linear_kinematics"),
            controller=ControllerSpec(type="mpc_agent", params={"horizon": 4}),
            coordinator=ControllerSpec(
                type="dmpc_threaded",
                params={
                    "horizon": 4,
                    "convergence_timeout_sec": 10.0,
                    "convergence_threshold": 1e-3,
                },
            ),
            drones=[
                DroneConfig(drone_id="d1", start=[0.0, -1.0, 0.0], target=[0.0, 1.0, 0.0]),
                DroneConfig(drone_id="d2", start=[0.0, 1.0, 0.0], target=[0.0, -1.0, 0.0]),
            ],
            room=RoomConfig(min=[-2.0, -2.0, -2.0], max=[2.0, 2.0, 2.0]),
        )

        sim = Simulator.from_config(cfg)

        # Run 3 batches of 10 steps each
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for batch in range(3):
                for _ in range(10):
                    sim.step()

                # Verify step count after each batch
                expected_steps = (batch + 1) * 10
                assert sim.step_count == expected_steps, (
                    f"Batch {batch}: expected {expected_steps} steps, got {sim.step_count}"
                )

                # Verify no NaN in states
                for drone in sim.drones:
                    assert not np.any(np.isnan(drone.x)), (
                        f"Batch {batch}: NaN in drone {drone.drone_id} state"
                    )

    def test_no_thread_leak_after_solve(self):
        """Validate no thread leaks after multiple solve_controls calls.

        Pattern: Record active thread count before/after 5 steps. Assert count
        returns to within 1 of pre-step count (tolerance for GC timing).
        Validates ThreadLifecycleManager shutdown works correctly.
        """
        cfg = ScenarioConfig(
            dt=0.1,
            physics=PhysicsSpec(type="linear_kinematics"),
            controller=ControllerSpec(type="mpc_agent", params={"horizon": 4}),
            coordinator=ControllerSpec(
                type="dmpc_threaded",
                params={
                    "horizon": 4,
                    "convergence_timeout_sec": 5.0,
                    "convergence_threshold": 1e-3,
                },
            ),
            drones=[
                DroneConfig(drone_id="d1", start=[0.0, -1.0, 0.0], target=[0.0, 1.0, 0.0]),
                DroneConfig(drone_id="d2", start=[0.0, 1.0, 0.0], target=[0.0, -1.0, 0.0]),
            ],
            room=RoomConfig(min=[-2.0, -2.0, -2.0], max=[2.0, 2.0, 2.0]),
        )

        sim = Simulator.from_config(cfg)

        # Record baseline thread count
        baseline_count = threading.active_count()

        # Run 5 steps
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for _ in range(5):
                sim.step()

        # Record thread count after steps
        final_count = threading.active_count()

        # Assert thread count returned to baseline (within 1 for GC timing)
        assert abs(final_count - baseline_count) <= 1, (
            f"Thread leak: baseline={baseline_count}, final={final_count}"
        )

    def test_gauss_seidel_mode_thread_safe(self):
        """Validate Gauss-Seidel staggered spawn mode is thread-safe.

        Pattern: 3-drone scenario with gauss_seidel=True. Run 15 steps.
        Assert no NaN, no exceptions. Verifies staggered spawn doesn't introduce
        race conditions.
        """
        cfg = ScenarioConfig(
            dt=0.1,
            physics=PhysicsSpec(type="linear_kinematics"),
            controller=ControllerSpec(type="mpc_agent", params={"horizon": 4}),
            coordinator=ControllerSpec(
                type="dmpc_threaded",
                params={
                    "horizon": 4,
                    "convergence_timeout_sec": 10.0,
                    "convergence_threshold": 1e-3,
                    "gauss_seidel": True,  # Enable staggered spawn
                },
            ),
            drones=[
                DroneConfig(drone_id="d1", start=[0.0, -1.0, 0.0], target=[0.0, 1.0, 0.0]),
                DroneConfig(drone_id="d2", start=[0.0, 1.0, 0.0], target=[0.0, -1.0, 0.0]),
                DroneConfig(drone_id="d3", start=[1.0, 0.0, 0.0], target=[-1.0, 0.0, 0.0]),
            ],
            room=RoomConfig(min=[-2.0, -2.0, -2.0], max=[2.0, 2.0, 2.0]),
        )

        sim = Simulator.from_config(cfg)

        # Run 15 steps with Gauss-Seidel mode
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for step in range(15):
                sim.step()

                # Check for NaN in drone states
                for drone in sim.drones:
                    assert not np.any(np.isnan(drone.x)), (
                        f"Step {step}: NaN in drone {drone.drone_id} state"
                    )

    def test_concurrent_simulator_instances(self):
        """Validate concurrent simulator instances don't interfere.

        Pattern: Create TWO separate Simulator instances with dmpc_threaded
        coordinator (2 drones each). Run them interleaved: sim1.step(),
        sim2.step() for 10 iterations. Assert both complete without error.
        Validates that separate coordinator instances don't interfere.
        """
        cfg1 = ScenarioConfig(
            dt=0.1,
            physics=PhysicsSpec(type="linear_kinematics"),
            controller=ControllerSpec(type="mpc_agent", params={"horizon": 4}),
            coordinator=ControllerSpec(
                type="dmpc_threaded",
                params={
                    "horizon": 4,
                    "convergence_timeout_sec": 5.0,
                    "convergence_threshold": 1e-3,
                },
            ),
            drones=[
                DroneConfig(drone_id="d1", start=[0.0, -1.0, 0.0], target=[0.0, 1.0, 0.0]),
                DroneConfig(drone_id="d2", start=[0.0, 1.0, 0.0], target=[0.0, -1.0, 0.0]),
            ],
            room=RoomConfig(min=[-2.0, -2.0, -2.0], max=[2.0, 2.0, 2.0]),
        )

        cfg2 = ScenarioConfig(
            dt=0.1,
            physics=PhysicsSpec(type="linear_kinematics"),
            controller=ControllerSpec(type="mpc_agent", params={"horizon": 4}),
            coordinator=ControllerSpec(
                type="dmpc_threaded",
                params={
                    "horizon": 4,
                    "convergence_timeout_sec": 5.0,
                    "convergence_threshold": 1e-3,
                },
            ),
            drones=[
                DroneConfig(drone_id="d3", start=[5.0, -1.0, 0.0], target=[5.0, 1.0, 0.0]),
                DroneConfig(drone_id="d4", start=[5.0, 1.0, 0.0], target=[5.0, -1.0, 0.0]),
            ],
            room=RoomConfig(min=[-10.0, -10.0, -10.0], max=[10.0, 10.0, 10.0]),
        )

        sim1 = Simulator.from_config(cfg1)
        sim2 = Simulator.from_config(cfg2)

        # Run both simulators interleaved
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for _ in range(10):
                sim1.step()
                sim2.step()

        # Verify both completed successfully
        assert sim1.step_count == 10, f"Sim1 step count: {sim1.step_count}"
        assert sim2.step_count == 10, f"Sim2 step count: {sim2.step_count}"

        # Verify no NaN in either simulator
        for drone in sim1.drones:
            assert not np.any(np.isnan(drone.x)), (
                f"Sim1 drone {drone.drone_id} has NaN state"
            )
        for drone in sim2.drones:
            assert not np.any(np.isnan(drone.x)), (
                f"Sim2 drone {drone.drone_id} has NaN state"
            )
