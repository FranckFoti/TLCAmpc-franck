"""Pure Python unit tests for DirectBackend — no Qt dependency."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from drone_sim.gui.backend import DroneState, SimState, StepResult
from drone_sim.gui.direct_backend import DirectBackend

MINIMAL_CONFIG = {
    "dt": 0.1,
    "physics": {"type": "linear_kinematics", "v_max": 2.0, "u_min": -1.0, "u_max": 1.0},
    "controller": {"type": "mpc_agent"},
    "coordinator": {"type": "mpc_central"},
    "drones": [
        {
            "drone_id": "d1",
            "start": [0.0, 0.0, 5.0],
            "target": [5.0, 5.0, 5.0],
        }
    ],
    "room": {"min": [-10.0, -10.0, 0.0], "max": [10.0, 10.0, 10.0]},
}


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    p = tmp_path / "test_config.json"
    p.write_text(json.dumps(MINIMAL_CONFIG), encoding="utf-8")
    return p


@pytest.fixture
def loaded_backend(config_file: Path) -> DirectBackend:
    backend = DirectBackend()
    backend.load_config(config_file)
    return backend


# --- load_config ---

def test_load_config_returns_sim_state(config_file: Path) -> None:
    backend = DirectBackend()
    state = backend.load_config(config_file)
    assert isinstance(state, SimState)


def test_load_config_drone_count(config_file: Path) -> None:
    backend = DirectBackend()
    state = backend.load_config(config_file)
    assert state.drone_count == 1


def test_load_config_step_count_zero(config_file: Path) -> None:
    backend = DirectBackend()
    state = backend.load_config(config_file)
    assert state.step_count == 0


def test_load_config_obstacle_count(config_file: Path) -> None:
    backend = DirectBackend()
    state = backend.load_config(config_file)
    assert state.obstacle_count == 0


def test_load_config_dt(config_file: Path) -> None:
    backend = DirectBackend()
    state = backend.load_config(config_file)
    assert state.dt == pytest.approx(0.1)


def test_load_config_coordinator_type_is_string(config_file: Path) -> None:
    backend = DirectBackend()
    state = backend.load_config(config_file)
    assert isinstance(state.coordinator_type, str)
    assert len(state.coordinator_type) > 0


def test_load_config_room_bounds_are_lists(config_file: Path) -> None:
    backend = DirectBackend()
    state = backend.load_config(config_file)
    assert isinstance(state.room_min, list)
    assert isinstance(state.room_max, list)
    assert len(state.room_min) == 3
    assert len(state.room_max) == 3


# --- step ---

def test_step_returns_step_result(loaded_backend: DirectBackend) -> None:
    result = loaded_backend.step()
    assert isinstance(result, StepResult)


def test_step_increments_step_count(loaded_backend: DirectBackend) -> None:
    result = loaded_backend.step()
    assert result.step_count == 1


def test_step_drone_count(loaded_backend: DirectBackend) -> None:
    result = loaded_backend.step()
    assert len(result.drones) == 1


def test_step_drone_state_types(loaded_backend: DirectBackend) -> None:
    result = loaded_backend.step()
    d = result.drones[0]
    assert isinstance(d, DroneState)
    assert isinstance(d.drone_id, str)
    assert isinstance(d.position, list)
    assert isinstance(d.velocity, list)
    assert len(d.position) == 3
    assert len(d.velocity) == 3


def test_step_position_values_are_floats(loaded_backend: DirectBackend) -> None:
    result = loaded_backend.step()
    d = result.drones[0]
    assert all(isinstance(v, float) for v in d.position), "position must be list[float], not numpy"
    assert all(isinstance(v, float) for v in d.velocity), "velocity must be list[float], not numpy"


def test_step_safety_radii_length(loaded_backend: DirectBackend) -> None:
    result = loaded_backend.step()
    assert len(result.safety_radii) == 1
    assert isinstance(result.safety_radii[0], float)


def test_step_infeasible_is_bool(loaded_backend: DirectBackend) -> None:
    result = loaded_backend.step()
    assert isinstance(result.infeasible, bool)


def test_step_last_collisions_is_list(loaded_backend: DirectBackend) -> None:
    result = loaded_backend.step()
    assert isinstance(result.last_collisions, list)


def test_step_t_is_float(loaded_backend: DirectBackend) -> None:
    result = loaded_backend.step()
    assert isinstance(result.t, float)


# --- get_state ---

def test_get_state_returns_sim_state(loaded_backend: DirectBackend) -> None:
    state = loaded_backend.get_state()
    assert isinstance(state, SimState)


def test_get_state_does_not_advance_step_count(loaded_backend: DirectBackend) -> None:
    loaded_backend.get_state()
    loaded_backend.get_state()
    state = loaded_backend.get_state()
    assert state.step_count == 0


# --- reset ---

def test_reset_after_steps_returns_step_count_to_zero(loaded_backend: DirectBackend) -> None:
    loaded_backend.step()
    loaded_backend.step()
    loaded_backend.reset()
    state = loaded_backend.get_state()
    assert state.step_count == 0


def test_reset_does_not_require_file_path(loaded_backend: DirectBackend, tmp_path: Path) -> None:
    """reset() must use cached config — original file can be deleted."""
    loaded_backend.step()
    # Delete the original file to prove reset does not re-read disk
    for f in tmp_path.iterdir():
        f.unlink()
    loaded_backend.reset()  # must not raise
    state = loaded_backend.get_state()
    assert state.step_count == 0


# --- all_reached ---

def test_step_all_reached_is_bool(loaded_backend: DirectBackend) -> None:
    result = loaded_backend.step()
    assert isinstance(result.all_reached, bool)


def test_step_all_reached_false_on_first_step(loaded_backend: DirectBackend) -> None:
    """Drone starts far from destination — should not be reached on step 1."""
    result = loaded_backend.step()
    assert result.all_reached is False


# --- admm_iteration_count ---

def test_step_admm_iteration_count_none_for_central_coordinator(loaded_backend: DirectBackend) -> None:
    """MINIMAL_CONFIG uses mpc_central coordinator — not ADMM — so admm_iteration_count must be None."""
    result = loaded_backend.step()
    assert result.admm_iteration_count is None


def test_step_admm_iteration_count_type_is_int_or_none(loaded_backend: DirectBackend) -> None:
    result = loaded_backend.step()
    assert result.admm_iteration_count is None or isinstance(result.admm_iteration_count, int)


# --- SimState.config_path ---

def test_load_config_sets_config_path(config_file: Path) -> None:
    backend = DirectBackend()
    state = backend.load_config(config_file)
    assert state.config_path is not None
    assert isinstance(state.config_path, str)


def test_load_config_path_ends_with_filename(config_file: Path) -> None:
    backend = DirectBackend()
    state = backend.load_config(config_file)
    assert state.config_path.endswith("test_config.json")


def test_config_path_none_before_load() -> None:
    """get_state() before load_config() should raise RuntimeError, but the field default is None."""
    # We only test the dataclass default — a fresh SimState() would have config_path=None.
    from drone_sim.gui.backend import SimState
    s = SimState(
        drone_count=0,
        obstacle_count=0,
        coordinator_type="none",
        dt=0.1,
        step_count=0,
        room_min=[0.0, 0.0, 0.0],
        room_max=[1.0, 1.0, 1.0],
    )
    assert s.config_path is None
