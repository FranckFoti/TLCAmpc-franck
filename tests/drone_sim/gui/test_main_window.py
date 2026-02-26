"""Smoke tests for MainWindow — run headless via QT_QPA_PLATFORM=offscreen."""
import os

import numpy as np

os.environ["QT_API"] = "pyside6"
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import sys

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from drone_sim.gui.main_window import MainWindow


@pytest.fixture(scope="module")
def app():
    """Single QApplication for all tests in this module."""
    existing = QApplication.instance()
    if existing:
        yield existing
    else:
        a = QApplication(sys.argv)
        yield a
        # Do NOT call a.quit() — other tests may reuse the app


@pytest.fixture
def window(app):
    """Fresh MainWindow for each test."""
    w = MainWindow()
    yield w
    w.close()


class TestMainWindowScaffold:
    def test_window_creates_without_crash(self, window):
        assert window is not None

    def test_canvas_attribute_exists(self, window):
        assert hasattr(window, "_canvas")

    def test_ax_is_3d(self, window):
        from mpl_toolkits.mplot3d import Axes3D
        assert isinstance(window._ax, Axes3D)

    def test_playing_starts_false(self, window):
        assert window._playing is False

    def test_interval_ms_default(self, window):
        assert window._interval_ms == 100

    def test_traces_starts_empty(self, window):
        assert window._traces == {}

    def test_step_label_initial_text(self, window):
        assert "Step: 0" in window._step_label.text()

    def test_speed_slider_range(self, window):
        assert window._speed_slider.minimum() == 10
        assert window._speed_slider.maximum() == 500

    def test_speed_slider_default_value(self, window):
        assert window._speed_slider.value() == 100


class TestSpeedSlot:
    def test_speed_changed_updates_interval_ms(self, window):
        window._speed_slider.setValue(250)
        assert window._interval_ms == 250

    def test_speed_changed_updates_label(self, window):
        window._speed_slider.setValue(300)
        assert "300" in window._speed_label.text()


class TestKeyboardShortcut:
    def test_spacebar_shortcut_is_registered(self, window):
        # QShortcut fires regardless of child focus; verify one is registered on the window.
        from PySide6.QtGui import QKeySequence, QShortcut
        shortcuts = window.findChildren(QShortcut)
        keys = [s.key() for s in shortcuts]
        assert QKeySequence(Qt.Key.Key_Space) in keys, "Spacebar QShortcut not found on MainWindow"

    def test_toggle_play_pause_starts_play_when_paused(self, window, monkeypatch):
        import drone_sim.gui.main_window as m
        fired = []
        monkeypatch.setattr(m.QTimer, "singleShot", staticmethod(lambda ms, fn: fired.append((ms, fn))))
        window._playing = False
        window._toggle_play_pause()
        assert window._playing is True
        assert len(fired) == 1

    def test_toggle_play_pause_pauses_when_playing(self, window):
        window._playing = True
        window._toggle_play_pause()
        assert window._playing is False


class TestPauseStopsPlayback:
    def test_pause_sets_playing_false(self, window):
        window._playing = True
        window._on_pause()
        assert window._playing is False

    def test_play_sets_playing_true(self, window, monkeypatch):
        # Monkeypatch QTimer.singleShot to avoid actual timer firing
        import drone_sim.gui.main_window as m
        fired = []
        monkeypatch.setattr(m.QTimer, "singleShot", staticmethod(lambda ms, fn: fired.append((ms, fn))))
        window._playing = False
        window._on_play()
        assert window._playing is True
        assert len(fired) == 1  # singleShot called once to start the loop

    def test_play_while_playing_is_noop(self, window, monkeypatch):
        import drone_sim.gui.main_window as m
        fired = []
        monkeypatch.setattr(m.QTimer, "singleShot", staticmethod(lambda ms, fn: fired.append((ms, fn))))
        window._playing = True
        window._on_play()
        assert len(fired) == 0  # already playing — no additional singleShot


class TestRunToCompletionFlag:
    def test_run_to_completion_starts_false(self, window):
        assert window._run_to_completion is False

    def test_on_run_to_completion_sets_flag(self, window, monkeypatch):
        import drone_sim.gui.main_window as m
        monkeypatch.setattr(m.QTimer, "singleShot", staticmethod(lambda ms, fn: None))
        window._on_run_to_completion()
        assert window._run_to_completion is True
        assert window._playing is True

    def test_on_pause_clears_run_to_completion_flag(self, window):
        window._run_to_completion = True
        window._playing = True
        window._on_pause()
        assert window._run_to_completion is False
        assert window._playing is False

    def test_max_run_steps_is_5000(self):
        from drone_sim.gui.main_window import _MAX_RUN_STEPS
        assert _MAX_RUN_STEPS == 5000


class TestCollisionIndicator:
    def test_collision_label_exists(self, window):
        assert hasattr(window, "_collision_label")

    def test_collision_label_default_text(self, window):
        assert window._collision_label.text() == "No Collisions"

    def test_update_collision_indicator_no_collisions(self, window):
        from drone_sim.gui.backend import StepResult, DroneState
        result = StepResult(
            drones=[],
            safety_radii=[],
            last_collisions=[],
            infeasible=False,
            infeasible_reason=None,
            step_count=1,
            t=0.1,
            all_reached=False,
            admm_iteration_count=None,
        )
        window._update_collision_indicator(result)
        assert window._collision_label.text() == "No Collisions"

    def test_update_collision_indicator_with_collisions(self, window):
        from drone_sim.gui.backend import StepResult
        result = StepResult(
            drones=[],
            safety_radii=[],
            last_collisions=[{"drone_a": "d1", "drone_b": "d2"}],
            infeasible=False,
            infeasible_reason=None,
            step_count=1,
            t=0.1,
            all_reached=False,
            admm_iteration_count=None,
        )
        window._update_collision_indicator(result)
        assert window._collision_label.text() == "COLLISION"


class TestAdmmIndicator:
    def test_admm_label_exists(self, window):
        assert hasattr(window, "_admm_label")

    def test_admm_label_hidden_by_default(self, window):
        # Use isHidden() — isVisible() depends on parent being shown (offscreen headless)
        assert window._admm_label.isHidden()

    def test_update_admm_indicator_none_keeps_hidden(self, window):
        from drone_sim.gui.backend import StepResult
        result = StepResult(
            drones=[],
            safety_radii=[],
            last_collisions=[],
            infeasible=False,
            infeasible_reason=None,
            step_count=1,
            t=0.1,
            all_reached=False,
            admm_iteration_count=None,
        )
        window._update_admm_indicator(result)
        assert window._admm_label.isHidden()

    def test_update_admm_indicator_int_shows_label(self, window):
        from drone_sim.gui.backend import StepResult
        result = StepResult(
            drones=[],
            safety_radii=[],
            last_collisions=[],
            infeasible=False,
            infeasible_reason=None,
            step_count=1,
            t=0.1,
            all_reached=False,
            admm_iteration_count=7,
        )
        window._update_admm_indicator(result)
        # Use not isHidden() — isVisible() depends on parent being shown (offscreen headless)
        assert not window._admm_label.isHidden()
        assert "7" in window._admm_label.text()


class TestInfoPanel:
    def test_info_path_label_exists(self, window):
        assert hasattr(window, "_info_path_label")

    def test_info_coord_label_exists(self, window):
        assert hasattr(window, "_info_coord_label")

    def test_info_dt_label_exists(self, window):
        assert hasattr(window, "_info_dt_label")

    def test_info_drones_label_exists(self, window):
        assert hasattr(window, "_info_drones_label")

    def test_info_obstacles_label_exists(self, window):
        assert hasattr(window, "_info_obstacles_label")

    def test_on_config_loaded_updates_labels(self, window):
        from drone_sim.gui.backend import SimState
        state = SimState(
            drone_count=3,
            obstacle_count=2,
            obstacles=[],
            coordinator_type="DistributedMPCCoordinator",
            dt=0.2,
            step_count=0,
            room_min=np.asarray([-10.0, -10.0, 0.0]),
            room_max=np.asarray([10.0, 10.0, 10.0]),
            config_path="/some/path/MyScenario.json",
        )
        window._on_config_loaded(state)
        assert window._info_path_label.text() == "MyScenario.json"
        assert "DistributedMPCCoordinator" in window._info_coord_label.text()
        assert "3" in window._info_drones_label.text()
        assert "2" in window._info_obstacles_label.text()

    def test_on_config_loaded_sets_tooltip_to_full_path(self, window):
        from drone_sim.gui.backend import SimState
        state = SimState(
            drone_count=1,
            obstacle_count=0,
            obstacles=[],
            coordinator_type="CentralMPCGlobalCoordinator",
            dt=0.1,
            step_count=0,
            room_min=np.asarray([-5.0, -5.0, 0.0]),
            room_max=np.asarray([5.0, 5.0, 5.0]),
            config_path="/full/absolute/path/config.json",
        )
        window._on_config_loaded(state)
        assert window._info_path_label.toolTip() == "/full/absolute/path/config.json"
