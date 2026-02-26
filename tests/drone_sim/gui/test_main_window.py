"""Smoke tests for MainWindow — run headless via QT_QPA_PLATFORM=offscreen."""
import os

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
    def test_space_calls_toggle(self, window, monkeypatch):
        calls = []
        monkeypatch.setattr(window, "_toggle_play_pause", lambda: calls.append(1))
        from PySide6.QtGui import QKeyEvent
        event = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier)
        window.keyPressEvent(event)
        assert len(calls) == 1

    def test_other_key_does_not_toggle(self, window, monkeypatch):
        calls = []
        monkeypatch.setattr(window, "_toggle_play_pause", lambda: calls.append(1))
        from PySide6.QtGui import QKeyEvent
        event = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_A, Qt.KeyboardModifier.NoModifier)
        window.keyPressEvent(event)
        assert len(calls) == 0


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
