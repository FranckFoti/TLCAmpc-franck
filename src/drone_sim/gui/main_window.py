from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPushButton, QSizePolicy, QSlider, QVBoxLayout, QWidget)

from drone_sim.gui.backend import StepResult, SimState
from drone_sim.gui.direct_backend import DirectBackend
from drone_sim.api.utils.render_helper import draw_room_wireframe, draw_sphere_wireframe, draw_trace, draw_obstacles

_MAX_RUN_STEPS = 5000  # run-to-completion cap (matches paper2_tools/scenarios.py default)

def _draw_ghost_max_sphere(ax: object, is_adaptive: bool, position, safety_zone:float, max_radius:float, safety_color):
   if is_adaptive:
      pos = np.asarray(position, dtype=float).reshape(3)

      # only draw if meaningfully larger
      if max_radius > safety_zone + 1e-4:
         draw_sphere_wireframe(ax, pos, max_radius, color=safety_color, alpha=0.12, lw=0.4)


class MainWindow(QMainWindow):
    """Main application window: 3D canvas + playback controls."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Drone Simulation")

        self._backend = DirectBackend()
        self._playing: bool = False
        self._interval_ms: int = 100
        self._traces: dict[str, list[list[float]]] = {}
        self._zoom: float = 1.0  # <1 zoomed in, >1 zoomed out; applied to room limits
        self._run_to_completion: bool = False

        # ---- Canvas ----
        fig = Figure()
        fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
        self._canvas = FigureCanvasQTAgg(fig)
        self._canvas.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self._canvas.setMinimumSize(400, 300)
        self._ax = fig.add_subplot(111, projection="3d")

        # ---- Controls ----
        self._btn_open = QPushButton("Open")
        self._btn_play = QPushButton("Play")
        self._btn_pause = QPushButton("Pause")
        self._btn_reset = QPushButton("Reset")
        self._btn_run_to_end = QPushButton("Run to Completion")

        # ---- Status widgets ----
        self._collision_label = QLabel("No Collisions")
        self._collision_label.setStyleSheet("color: green;")

        self._admm_label = QLabel("")
        self._admm_label.setVisible(False)

        # ---- Scenario info panel ----
        info_box = QGroupBox("Scenario Info")
        _form = QFormLayout()
        self._info_path_label = QLabel("—")
        self._info_coord_label = QLabel("—")
        self._info_dt_label = QLabel("—")
        self._info_drones_label = QLabel("—")
        self._info_obstacles_label = QLabel("—")
        _form.addRow("Config:", self._info_path_label)
        _form.addRow("Coordinator:", self._info_coord_label)
        _form.addRow("dt:", self._info_dt_label)
        _form.addRow("Drones:", self._info_drones_label)
        _form.addRow("Obstacles:", self._info_obstacles_label)
        info_box.setLayout(_form)
        self._info_box = info_box

        self._speed_slider = QSlider(Qt.Orientation.Horizontal)
        self._speed_slider.setRange(10, 500)
        self._speed_slider.setValue(100)
        self._speed_label = QLabel("100 ms")

        # Responsive layout: controls left (vertical) when wide, bottom (horizontal) when tall
        self._controls_max_width = 180  # max width for controls panel in landscape mode
        self._is_landscape: bool = False  # track current layout mode

        # Create central widget and main layout placeholder
        self._central = QWidget()
        self._main_layout: QHBoxLayout | QVBoxLayout | None = None
        self._ctrl_container: QWidget | None = None
        self.setCentralWidget(self._central)
        self._rebuild_layout()

        # ---- Status bar ----
        self._step_label = QLabel("Step: 0 | t: 0.00 s")
        self.statusBar().addPermanentWidget(self._step_label)

        # ---- Signal connections ----
        self._btn_open.clicked.connect(self._on_open_file)
        self._btn_play.clicked.connect(self._on_play)
        self._btn_pause.clicked.connect(self._on_pause)
        self._btn_reset.clicked.connect(self._on_reset)
        self._btn_run_to_end.clicked.connect(self._on_run_to_completion)
        self._speed_slider.valueChanged.connect(self._on_speed_changed)

        # ---- Keyboard shortcuts ----
        # QShortcut fires regardless of which child widget has focus (unlike keyPressEvent
        # which only fires when MainWindow itself has focus — canvas steals focus after orbit).
        spacebar = QShortcut(QKeySequence(Qt.Key.Key_Space), self)
        spacebar.activated.connect(self._toggle_play_pause)

        # ---- Scroll zoom ----
        # Axes3D scroll zoom is not wired automatically in the Qt backend without a
        # NavigationToolbar. Connect it manually via mpl_connect.
        self._canvas.mpl_connect("scroll_event", self._on_scroll)

    # ------------------------------------------------------------------ #
    # Playback slots                                                       #
    # ------------------------------------------------------------------ #

    def _on_open_file(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(self, "Open Scenario JSON", "configs", "JSON Files (*.json)")
        if not path_str:
            return
        self._backend.load_config(Path(path_str))
        self._playing = False
        self._traces = {}
        self._zoom = 1.0
        # Call step() once to get initial drone positions for first render.
        # (SimState from load_config has counts only, no per-drone positions.)
        # Research note (open question 3): simplest approach is one probe step on load.
        result = self._backend.step()
        for drone in result.drones:
            self._traces.setdefault(drone.drone_id, []).append(drone.position.tolist())
        self._redraw(result)
        self._update_step_label(result)
        self._update_collision_indicator(result)
        state = self._backend.get_state()
        self._on_config_loaded(state)

    def _on_play(self) -> None:
        if not self._playing:
            self._playing = True
            QTimer.singleShot(0, self._tick)  # fire immediately for responsiveness

    def _on_pause(self) -> None:
        self._playing = False  # _tick() checks this flag and exits without rescheduling
        self._run_to_completion = False  # Pause cancels run-to-completion (research Q3 resolution)

    def _toggle_play_pause(self) -> None:
        if self._playing:
            self._on_pause()
        else:
            self._on_play()

    def _on_reset(self) -> None:
        self._playing = False
        self._traces = {}  # CRITICAL: clear GUI traces BEFORE backend reset (pitfall 7)
        self._zoom = 1.0
        self._backend.reset()
        # Draw initial frame after reset via one probe step.
        # Intentional: reset always shows step 1 (research open question 3).
        result = self._backend.step()
        for drone in result.drones:
            self._traces.setdefault(drone.drone_id, []).append(drone.position.tolist())
        self._redraw(result)
        self._update_step_label(result)
        self._update_collision_indicator(result)
        # NOTE: _on_config_loaded is NOT called on reset — config metadata is unchanged

    def _on_run_to_completion(self) -> None:
        """Start playing and auto-stop when all drones reach destination or max steps hit."""
        self._run_to_completion = True
        self._on_play()

    def _on_config_loaded(self, state: SimState) -> None:
        """Update scenario info panel after load_config(). NOT called on reset — config metadata unchanged."""
        path_str = state.config_path or "—"
        fname = Path(path_str).name if state.config_path else "—"
        self._info_path_label.setText(fname)
        self._info_path_label.setToolTip(path_str)
        self._info_coord_label.setText(state.coordinator_type)
        self._info_dt_label.setText(f"{state.dt} s")
        self._info_drones_label.setText(str(state.drone_count))
        self._info_obstacles_label.setText(str(state.obstacle_count))

    def _update_collision_indicator(self, result: StepResult) -> None:
        if result.last_collisions:
            self._collision_label.setText("COLLISION")
            self._collision_label.setStyleSheet("color: red; font-weight: bold;")
        else:
            self._collision_label.setText("No Collisions")
            self._collision_label.setStyleSheet("color: green;")

    def _update_admm_indicator(self, result: StepResult) -> None:
        if result.admm_iteration_count is None:
            self._admm_label.setVisible(False)
        else:
            self._admm_label.setVisible(True)
            self._admm_label.setText(f"ADMM: {result.admm_iteration_count} iters")

    def _tick(self) -> None:
        if not self._playing:
            return  # Pause was pressed — do not reschedule
        result = self._backend.step()
        if result.infeasible:
            self._playing = False
            self._run_to_completion = False
            reason = result.infeasible_reason or "Solver reported infeasible controls."
            QMessageBox.warning(self, "Solver Infeasible", reason)
            return  # do NOT reschedule (pitfall 6)
        for drone in result.drones:
            self._traces.setdefault(drone.drone_id, []).append(drone.position)
        self._redraw(result)
        self._update_step_label(result)
        self._update_collision_indicator(result)
        self._update_admm_indicator(result)

        # Run-to-completion termination check:
        if self._run_to_completion:
            if result.all_reached or result.step_count >= _MAX_RUN_STEPS:
                self._playing = False
                self._run_to_completion = False
                return  # do NOT reschedule (auto-stop)

        # Reschedule AFTER current tick completes — prevents event accumulation (locked decision)
        QTimer.singleShot(self._interval_ms, self._tick)

    def _redraw(self, result: StepResult) -> None:
        # Save orbit angle BEFORE cla() — ax.cla() resets elev/azim to defaults (pitfall 3)
        elev = self._ax.elev
        azim = self._ax.azim

        self._ax.cla()

        # Restore orbit angle IMMEDIATELY after cla()
        self._ax.view_init(elev=elev, azim=azim)

        sim_state = self._backend.get_state()
        room_min = sim_state.room_min
        room_max = sim_state.room_max

        draw_room_wireframe(self._ax, room_min, room_max)

        draw_obstacles(self._ax, sim_state.obstacles)

        # Draw drones
        for drone in result.drones:
            pos = drone.position
            color = drone.color
            safety_r = (drone.adaptive_safety_radius if drone.adaptive_safety_radius is not None else drone.safety_zone)
            safety_color = drone.safety_color

            self._ax.scatter([pos[0]], [pos[1]], [pos[2]], s=80, c=[color] if isinstance(color, str) else [color], depthshade=True, label=drone.drone_id)
            draw_sphere_wireframe(self._ax, pos, safety_r, color=safety_color, alpha=0.6, lw=0.6)

            trace = self._traces.get(drone.drone_id, [])
            if trace:
                draw_trace(self._ax, trace, drone.trace_color)

            # Ghost sphere: maximum adaptive radius (only when larger than current zone)
            _draw_ghost_max_sphere(self._ax, drone.adaptive_safety_radius is not None, pos, drone.adaptive_safety_radius, drone.max_adaptive_safety_radius, safety_color)

        # Set axis limits from room bounds, scaled by zoom level.
        # self._zoom is stored on self (not ax) so ax.cla() never resets it.
        cx = (room_min[0] + room_max[0]) / 2
        cy = (room_min[1] + room_max[1]) / 2
        cz = (room_min[2] + room_max[2]) / 2
        hx = (room_max[0] - room_min[0]) / 2 * self._zoom
        hy = (room_max[1] - room_min[1]) / 2 * self._zoom
        hz = (room_max[2] - room_min[2]) / 2 * self._zoom
        self._ax.set_xlim(cx - hx, cx + hx)
        self._ax.set_ylim(cy - hy, cy + hy)
        self._ax.set_zlim(cz - hz, cz + hz)
        self._ax.set_xlabel("x")
        self._ax.set_ylabel("y")
        self._ax.set_zlabel("z")

        box = [room_max[i] - room_min[i] for i in range(3)]
        if all(b > 0 for b in box):
            self._ax.set_box_aspect(box)

        # Use draw_idle — NOT draw() — to avoid blocking the event loop (pitfall)
        self._canvas.draw_idle()

    def _update_step_label(self, result: StepResult) -> None:
        self._step_label.setText(f"Step: {result.step_count} | t: {result.t:.2f} s")

    # ------------------------------------------------------------------ #
    # Speed slider                                                        #
    # ------------------------------------------------------------------ #

    def _on_speed_changed(self, value: int) -> None:
        self._interval_ms = value
        self._speed_label.setText(f"{value} ms")

    # ------------------------------------------------------------------ #
    # Scroll zoom                                                          #
    # ------------------------------------------------------------------ #

    def _on_scroll(self, event) -> None:
        """Zoom 3D view by scaling room limits stored in self._zoom.
        Stored in Python (not on ax) so ax.cla() in _redraw() cannot reset it."""
        if event.button == "up":
            self._zoom = max(0.2, self._zoom * 0.85)
        elif event.button == "down":
            self._zoom = min(5.0, self._zoom * (1.0 / 0.85))
        self._canvas.draw_idle()

    # ------------------------------------------------------------------ #
    # Responsive layout                                                    #
    # ------------------------------------------------------------------ #

    def resizeEvent(self, event) -> None:
        """Rebuild layout when aspect ratio crosses landscape/portrait threshold."""
        super().resizeEvent(event)
        w = self.width()
        h = self.height()
        landscape = w > h
        if landscape != self._is_landscape:
            self._is_landscape = landscape
            self._rebuild_layout()

    def _rebuild_layout(self) -> None:
        """Rebuild the main layout based on current aspect ratio."""
        # Remove old layout if exists
        if self._main_layout is not None:
            # Reparent widgets before deleting layout
            self._canvas.setParent(None)
            self._btn_open.setParent(None)
            self._btn_play.setParent(None)
            self._btn_pause.setParent(None)
            self._btn_reset.setParent(None)
            self._btn_run_to_end.setParent(None)
            self._speed_slider.setParent(None)
            self._speed_label.setParent(None)
            self._collision_label.setParent(None)
            self._admm_label.setParent(None)
            self._info_box.setParent(None)
            # Delete old layout
            QWidget().setLayout(self._central.layout())

        w = self.width()
        h = self.height()
        landscape = w > h
        self._is_landscape = landscape

        if landscape:
            # Landscape: controls on left, vertical, max width limited
            ctrl_layout = QVBoxLayout()
            ctrl_layout.addWidget(self._btn_open)
            ctrl_layout.addWidget(self._btn_play)
            ctrl_layout.addWidget(self._btn_pause)
            ctrl_layout.addWidget(self._btn_reset)
            ctrl_layout.addWidget(self._btn_run_to_end)
            ctrl_layout.addSpacing(15)
            ctrl_layout.addWidget(self._info_box)
            ctrl_layout.addWidget(self._collision_label)
            ctrl_layout.addWidget(self._admm_label)
            ctrl_layout.addStretch()
            ctrl_layout.addWidget(QLabel("Speed:"))
            self._speed_slider.setOrientation(Qt.Orientation.Horizontal)
            ctrl_layout.addWidget(self._speed_slider)
            ctrl_layout.addWidget(self._speed_label)

            ctrl_container = QWidget()
            ctrl_container.setLayout(ctrl_layout)
            ctrl_container.setMaximumWidth(self._controls_max_width)
            ctrl_container.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Expanding)

            main_layout = QHBoxLayout()
            main_layout.setContentsMargins(2, 2, 2, 2)
            main_layout.setSpacing(4)
            main_layout.addWidget(ctrl_container)
            main_layout.addWidget(self._canvas, stretch=1)
            self._canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        else:
            # Portrait: controls at bottom, horizontal
            ctrl_layout = QHBoxLayout()
            ctrl_layout.addWidget(self._btn_open)
            ctrl_layout.addWidget(self._btn_play)
            ctrl_layout.addWidget(self._btn_pause)
            ctrl_layout.addWidget(self._btn_reset)
            ctrl_layout.addWidget(self._btn_run_to_end)
            ctrl_layout.addStretch()
            ctrl_layout.addWidget(QLabel("Speed:"))
            self._speed_slider.setOrientation(Qt.Orientation.Horizontal)
            ctrl_layout.addWidget(self._speed_slider)
            ctrl_layout.addWidget(self._speed_label)

            ctrl_container = QWidget()
            ctrl_container.setLayout(ctrl_layout)

            status_row = QHBoxLayout()
            status_row.addWidget(self._info_box)
            status_row.addStretch()
            status_row.addWidget(self._collision_label)
            status_row.addWidget(self._admm_label)

            main_layout = QVBoxLayout()
            main_layout.setContentsMargins(2, 2, 2, 2)
            main_layout.setSpacing(4)
            main_layout.addWidget(self._canvas, stretch=1)
            main_layout.addLayout(status_row)
            main_layout.addWidget(ctrl_container)
            self._canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._central.setLayout(main_layout)
        self._main_layout = main_layout
        self._ctrl_container = ctrl_container
