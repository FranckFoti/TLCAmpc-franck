from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator

from drone_sim.simulation.coordinator import Coordinator

ColorValue = str | list[float]


class PhysicsSpec(BaseModel):
   type: str
   id: str | None = None
   params: dict[str, Any] = Field(default_factory=dict)


class ControllerSpec(BaseModel):
   type: str
   params: dict[str, Any] = Field(default_factory=dict)


class DroneConfig(BaseModel):
   drone_id: str
   start: list[float] = Field(..., min_length=3, max_length=3)
   waypoints: list[list[float]] = Field(default_factory=list)
   target: list[float] = Field(..., min_length=3, max_length=3)

   # Optional per-drone controller override (otherwise ScenarioConfig.controller is used).
   controller: ControllerSpec | None = None

   # Physics model ID referencing a PhysicsSpec by name (None = use global physics).
   physics: str | None = None

   # Drone physical radius (used for room clamping and visualization).
   radius: float = 0.2

   # Visualization / safety bubble radius around the drone.
   safety_zone: float = 1.0

   # Conservative stopping addition, like it is shown in the paper
   cons_stop: float = 0.0

   # Adaptive safety zone parameter. When set, the drone uses a velocity-dependent
   # safety radius: r(t) = r_min + alpha * ||v||^2 / (2 * U_max).
   # When None, the fixed safety_zone is used instead.
   alpha: float | None = None

   # Colors used by the renderer. Each field accepts either:
   # - a matplotlib-compatible color string (e.g. "red", "tab:blue", "#ff00aa")
   # - an RGB list [r,g,b] either in 0..1 or 0..255.
   drone_color: ColorValue = "tab:blue"
   # If omitted, the renderer uses the drone_color.
   safety_color: ColorValue | None = None
   # If omitted, the renderer uses the drone_color.
   trace_color: ColorValue | None = None

   @model_validator(mode="after")
   def _validate_alpha(self) -> DroneConfig:
      if self.alpha is not None and self.alpha <= 0:
         raise ValueError("alpha must be positive when set")
      return self


class ObstacleConfig(BaseModel):
   center: list[float] = Field(..., min_length=3, max_length=3)
   half_extents: list[float] = Field(..., min_length=3, max_length=3)


class RoomConfig(BaseModel):
   """Axis-aligned room bounds used for visualization (and later constraints).
   """

   min: list[float] = Field(..., min_length=3, max_length=3)
   max: list[float] = Field(..., min_length=3, max_length=3)


class ScenarioConfig(BaseModel):
   dt: float = 0.1
   physics: PhysicsSpec | list[PhysicsSpec]

   # Default controller used for drones that do not define DroneConfig.controller.
   controller: ControllerSpec

   # Optional coordinator used for distributed MPC neighbor discovery.
   coordinator: ControllerSpec | None = None

   # Communication radius for distributed MPC neighbor discovery.
   # When None, all drones are considered neighbors (backward compatible with centralized mode).
   # When set to a positive float, only drones within this distance are neighbors.
   comm_radius: float | None = None

   drones: list[DroneConfig]
   obstacles: list[ObstacleConfig] = Field(default_factory=list)

   # Optional visualization bounds.
   room: RoomConfig | None = None
