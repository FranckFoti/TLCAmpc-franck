from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
   status: str


class ConfigResponse(BaseModel):
   status: str
   num_drones: int


class StepResponse(BaseModel):
   status: str
   t: float
   dt: float
   # Optional detail about the step outcome (e.g. infeasibility reason).
   detail: str | None = None


class DroneState(BaseModel):
   drone_id: str
   x: list[float] = Field(..., min_length=6, max_length=6)
   route_idx: int
   p_ref: list[float] = Field(..., min_length=3, max_length=3)
   radius: float
   safety_zone: float

   drone_color: str | list[float]
   safety_color: str | list[float]
   trace_color: str | list[float]


class ObstacleState(BaseModel):
   center: list[float] = Field(..., min_length=3, max_length=3)
   radius: float


class RoomState(BaseModel):
   min: list[float] = Field(..., min_length=3, max_length=3)
   max: list[float] = Field(..., min_length=3, max_length=3)


class CollisionEvent(BaseModel):
   kind: str
   owner: str

   # drone_drone
   intruder: str | None = None

   # drone_obstacle
   obstacle_idx: int | None = None

   distance: float
   threshold: float


class StateResponse(BaseModel):
   t: float
   dt: float
   room: RoomState
   drones: list[DroneState]
   obstacles: list[ObstacleState]
   collisions: list[CollisionEvent] = []
