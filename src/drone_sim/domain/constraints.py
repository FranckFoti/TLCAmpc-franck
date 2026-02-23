from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np

from drone_sim.domain.drone import Drone


def point_to_box_dist(
    point: np.ndarray,
    center: np.ndarray,
    half_extents: np.ndarray,
    eps: float = 1e-3,
) -> float:
    """Smooth point-to-axis-aligned-box distance, SLSQP-compatible.

    Uses a regularized L2 norm of per-axis excess: sqrt(sum(relu(d)^2) + eps^2) - eps
    where d_i = |point_i - center_i| - half_extents_i.

    This is positive outside the box, approaches 0 at the box surface, and
    has a non-zero gradient everywhere (avoiding the flat zero-gradient region
    inside the box that would stall SLSQP).

    :param point: 3D query point.
    :param center: Box center (3,).
    :param half_extents: Box half-sizes per axis (3,).
    :param eps: Regularization constant — controls smoothness near the surface.
    :return: Smooth distance >= 0.
    """
    d = np.abs(point - center) - half_extents         # signed per-axis excess
    relu_d = np.maximum(d, 0.0)                        # only positive excess matters
    return float(np.sqrt(np.dot(relu_d, relu_d) + eps * eps) - eps)


def _velocity_at_step(pred_vel: np.ndarray | None, step: int) -> np.ndarray | None:
   """Extract the velocity vector at a given step, or ``None`` if unavailable."""
   if pred_vel is not None:
      return pred_vel[step]
   return None


def _safety_radius(drone: Drone, velocity: np.ndarray | None) -> float:
   """Compute the safety radius for a drone at a given velocity.

   :param drone: the drone to compute the radius for.
   :param velocity: 3D velocity vector, or ``None`` to use the fixed safety zone.
   :return: adaptive radius when the drone is adaptive and velocity is provided,
            otherwise the fixed ``drone.safety_zone``.
   """
   if velocity is not None and drone.is_adaptive:
      return drone.compute_adaptive_radius(velocity)
   return drone.safety_zone


class MPCConstraints(ABC):
   """Base class for constraints."""

   def __init__(self, horizon: int):
      self._horizon = horizon

   @abstractmethod
   def label(self) -> str:
      pass


class VelocityConstraints(MPCConstraints):
   def label(self) -> str:
      return "velocity"

   def evaluate_single(self, drone_state: Drone, v_pred: np.ndarray, values: np.ndarray) -> np.ndarray:
      """Evaluate velocity constraints for a single drone.

      :param drone_state: the drone.
      :param v_pred: predicted velocity per time step (shape: (horizon, 3)).
      :param values: combined constraint values to append to.
      :return: updated values list.
      """
      speed_sq = np.sum(v_pred[:self._horizon] ** 2, axis=1)
      result = drone_state.v_max ** 2 - speed_sq
      return np.concatenate([values, result])

   def evaluate_multi(self, drone_states: list[Drone], v_pred: np.ndarray, values: np.ndarray) -> np.ndarray:
      """Evaluate velocity constraints for multiple drones.

      :param drone_states: list of drones.
      :param v_pred: predicted velocity per drone per time step (shape: (num_drones, horizon, 3)).
      :param values: combined constraint values to append to.
      :return: updated values list.
      """
      parts = []
      for drone_idx, drone in enumerate(drone_states):
         speed_sq = np.sum(v_pred[drone_idx, :self._horizon] ** 2, axis=1)
         parts.append(drone.v_max ** 2 - speed_sq)
      return np.concatenate([values, *parts])


class MovingObstacleAvoidanceConstraints(MPCConstraints):
   def label(self) -> str:
      return "moving_obstacle_avoidance"

   def evaluate_single(
      self,
      drone: Drone,
      pred_pos: np.ndarray,
      neighbor_trajectories: dict[str, tuple[np.ndarray, np.ndarray | None]],
      values: np.ndarray,
      pred_vel: np.ndarray | None = None,
   ) -> np.ndarray:
      """Evaluate moving obstacle avoidance constraints for a single drone.

      :param drone: the drone.
      :param pred_pos: predicted positions per time step (shape: (horizon, 3)).
      :param neighbor_trajectories: mapping of neighbor id to
             (trajectory (H,3), predicted_velocities (H,3) or None).
             Uses ego drone's config to compute neighbor adaptive radii.
      :param values: combined constraint values to append to.
      :param pred_vel: optional predicted velocities (shape: (horizon, 3)).
                       When provided and the drone is adaptive, per-step adaptive radii are used.
      :return: updated values list.
      """
      result = self._evaluate(drone, pred_pos, neighbor_trajectories, pred_vel=pred_vel)
      return np.concatenate([values, result])

   def evaluate_multi(
      self,
      drones: list[Drone],
      pred_pos: dict[str, np.ndarray],
      values: np.ndarray,
      pred_vel: dict[str, np.ndarray] | None = None,
   ) -> np.ndarray:
      """Evaluate collision avoidance for all drone pairs.

      Uses i<j pairs to avoid duplicates. Includes cons_stop since all drone
      info is available in the multi-drone case.

      :param drones: list of drones.
      :param pred_pos: predicted positions keyed by drone_id.
      :param values: combined constraint values to append to.
      :param pred_vel: optional dict mapping drone_id to predicted velocity arrays
                       (shape per drone: (horizon, 3)). When provided, adaptive
                       drones use per-step adaptive radii.
      :return: updated values list.
      """
      parts = []
      for i in range(len(drones)):
         for j in range(i + 1, len(drones)):
            drone_i = drones[i]
            drone_j = drones[j]
            traj_i = pred_pos[drone_i.drone_id]
            traj_j = pred_pos[drone_j.drone_id]
            vel_i = pred_vel[drone_i.drone_id] if pred_vel is not None else None
            vel_j = pred_vel[drone_j.drone_id] if pred_vel is not None else None

            result = np.zeros(self._horizon)
            for step in range(self._horizon):
               safety_i = _safety_radius(drone_i, _velocity_at_step(vel_i, step))
               safety_j = _safety_radius(drone_j, _velocity_at_step(vel_j, step))
               dist = float(np.linalg.norm(traj_i[step] - traj_j[step]))
               result[step] = dist - (safety_i + safety_j + drone_i.cons_stop + drone_j.cons_stop)
            parts.append(result)
      return np.concatenate([values, *parts])

   def _evaluate(
      self,
      drone: Drone,
      pred_pos: np.ndarray,
      neighbor_trajectories: dict[str, tuple[np.ndarray, np.ndarray | None]],
      pred_vel: np.ndarray | None = None,
   ):
      """Evaluate collision constraints against neighbor trajectories.

      :param drone: the ego drone.
      :param pred_pos: predicted positions (horizon, 3).
      :param neighbor_trajectories: neighbor data mapping id to
             (trajectory (H,3), predicted_velocities (H,3) or None).
             Neighbor adaptive radii are computed using the ego drone's config
             ("same drone type" assumption).
      :param pred_vel: optional predicted velocities for the ego drone (horizon, 3).
      :return: constraint margin array.
      """
      parts = []
      for neighbor_traj, neighbor_vel in neighbor_trajectories.values():
         neighbor_traj = np.asarray(neighbor_traj, dtype=float).reshape((self._horizon, 3))
         result = np.zeros(self._horizon)
         for step in range(self._horizon):
            safety = _safety_radius(drone, _velocity_at_step(pred_vel, step))
            neighbor_safety = _safety_radius(drone, _velocity_at_step(neighbor_vel, step))
            dist = float(np.linalg.norm(pred_pos[step] - neighbor_traj[step]))
            result[step] = dist - (safety + neighbor_safety)
         parts.append(result)
      if not parts:
         return np.zeros(0)
      return np.concatenate(parts)


class ObstacleAvoidanceConstraints(MPCConstraints):
   def label(self) -> str:
      return "obstacle_avoidance"

   def evaluate_single(
      self,
      drone: Drone,
      pred_pos: np.ndarray,
      obstacles: list[tuple[np.ndarray, np.ndarray]],
      values: np.ndarray,
      pred_vel: np.ndarray | None = None,
   ) -> np.ndarray:
      """Evaluate obstacle avoidance constraints for a single drone.

      :param drone: the drone.
      :param pred_pos: predicted position per time step (shape: (horizon, 3)).
      :param obstacles: list of obstacles, each a (center, half_extents) tuple.
      :param values: combined constraint values to append to.
      :param pred_vel: optional predicted velocities (shape: (horizon, 3)).
                       When provided and the drone is adaptive, per-step adaptive radii are used.
      :return: updated values list.
      """
      result = self._evaluate(drone, pred_pos, obstacles, pred_vel=pred_vel)
      return np.concatenate([values, result])

   def evaluate_multi(
      self,
      drones: list[Drone],
      pred_pos: dict[str, np.ndarray],
      obstacles: list[tuple[np.ndarray, np.ndarray]],
      values: np.ndarray,
      pred_vel: dict[str, np.ndarray] | None = None,
   ) -> np.ndarray:
      """Evaluate obstacle avoidance constraints for multiple drones.

      :param drones: list of Drones involved.
      :param pred_pos: predicted positions for the drones keyed by drone_id.
      :param obstacles: list of obstacles, each a (center, half_extents) tuple.
      :param values: combined constraint values to append to.
      :param pred_vel: optional dict mapping drone_id to predicted velocity arrays (shape: (horizon, 3)).
      :return: updated values list.
      """
      parts = []
      for drone in drones:
         vel = pred_vel[drone.drone_id] if pred_vel is not None else None
         parts.append(self._evaluate(drone, pred_pos[drone.drone_id], obstacles, pred_vel=vel))
      return np.concatenate([values, *parts])

   def _evaluate(
      self,
      drone: Drone,
      pred_pos: np.ndarray,
      obstacles: list[tuple[np.ndarray, np.ndarray]],
      pred_vel: np.ndarray | None = None,
   ) -> np.ndarray:
      """Evaluate static obstacle constraints.

      :param drone: the drone.
      :param pred_pos: predicted positions (horizon, 3).
      :param obstacles: list of (center, half_extents) tuples.
      :param pred_vel: optional predicted velocities (horizon, 3).
      :return: constraint margin array.
      """
      result = np.zeros(self._horizon)
      for center, half_extents in obstacles:
         obstacle_center = np.asarray(center, dtype=float).reshape(3)
         obstacle_half_extents = np.asarray(half_extents, dtype=float).reshape(3)
         for step in range(self._horizon):
            safety = _safety_radius(drone, _velocity_at_step(pred_vel, step))
            dist = point_to_box_dist(pred_pos[step], obstacle_center, obstacle_half_extents)
            result[step] = dist - safety
      return result


class RoomConstraints(MPCConstraints):
   def __init__(self, horizon: int, wall_tolerance: float = 0.0):
      super().__init__(horizon)
      self._wall_tolerance = wall_tolerance

   def label(self) -> str:
      return "room"

   def evaluate_single(
      self,
      drone: Drone,
      pred_pos: np.ndarray,
      room_max: float,
      room_min: float,
      values: np.ndarray,
      room_is_sphere: bool = False,
      pred_vel: np.ndarray | None = None,
   ) -> np.ndarray:
      """Evaluate room boundary constraints for a single drone.

      :param drone: the drone.
      :param pred_pos: predicted positions per time step (shape: (horizon, 3)).
      :param room_max: room upper bounds (3,) for box or radius for sphere.
      :param room_min: room lower bounds (3,) for box or unused for sphere.
      :param values: combined constraint values to append to.
      :param room_is_sphere: if True, room_max is treated as sphere radius.
      :param pred_vel: optional predicted velocities (shape: (horizon, 3)).
                       When provided and the drone is adaptive, per-step adaptive radii are used.
      :return: updated values list.
      """
      result = self._evaluate(drone, pred_pos, room_max, room_min, room_is_sphere, pred_vel=pred_vel)
      return np.concatenate([values, result])

   def evaluate_multi(
      self,
      drones: list[Drone],
      pred_pos: dict[str, np.ndarray],
      room_max: float,
      room_min: float,
      values: np.ndarray,
      room_is_sphere: bool = False,
      pred_vel: dict[str, np.ndarray] | None = None,
   ) -> np.ndarray:
      """Evaluate room boundary constraints for multiple drones.

      :param drones: list of Drones involved.
      :param pred_pos: predicted positions for the drones keyed by drone_id.
      :param room_max: room upper bounds (3,) for box or radius for sphere.
      :param room_min: room lower bounds (3,) for box or unused for sphere.
      :param values: combined constraint values to append to.
      :param room_is_sphere: if True, room_max is treated as sphere radius.
      :param pred_vel: optional dict mapping drone_id to predicted velocity arrays (shape: (horizon, 3)).
      :return: updated values list.
      """
      parts = []
      for drone in drones:
         vel = pred_vel[drone.drone_id] if pred_vel is not None else None
         parts.append(self._evaluate(drone, pred_pos[drone.drone_id], room_max, room_min, room_is_sphere, pred_vel=vel))
      return np.concatenate([values, *parts])

   def _evaluate(
      self,
      drone: Drone,
      pred_pos: np.ndarray,
      room_max: float,
      room_min: float,
      room_is_sphere: bool = False,
      pred_vel: np.ndarray | None = None,
   ) -> np.ndarray:
      if room_is_sphere:
         return self._evaluate_sphere(drone, pred_pos, room_radius=room_max, pred_vel=pred_vel)
      return self._evaluate_box(drone, pred_pos, room_max, room_min, pred_vel=pred_vel)

   def _evaluate_box(
      self,
      drone: Drone,
      pred_pos: np.ndarray,
      room_max: float,
      room_min: float,
      pred_vel: np.ndarray | None = None,
   ) -> np.ndarray:
      lower_bounds = np.asarray(room_min, dtype=float).reshape(3)
      upper_bounds = np.asarray(room_max, dtype=float).reshape(3)
      result = np.zeros(self._horizon * 6)
      count = 0
      for step in range(self._horizon):
         pos = pred_pos[step]
         safety = _safety_radius(drone, _velocity_at_step(pred_vel, step))
         for axis in range(3):
            result[count] = float(pos[axis] - safety - lower_bounds[axis]) + self._wall_tolerance
            count += 1
         for axis in range(3):
            result[count] = float(upper_bounds[axis] - (pos[axis] + safety)) + self._wall_tolerance
            count += 1
      return result

   def _evaluate_sphere(
      self,
      drone: Drone,
      pred_pos: np.ndarray,
      room_radius: float,
      pred_vel: np.ndarray | None = None,
   ) -> np.ndarray:
      result = np.zeros(self._horizon)
      for step in range(self._horizon):
         dist = float(np.linalg.norm(pred_pos[step]))
         safety = _safety_radius(drone, _velocity_at_step(pred_vel, step))
         result[step] = room_radius - dist - safety
      return result
