from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np

from drone_sim.domain.drone import Drone


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
      """ Evaluate velocity constraints for a single drone.
      :param drone_state: the drone
      :param v_pred: predicted velocity for the drone for each time step (shape: (horizon, 3))
      :param values: combined list for all constraints, will be filled in row, this step appends velocity constraints for the drone
      :return: updated values list
      """
      result = np.zeros(self._horizon)
      for h in range(self._horizon):
         vel = v_pred[h, :]
         v_max = drone_state.v_max
         vel_margin = v_max ** 2 - float(vel[0] ** 2 + vel[1] ** 2 + vel[2] ** 2)
         result[h] = max(0, vel_margin)
      return np.concatenate([values, result])

   def evaluate_multi(self, drone_states: list[Drone], v_pred: np.ndarray, values: np.ndarray) -> np.ndarray:
      """ Evaluate velocity constraints for multiple drones.
      :param drone_states: list of drones
      :param v_pred: predicted velocity for each drone for each time step (shape: (num_drones, horizon, 3))
      :param values: combined list for all constraints, will be filled in row, this step appends velocity constraints for each drone
      :return: updated values list
      """
      result = np.zeros(self._horizon * len(drone_states))
      count = 0
      for d in range(len(drone_states)):
         for h in range(self._horizon):
            vel = v_pred[d][h]  # (vx, vy, vz)
            v_max = drone_states[d].v_max
            # Constraint: v_max^2 - (vx^2 + vy^2 + vz^2) >= 0 (is always correct .. do not run into square root issue)
            velocity_margin = v_max ** 2 - float(vel[0] ** 2 + vel[1] ** 2 + vel[2] ** 2)
            result[count] = max(0, velocity_margin)
            count += 1
      return np.concatenate([values, result])


class MovingObstacleAvoidanceConstraints(MPCConstraints):
   def label(self) -> str:
      return "moving_obstacle_avoidance"

   def evaluate_single(self, drone: Drone, pred_pos: np.ndarray, neighbor_trajectories: dict[str, tuple[np.ndarray, float]], values: np.ndarray) -> np.ndarray:
      """ Evaluate moving obstacle avoidance constraints for a single drone.
      :param drone: the drone
      :param pred_pos: predicted positions for the drone for each time step (shape: (horizon, 3))
      :param neighbor_trajectories: predicted trajectories of neighbors (shape: (num_neighbors, horizon, 3))
      :param values: combined list for all constraints, will be filled in row, this step appends velocity constraints for the drone
      :return: updated values list
      """
      result = self._evaluate(drone, pred_pos, neighbor_trajectories)

      return np.concatenate([values, result])

   def evaluate_multi(self, drones: list[Drone], pred_pos: dict[str, np.ndarray], values: np.ndarray) -> np.ndarray:
      for drone in drones:
         neighbor_trajectories = self._get_neighbor_trajectories(drone.drone_id, drones, pred_pos)
         result = self._evaluate(drone, pred_pos[drone.drone_id], neighbor_trajectories)
         values = np.concatenate([values, result])
      return values

   def _get_neighbor_trajectories(self, current_id: str, drones: list[Drone], pred_pos: dict[str, np.ndarray]):
      res = {}
      for drone in drones:
         if drone.drone_id == current_id:
            continue
         res[drone.drone_id] = (pred_pos[drone.drone_id], drone.safety_zone)
      return res

   def _evaluate(self, drone: Drone, pred_pos: np.ndarray, neighbor_trajectories: dict[str, tuple[np.ndarray, float]]):
      result = np.zeros(self._horizon * len(neighbor_trajectories))
      count = 0
      for neighbor_id, (neighbor_traj, neighbor_sz) in neighbor_trajectories.items():
         neighbor_traj = np.asarray(neighbor_traj, dtype=float).reshape((self._horizon, 3))
         min_dist = drone.safety_zone + neighbor_sz

         for h in range(self._horizon):
            dist = float(np.linalg.norm(pred_pos[h] - neighbor_traj[h]))
            result[count] = dist - min_dist
            count += 1
      return result


class ObstacleAvoidanceConstraints(MPCConstraints):
   def label(self) -> str:
      return "obstacle_avoidance"

   def evaluate_single(self, drone: Drone, pred_pos: np.ndarray, obstacles: list[tuple[np.ndarray, float]], values: np.ndarray) -> np.ndarray:
      """ Evaluate obstacle avoidance constraints for a single drone.
      :param drone: the drone
      :param pred_pos: one predicted position for each time step (shape: (horizon, 3))
      :param obstacles: list of obstacles, each obstacle is a tuple of (center, radius)
      :param values: combined list for all constraints, will be filled in row, this step appends velocity constraints for the drone
      :return: updated values list
      """
      result = self._evaluate(drone, pred_pos, obstacles)
      return np.concatenate([values, result])

   def evaluate_multi(self, drones: list[Drone], pred_pos: dict[str, np.ndarray], obstacles: list[tuple[np.ndarray, float]], values: np.ndarray) -> np.ndarray:
      """ Evaluate obstacle avoidance constraints for a drones.
      :param drones: list of Drones involved
      :param pred_pos: predicted positions for the drones (shape: (num_drones, horizon, 3))
      :param obstacles: list of obstacles, each obstacle is a tuple of (center, radius)
      :param values: combined list for all constraints, will be filled in row, this step appends velocity constraints for each drone
      :return: updated values list
      """
      result = np.zeros(self._horizon * len(drones))
      count = 0
      for d in range(len(drones)):
         result[count:count + self._horizon] = self._evaluate(drones[d], pred_pos[drones[d].drone_id], obstacles)
         count += self._horizon
      return np.concatenate([values, result])

   def _evaluate(self, drone: Drone, pred_pos: np.ndarray, obstacles: list[tuple[np.ndarray, float]]) -> np.ndarray:
      result = np.zeros(self._horizon)
      for center, radius in obstacles:
         c = np.asarray(center, dtype=float).reshape(3)
         min_dist = drone.safety_zone + radius
         for h in range(self._horizon):
            dist = float(np.linalg.norm(pred_pos[h] - c))
            result[h] = dist - min_dist
      return result


class RoomConstraints(MPCConstraints):
   def label(self) -> str:
      return "room"

   def evaluate_single(self, drone: Drone, pred_pos: np.ndarray, room_max: float, room_min: float, values: np.ndarray, room_is_sphere: bool = False) -> np.ndarray:
      """ Evaluate room boundary constraints for a single drone.
      :param drone: the drone
      :param pred_pos: predicted positions for the drone for each time step (shape: (horizon, 3))
      :param room: the room defining the boundaries
      :param values: combined list for all constraints, will be filled in row, this step appends room constraints for the drone
      :return: updated values list
      """
      result = self._evaluate(drone, pred_pos, room_max, room_min, room_is_sphere)
      return np.concatenate([values, result])

   def evaluate_multi(self, drones: list[Drone], pred_pos: dict[str, np.ndarray], room_max: float, room_min: float, values: np.ndarray, room_is_sphere: bool = False) -> np.ndarray:
      """ Evaluate room boundary constraints for multiple drones.
      :param drones: list of Drones involved
      :param pred_pos: predicted positions for the drones keyed by drone_id
      :param room: the room defining the boundaries
      :param values: combined list for all constraints, will be filled in row, this step appends room constraints for each drone
      :return: updated values list
      """
      result = np.zeros(self._horizon * len(drones))
      count = 0
      for d in range(len(drones)):
         result[count:count + self._horizon] = self._evaluate(drones[d], pred_pos[drones[d].drone_id], room_max, room_min, room_is_sphere)
         count += self._horizon
      return np.concatenate([values, result])

   def _evaluate(self, drone: Drone, pred_pos: np.ndarray, room_max: float, room_min: float, room_is_sphere: bool = False) -> np.ndarray:
      if room_is_sphere:
         return self._evaluate_sphere(drone, pred_pos, room_radius=room_max)
      return self._evaluate_box(drone, pred_pos, room_max, room_min)

   def _evaluate_box(self, drone: Drone, pred_pos: np.ndarray, room_max: float, room_min: float) -> np.ndarray:
      min_c = np.asarray(room_min, dtype=float).reshape(3)
      max_c = np.asarray(room_max, dtype=float).reshape(3)
      result = np.zeros(self._horizon)
      for h in range(self._horizon):
         pos = pred_pos[h]
         # Minimum margin across all 6 room faces, accounting for safety zone
         lower_margins = pos - min_c - drone.safety_zone
         upper_margins = max_c - pos - drone.safety_zone
         result[h] = float(min(np.min(lower_margins), np.min(upper_margins)))
      return result

   def _evaluate_sphere(self, drone: Drone, pred_pos: np.ndarray, room_radius: float) -> np.ndarray:
      result = np.zeros(self._horizon)
      for h in range(self._horizon):
         dist = float(np.linalg.norm(pred_pos[h])) # center is always (0,0,0)
         result[h] = room_radius - dist - drone.safety_zone
      return result

