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
      for step in range(self._horizon):
         vel = v_pred[step, :]
         v_max = drone_state.v_max
         result[step] = v_max ** 2 - float(vel[0] ** 2 + vel[1] ** 2 + vel[2] ** 2)
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
      for drone_idx in range(len(drone_states)):
         for step in range(self._horizon):
            vel = v_pred[drone_idx][step]  # (vx, vy, vz)
            v_max = drone_states[drone_idx].v_max
            result[count] = v_max ** 2 - float(vel[0] ** 2 + vel[1] ** 2 + vel[2] ** 2)
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
      """ Evaluate collision avoidance for all drone pairs.

      Uses i<j pairs to avoid duplicates. Includes cons_stop since all drone
      info is available in the multi-drone case.
      """
      for i in range(len(drones)):
         for j in range(i + 1, len(drones)):
            drone_i = drones[i]
            drone_j = drones[j]
            traj_i = pred_pos[drone_i.drone_id]
            traj_j = pred_pos[drone_j.drone_id]
            min_dist = drone_i.safety_zone + drone_j.safety_zone + drone_i.cons_stop + drone_j.cons_stop

            result = np.zeros(self._horizon)
            for step in range(self._horizon):
               dist = float(np.linalg.norm(traj_i[step] - traj_j[step]))
               result[step] = dist - min_dist
            values = np.concatenate([values, result])
      return values

   def _get_neighbor_trajectories(self, current_id: str, drones: list[Drone], pred_pos: dict[str, np.ndarray]):
      neighbors = {}
      for drone in drones:
         if drone.drone_id == current_id:
            continue
         neighbors[drone.drone_id] = (pred_pos[drone.drone_id], drone.safety_zone)
      return neighbors

   def _evaluate(self, drone: Drone, pred_pos: np.ndarray, neighbor_trajectories: dict[str, tuple[np.ndarray, float]]):
      result = np.zeros(self._horizon * len(neighbor_trajectories))
      count = 0
      for neighbor_id, (neighbor_traj, neighbor_sz) in neighbor_trajectories.items():
         neighbor_traj = np.asarray(neighbor_traj, dtype=float).reshape((self._horizon, 3))
         min_dist = drone.safety_zone + neighbor_sz

         for step in range(self._horizon):
            dist = float(np.linalg.norm(pred_pos[step] - neighbor_traj[step]))
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
      for drone_idx in range(len(drones)):
         result[count:count + self._horizon] = self._evaluate(drones[drone_idx], pred_pos[drones[drone_idx].drone_id], obstacles)
         count += self._horizon
      return np.concatenate([values, result])

   def _evaluate(self, drone: Drone, pred_pos: np.ndarray, obstacles: list[tuple[np.ndarray, float]]) -> np.ndarray:
      result = np.zeros(self._horizon)
      for center, radius in obstacles:
         obstacle_center = np.asarray(center, dtype=float).reshape(3)
         min_dist = drone.safety_zone + radius
         for step in range(self._horizon):
            dist = float(np.linalg.norm(pred_pos[step] - obstacle_center))
            result[step] = dist - min_dist
      return result


class RoomConstraints(MPCConstraints):
   def __init__(self, horizon: int, wall_tolerance: float = 0.0):
      super().__init__(horizon)
      self._wall_tolerance = wall_tolerance

   def label(self) -> str:
      return "room"

   def evaluate_single(self, drone: Drone, pred_pos: np.ndarray, room_max: float, room_min: float, values: np.ndarray,
                       room_is_sphere: bool = False) -> np.ndarray:
      """ Evaluate room boundary constraints for a single drone.
      :param drone: the drone
      :param pred_pos: predicted positions for the drone for each time step (shape: (horizon, 3))
      :param room_max: room upper bounds (3,) for box or radius for sphere
      :param room_min: room lower bounds (3,) for box or unused for sphere
      :param values: combined list for all constraints, will be filled in row
      :param room_is_sphere: if True, room_max is treated as sphere radius
      :return: updated values list
      """
      result = self._evaluate(drone, pred_pos, room_max, room_min, room_is_sphere)
      return np.concatenate([values, result])

   def evaluate_multi(self, drones: list[Drone], pred_pos: dict[str, np.ndarray], room_max: float, room_min: float, values: np.ndarray,
                      room_is_sphere: bool = False) -> np.ndarray:
      """ Evaluate room boundary constraints for multiple drones.
      :param drones: list of Drones involved
      :param pred_pos: predicted positions for the drones keyed by drone_id
      :param room_max: room upper bounds (3,) for box or radius for sphere
      :param room_min: room lower bounds (3,) for box or unused for sphere
      :param values: combined list for all constraints, will be filled in row
      :param room_is_sphere: if True, room_max is treated as sphere radius
      :return: updated values list
      """
      for drone_idx in range(len(drones)):
         result = self._evaluate(drones[drone_idx], pred_pos[drones[drone_idx].drone_id], room_max, room_min, room_is_sphere)
         values = np.concatenate([values, result])
      return values

   def _evaluate(self, drone: Drone, pred_pos: np.ndarray, room_max: float, room_min: float, room_is_sphere: bool = False) -> np.ndarray:
      if room_is_sphere:
         return self._evaluate_sphere(drone, pred_pos, room_radius=room_max)
      return self._evaluate_box(drone, pred_pos, room_max, room_min)

   def _evaluate_box(self, drone: Drone, pred_pos: np.ndarray, room_max: float, room_min: float) -> np.ndarray:
      lower_bounds = np.asarray(room_min, dtype=float).reshape(3)
      upper_bounds = np.asarray(room_max, dtype=float).reshape(3)
      result = np.zeros(self._horizon * 6)
      count = 0
      for step in range(self._horizon):
         pos = pred_pos[step]
         for axis in range(3):
            result[count] = float(pos[axis] - drone.safety_zone - lower_bounds[axis]) + self._wall_tolerance
            count += 1
         for axis in range(3):
            result[count] = float(upper_bounds[axis] - (pos[axis] + drone.safety_zone)) + self._wall_tolerance
            count += 1
      return result

   def _evaluate_sphere(self, drone: Drone, pred_pos: np.ndarray, room_radius: float) -> np.ndarray:
      result = np.zeros(self._horizon)
      for step in range(self._horizon):
         dist = float(np.linalg.norm(pred_pos[step]))  # center is always (0,0,0)
         result[step] = room_radius - dist - drone.safety_zone
      return result
