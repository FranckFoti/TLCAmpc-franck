# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 11:02:19 2026

@author: Hamekong
"""
# %% imports

import numpy as np


# %% GENERATE RANDOM POINTS WITHIN THE AIRSPACE
def generate_random_points(drone_radius, cube_side, N: int, d_r, seed, max_attempts: int = 1000) -> np.ndarray:
   """
   Randomly generates a number of points, N, within the bounded airspace with
   a minimum specified distance, d_r, between every point

   Args:
       drone_radius (float): Drone radius
       cube_side (float): Side of cubic airspace
       N (int): Number of points to be generated
       d_r (float): Minimum distance between generated points
       seed (int): Seed used for the reproduceability of the random selection of points
       max_attempts (int): Maximum number of attemps to generating the N number of points
                   before concluding that it is impossible to place this N number of drones,
                   d_r apart within the airspace

   Returns:
       points (list of np.array): Positions of randomly generations points satisfying all specified requirements
   """

   if N <= 0 or d_r <= 0:
      raise ValueError("N and d_r must be positive.\n")

   # --- seed for reproduceability of random points ---
   np.random.seed(seed)

   # --- set airspace bounds ---
   half_side = cube_side / 2
   min_bound = -half_side + drone_radius
   max_bound = half_side - drone_radius

   # --- Calculate the minimum required squared distance ---
   d_r_sq = d_r ** 2

   # --- Obatin array of accepted points ---
   points = np.empty((0, 3))  # returns an empty array of shape (0,3)

   for i in range(N):
      attempts = 0

      while attempts < max_attempts:
         # Generate random point on a unit sphere (using uniform sampling of the sphere)
         new_point = (2 * max_bound * np.random.rand(3)) + min_bound  # np.random.rand(3) generates a (1,3) array with elements btw 0.0 and 1.0, excluding 1.0

         # Check distance constraint
         is_valid = True
         if points.shape[0] > 0:  # check that "points" is not empty
            # Calculate squared Euclidean distances to all existing points
            distances_sq_point = np.sum((points - new_point) ** 2, axis=1)

            # Check if any distance is less than the required minimum
            if np.any(distances_sq_point < d_r_sq):
               is_valid = False

         if is_valid:
            # Append the valid point
            points = np.vstack([points, new_point])
            break  # Move to the next point

         attempts += 1

      if attempts == max_attempts:
         print(f"Warning: Could not place target point {i + 1} after {max_attempts} attempts. Stopping at {points.shape[0]} points.\n")
         break

   # --- Display generated points and their distances apart ---
   # print(f'\nGenerated points: {points}\n')
   # distances = np.linalg.norm(points[:,np.newaxis] - points, axis=2)
   # print(f'Pairwise distances between points: {distances}\n')

   return points


# %% test

if __name__ == '__main__':
   x = True