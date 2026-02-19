from __future__ import annotations

import math
import random


def generate_positions(num_drones: int, room_min=(-2.5, -2.5, -2.5), room_max=(2.5, 2.5, 2.5),
      min_pair_distance: float = 2.41, wall_margin: float = 1.0, max_attempts: int = 100000) -> list[
   tuple[list[float], list[float]]]:
   """
   Generate (start, target) pairs:

   - room is an axis-aligned box [room_min, room_max]
   - every start is at least `wall_margin` away from every wall
   - pairwise distance between all starts >= min_pair_distance
   - target is simply -start (mirror through origin)
   """

   # Shrink the usable box by wall_margin on each side.
   inner_min = [v + wall_margin for v in room_min]
   inner_max = [v - wall_margin for v in room_max]

   def sample_point() -> list[float]:
      return [random.uniform(lo, hi) for lo, hi in zip(inner_min, inner_max)]

   def dist(a: list[float], b: list[float]) -> float:
      return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)

   starts: list[list[float]] = []
   attempts = 0

   while len(starts) < num_drones and attempts < max_attempts:
      attempts += 1
      p = sample_point()

      # Check distance to all previous starts
      if all(dist(p, q) >= min_pair_distance for q in starts):
         starts.append(p)

   if len(starts) < num_drones:
      raise RuntimeError(f"Failed to place {num_drones} drones with "
                         f"d_min={min_pair_distance} and wall_margin={wall_margin} "
                         f"after {max_attempts} attempts.")

   pattern: list[tuple[list[float], list[float]]] = []
   for s in starts:
      t = [-s[0], -s[1], -s[2]]  # mirror through origin
      pattern.append((s, t))

   return pattern


def main():
   random.seed(0)  # make results reproducible; remove or change if you want variability

   room_min = (-2.5, -2.5, -2.5)
   room_max = (2.5, 2.5, 2.5)
   min_pair_distance = 2.1
   wall_margin = 1.1

   patterns: dict[int, list[tuple[list[float], list[float]]]] = {}
   failed: list[int] = []

   for n in range(2, 23):  # 2..22 -> 2..6 (2.1/1.1) -> 2..7 (2.01/1.01)
      try:
         pattern = generate_positions(num_drones=n, room_min=room_min, room_max=room_max,
               min_pair_distance=min_pair_distance, wall_margin=wall_margin, max_attempts=100000)
         patterns[n] = pattern
         print(f"Successfully generated pattern for N={n}")
      except RuntimeError as e:
         print(f"FAILED for N={n}: {e}")
         failed.append(n)

   print("\n\nPREDEFINED_PATTERNS = {")
   for n in sorted(patterns.keys()):
      print(f"    {n}: [")
      for (start, target) in patterns[n]:
         # Format with limited decimals for readability
         s = [round(v, 3) for v in start]
         t = [round(v, 3) for v in target]
         print(f"        ({s}, {t}),")
      print("    ],")
   print("}\n")

   if failed:
      print(f"Could not generate valid patterns for N={failed} "
            f"with d_min={min_pair_distance}, wall_margin={wall_margin}.")


if __name__ == "__main__":
   main()
