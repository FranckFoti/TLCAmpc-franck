from __future__ import annotations

import io

import numpy as np
from drone_sim.domain.drone import Drone


def _draw_room_wireframe(ax: object, room_min: np.ndarray, room_max: np.ndarray) -> None:
   x0, y0, z0 = room_min.tolist()
   x1, y1, z1 = room_max.tolist()

   # 12 edges of the axis-aligned box
   edges = [  # bottom rectangle
         ((x0, y0, z0), (x1, y0, z0)), ((x1, y0, z0), (x1, y1, z0)), ((x1, y1, z0), (x0, y1, z0)), ((x0, y1, z0), (x0, y0, z0)),  # top rectangle
         ((x0, y0, z1), (x1, y0, z1)), ((x1, y0, z1), (x1, y1, z1)), ((x1, y1, z1), (x0, y1, z1)), ((x0, y1, z1), (x0, y0, z1)),  # vertical edges
         ((x0, y0, z0), (x0, y0, z1)), ((x1, y0, z0), (x1, y0, z1)), ((x1, y1, z0), (x1, y1, z1)), ((x0, y1, z0), (x0, y1, z1))]

   for (xa, ya, za), (xb, yb, zb) in edges:
      ax.plot([xa, xb], [ya, yb], [za, zb], color="black", linewidth=1.0, alpha=0.4)


def _draw_sphere_wireframe(ax: object, center: np.ndarray, radius: float, *, color: str, alpha: float, lw: float, resolution: int = 18) -> None:
   """Draw a simple sphere wireframe."""

   u = np.linspace(0.0, 2.0 * np.pi, resolution)
   v = np.linspace(0.0, np.pi, resolution)
   x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
   y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
   z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))

   ax.plot_wireframe(x, y, z, color=color, linewidth=lw, alpha=alpha, rstride=2, cstride=2)


def _draw_box_wireframe(ax: object, center: np.ndarray, half_extents: np.ndarray, *, color: str, alpha: float, lw: float) -> None:
   """Draw an axis-aligned box wireframe (12 edges) given center and half-extents."""
   cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
   hx, hy, hz = float(half_extents[0]), float(half_extents[1]), float(half_extents[2])
   x0, x1 = cx - hx, cx + hx
   y0, y1 = cy - hy, cy + hy
   z0, z1 = cz - hz, cz + hz
   edges = [# bottom face
         ((x0, y0, z0), (x1, y0, z0)), ((x1, y0, z0), (x1, y1, z0)), ((x1, y1, z0), (x0, y1, z0)), ((x0, y1, z0), (x0, y0, z0)), # top face
         ((x0, y0, z1), (x1, y0, z1)), ((x1, y0, z1), (x1, y1, z1)), ((x1, y1, z1), (x0, y1, z1)), ((x0, y1, z1), (x0, y0, z1)), # vertical edges
         ((x0, y0, z0), (x0, y0, z1)), ((x1, y0, z0), (x1, y0, z1)), ((x1, y1, z0), (x1, y1, z1)), ((x0, y1, z0), (x0, y1, z1)), ]
   for (xa, ya, za), (xb, yb, zb) in edges:
      ax.plot([xa, xb], [ya, yb], [za, zb], color=color, linewidth=lw, alpha=alpha)


def _has_drones_safety_zones(drones: list[Drone]) -> bool:
   return any(d.safety_zone is not None for d in drones)


def _draw_ghost_max_sphere(ax: object, drone: Drone, print_always: bool = False):
   if drone.is_adaptive:
      pos = np.asarray(drone.position(), dtype=float).reshape(3)
      max_r = drone.compute_max_adaptive_radius()
      # only draw if meaningfully larger
      if print_always or max_r > float(drone.safety_zone) + 1e-4:
         _draw_sphere_wireframe(ax, pos, max_r, color=drone.safety_color, alpha=0.12, lw=0.4)


def _draw_trace(ax: object, trace: list[np.ndarray], drone: Drone) -> None:
   # Trace as a dotted/pointed line + a small arrow.
   if trace and len(trace) >= 2:
      T = np.stack([np.asarray(tp, dtype=float).reshape(3) for tp in trace], axis=0)
      ax.plot(T[:, 0], T[:, 1], T[:, 2], color=drone.trace_color, linewidth=1.2, alpha=0.9, linestyle=":", marker=".", markersize=2.5)

      p1 = T[-1]
      d = p1 - T[-2]
      dn = float(np.linalg.norm(d))
      if dn > 1e-9:
         d = d / dn
         arrow_len = 0.35
         ax.quiver(p1[0] - arrow_len * d[0], p1[1] - arrow_len * d[1], p1[2] - arrow_len * d[2], arrow_len * d[0], arrow_len * d[1], arrow_len * d[2],
                   color=drone.trace_color, linewidth=1.0, arrow_length_ratio=0.35)


def _draw_neighbor_links(ax: object, neighbor_links: list[tuple[int, int]], drone_positions: list[np.ndarray]) -> None:
   if neighbor_links:
      for i, j in neighbor_links:
         if i < len(drone_positions) and j < len(drone_positions):
            p1 = np.asarray(drone_positions[i], dtype=float)
            p2 = np.asarray(drone_positions[j], dtype=float)
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color="gray", linewidth=0.8, alpha=0.5, linestyle="--")


def _draw_obstacles(ax: object, obstacles: list[tuple[np.ndarray, np.ndarray]]):
   # Obstacles — draw as 3D wireframe cuboids
   if obstacles:
      for i, (c, he) in enumerate(obstacles):
         center_arr = np.asarray(c, dtype=float).reshape(3)
         he_arr = np.asarray(he, dtype=float).reshape(3)
         label = "obstacles" if i == 0 else None
         _draw_box_wireframe(ax, center_arr, he_arr, color="tab:red", alpha=0.7, lw=1.2)
         if label:
            # Invisible scatter point to register the legend entry
            ax.scatter([float(center_arr[0])], [float(center_arr[1])], [float(center_arr[2])], s=1, c="tab:red", alpha=0.0, label=label)


def render_png(*, room_min: np.ndarray, room_max: np.ndarray, drones: list[Drone], drone_traces: dict[str, list[np.ndarray]],
               obstacles: list[tuple[np.ndarray, np.ndarray]], step_count: int, compute_time_s: float, neighbor_links: list[tuple[int, int]] | None = None,
               admm_iteration_count: int | None = None, admm_converged: bool | None = None, safety_alphas: list[float] | None = None, width: int = 900,
               height: int = 700, dpi: int = 120, elev: float = 20.0, azim: float = -60.0) -> bytes:
   """Render a 3D scene to PNG bytes.

   This is intentionally simple (matplotlib wireframe room + points) to keep the REST API self-contained.
   """

   # Import matplotlib lazily so the simulation core remains lightweight.
   from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
   from matplotlib.figure import Figure

   fig = Figure(figsize=(width / dpi, height / dpi), dpi=dpi)
   _ = FigureCanvas(fig)

   ax = fig.add_subplot(111, projection="3d")

   _draw_room_wireframe(ax, room_min, room_max)

   _draw_obstacles(ax, obstacles)

   # Draw neighbor communication links
   _draw_neighbor_links(ax, neighbor_links, [d.x for d in drones])

   if _has_drones_safety_zones(drones):
      for drone in drones:
         if drone.safety_zone is not None:
            r = drone.radius
            safety_zone = drone.compute_adaptive_radius(drone.velocity())

            pos = np.asarray(drone.position(), dtype=float).reshape(3)
            s_to_print = max(20.0, float(r) * 250.0)  # scale radius to scatter size TODO, check if this complicate stuff is needed
            ax.scatter([pos[0]], [pos[1]], [pos[2]], s=s_to_print, c=[drone.color], depthshade=True, label=drone.drone_id)

            # Safety zones as wireframe spheres.
            alpha_val = drone.alpha if safety_alphas else 0.8
            _draw_sphere_wireframe(ax, pos, radius=safety_zone, color=drone.safety_color, alpha=alpha_val, lw=0.6)
            _draw_trace(ax, drone_traces.get(drone.drone_id, []), drone)

            # Ghost sphere: maximum adaptive radius (only when larger than current zone)
            _draw_ghost_max_sphere(ax, drone, print_always=True)

   ax.set_xlabel("x")
   ax.set_ylabel("y")
   ax.set_zlabel("z")

   ax.set_xlim(float(room_min[0]), float(room_max[0]))
   ax.set_ylim(float(room_min[1]), float(room_max[1]))
   ax.set_zlim(float(room_min[2]), float(room_max[2]))

   # Keep aspect reasonable for non-cubic rooms.
   box = np.asarray(room_max) - np.asarray(room_min)
   if np.all(np.isfinite(box)) and np.all(box > 0):
      ax.set_box_aspect((float(box[0]), float(box[1]), float(box[2])))

   ax.view_init(elev=float(elev), azim=float(azim))
   title = f"t = {int(step_count)} steps | compute = {float(compute_time_s):.2f}s"
   if admm_iteration_count is not None:
      status = "Y" if admm_converged else "!"
      title += f" | ADMM: {admm_iteration_count} iter {status}"
   ax.set_title(title)

   if _has_drones_safety_zones(drones) or obstacles:
      ax.legend(loc="upper right")

   buf = io.BytesIO()
   fig.savefig(buf, format="png", bbox_inches="tight")
   return buf.getvalue()
