from __future__ import annotations

import io

import numpy as np
from drone_sim.domain.drone import Drone
from drone_sim.api.utils.render_helper import draw_room_wireframe, draw_obstacles, draw_neighbor_links, draw_sphere_wireframe, draw_trace, draw_ghost_max_sphere, has_drones_safety_zones

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

   draw_room_wireframe(ax, room_min, room_max)

   draw_obstacles(ax, obstacles)

   # Draw neighbor communication links
   draw_neighbor_links(ax, neighbor_links, [d.x for d in drones])

   if has_drones_safety_zones(drones):
      for drone in drones:
         if drone.safety_zone is not None:
            r = drone.radius
            safety_zone = drone.compute_adaptive_radius(drone.velocity())

            pos = np.asarray(drone.position(), dtype=float).reshape(3)
            s_to_print = max(20.0, float(r) * 250.0)  # scale radius to scatter size TODO, check if this complicate stuff is needed
            ax.scatter([pos[0]], [pos[1]], [pos[2]], s=s_to_print, c=[drone.color], depthshade=True, label=drone.drone_id)

            # Safety zones as wireframe spheres.
            alpha_val = drone.alpha if safety_alphas else 0.8
            draw_sphere_wireframe(ax, pos, radius=safety_zone, color=drone.safety_color, alpha=alpha_val, lw=0.6)
            draw_trace(ax, drone_traces.get(drone.drone_id, []), drone.trace_color)

            # Ghost sphere: maximum adaptive radius (only when larger than current zone)
            draw_ghost_max_sphere(ax, drone, print_always=False)

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

   if has_drones_safety_zones(drones) or obstacles:
      ax.legend(loc="upper right")

   buf = io.BytesIO()
   fig.savefig(buf, format="png", bbox_inches="tight")
   return buf.getvalue()
