from __future__ import annotations

import io

import numpy as np


def _draw_room_wireframe(ax: object, room_min: np.ndarray, room_max: np.ndarray) -> None:
   x0, y0, z0 = room_min.tolist()
   x1, y1, z1 = room_max.tolist()

   # 12 edges of the axis-aligned box
   edges = [# bottom rectangle
         ((x0, y0, z0), (x1, y0, z0)), ((x1, y0, z0), (x1, y1, z0)), ((x1, y1, z0), (x0, y1, z0)),
         ((x0, y1, z0), (x0, y0, z0)), # top rectangle
         ((x0, y0, z1), (x1, y0, z1)), ((x1, y0, z1), (x1, y1, z1)), ((x1, y1, z1), (x0, y1, z1)),
         ((x0, y1, z1), (x0, y0, z1)), # vertical edges
         ((x0, y0, z0), (x0, y0, z1)), ((x1, y0, z0), (x1, y0, z1)), ((x1, y1, z0), (x1, y1, z1)),
         ((x0, y1, z0), (x0, y1, z1))]

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


def render_png(*, room_min: np.ndarray, room_max: np.ndarray, drone_positions: list[np.ndarray],
               drone_radii: list[float], drone_safety_zones: list[float], drone_colors: list[object],
               safety_colors: list[object], trace_colors: list[object], drone_traces: list[list[np.ndarray]],
               obstacles: list[tuple[np.ndarray, float]], step_count: int, compute_time_s: float,
               neighbor_links: list[tuple[int, int]] | None = None,
               admm_iteration_count: int | None = None,
               admm_converged: bool | None = None,
               width: int = 900,
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

   # Draw neighbor communication links
   if neighbor_links:
      for i, j in neighbor_links:
         if i < len(drone_positions) and j < len(drone_positions):
            p1 = np.asarray(drone_positions[i], dtype=float)
            p2 = np.asarray(drone_positions[j], dtype=float)
            ax.plot(
               [p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
               color="gray", linewidth=0.8, alpha=0.5, linestyle="--"
            )

   # Drones (and safety zones)
   if drone_positions:
      for i, (p, r, rz, c_drone, c_safety, c_trace, trace) in enumerate(
            zip(drone_positions, drone_radii, drone_safety_zones, drone_colors, safety_colors, trace_colors,
                drone_traces, strict=True)):
         pos = np.asarray(p, dtype=float).reshape(3)

         ax.scatter([pos[0]], [pos[1]], [pos[2]], s=max(20.0, float(r) * 250.0), c=[c_drone], depthshade=True,
                    label=f"drone-{i + 1}")

         # Safety zones as wireframe spheres.
         _draw_sphere_wireframe(ax, pos, float(rz), color=c_safety, alpha=0.8, lw=0.6)

         # Trace as a dotted/pointed line + a small arrow.
         if trace and len(trace) >= 2:
            T = np.stack([np.asarray(tp, dtype=float).reshape(3) for tp in trace], axis=0)
            ax.plot(T[:, 0], T[:, 1], T[:, 2], color=c_trace, linewidth=1.2, alpha=0.9, linestyle=":", marker=".", markersize=2.5)

            p0 = T[-2]
            p1 = T[-1]
            d = p1 - p0
            dn = float(np.linalg.norm(d))
            if dn > 1e-9:
               d = d / dn
               arrow_len = 0.35
               ax.quiver(p1[0] - arrow_len * d[0], p1[1] - arrow_len * d[1], p1[2] - arrow_len * d[2], arrow_len * d[0],
                         arrow_len * d[1], arrow_len * d[2], color=c_trace, linewidth=1.0, arrow_length_ratio=0.35)

   # Obstacles (render centers; radius shown via point size)
   if obstacles:
      C = np.stack([c for c, _r in obstacles], axis=0)
      sizes = [max(20.0, float(r) * 300.0) for _c, r in obstacles]
      ax.scatter(C[:, 0], C[:, 1], C[:, 2], s=sizes, c="tab:red", alpha=0.6, depthshade=True, label="obstacles")

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

   if drone_positions or obstacles:
      ax.legend(loc="upper right")

   buf = io.BytesIO()
   fig.savefig(buf, format="png", bbox_inches="tight")
   return buf.getvalue()
