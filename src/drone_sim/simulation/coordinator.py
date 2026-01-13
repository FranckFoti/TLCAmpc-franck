from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from drone_sim.domain.registry import register_coordinator
from drone_sim.physics.linear_kinematics import LinearKinematicsPhysics


def _predict_external_const_vel(
    *, p0: np.ndarray, v0: np.ndarray, dt: float, horizon: int
) -> np.ndarray:
    p0 = np.asarray(p0, dtype=float).reshape(3)
    v0 = np.asarray(v0, dtype=float).reshape(3)
    Ps = [p0 + (k + 1) * float(dt) * v0 for k in range(horizon)]
    return np.stack(Ps, axis=0)


def _has_central_cost(ctrl: object) -> bool:
    return all(
        hasattr(ctrl, name)
        for name in (
            "central_cost",
            "central_bounds",
            "central_initial_guess",
            "horizon",
        )
    )


@register_coordinator("mpc_central")
@dataclass
class CentralMPCGlobalCoordinator:
    """Central coordinator for mixed controllers.

    Optimizes only drones whose controller implements the central-cost interface.

    Safety constraints use owner-only rule:
        dist(owner, other) >= owner.safety_zone + other.radius + safety_buffer

    External-drone prediction default: constant velocity.
    External predictions can be overridden by passing `external_predictions`.
    """

    dt: float
    horizon: int = 12
    safety_buffer: float = 0.5

    # Small lateral acceleration used only for warm-start / initial-guess symmetry breaking.
    # This helps SLSQP escape the "head-on, perfectly collinear" deadlock where the distance
    # constraint gradient is zero in lateral directions at the symmetric point.
    symmetry_break_accel: float = 0.05

    maxiter: int = 120
    ftol: float = 1e-3

    def __post_init__(self) -> None:
        self._phys = LinearKinematicsPhysics(dt=self.dt)
        self._u_prev: dict[str, np.ndarray] = {}

    def _pack(self, u: np.ndarray) -> np.ndarray:
        return np.asarray(u, dtype=float).reshape(-1)

    def _unpack(self, u_flat: np.ndarray, m: int) -> np.ndarray:
        return np.asarray(u_flat, dtype=float).reshape((m, self.horizon, 3))

    def _predict_states(self, xs0: np.ndarray, u: np.ndarray) -> np.ndarray:
        # xs0: (M,6), u: (M,H,3) => X: (M,H,6)
        A = self._phys.A
        B = self._phys.B
        M = xs0.shape[0]
        Xk = np.asarray(xs0, dtype=float).copy()
        X = np.zeros((M, self.horizon, 6), dtype=float)
        for k in range(self.horizon):
            for i in range(M):
                Xk[i] = A @ Xk[i] + B @ u[i, k]
                X[i, k] = Xk[i]
        return X

    def _predict_positions(self, xs0: np.ndarray, u: np.ndarray) -> np.ndarray:
        X = self._predict_states(xs0, u)
        return X[:, :, :3]

    def _apply_symmetry_break(self, u0: np.ndarray) -> np.ndarray:
        eps = float(self.symmetry_break_accel)
        if eps <= 0.0:
            return u0

        M = u0.shape[0]
        if M < 2:
            return u0

        u = np.asarray(u0, dtype=float).copy()

        # Alternate +/- in x acceleration (axis 0) to force a tiny lateral separation.
        for i in range(M):
            sign = 1.0 if (i % 2) == 0 else -1.0
            u[i, :, 0] = u[i, :, 0] + sign * eps

        return u

    def solve_controls(
        self,
        *,
        drone_ids: list[str],
        xs: list[np.ndarray],
        prefs: list[np.ndarray],
        radii: list[float],
        safety_zones: list[float],
        controllers: list[object],
        obstacles: list[tuple[np.ndarray, float]],
        # Optional override trajectories for external drones: id -> (H,3)
        external_predictions: dict[str, np.ndarray] | None = None,
        # External drones state (all drones, including optimized): id -> (p0, v0, radius)
        all_drone_state: dict[str, tuple[np.ndarray, np.ndarray, float]] | None = None,
    ) -> dict[str, np.ndarray]:

        from scipy.optimize import minimize

        n = len(drone_ids)
        idx_opt = [i for i in range(n) if _has_central_cost(controllers[i])]
        idx_ext = [i for i in range(n) if i not in idx_opt]

        # If nothing to optimize, return empty.
        if not idx_opt:
            return {}

        safety_by_id = {did: float(safety_zones[i]) for i, did in enumerate(drone_ids)}
        radii_by_id = {did: float(radii[i]) for i, did in enumerate(drone_ids)}

        opt_ids = [drone_ids[i] for i in idx_opt]
        M = len(idx_opt)

        xs0 = np.stack([np.asarray(xs[i], dtype=float).reshape(6) for i in idx_opt], axis=0)
        prefs0 = np.stack([np.asarray(prefs[i], dtype=float).reshape(3) for i in idx_opt], axis=0)

        # Per-optimized-drone bounds
        umins = []
        umaxs = []
        for i in idx_opt:
            umin, umax = controllers[i].central_bounds()  # type: ignore[attr-defined]
            umins.append(np.asarray(umin, dtype=float).reshape(3))
            umaxs.append(np.asarray(umax, dtype=float).reshape(3))
        umins = np.stack(umins, axis=0)
        umaxs = np.stack(umaxs, axis=0)

        def clip_u(u: np.ndarray) -> np.ndarray:
            return np.clip(u, umins[:, None, :], umaxs[:, None, :])

        # External predictions for constraints
        ext_pred: dict[str, np.ndarray] = {}
        if external_predictions:
            for k, v in external_predictions.items():
                ext_pred[k] = np.asarray(v, dtype=float).reshape((self.horizon, 3))

        if all_drone_state is None:
            all_drone_state = {}

        for i in idx_ext:
            did = drone_ids[i]
            if did in ext_pred:
                continue
            if did not in all_drone_state:
                continue
            p0, v0, _r = all_drone_state[did]
            ext_pred[did] = _predict_external_const_vel(
                p0=p0, v0=v0, dt=self.dt, horizon=self.horizon
            )

        # Warm-start: shift previous solution if available
        u0 = np.zeros((M, self.horizon, 3), dtype=float)
        have_prev = all(did in self._u_prev for did in opt_ids)
        if have_prev:
            prev = np.stack([self._u_prev[did] for did in opt_ids], axis=0)
            u0 = np.concatenate([prev[:, 1:, :], prev[:, -1:, :]], axis=1)
            u0 = self._apply_symmetry_break(u0)
        else:
            # Build per-drone initial guesses and backtrack to feasibility.
            u_guess = []
            for j, i in enumerate(idx_opt):
                ug = controllers[i].central_initial_guess(xs[i], prefs[i])  # type: ignore[attr-defined]
                ug = np.asarray(ug, dtype=float).reshape((-1, 3))

                # Controllers may have their own configured horizon; the coordinator owns the
                # optimization horizon. Trim/pad initial guesses accordingly.
                if ug.shape[0] >= self.horizon:
                    ug = ug[: self.horizon]
                else:
                    pad = np.repeat(ug[-1:, :], self.horizon - ug.shape[0], axis=0)
                    ug = np.concatenate([ug, pad], axis=0)

                u_guess.append(ug)
            u_guess = np.stack(u_guess, axis=0)

            # Apply a tiny symmetry-breaking lateral component to the guess.
            u_guess = self._apply_symmetry_break(u_guess)

            alpha = 1.0
            for _ in range(12):
                u0 = clip_u(alpha * u_guess)
                if (
                    self._constraints(
                        self._pack(u0),
                        xs0=xs0,
                        opt_ids=opt_ids,
                        safety_by_id=safety_by_id,
                        radii_by_id=radii_by_id,
                        P_ext=ext_pred,
                        obstacles=obstacles,
                        amax=np.linalg.norm(umaxs, axis=1),
                    ).min(initial=0.0)
                    >= 0.0
                ):
                    break
                alpha *= 0.5
            else:
                u0 = np.zeros_like(u0)

        bounds = []
        for j in range(M):
            for _k in range(self.horizon):
                for d in range(3):
                    bounds.append((float(umins[j, d]), float(umaxs[j, d])))

        cons = {
            "type": "ineq",
            "fun": lambda u_flat: self._constraints(
                u_flat,
                xs0=xs0,
                opt_ids=opt_ids,
                safety_by_id=safety_by_id,
                radii_by_id=radii_by_id,
                P_ext=ext_pred,
                obstacles=obstacles,
                amax=np.linalg.norm(umaxs, axis=1),
            ),
        }

        res = minimize(
            lambda u_flat: self._cost(
                u_flat,
                xs0=xs0,
                prefs0=prefs0,
                controllers=[controllers[i] for i in idx_opt],
                clip_u=clip_u,
            ),
            self._pack(u0),
            method="SLSQP",
            bounds=bounds,
            constraints=[cons],
            options={"maxiter": int(self.maxiter), "ftol": float(self.ftol), "disp": False},
        )

        u_opt = clip_u(self._unpack(res.x, M))
        for did, u_seq in zip(opt_ids, u_opt, strict=True):
            self._u_prev[did] = u_seq

        return {did: u_opt[k, 0].copy() for k, did in enumerate(opt_ids)}

    def _cost(
        self,
        u_flat: np.ndarray,
        *,
        xs0: np.ndarray,
        prefs0: np.ndarray,
        controllers: list[object],
        clip_u,
    ) -> float:
        u = clip_u(self._unpack(u_flat, xs0.shape[0]))
        total = 0.0
        for i in range(xs0.shape[0]):
            total += float(controllers[i].central_cost(u[i], xs0[i], prefs0[i]))  # type: ignore[attr-defined]
        return float(total)

    def _constraints(
        self,
        u_flat: np.ndarray,
        *,
        xs0: np.ndarray,
        opt_ids: list[str],
        safety_by_id: dict[str, float],
        radii_by_id: dict[str, float],
        P_ext: dict[str, np.ndarray],
        obstacles: list[tuple[np.ndarray, float]],
        amax: np.ndarray,
    ) -> np.ndarray:
        """Inequality constraints c(u) >= 0 using owner-only safety-zone rule.

        For each optimized drone A and any other object B:
            ||p_A - p_B|| >= A.safety_zone + B.radius + safety_buffer
        """

        # Build predicted state/position/velocity for optimized drones
        M = xs0.shape[0]
        u = self._unpack(u_flat, M)
        X_opt = self._predict_states(xs0, u)
        P_opt = X_opt[:, :, :3]
        # V_opt = X_opt[:, :, 3:]

        vals: list[float] = []

        # Optimized vs optimized (pairwise): add asymmetric constraints for both owners.
        for kk in range(self.horizon):
            for i in range(M):
                for j in range(i + 1, M):
                    pi = P_opt[i, kk]
                    pj = P_opt[j, kk]

                    d = pi - pj
                    dist = float(np.linalg.norm(d))

                    id_i = opt_ids[i]
                    id_j = opt_ids[j]

                    thresh_i = float(safety_by_id[id_i] + radii_by_id[id_j] + self.safety_buffer)
                    thresh_j = float(safety_by_id[id_j] + radii_by_id[id_i] + self.safety_buffer)

                    vals.append(dist - thresh_i)
                    vals.append(dist - thresh_j)

        # Optimized vs external predictions
        for kk in range(self.horizon):
            for i in range(M):
                pi = P_opt[i, kk]
                id_i = opt_ids[i]
                for other_id, Pj in P_ext.items():
                    dist = float(np.linalg.norm(pi - Pj[kk]))
                    thresh = float(safety_by_id[id_i] + radii_by_id[other_id] + self.safety_buffer)
                    vals.append(dist - thresh)

        # Optimized vs obstacles
        for kk in range(self.horizon):
            for i in range(M):
                pi = P_opt[i, kk]
                id_i = opt_ids[i]
                for c, r in obstacles:
                    c = np.asarray(c, dtype=float).reshape(3)
                    dist = float(np.linalg.norm(pi - c))
                    thresh = float(safety_by_id[id_i] + float(r) + self.safety_buffer)
                    vals.append(dist - thresh)

        return np.asarray(vals, dtype=float)
