"""
Thrust Allocation -- Project Part 2 (constrained allocation)

The simulator calls, once per step:

    allocator.allocate(t, dt, tau_d, u_now, alpha_now) -> (u_cmd, alpha_cmd)

Inputs (full actuator state -- use what your algorithm needs):
    t         : current simulation time [s]
    dt        : time step [s]              (unused: steady-state allocation)
    tau_d     : (6,) desired BODY wrench [Fx, Fy, Fz, Mx, My, Mz]
                (the 3-DOF wrench to allocate is tau_d[[0, 1, 5]]
                 = [Fx, Fy, Mz]; the other components are zero)
    u_now     : current actual thrusts [N]     (sign / branch selection)
    alpha_now : current thruster angles [rad]  (minimise azimuth slewing)

Outputs:
    u_cmd     : signed thrust command for each thruster [N]
    alpha_cmd : thruster angle command for each thruster [rad]

Contract with the checks, exactly as the public checks test it: ``allocate()``
must never command more thrust than a thruster can deliver, so for an
infeasible demand (the check uses 400 kN of sway) every |u_cmd| <= u_max.
*How* you handle an infeasible demand beyond that limit is your design
decision, not a checked criterion -- state your rule and justify it in the
report.  The check also calls ``allocate()`` repeatedly from the initial
actuator state, applies each command through IDEAL actuators and feeds the
resulting actuator state back as ``u_now``/``alpha_now``; after 30 s of such
calls the achieved wrench must equal the requested one.  A steady-state
allocator therefore passes on its first call.  In the simulation the transient
(thrust ramps, 2 rpm azimuth slews) is shaped by the actuator model in
``models/thruster_dynamics.py`` whatever you command, so this allocator does
NOT rate limit its own commands: it is a steady-state allocator and the
actuator dynamics are kept strictly separate from it.

=============================================================================
DESIGN SUMMARY (for the report)
=============================================================================
1. Decision variables (linear problem, constant matrix).
   Each fixed thruster (tunnel) contributes its signed thrust u_T; each
   azimuth thruster j contributes its two body-frame force components
   (Fx_j, Fy_j) instead of (u_j, alpha_j).  For the Table 1 layout

        z = [u_T, Fx_1, Fy_1, Fx_2, Fy_2]^T,        tau = Be z,

        Be = [ 0    1     0     1     0  ]      Fx
             [ 1    0     1     0     1  ]      Fy
             [ x_T  -y_1  x_1  -y_2  x_2 ]      Mz

   Be does not depend on the azimuth angles, so it is constant and
   rank(Be) = 3 (checked in the constructor).  There are no angle
   singularities.

2. Steady-state criterion (weighted least squares, Sordalen 1997).
   With every variable normalised by its thruster's u_max,

        min  1/2 zn^T Q zn          s.t.  Be_n zn = tau_n,
        Q = blockdiag( w_i * (I + rho * n_i n_i^T) )

   * w_i       : per-thruster weight (default 1): minimises the sum of squared
                 UTILISATIONS (u_i/u_max_i)^2, so the load is shared in
                 proportion to capability and no unit is pushed to its limit
                 early.
   * rho n n^T : extra cost on the force component PERPENDICULAR to the
                 azimuth's current axis (n_i = [-sin a_now, cos a_now]).
                 It does not change the delivered wrench; it only picks, among
                 the redundant solutions, the one that needs the least azimuth
                 rotation ("avoid unnecessary azimuth rotations").

3. Thrust limits (hard, in the steady-state solution).
       |u_T| <= u_max_T ,        sqrt(Fx_j^2 + Fy_j^2) <= u_max_j .
   The azimuth limit is a disc (second-order-cone constraint) on the force
   vector, i.e. a limit on the thrust MAGNITUDE, independent of direction.
   The constrained problem is convex; it is solved with SLSQP (scipy).  The
   unconstrained closed form  z = Q^-1 Be^T (Be Q^-1 Be^T)^-1 tau  is tried
   first and accepted whenever it respects every limit (the common case).

4. Infeasible demand -- rule: ONE common scale factor, never per-thruster
   clipping.  If tau cannot be produced within the limits, the allocator
   finds the smallest max-utilisation mu* with Be z = tau and
   |z_i| <= mu* u_max_i (a convex min-max problem) and commands the wrench

        tau_alloc = tau / mu*     (0 < 1/mu* < 1).

   Preserved : the DIRECTION of the full wrench, i.e. the force direction and
               the force : moment ratio; the achieved wrench is the largest
               multiple of the request that is attainable.
   Sacrificed: magnitude (surge, sway and yaw are all reduced by the same
               factor).  The shortfall tau_d - tau_alloc is the allocation
               residual r_tau that the controller / anti-windup must handle.
   Independent clipping of single thrusters, by contrast, would rotate the
   achieved force and moment away from the requested direction.

5. Angle ambiguity (u, alpha) == (-u, alpha + pi).  The force vector of an
   azimuth is converted to a thrust magnitude and a direction
   alpha = atan2(Fy, Fx); the representation needing the SMALLER rotation
   from the actual angle alpha_now is commanded (so a reversal is made by
   reversing the thrust instead of swinging the azimuth 180 deg at 2 rpm).
   A hysteresis band (flip_margin) prevents the branch from chattering when
   the demanded direction is ~90 deg from the current axis.  Below a small
   force (hold_fraction * u_max) the direction is undefined, so the azimuth
   holds alpha_now and only the force component along that axis is commanded.

6. What it does NOT do (kept out on purpose): thrust-ramp and azimuth-slew
   limits.  Those are applied by the actuator model, so the wrench delivered
   to the vessel is B(alpha_actual) u_actual and differs from the steady-state
   allocation during transients.  Quantify that difference through the
   residual  r_tau = tau_c - tau_thr  rather than hiding it in the allocator.

Diagnostics of the last call are available in ``allocator.last_info``
(mode, scale, utilisation, steady-state wrench and residual).
"""
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
from scipy.optimize import minimize
from simulation.utils import wrap_angle_pi
from models.thruster_dynamics import ThrusterConfig


def _is_fixed(th: ThrusterConfig) -> bool:
    """Same rule the actuator model uses: tunnels / zero rotation speed."""
    return (th.rot_speed is None) or (th.rot_speed == 0.0) or (th.kind.lower() == "tunnel")


class ThrustAllocator:
    """Constrained weighted-least-squares thrust allocator (steady state)."""

    # numerical tolerances (normalised units: 1.0 == a thruster at u_max)
    _TOL_FEAS = 1e-9      # accept a candidate as within limits
    _TOL_EQ = 1e-6        # max accepted wrench error of an SLSQP solution
    _EPS_MINMAX = 1e-3    # tie-break weight of the cost in the min-max problem

    def __init__(
        self,
        thrusters: List[ThrusterConfig],
        weights: Optional[Sequence[float]] = None,
        rotation_penalty: float = 1.0,
        hold_fraction: float = 0.001,
        flip_margin: float = np.radians(10.0),
    ):
        """
        Parameters
        ----------
        thrusters        : thruster configurations (Table 1), any number >= 2.
        weights          : per-thruster weight w_i on the squared utilisation
                           (default all 1).  Raise one to use that unit less.
        rotation_penalty : rho >= 0, extra relative cost of the force component
                           perpendicular to an azimuth's current axis
                           (0 switches the rotation-avoidance term off).
        hold_fraction    : an azimuth force below hold_fraction * u_max has no
                           reliable direction; the angle is then held.
        flip_margin      : hysteresis [rad] on the (u, alpha) / (-u, alpha+pi)
                           branch choice.
        """
        self.thrusters = list(thrusters)
        self.n = len(self.thrusters)
        if self.n < 2:
            raise ValueError("at least two thrusters are required")

        self.u_max = np.array([th.u_max for th in self.thrusters], dtype=float)
        self.fixed = np.array([_is_fixed(th) for th in self.thrusters], dtype=bool)
        self.alpha0 = np.array([th.alpha0 for th in self.thrusters], dtype=float)

        w = np.ones(self.n) if weights is None else np.asarray(weights, dtype=float)
        if w.shape != (self.n,) or np.any(w <= 0.0):
            raise ValueError("weights must be positive, one per thruster")
        self.weights = w
        self.rho = float(max(rotation_penalty, 0.0))
        self.hold = float(hold_fraction) * self.u_max
        self.flip_margin = float(flip_margin)

        # ---- allocation variables z and extended configuration matrix Be ----
        cols: List[np.ndarray] = []
        scale: List[float] = []
        self._vars: List[np.ndarray] = []            # variable indices per thruster
        for i, th in enumerate(self.thrusters):
            k0 = len(cols)
            if self.fixed[i]:
                a = float(th.alpha0)
                cols.append(np.array([np.cos(a), np.sin(a),
                                      th.x * np.sin(a) - th.y * np.cos(a)]))
                scale.append(th.u_max)
            else:
                cols.append(np.array([1.0, 0.0, -th.y]))      # Fx_j
                cols.append(np.array([0.0, 1.0, th.x]))       # Fy_j
                scale += [th.u_max, th.u_max]
            self._vars.append(np.arange(k0, len(cols)))
        self.Be = np.column_stack(cols)                       # (3, nv)  [N -> N, Nm]
        self.nv = self.Be.shape[1]
        if np.linalg.matrix_rank(self.Be) < 3:
            raise ValueError("rank(Be) < 3: the layout cannot produce surge, sway and yaw")

        # normalisation: z_n = z / u_max, tau_n = tau / [F0, F0, F0*L0]
        self._zscale = np.array(scale, dtype=float)
        F0 = float(np.max(self.u_max))
        L0 = max(float(np.max([np.hypot(th.x, th.y) for th in self.thrusters])), 1.0)
        self._tscale = np.array([F0, F0, F0 * L0])
        self._A = (self.Be * self._zscale) / self._tscale[:, None]   # (3, nv)

        # state: branch (+1: u>0 at alpha=phi, -1: u<0 at alpha=phi+pi)
        self._sign = np.ones(self.n)
        self._seeded = False
        self._t_prev = -np.inf
        self.last_info: Dict[str, object] = {}

    # ------------------------------------------------------------------ API
    def reset(self) -> None:
        """Forget the branch memory (called automatically if time restarts)."""
        self._sign[:] = 1.0
        self._seeded = False
        self._t_prev = -np.inf

    def configuration_matrix(self, alpha: np.ndarray) -> np.ndarray:
        """B(alpha) of Eq. (3): 3 x n matrix mapping signed thrusts u to [Fx, Fy, Mz]."""
        alpha = np.asarray(alpha, dtype=float).reshape(self.n)
        B = np.zeros((3, self.n))
        for i, th in enumerate(self.thrusters):
            c, s = np.cos(alpha[i]), np.sin(alpha[i])
            B[:, i] = [c, s, th.x * s - th.y * c]
        return B

    def wrench(self, u: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """Steady-state wrench B(alpha) u = [Fx, Fy, Mz]."""
        return self.configuration_matrix(alpha) @ np.asarray(u, dtype=float).reshape(self.n)

    def allocate(
        self,
        t: float,
        dt: float,
        tau_d: np.ndarray,
        u_now: Optional[np.ndarray] = None,
        alpha_now: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = self.n
        if t < self._t_prev:            # a new run reusing this object
            self.reset()
        self._t_prev = float(t)

        # ---- inputs ----
        tau_d = np.asarray(tau_d, dtype=float).ravel()
        if tau_d.size >= 6:
            tau = tau_d[[0, 1, 5]]
        elif tau_d.size == 3:
            tau = tau_d.copy()
        else:
            raise ValueError("tau_d must have 6 (or 3) components")
        a_now = self.alpha0.copy() if alpha_now is None else np.asarray(alpha_now, float).reshape(n)
        u_act = np.zeros(n) if u_now is None else np.asarray(u_now, float).reshape(n)
        if not (np.all(np.isfinite(tau)) and np.all(np.isfinite(a_now)) and np.all(np.isfinite(u_act))):
            # never pass NaN/inf on to the actuators: ask for zero thrust, hold angles
            a_now = np.where(np.isfinite(a_now), a_now, self.alpha0)
            self.last_info = dict(mode="invalid-input", scale=0.0)
            return np.zeros(n), a_now
        if not self._seeded:
            self._sign = np.where(u_act < 0.0, -1.0, 1.0)
            self._seeded = True

        # ---- steady-state allocation (Cartesian variables, normalised) ----
        tau_n = tau / self._tscale
        Q = self._cost_matrix(a_now)
        zn, scale, mode = self._solve(Q, tau_n)

        # ---- back to thrust magnitude / direction ----
        u_cmd, alpha_cmd = self._commands(zn * self._zscale, a_now)

        tau_ss = self.wrench(u_cmd, alpha_cmd)
        self.last_info = dict(
            mode=mode,                                   # wls | constrained | scaled | fallback
            scale=float(scale),                          # achievable fraction of the request (1 = feasible)
            infeasible=bool(scale < 1.0 - 1e-6),
            utilisation=np.abs(u_cmd) / self.u_max,
            tau_requested=tau,
            tau_alloc=tau_ss,                            # B(alpha_cmd) u_cmd
            residual=tau - tau_ss,                       # steady-state r_tau
        )
        return u_cmd, alpha_cmd

    # ------------------------------------------------------- cost / solvers
    def _cost_matrix(self, alpha_ref: np.ndarray) -> np.ndarray:
        """Q = blockdiag(w_i (I + rho n_i n_i^T)) in normalised variables."""
        Q = np.zeros((self.nv, self.nv))
        for i in range(self.n):
            v = self._vars[i]
            if self.fixed[i]:
                Q[v[0], v[0]] = self.weights[i]
            else:
                nrm = np.array([-np.sin(alpha_ref[i]), np.cos(alpha_ref[i])])
                Q[np.ix_(v, v)] = self.weights[i] * (np.eye(2) + self.rho * np.outer(nrm, nrm))
        return Q

    def _util(self, zn: np.ndarray) -> np.ndarray:
        """Utilisation |u_i|/u_max_i of every thruster for normalised variables."""
        return np.array([np.linalg.norm(zn[v]) for v in self._vars])

    def _wls(self, Q: np.ndarray, tau_n: np.ndarray) -> np.ndarray:
        """Closed form min 1/2 z'Qz s.t. A z = tau (no limits): z = Q^-1 A'(A Q^-1 A')^-1 tau."""
        X = np.linalg.solve(Q, self._A.T)                    # Q^-1 A'
        lam = np.linalg.solve(self._A @ X, tau_n)
        return X @ lam

    def _solve(self, Q: np.ndarray, tau_n: np.ndarray):
        z0 = self._wls(Q, tau_n)
        mu0 = float(np.max(self._util(z0)))
        if mu0 <= 1.0 + self._TOL_FEAS:                      # limits respected: done
            return z0, 1.0, "wls"

        # min-max utilisation: smallest mu with A z = tau, |z_i| <= mu
        z1, mu1 = self._minmax(Q, tau_n, z0, mu0)
        if z1 is None:                                       # solver failure: safe scaling
            return z0 / mu0, 1.0 / mu0, "fallback"
        if mu1 >= 1.0 - 1e-6:                                # infeasible: scale, keep direction
            s = 1.0 / max(1.0, mu1)
            return z1 * s, s, "scaled"

        # feasible but the unconstrained optimum violates a limit: constrained WLS
        z2 = self._constrained_wls(Q, tau_n, z1)
        if z2 is None:
            return z1, 1.0, "constrained"                    # strictly feasible min-max point
        return z2, 1.0, "constrained"

    def _group_ineq(self, zn: np.ndarray, mu: Optional[float]):
        """Values/Jacobian of g_i = mu^2 - |z_i|^2  (mu=None -> 1)."""
        ng = self.n
        g = np.empty(ng)
        J = np.zeros((ng, self.nv + (1 if mu is not None else 0)))
        m2 = 1.0 if mu is None else mu * mu
        for i, v in enumerate(self._vars):
            g[i] = m2 - float(zn[v] @ zn[v])
            J[i, v] = -2.0 * zn[v]
            if mu is not None:
                J[i, self.nv] = 2.0 * mu
        return g, J

    def _minmax(self, Q, tau_n, z0, mu0):
        nv, A, eps = self.nv, self._A, self._EPS_MINMAX
        cons = [
            {"type": "eq", "fun": lambda x: A @ x[:nv] - tau_n,
             "jac": lambda x: np.hstack([A, np.zeros((3, 1))])},
            {"type": "ineq", "fun": lambda x: self._group_ineq(x[:nv], x[nv])[0],
             "jac": lambda x: self._group_ineq(x[:nv], x[nv])[1]},
        ]
        f = lambda x: x[nv] + eps * 0.5 * x[:nv] @ Q @ x[:nv]
        g = lambda x: np.concatenate([eps * (Q @ x[:nv]), [1.0]])
        x0 = np.concatenate([z0, [max(mu0, 1e-3)]])
        res = minimize(f, x0, jac=g, constraints=cons, method="SLSQP",
                       bounds=[(None, None)] * nv + [(0.0, None)],
                       options={"maxiter": 200, "ftol": 1e-12})
        z = res.x[:nv]
        if not (np.all(np.isfinite(res.x)) and np.linalg.norm(A @ z - tau_n) < self._TOL_EQ):
            return None, None
        return z, float(np.max(self._util(z)))               # mu actually attained

    def _constrained_wls(self, Q, tau_n, z_start):
        A = self._A
        cons = [
            {"type": "eq", "fun": lambda z: A @ z - tau_n, "jac": lambda z: A},
            {"type": "ineq", "fun": lambda z: self._group_ineq(z, None)[0],
             "jac": lambda z: self._group_ineq(z, None)[1]},
        ]
        res = minimize(lambda z: 0.5 * z @ Q @ z, z_start, jac=lambda z: Q @ z,
                       constraints=cons, method="SLSQP",
                       options={"maxiter": 200, "ftol": 1e-12})
        z = res.x
        ok = (np.all(np.isfinite(z)) and np.linalg.norm(A @ z - tau_n) < self._TOL_EQ
              and np.max(self._util(z)) <= 1.0 + 1e-7)
        return z if ok else None

    # ------------------------------------------------- force -> (u, alpha)
    def _commands(self, F: np.ndarray, a_now: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        u_cmd = np.zeros(self.n)
        a_cmd = np.zeros(self.n)
        half = 0.5 * np.pi
        for i, th in enumerate(self.thrusters):
            v = self._vars[i]
            if self.fixed[i]:
                u_cmd[i] = F[v[0]]
                a_cmd[i] = th.alpha0
                continue
            fx, fy = float(F[v[0]]), float(F[v[1]])
            mag = float(np.hypot(fx, fy))
            if mag < self.hold[i]:
                # direction undefined: hold the angle, command the component along it
                a_cmd[i] = a_now[i]
                u_cmd[i] = fx * np.cos(a_now[i]) + fy * np.sin(a_now[i])
                continue
            phi = np.arctan2(fy, fx)
            d_plus = abs(wrap_angle_pi(phi - a_now[i]))      # rotation needed for (+u, phi)
            if self._sign[i] > 0 and d_plus > half + 0.5 * self.flip_margin:
                self._sign[i] = -1.0                          # (-u, phi+pi) is clearly closer
            elif self._sign[i] < 0 and d_plus < half - 0.5 * self.flip_margin:
                self._sign[i] = 1.0
            if self._sign[i] > 0:
                u_cmd[i], a_cmd[i] = mag, wrap_angle_pi(phi)
            else:
                u_cmd[i], a_cmd[i] = -mag, wrap_angle_pi(phi + np.pi)
        return np.clip(u_cmd, -self.u_max, self.u_max), a_cmd