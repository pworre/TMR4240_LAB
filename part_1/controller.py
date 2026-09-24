"""
Controller template

Students should implement a controller that maps the vessel state and the
full reference to a body-frame wrench. The simulator calls, once per step:

    controller.compute(t, dt, eta, nu, eta_ref, nu_ref, acc_ref) -> tau_d

All generalized vectors are 6-DOF, ordered [surge, sway, heave, roll, pitch,
yaw]. The 3-DOF model uses indices [0, 1, 5]; the remaining components are
zero on input and ignored on output.

Inputs (full loop state and full reference):
    t       : current simulation time [s]
    dt      : time step [s]
    eta     : (6,) vessel NED state [N, E, z, phi, theta, psi]
              (use N = eta[0], E = eta[1], psi = eta[5])
    nu      : (6,) vessel BODY velocities [u, v, w, p, q, r]
              (use u = nu[0], v = nu[1], r = nu[5])
    eta_ref : (6,) NED reference state
              (use N_d = eta_ref[0], E_d = eta_ref[1], psi_d = eta_ref[5])
    nu_ref  : (6,) NED-frame reference velocities
              (use Ndot_d = nu_ref[0], Edot_d = nu_ref[1], psidot_d = nu_ref[5])
    acc_ref : (6,) NED-frame reference accelerations, same layout as nu_ref
              (use for model-based / inertia feedforward)

Output:
    tau_d   : (6,) desired BODY wrench [Fx, Fy, Fz, Mx, My, Mz] (N, Nm)
              (fill in Fx = tau_d[0], Fy = tau_d[1], Mz = tau_d[5];
               leave the other components zero)

Optional hooks the simulator will use IF you define them (safe to omit):
    reset()                                  — called before each run
    apply_external_aw(tau_applied, psi, dt)  — anti-windup with the (6,)
                                               wrench actually applied after
                                               allocation and the actuator
                                               model (ideal in Part 1)
    last_pid_body  : {"P","I","D"} -> (6,) BODY components   (logged)
    int_ned (2,), int_psi (float)            — integrator states (logged)

Constructor contract — the automated checks (``python check.py``, ``pytest``,
``notebooks/part_1_demo.ipynb``) construct your controller as
``DPController()`` with NO arguments, so your final tuned gains must be the
constructor defaults. Tuning only inside ``run_case_part1.py`` will pass your
own runs but fail the checks.
"""
import numpy as np
import pickle
from importlib.resources import files
from simulation.utils import Rz, wrap_angle_pi

class DPController:
    """
    Template for student DP controller.

    Students may implement any type of controller (PID, LQR, backstepping,
    ...). Only compute() is required; everything else is optional.
    """

    def __init__(self, *args, **kwargs):

        # Load vessel data
        pkl = files("mcsimpy.vessel_data.gunnerus") / "parV_RVG3DOF.pkl"
        with open(str(pkl), "rb") as f:
            data = pickle.load(f)
        M_RB, M_A, Dl = data["Mrb"], data["Ma"], data["Dl"]
        M = M_RB + M_A
        M_diag = np.diag(M)[:3]
        D_diag = np.diag(Dl)[:3]
        T_b = np.array([20, 20, 20]) # Bandwidth time constants
        w_b = 2 * np.pi / T_b # Bandwidth frecuencies
        self.zeta = 1.0              # Relative damping factors
        self.wn = w_b / np.sqrt(
            1 - 2*pow(self.zeta, 2) + np.sqrt(
                4*pow(self.zeta, 4) - 4*pow(self.zeta, 2) + 2
        )); # Closed-loop natural frequency [rad/s]
        # PID Gains
        self.Kp = np.diag(M_diag * self.wn**2)
        self.Kd = np.diag(2 * self.zeta * self.wn * M_diag - D_diag)
        self.Ki = self.wn / 8 * self.Kp

        # Integral states
        self.int = np.zeros(3)

        # Anti-windup states
        self.Kaw = np.array(1.0 / np.diag(self.Ki))
        self.tau_cmd = np.zeros(6)

    def reset(self) -> None:
        self.int = np.zeros(3)
        self.tau_cmd = np.zeros(6)

    def compute(
        self,
        t: float,
        dt: float,
        eta: np.ndarray, #Position/attitude	[N, E, z, phi, theta, psi]	NED
        nu: np.ndarray, #Velocity [u, v, w, p, q, r]	BODY
        eta_ref: np.ndarray,
        nu_ref: np.ndarray | None = None,
        acc_ref: np.ndarray | None = None,
    ) -> np.ndarray:
        if nu_ref is None: 
            nu_ref = np.zeros(6)

        # Position error
        e_eta = np.array([
            eta_ref[0] - eta[0],
            eta_ref[1] - eta[1],
            wrap_angle_pi(eta_ref[5] - eta[5])
        ])

        # Define J to rotate from body frame to NED frame
        J = Rz(eta[5])
        nu_ref_3dof = [
            nu_ref[0],
            nu_ref[1],
            nu_ref[5]
        ]
        nu_3dof = np.array([
            nu[0],
            nu[1],
            nu[5]
        ])
        # Reference velocity
        e_nu = J.T @ nu_ref_3dof - nu_3dof

        # Control law
        [Fx, Fy, Mz] = J.T @ (
        self.Kp @ e_eta
        + self.Ki @ self.int
        ) + self.Kd @ e_nu

        # Euler's method (k+1)
        self.int += dt * e_eta # position tracking error

        # Return computed tau_d
        tau_d = np.zeros(6)
        tau_d[[0, 1, 5]] = Fx, Fy, Mz

        self.tau_cmd = tau_d.copy()
        if np.any(np.isnan(tau_d)):
            tau_d = np.zeros(6)

        return tau_d

    def apply_external_aw(
        self,
        tau_applied: np.ndarray,
        psi: float,
        dt: float
    ) -> None:
        """
        Back-calculation anti-windup.

        tau_applied is the actual BODY wrench after allocation and
        actuator saturation.

        The difference between the applied and commanded wrench
        is fed back into the integral states.
        """

        # Saturation/allocation error
        aw_error = tau_applied - self.tau_cmd

        # BODY -> NED rotation
        J = Rz(psi)

        # Translational anti-windup correction
        aw_error_ned = J @ [
            aw_error[0],
            aw_error[1],
            aw_error[5]
        ]

        self.int += self.Kaw * aw_error_ned * dt
