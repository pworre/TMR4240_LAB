"""Student observer templates and their common simulator interface.

The simulator calls, once per step (pose y only, no velocity):

    observer.step(t, dt, eta_measured, tau_est) -> ObserverEstimate(eta, nu, bias)

    eta_measured : (6,) measured NED pose [N, E, z, phi, theta, psi]
    tau_est      : (6,) desired controller wrench (before thruster dynamics)
                   of the PREVIOUS step (zero at the first step)
    eta, nu      : (6,) low-frequency NED pose and BODY velocity estimates
    bias         : (6,) slowly varying bias estimate (NED force) — may be zeros

A plain ``(eta, nu, bias)`` tuple is accepted as well.  Before every run the
engine calls ``reset(eta0)`` with the TRUE initial pose, so an observer can
start from the vessel's actual position.  Select an implementation with
``Part2SimConfig(observer_type=...)`` (Simulations 4-7 use the
``SELECTED_OBSERVER`` constant of ``run_case_part_2.py``); extra constructor
arguments go through ``Part2SimConfig(observer_kwargs=...)``.  Both classes
intentionally start as pass-through placeholders; students must implement and
tune them, keeping the tuned parameters as constructor defaults because the
checks call ``select_observer(kind)`` without arguments.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pickle
from importlib.resources import files
from scipy.differentiate import jacobian
from models.sensors import SENSOR_ETA_STD
from simulation.utils import Rz, wrap_angle_pi

@dataclass
class ObserverEstimate:
    eta: np.ndarray
    nu: np.ndarray
    bias: np.ndarray


class Observer:
    """Interface shared by all Part 2 observers."""

    name = "observer"

    def reset(self, eta0: np.ndarray | None = None):
        pass

    def step(
        self, t: float, dt: float, eta_measured: np.ndarray, tau_est: np.ndarray
    ) -> ObserverEstimate:
        raise NotImplementedError

    @staticmethod
    def _placeholder(eta_measured: np.ndarray) -> ObserverEstimate:
        return ObserverEstimate(
            eta=np.asarray(eta_measured, dtype=float).reshape(6).copy(),
            nu=np.zeros(6),
            bias=np.zeros(6),
        )


class NonlinearPassiveObserver(Observer):
    """Template for a nonlinear passive wave-filtering observer.

    Suggested states are wave-frequency motion, low-frequency pose, BODY
    velocity, and slowly-varying bias.  Implement the correction and model
    propagation in :meth:`step`.
    """

    name = "nonlinear_passive"



    def __init__(self, *args, **kwargs):
        #First, loading vessel data
        pkl = files("mcsimpy.vessel_data.gunnerus") / "parV_RVG3DOF.pkl"
        with open(str(pkl), "rb") as f:
            data = pickle.load(f)


        #Defining identity matrix and zero matrix
        self.I3 = np.eye(3)
        self.Z3 = np.zeros((3,3))
        
        #Defining states for the NPO
        self.xi_hat = np.zeros(6)
        self.eta_hat = np.zeros(3)
        self.nu_hat = np.zeros(3)
        self.bias_hat = np.zeros(3)


        #Tuneable parameters
        self.T_b = np.diag([1000.0, 1000.0, 1000.0])
        self.T_b_inv = np.linalg.inv(self.T_b)

        self.w_w = 2 * np.pi / 8.0  
        self.zeta_w = 0.1
        self.zeta_n = 1.0
        self.w_c = 1.5


        delta_zeta = self.zeta_n - self.zeta_w

        k1 = -2 * delta_zeta * self.w_c / self.w_w
        k4 = 2 * self.w_w * delta_zeta
        k7 = self.w_c


        #Correction gains
        self.K1 = np.vstack([
            k1 * self.I3,
            k4 * self.I3
        ])

        self.K2 = k7 * self.I3
        self.K3 = np.diag([
            10000.0,
            10000.0,
            1000000.0
        ])

        self.K4 = np.diag([
            100000.0,
            100000.0,
            10000000.0
        ])
        
        self.A_w = np.block([
        [self.Z3, self.I3],
        [(-(self.w_w**2)*self.I3), (-2*self.zeta_w*self.w_w)*self.I3]
        ])

        self.E_w = np.vstack([self.Z3, self.I3])
        self.C_w = np.hstack([self.Z3, self.I3])

        M_RB, M_A, Dl = data["Mrb"], data["Ma"], data["Dl"]
        M = np.diag(np.diag(M_RB + M_A))

        self.M_inv = np.linalg.inv(M)
        self.D = np.diag(np.diag(Dl))

        self.reset()


    def passivity_matrices(self):
        A_o = np.block([
            [self.A_w - self.K1 @ self.C_w, -self.K1, np.zeros((6,3))],
            [-self.K2 @ self.C_w, -self.K2, self.Z3],
            [-self.K3 @ self.C_w, -self.K3, -self.T_b_inv]
        ])

        B_o = np.vstack([
            np.zeros((6, 3)),
            self.I3,
            self.Z3
        ])

        C_o = np.hstack([
            self.K4 @ self.C_w,
            self.K4,
            -self.I3
        ])

        return A_o, B_o, C_o


    @staticmethod
    def J(psi):
        return np.array([
                [np.cos(psi), -np.sin(psi), 0],
                [np.sin(psi), np.cos(psi), 0],
                [0,0,1]
                    ])


    def reset(self, eta0=None):
        self.xi_hat = np.zeros(6)

        if eta0 is not None:
            eta0 = np.asarray(eta0, dtype=float).reshape(6)
            self.eta_hat = eta0[[0, 1, 5]].copy()
        else:
            self.eta_hat = np.zeros(3)

        self.nu_hat = np.zeros(3)
        self.bias_hat = np.zeros(3)
        
    def step(self, t, dt, eta_measured, tau_est):

        y_m = eta_measured[[0, 1, 5]]
        y_hat = self.eta_hat + self.C_w@self.xi_hat 
         
        y_tilde = y_m - y_hat
        y_tilde[2] = wrap_angle_pi(y_tilde[2])

        psi_hat = self.eta_hat[2]

        J_hat = self.J(psi_hat)

        tau_3dof = tau_est[[0, 1, 5]]


        xi_dot_hat = self.A_w @ self.xi_hat + self.K1 @ y_tilde

        eta_dot_hat = J_hat @ self.nu_hat + self.K2 @ y_tilde

        nu_dot_hat = self.M_inv @ (-self.D @ self.nu_hat + J_hat.T @ self.bias_hat + tau_3dof + J_hat.T @ self.K4 @ y_tilde)
        
        bias_dot_hat = -self.T_b_inv@self.bias_hat + self.K3 @ y_tilde


        self.xi_hat += dt * xi_dot_hat
        self.eta_hat += dt * eta_dot_hat
        self.nu_hat += dt * nu_dot_hat
        self.bias_hat += dt * bias_dot_hat

        self.eta_hat[2] = wrap_angle_pi(self.eta_hat[2])

        eta_out = np.zeros(6)
        nu_out = np.zeros(6)
        bias_out = np.zeros(6)

        eta_out[[0, 1, 5]] = self.eta_hat
        nu_out[[0, 1, 5]] = self.nu_hat
        bias_out[[0, 1, 5]] = self.bias_hat

        return ObserverEstimate(
            eta=eta_out,
            nu=nu_out,
            bias=bias_out
        )

class KalmanFilterObserver(Observer):
    """Template for a discrete KF/EKF observer.

    Students define the process/y models and tune Q, R and P0.
    The input ``tau_est`` is the desired controller wrench before thruster
    dynamics.
    """

    name = "kalman"

    def __init__(self, *args, **kwargs):
        # Utility
        self.n = 15                     # Total dimension number
        self.I3 = np.eye(3)
        self.Z3 = np.zeros((3,3))

        # Load vessel data
        pkl = files("mcsimpy.vessel_data.gunnerus") / "parV_RVG3DOF.pkl"
        with open(str(pkl), "rb") as f:
            data = pickle.load(f)
        M_RB, M_A, Dl = data["Mrb"], data["Ma"], data["Dl"]
        M = M_RB + M_A
        self.M_inv = np.linalg.inv(M)
        self.damping = Dl

        # Estimated model parameters
        self.wave_wn = np.array([0.8, 0.8, 0.8])
        self.wave_zeta = np.array([0.1, 0.1, 0.1])
        self.bias_tau = np.array([300, 300, 300])
        # Model and measurement noise
        self.q_wave_accel = np.array([0.02**2, 0.02**2, np.deg2rad(0.02)**2])

        sigma_Fx, sigma_Yx, sigma_Mz = 100 * np.array([2.5, 5, 10])
        self.q_force = 0.5 * np.array([sigma_Fx**2, sigma_Yx**2, sigma_Mz**2])

        self.q_bias = np.array([100.0**2, 100.0**2, 200.0**2])

        self.Q = np.diag(
            np.concatenate([
                self.q_wave_accel,
                self.q_force,
                self.q_bias,
            ])
        )   # Model uncertainty matrix
        self.R = np.diag(SENSOR_ETA_STD[[0, 1, 5]] ** 2) # Measurement uncertainty matrix

        Cw = np.hstack([self.Z3, self.I3]) # Linear wave response
        self.H = np.hstack([Cw, self.I3, self.Z3, self.Z3])
        E_w = np.block([
            [self.Z3],
            [self.I3]
        ])
        self.E = np.block([
            [E_w, np.zeros((6, 6))],
            [self.Z3, self.Z3, self.Z3],
            [self.Z3, self.M_inv, self.Z3],
            [self.Z3, self.Z3, self.I3],
        ])
        self.A_w = np.block([
            [self.Z3,           self.I3],
            [-np.diag(self.wave_wn**2), -np.diag(2.0 * self.wave_zeta * self.wave_wn)],
        ])
        self.P0 = np.diag([
            0.25, 0.25, np.deg2rad(0.25)**2, #WF
            0.04, 0.04, np.deg2rad(0.05)**2, #WF
            0.08, 0.08, np.deg2rad(0.05)**2, #eta
            0.20, 0.20, np.deg2rad(0.05)**2, #nu
            1000, 1000, 1000,                #bias
        ])
        self.P = self.P0.copy()

    def reset(self, eta0: np.ndarray | None = None):
        eta0 = (
            np.zeros(6)
            if eta0 is None
            else np.asarray(eta0)
        )
        self.x = np.zeros((self.n, 1))
        self.x[6:9, 0] = [eta0[0], eta0[1], eta0[5]]
        self.P = self.P0.copy()

    def _continuous_dynamics(
        self, x: np.ndarray, tau_body: np.ndarray,
    ) -> np.ndarray:
        eta = x[6:9]
        nu = x[9:12]
        bias_ned = x[12:15]
        J = Rz(eta[2])

        return np.concatenate((
            self.A_w @ x[:6],
            J @ nu,
            self.M_inv @ (-self.damping @ nu + J.T @ bias_ned + tau_body),
            -bias_ned / self.bias_tau,
        ))

    def step(self, t, dt, eta_measured, tau_est):
        tau_body = tau_est[[0, 1, 5]]

        # System dynamics
        f = self._continuous_dynamics(self.x[:, 0], tau_body).reshape(self.n, 1)
        A = jacobian(
            lambda states: np.apply_along_axis(
                self._continuous_dynamics,
                axis=0,
                arr=states,
                tau_body=tau_body,
            ),
            self.x[:, 0],
        ).df
        
        # Discretize
        F = np.eye(self.n) + dt * A # First order Euler
        Q_d = dt * self.E @ self.Q @ self.E.T

        # Prediction
        x_pred = self.x + dt * f        # First order Euler method with non-linear f
        x_pred[8, 0] = wrap_angle_pi(float(x_pred[8, 0])) # Wrap heading
        P_pred = F @ self.P @ F.T + Q_d # Update predictor with linearized f

        # Finding prediction error
        y = np.array([eta_measured[0], eta_measured[1], eta_measured[5]]).reshape(3, 1)
        z = y - self.H @ x_pred # Prediction error
        z[2, 0] = wrap_angle_pi(float(z[2, 0]))

        S = self.H @ P_pred @ self.H.T + self.R
        L = np.linalg.solve(S, self.H @ P_pred).T

        # Update
        self.x = x_pred + L @ z
        self.x[8, 0] = wrap_angle_pi(float(self.x[8, 0]))
        I_LH = np.eye(self.n) - L @ self.H
        self.P = I_LH @ P_pred @ I_LH.T + L @ self.R @ L.T

        # Return final values
        eta_est = np.zeros(6)
        eta_est[[0, 1, 5]] = self.x[6:9, 0]

        nu_est = np.zeros(6)
        nu_est[[0, 1, 5]] = self.x[9:12, 0]

        bias_est = np.zeros(6)
        bias_est[[0, 1, 5]] = self.x[12:15, 0]
        return ObserverEstimate(
            eta=eta_est,
            nu=nu_est,
            bias=bias_est,
        )



def select_observer(kind: str, **kwargs) -> Observer:
    """Construct one observer without changing simulator code."""
    choices = {
        "nonlinear_passive": NonlinearPassiveObserver,
        "kalman": KalmanFilterObserver,
    }
    try:
        return choices[kind](**kwargs)
    except KeyError as exc:
        raise ValueError(f"observer_type must be one of {tuple(choices)}") from exc

