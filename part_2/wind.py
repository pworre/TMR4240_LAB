"""
Wind template (Project Part 2)

Part 2 extends the Part 1 wind model.  In addition to the mean and slowly
varying wind speed, the wind field must now contain

* a fluctuating **gust** component of the wind speed, generated from a wind
  spectrum of your choice (Harris, Davenport, NORSOK, ...), and
* a **slowly varying wind direction**, without a gust component, bounded to
  at most +/- 5 degrees around the mean direction.

You built such a wind generator in Assignment 2 -- reuse it.  The coefficient
lookup (relative wind -> loads) is unchanged from Part 1; only the ambient
wind field (speed and direction as functions of time) becomes richer.

Students should compute generalized BODY-frame wind loads:
    tau_w6 = [Fx, Fy, Fz, Mx, My, Mz]

The simulator uses the 3-DOF subset [Fx, Fy, Mz] = tau_w6 indices [0, 1, 5]
and calls, once per step:

    wind.step(t, dt, eta, nu) -> (tau_w6, info)

Inputs (full 6-DOF state — use what your model needs):
    t    : current simulation time [s]        (gust spectra, time variation)
    dt   : time step [s]                      (slowly-varying components)
    eta  : (6,) vessel state [N, E, z, phi, theta, psi] in NED
           (heading is eta[5])
    nu   : (6,) vessel BODY velocities [u, v, w, p, q, r]
           (RELATIVE wind: compute the loads from V_rw = V_wind - V_vessel,
            using the horizontal components nu[0], nu[1])

Outputs:
    tau_w6 : (6,) BODY loads
    info   : optional dict for logging, e.g.
             {"U": ambient speed (total), "beta_ned": direction (towards, rad),
              "alpha_body": relative wind angle in BODY (rad),
              "U_mean": mean + slowly varying speed, "U_gust": gust speed}
             The Part 2 checks read "U_mean", "U_gust" and "beta_ned" to
             verify your wind field (gust present, direction within limit),
             so fill them in.
             Return {} (or None) if you do not need it.
             NOTE: "beta_ned" is always the direction the wind blows
             TOWARDS, even when the constructor semantics is "from" —
             convert before logging, do not log the raw constructor value.

Wind coefficient data
---------------------
The vessel wind coefficients C(alpha) = [Cx, Cy, Cz, Cphi, Ctheta, Cpsi] are
provided in `data/wind_coeff.csv` (repository root), tabulated against the relative
wind angle alpha in degrees (0..360). Load them with:

    alpha_deg, C6 = load_wind_coefficients()

The wind loads are then computed as F_wind = U_rw^2 * C(alpha_rw), where U_rw
and alpha_rw are the relative wind speed and angle in the BODY frame.
"""
from pathlib import Path
from typing import Dict, Tuple
import numpy as np

_WIND_COEFF_FILE = Path(__file__).resolve().parent.parent / "data" / "wind_coeff.csv"


def load_wind_coefficients() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the vessel wind coefficient table.

    Returns
    -------
    alpha_deg : (M,) ndarray
        Relative wind angle grid [deg], from 0 to 360.
    C6 : (M, 6) ndarray
        Coefficients [Cx, Cy, Cz, Cphi, Ctheta, Cpsi] at each angle.
    """
    table = np.loadtxt(_WIND_COEFF_FILE, delimiter=",", skiprows=1)
    return table[:, 0], table[:, 1:]


class Wind:
    """Template for the Part 2 wind model.

    Constructor contract -- the automated checks (``python check.py``,
    ``pytest``) and the Part 2 runner construct your model with this
    signature, so keep it working.  It is a superset of the Part 1 contract:
    your Part 1 model can be pasted in and extended.

        Wind(mean_speed, beta, semantics=..., sigma_slow=..., tau_slow=...,
             gust=..., gust_params=..., sigma_dir=..., tau_dir=...,
             dir_limit=..., seed=...)

    Parameters
    ----------
    mean_speed : mean wind speed [m/s].
    beta : mean direction [rad] in NED (0 = North, pi/2 = East).
    semantics : ``"from"`` (default, the usual meteorological convention --
        "wind from south" blows northward) or ``"towards"``.
    sigma_slow : standard deviation of the slowly varying wind speed
        component [m/s] (Part 1; 0 disables it).
    tau_slow : time constant of the slow speed variation [s].
    gust : switch the fluctuating (gust) speed component on.  The gust must
        be zero-mean and follow a wind spectrum; the spectrum and its
        parameters are your choice (see Assignment 2).  ``False`` gives the
        Part 1 wind field, which the checks use to isolate the gust.
    gust_params : dict with the parameters of *your* gust model (spectrum
        type, elevation, turbulence intensity, number of harmonics, ...).
        ``None`` means your own defaults.
    sigma_dir : standard deviation of the slowly varying wind direction
        [rad] (0 disables it).  The direction has NO gust component.
    tau_dir : time constant of the slow direction variation [s].
    dir_limit : hard bound on the direction excursion |beta(t) - beta| [rad];
        the project requires at most 5 degrees.
    seed : random seed for all stochastic components, so runs are
        reproducible.
    """

    def __init__(self, mean_speed: float = 0.0, beta: float = 0.0, *,
                 semantics: str = "from", sigma_slow: float = 0.0,
                 tau_slow: float = 120.0, gust: bool = False,
                 gust_params: Dict | None = None, sigma_dir: float = 0.0,
                 tau_dir: float = 300.0, dir_limit: float = np.deg2rad(5.0),
                 seed: int | None = None):
        # TODO: Store and use the parameters above in step().
        self.mean_speed = float(mean_speed)
        self.beta = float(beta)
        self.semantics = semantics
        self.sigma_slow = float(sigma_slow)
        self.tau_slow = float(tau_slow)
        self.gust = bool(gust)
        self.gust_params = dict(gust_params or {})
        self.sigma_dir = float(sigma_dir)
        self.tau_dir = float(tau_dir)
        self.dir_limit = float(dir_limit)
        self.seed = seed

        # Semantics
        if self.semantics == "from":
            self.beta = (self.beta + np.pi) % (2*np.pi)
        else:
            self.beta = float(self.beta) % (2*np.pi)

        self.random_generator = np.random.default_rng(seed)

        # Stochastic states
        self.U_slow = 0.0
        self.beta_slow = 0.0

        # Wind Coefficient table    
        self.alpha_deg, self.C6 = load_wind_coefficients()

        self.random_generator = np.random.default_rng(seed)

        # Gust params
        self.gust_init()

    def gust_init(self):

        if not self.gust:
            self.gust_frequencies = np.array([])
            self.U_gi = np.array([])
            self.gust_phases = np.array([])
            return

        params = self.gust_params

        self.spectrum_type = str(params.get("spectrum_type", "NORSOK")).upper()
        if self.spectrum_type not in ("NORSOK", "HARRIS"):
            raise ValueError("Spectrum type must be 'NORSOK' or 'HARRIS'!")

        # gust parameters
        n_components = int(params.get("n_components", 100))
        f_min = float(params.get("f_min", 0.001))
        f_max = float(params.get("f_max", 1.0))

        # 10 m standard for NORSOK
        self.elevation = int(params.get("elevation", 10))

        self.L = float(params.get("L", 1800.0))
        self.kappa = float(params.get("kappa", 0.0026))

        # Check valid components
        if n_components < 1:
            raise ValueError("Number of frequency components 'n_components' must be greater than 0!")
        if f_min <= 0 or f_min >= f_max:
            raise ValueError("f_min is less or equal to zero or equal or greater than f_max")

        # Frequency bins
        # Remember we talk in bins, NOT components. Hence, bins = n_components + 1
        bins = np.linspace(f_min, f_max, n_components + 1)
        # Centre frequencies = (lower + upper) / 2
        self.gust_frequencies = (bins[:-1] + bins[1:])
        df = np.diff(bins)

        S = self.wind_spectrum(self.gust_frequencies)

        self.U_g = np.sqrt(2 * S * df)

        self.gust_phases = self.random_generator.uniform(0.0, 2.0 * np.pi, size=n_components)


    def wind_spectrum(self, f: np.ndarray) -> np.ndarray:
        S = 0
        U10 = self.mean_speed

        if self.spectrum_type == "HARRIS":
            S = (4 * self.kappa * self.L * U10 / (2 + f * self.L / U10)**(5/3))

        elif self.spectrum_type == "NORSOK":
            # NORSOK spectrum with elevation z and frequency f
            # Defined p. 115 in Marine Cybernetics
            n = 0.468
            x = 172 * f * (self.elevation / 10)**(2/3) * (U10/10)**(-3/4)
            S = 320.0 * (U10 / 10.0) ** 2 * (self.elevation/10) ** 0.45 / (1.0 + x**n) ** (5.0 / (3.0 * n))

        else:
            raise RuntimeError("Unknown type of wind spectrum!")

        return S

    def gust_speed(self, t: float) -> float:
        if not self.gust:
            return 0.0

        return float(np.sum(self.U_g * np.cos(2 * np.pi * self.gust_frequencies * t + self.gust_phases)))

    def update_slow_speed(self, dt: float):

        if self.sigma_slow == 0.0:
            self.U_slow = 0.0
            return

         w = self.random_generator.normal(0,1)
        self.U_slow += (-self.U_slow / self.tau_slow * dt + np.sqrt(2*self.sigma_slow**2 / self.tau_slow * dt) * w)

    def update_slow_direction(self, dt: float):

        if self.sigma_dir == 0.0 or self.dir_limit == 0.0:
            self.beta_slow = 0.0
            return

        w = self.random_generator.normal()

        # Gauss-Markov principle to go back to origin + standard deviation in direction of gust amplitude
        self.beta_slow += (
            -self.beta_slow / self.tau_dir * dt
            + self.sigma_dir
            * np.sqrt(2.0 * dt / self.tau_dir)
            * w
        )

        # Maximum +/- 5 degree
        if self.beta_slow > self.dir_limit:
            self.beta_slow = self.dir_limit
        elif self.beta_slow < -self.dir_limit:
            self.beta_slow = -self

    def step(
        self,
        t: float,
        dt: float,
        eta: np.ndarray,
        nu: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        psi = eta[5]

        # Slowly varying wind components
        self.update_slow_speed(dt)
        self.update_slow_direction(dt)

        # Ambient wind speed and direction
        U_mean = self.mean_speed + self.U_slow
        U_gust = self.gust_speed(t)

        # Complete Wind Speed
        U_w = U_mean + U_gust

        # Include slow wind direction
        beta_ned = (self.beta + self.beta_slow) % (2 * np.pi)

        # Wind components in NED
        V_w = np.array([
            [U_w * np.cos(beta_ned)],
            [U_w * np.sin(beta_ned)]
        ])

        # Relative wind
        V_rw = np.zeros((2,1))
        R_ned_to_body = np.array([
            [np.cos(psi), np.sin(psi)],
            [-np.sin(psi), np.cos(psi)]
        ])
        V_rw_body = R_ned_to_body @ V_w
        V_rw = V_rw_body - nu[:2].reshape(2,1)
        U_rw = np.sqrt(V_rw[0,0]**2 + V_rw[1,0]**2)

        # Relative wind angle
        alpha_rw = np.arctan2(V_rw[1,0], V_rw[0,0])
        alpha_rw = alpha_rw % (2*np.pi)
        alpha_rw_deg = np.rad2deg(alpha_rw) # index to use in table

        # Wind coefficients
        C_w = np.array([
            np.interp(alpha_rw_deg, self.alpha_deg, self.C6[:, i])
            for i in range(6)
        ])

        # Wind Forces
        tau_w6 = np.zeros(6)
        tau_w6 = U_rw**2 * C_w
        
        info = {"U": float(U_w), "beta_ned": float(beta_ned), "alpha_body": float(alpha_rw),
                "U_mean": float(U_mean), "U_gust": float(U_gust)}
        return tau_w6, info
