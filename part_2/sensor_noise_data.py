
from run_case_part_2 import bonus_sensor_noise, sim4
import numpy as np


def calculate_metrics(name, logs):
    # Position RMSE
    N = logs.eta[:, 0]
    E = logs.eta[:, 1]
    rmse_pos = np.sqrt(np.mean(N**2 + E**2))

    # Heading RMSE
    psi = logs.eta[:, 5]
    psi_sp = logs.sp[:, 5]

    error = np.arctan2(
        np.sin(psi - psi_sp),
        np.cos(psi - psi_sp)
    )
    rmse_heading = np.rad2deg(np.sqrt(np.mean(error**2)))

    # Thrust activity
    thrust = logs.u_cmd
    changes = np.diff(thrust, axis=0)
    duration = logs.t[-1] - logs.t[0]
    activity = np.sum(np.abs(changes)) / duration / 1000

    print(f"\n{name}")
    print(f"Position RMSE = {rmse_pos:.3f} m")
    print(f"Heading RMSE = {rmse_heading:.3f} deg")
    print(f"Thrust activity = {activity:.3f} kN/s")


# sensor noise ON
results = bonus_sensor_noise()

calculate_metrics(
    "Raw WITH noise",
    results[("closed loop", "raw noisy measurements")]
)

calculate_metrics(
    "Kalman WITH noise",
    results[("closed loop", "observer kalman")]
)


# sensor noise OFF
results_no_noise = sim4()

calculate_metrics(
    "Raw WITHOUT noise",
    results_no_noise[("closed loop", "raw measurements")]
)

calculate_metrics(
    "Kalman WITHOUT noise",
    results_no_noise[("closed loop", "observer kalman")]
)
