"""Run the Kalman observer on synthetic pose measurements for debugging."""
import numpy as np

from part_2.observer import KalmanFilterObserver


DT = 0.1
DURATION = 600.0

t = np.arange(0.0, DURATION, DT)
measurements = np.zeros((len(t), 6), dtype=float)
measurements[:, 0] = 0.5 * np.sin(0.70 * t) + 0.3 * np.sin(0.95 * t + 1.0)
measurements[:, 1] = 0.6 * np.sin(0.75 * t + 0.5)
measurements[:, 5] = np.deg2rad(1.5) * np.sin(0.80 * t + 2.0)

observer = KalmanFilterObserver()
observer.reset(np.zeros(6))
tau_est = np.zeros(6, dtype=float)

print(f"Synthetic measurement series: {measurements.shape}", flush=True)
for k, eta_measured in enumerate(measurements):
    assert eta_measured.shape == (6,)
    assert tau_est.shape == (6,)
    observer.step(t[k], DT, eta_measured, tau_est)

print("Observer completed the synthetic run.")
