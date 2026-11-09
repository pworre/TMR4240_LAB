#!/usr/bin/env python3
"""
tb_sweep.py -- sweep the PID bandwidth period T_b through every Part 2
simulation that uses the DP controller, and log tracking error and overshoot.

Put this file in the repository root (next to ``run_case_part_2.py``) and run

    python tb_sweep.py                          # default sweep, all controller sims
    python tb_sweep.py --tb 30 45 60 90         # same T_b on all three DOFs
    python tb_sweep.py --tb-vec 20,20,20 40,20,30 60,60,60    # per-DOF [surge,sway,yaw]
    python tb_sweep.py --sims sim2 sim3 --zeta 0.8 1.0
    python tb_sweep.py --observer nonlinear_passive --jobs 4
    python tb_sweep.py --list                   # list the simulations

What is swept
-------------
``T_b`` is the bandwidth PERIOD in seconds (w_b = 2*pi/T_b), one value per DOF
[surge, sway, yaw].  ``--tb`` applies one value to all three DOFs, ``--tb-vec``
sets all three explicitly.  zeta is the same on all DOFs for each run.  Every
run uses the same environment seeds as the presets, so the disturbance
realisation is identical for every T_b and only the tuning changes.

Which simulations (Part 2 presets in ``run_case_part_2.py``)
------------------------------------------------------------
    sim2     four-corner test, constrained thrusters, no environment, no observer
    sim3     four-corner test in the Simulation 1 environment, raw feedback
    sim4raw  station keeping in the environment, raw feedback   (Simulation 4, stage c)
    sim4obs  station keeping in the environment, SELECTED observer (Simulation 4, stage c)
    sim5     thrust-capability polar sweep (heavy: one run per direction;
             --sim5-step-deg sets the direction spacing, default 30 deg)
    sim6     observer robustness: heavy sea, 1000 s
    sim7     your own showcase (only with --sims sim7)
Simulations 1 and 4(a,b) do not use the controller and are not swept.  The
presets are CALLED, not copied, so edits you make to them are picked up.  Those
that need an observer (sim4obs, sim5, sim6, sim7) use ``--observer`` or else
``SELECTED_OBSERVER`` from ``run_case_part_2.py``, and are skipped, not
guessed, when neither is set.

Metrics (all from the true vessel state in the logs)
----------------------------------------------------
Tracking error   e = eta - eta_ref, the controller's own reference (the output of
                 the reference model), in NED: N and E in m, psi in degrees
                 (wrapped).  Reported as RMS and max per DOF, the horizontal norm
                 sqrt(eN^2 + eE^2), the low-frequency RMS (30 s moving average, the
                 same window as simulation.capability.low_frequency_motion) and the
                 final error (mean over the last 30 s, i.e. has it converged).
Overshoot        per DOF and per set-point leg: the largest excursion beyond the
                 new set-point in the direction of travel, divided by the size of
                 the step, in percent; the table keeps the worst leg.  A leg
                 runs from one set-point change to the next change of ANY
                 set-point, so the cross-coupling of the following manoeuvre is
                 not counted.  It is measured against the COMMANDED set-point,
                 not the smoothed reference.  Only defined where the command changes (sim2, sim3):
                 empty for station keeping.  With waves the response is the 30 s
                 moving average so wave-frequency motion is not counted as
                 overshoot; without waves it is the raw response.
Effort           peak thrust utilisation, % of time a thruster is at its limit,
                 and the RMS wrench shortfall tau_d - tau_thr.  Slowing the
                 controller trades tracking for effort, so report both.
sim5 (capability) worst-direction low-frequency position/heading excursion,
                 mean utilisation and the number of directions inside the watch
                 circle.

Note on axes: the errors are NED (N, E).  They coincide with surge and sway only
at heading 0; the T_b vector is per BODY axis [surge, sway, yaw].

How T_b reaches your controller
-------------------------------
Preferred: let ``DPController.__init__`` take ``T_b`` and ``zeta`` keyword
arguments whose DEFAULTS are your tuned values (the checks still call
``DPController()`` with no arguments):

    def __init__(self, T_b=(40., 20., 30.), zeta=(1., 1., 1.)): ...

If it takes no such arguments, the script builds ``DPController()`` and
overwrites ``zeta, wn, Kp, Kd, Ki`` with the formulas from the project:

    w_b = 2*pi/T_b ;  wn = w_b / sqrt(1 - 2 z^2 + sqrt(4 z^4 - 4 z^2 + 2))
    Kp = M wn^2 ; Kd = 2 z wn M - D ; Ki = wn/10 Kp      (M, D: diagonal, 3 DOF)

with M = M_RB + M_A and D = D_L read from the vessel model.  That is only right
if your compute() uses those attributes and nothing derived from them, so
the script checks that the gains it reads back change with T_b and writes them
to ``sweep_gains.csv`` for you to compare with your controller.

Runs that diverge are aborted as soon as the horizontal tracking error exceeds
``--abort-m`` and logged as ``diverged`` (a diverging DP loop is otherwise very
slow to simulate).

Outputs (in ``--out``, default ``tb_sweep_results/``)
    sweep_summary.csv   one row per (sim, T_b, zeta): the metrics above
    sweep_legs.csv      overshoot per DOF and per set-point leg
    sweep_gains.csv     the gains the controller actually used
    *.png               summary plots (unless --no-plot)
"""
from __future__ import annotations

import os
os.environ.setdefault("MPLBACKEND", "Agg")          # the presets must not open windows

import argparse
import csv
import functools
import inspect
import math
import sys
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
DEFAULT_TB = [20.0, 30.0, 45.0, 60.0, 90.0, 120]     # s, applied to all three DOFs
DEFAULT_ZETA = [1.0]
LF_WINDOW_S = 30.0          # moving-average window, as in simulation.capability
FINAL_WINDOW_S = 30.0       # "final error" = mean error vector over this last window
MIN_STEP_M = 1.0            # smaller set-point changes are not counted as steps
MIN_STEP_DEG = 2.0
ABORT_M = 150.0             # horizontal tracking error that counts as divergence
DOF_IDX = (0, 1, 5)         # 6-DOF indices of the 3 controlled DOFs
DOF = ("N", "E", "psi")

SIMS = {
    "sim2":    dict(desc="four-corner, no environment, no observer", preset="sim2",
                    lf=False, needs_observer=False),
    "sim3":    dict(desc="four-corner + environment, raw feedback", preset="sim3",
                    lf=True, needs_observer=False),
    "sim4raw": dict(desc="station keeping + environment, raw feedback", custom="sim4",
                    use_observer=False, lf=True, needs_observer=False),
    "sim4obs": dict(desc="station keeping + environment, selected observer", custom="sim4",
                    use_observer=True, lf=True, needs_observer=True),
    "sim5":    dict(desc="thrust-capability polar sweep", preset="sim5",
                    capability=True, needs_observer=True),
    "sim6":    dict(desc="heavy sea station keeping, 1000 s", preset="sim6",
                    lf=True, needs_observer=True),
    "sim7":    dict(desc="your own showcase", preset="sim7",
                    lf=True, needs_observer=True),
}
DEFAULT_SIMS = ["sim2", "sim3", "sim4raw", "sim4obs", "sim6"]

SUMMARY_COLS = [
    "sim", "T_b_surge", "T_b_sway", "T_b_yaw", "zeta", "status", "runtime_s",
    "rms_pos_m", "max_pos_m", "rms_pos_lf_m", "final_pos_m",
    "rms_N_m", "rms_E_m", "rms_psi_deg", "max_N_m", "max_E_m", "max_psi_deg",
    "rms_lf_N_m", "rms_lf_E_m", "rms_lf_psi_deg", "final_psi_deg",
    "os_N_pct", "os_E_pct", "os_psi_pct",
    "peak_util_pct", "at_limit_pct", "rms_resid_F_kN", "rms_resid_Mz_kNm",
    "cap_worst_pos_lf_m", "cap_worst_psi_lf_deg", "cap_mean_util_pct",
    "cap_dirs_ok", "cap_dirs",
]
LEG_COLS = ["sim", "T_b_surge", "T_b_sway", "T_b_yaw", "zeta", "dof", "t_start_s",
            "step", "unit", "overshoot_pct", "peak_excursion"]
GAIN_COLS = (["T_b_surge", "T_b_sway", "T_b_yaw", "zeta", "source"]
             + [f"{g}_{d}" for g in ("wn", "Kp", "Kd", "Ki") for d in ("surge", "sway", "yaw")])


class RunDiverged(RuntimeError):
    """Raised inside the controller call when the tracking error blows up."""

    def __init__(self, t: float, err: float):
        super().__init__(f"tracking error {err:.0f} m at t = {t:.0f} s")
        self.t, self.err = t, err


# --------------------------------------------------------------------------
# Small numerical helpers
# --------------------------------------------------------------------------
def wrap(a):
    """Wrap angle(s) to (-pi, pi]."""
    return np.arctan2(np.sin(a), np.cos(a))


def moving_average(x: np.ndarray, dt: float, window: float) -> np.ndarray:
    """Centred moving average, edge-padded (same definition as the repo's)."""
    k = max(int(round(window / dt)), 1)
    if k == 1:
        return np.asarray(x, float).copy()
    pad = k // 2
    xp = np.pad(np.asarray(x, float), (pad, k - 1 - pad), mode="edge")
    c = np.cumsum(np.insert(xp, 0, 0.0))
    return (c[k:] - c[:-k]) / k


def rms(x) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


# --------------------------------------------------------------------------
# Getting T_b and zeta into the controller
# --------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def vessel_diagonals():
    """Diagonal of M = M_RB + M_A and of the linear damping D_L (3 DOF)."""
    from models.gunnerus_3dof import Gunnerus3DOF
    v = Gunnerus3DOF(0.1, "Euler")._v
    try:
        return np.diag(v._Mrb + v._Ma).astype(float), np.diag(v._Dl).astype(float)
    except AttributeError as exc:                      # pragma: no cover
        raise RuntimeError(
            "could not read M and D_L from the vessel model; give your DPController "
            "T_b/zeta keyword arguments so the sweep does not have to") from exc


def bandwidth_gains(T_b, zeta):
    """The project's gain formulas: returns wn, Kp, Kd, Ki as 3-vectors."""
    T_b, zeta = np.asarray(T_b, float), np.asarray(zeta, float)
    M, D = vessel_diagonals()
    w_b = 2.0 * np.pi / T_b
    wn = w_b / np.sqrt(1 - 2 * zeta**2 + np.sqrt(4 * zeta**4 - 4 * zeta**2 + 2))
    Kp = M * wn**2
    Kd = 2 * zeta * wn * M - D
    if np.any(Kd <= 0):
        raise ValueError(f"Kd <= 0 for T_b={T_b}, zeta={zeta}: the bandwidth is lower "
                         "than the linear damping alone already gives")
    return wn, Kp, Kd, wn / 10.0 * Kp


def _store(ctrl, name: str, diag3: np.ndarray):
    """Overwrite a gain attribute keeping its existing shape (3x3, 6x6 or vector)."""
    old = getattr(ctrl, name, None)
    old = None if old is None else np.asarray(old)
    if old is not None and old.shape == (6, 6):
        new = old.copy()
        new[list(DOF_IDX), list(DOF_IDX)] = diag3
    elif old is not None and old.ndim == 1 and name != "wn":
        new = diag3.copy() if old.size == 3 else old
    elif name in ("wn", "zeta"):
        new = diag3.copy()
    else:
        new = np.diag(diag3)
    setattr(ctrl, name, new)


_warned_patch = False


def build_controller(T_b, zeta):
    """Return (controller, how) for the given per-DOF T_b and zeta."""
    global _warned_patch
    from part_2.controller import DPController
    params = inspect.signature(DPController.__init__).parameters
    kwargs, complete = {}, True
    for key, val in (("T_b", T_b), ("zeta", zeta)):
        if key in params:
            kwargs[key] = np.array(val, float)
        else:
            complete = False
    ctrl = DPController(**kwargs)
    if complete:
        return ctrl, "kwargs"
    if not _warned_patch:
        _warned_patch = True
        warnings.warn(
            "DPController.__init__ has no T_b/zeta keyword arguments: overwriting its "
            "zeta, wn, Kp, Kd, Ki with the project formulas.  Check sweep_gains.csv "
            "against your controller, or add T_b/zeta arguments (defaults = your tuned "
            "values) to __init__ and this fallback is not needed.", stacklevel=2)
    wn, Kp, Kd, Ki = bandwidth_gains(T_b, zeta)
    for name, val in (("zeta", np.asarray(zeta, float)), ("wn", wn),
                      ("Kp", Kp), ("Kd", Kd), ("Ki", Ki)):
        _store(ctrl, name, val)
    return ctrl, "patched"


def read_gains(ctrl) -> dict:
    """Diagonal of the gains actually held by the controller (NaN if not found)."""
    out = {}
    for name in ("wn", "Kp", "Kd", "Ki"):
        val = getattr(ctrl, name, None)
        if val is None:
            out[name] = [math.nan] * 3
            continue
        a = np.asarray(val, float)
        d = np.diag(a) if a.ndim == 2 else a.ravel()
        out[name] = [float(d[i]) for i in (DOF_IDX if d.size == 6 else range(3))]
    return out


class DivergenceGuard:
    """Wraps the controller; raises RunDiverged if the loop clearly blows up."""

    def __init__(self, ctrl, limit_m: float):
        self.ctrl, self.limit = ctrl, float(limit_m)

    def compute(self, t, dt, eta, nu, eta_ref, nu_ref=None, acc_ref=None):
        err = float(np.hypot(eta[0] - eta_ref[0], eta[1] - eta_ref[1]))
        if not np.isfinite(err) or err > self.limit:
            raise RunDiverged(float(t), err)
        return self.ctrl.compute(t, dt, eta, nu, eta_ref, nu_ref, acc_ref)

    def __getattr__(self, name):                 # reset(), apply_external_aw(), last_pid_body, ...
        return getattr(self.ctrl, name)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def leg_bounds(cmd: np.ndarray) -> list:
    """Sample indices where a new leg starts: t = 0 and every change of ANY set-point."""
    d = np.abs(np.diff(cmd, axis=0))
    d[:, 2] = np.abs(wrap(np.diff(cmd[:, 2])))
    changed = np.where(d.max(axis=1) > 1e-9)[0] + 1
    return [0] + [int(k) for k in changed] + [len(cmd)]


def overshoot_legs(resp, cmd, eta0, bounds, *, is_angle: bool, min_step: float):
    """Overshoot of one DOF in every leg where its own set-point steps.

    A leg runs from one set-point change to the next change of ANY DOF, so the
    cross-coupling caused by the next manoeuvre is not counted as overshoot.  The
    first leg starts from the initial pose.  Overshoot = largest excursion beyond
    the new set-point in the direction of travel, as a percentage of the step.
    Returns a list of dicts: k0, step, overshoot_pct, peak_excursion.
    """
    legs = []
    for i in range(len(bounds) - 1):
        k0, k1 = bounds[i], bounds[i + 1]
        prev = eta0 if i == 0 else cmd[k0 - 1]
        step = float(wrap(cmd[k0] - prev) if is_angle else cmd[k0] - prev)
        if abs(step) < min_step:
            continue
        dev = resp[k0:k1] - cmd[k0]
        if is_angle:
            dev = wrap(dev)
        peak = float(np.max(np.sign(step) * dev))
        legs.append(dict(k0=k0, step=step, peak_excursion=max(peak, 0.0),
                         overshoot_pct=100.0 * max(peak, 0.0) / abs(step)))
    return legs


def log_metrics(logs, spec: dict, umax: np.ndarray):
    """Tracking error, overshoot and effort from one simulation's logs."""
    t = np.asarray(logs.t, float)
    dt = float(t[1] - t[0])
    eta = np.asarray(logs.eta, float)[:, DOF_IDX]
    sp = np.asarray(logs.sp, float)[:, DOF_IDX]
    cmd = np.asarray(logs.cmd, float)[:, DOF_IDX]
    if not np.all(np.isfinite(eta)):
        raise RunDiverged(float(t[np.argmax(~np.isfinite(eta).all(axis=1))]), math.inf)

    e = eta - sp
    e[:, 2] = wrap(e[:, 2])
    e_lf = np.column_stack([moving_average(e[:, i], dt, LF_WINDOW_S) for i in range(3)])
    ep, ep_lf = np.hypot(e[:, 0], e[:, 1]), np.hypot(e_lf[:, 0], e_lf[:, 1])
    k_fin = max(int(round(FINAL_WINDOW_S / dt)), 1)
    deg = np.degrees

    m = dict(rms_pos_m=rms(ep), max_pos_m=float(ep.max()), rms_pos_lf_m=rms(ep_lf),
             final_pos_m=float(np.hypot(*e[-k_fin:, :2].mean(axis=0))),
             final_psi_deg=float(deg(e[-k_fin:, 2].mean())))
    for i, name in enumerate(DOF):
        sc = 1.0 if i < 2 else 180.0 / np.pi
        unit = "m" if i < 2 else "deg"
        m[f"rms_{name}_{unit}"] = rms(e[:, i]) * sc
        m[f"max_{name}_{unit}"] = float(np.abs(e[:, i]).max()) * sc
        m[f"rms_lf_{name}_{unit}"] = rms(e_lf[:, i]) * sc

    # overshoot against the commanded set-point
    resp = eta.copy()
    resp[:, 2] = np.unwrap(resp[:, 2])
    if spec.get("lf", False):
        resp = np.column_stack([moving_average(resp[:, i], dt, LF_WINDOW_S) for i in range(3)])
    legs, bounds = [], leg_bounds(cmd)
    for i, name in enumerate(DOF):
        is_angle = i == 2
        found = overshoot_legs(resp[:, i], cmd[:, i], eta[0, i], bounds, is_angle=is_angle,
                               min_step=np.radians(MIN_STEP_DEG) if is_angle else MIN_STEP_M)
        m[f"os_{name}_pct"] = max((l["overshoot_pct"] for l in found), default=math.nan)
        for l in found:
            sc = 180.0 / np.pi if is_angle else 1.0
            legs.append(dict(dof=name, t_start_s=float(t[l["k0"]]), step=l["step"] * sc,
                             unit="deg" if is_angle else "m",
                             overshoot_pct=l["overshoot_pct"],
                             peak_excursion=l["peak_excursion"] * sc))

    # effort and limits
    util = np.abs(np.asarray(logs.u, float)) / umax
    resid = (np.asarray(logs.tau_d, float) - np.asarray(logs.tau_thr, float))[:, DOF_IDX]
    m.update(peak_util_pct=100.0 * float(util.max()),
             at_limit_pct=100.0 * float(np.mean(util.max(axis=1) >= 0.995)),
             rms_resid_F_kN=rms(np.hypot(resid[:, 0], resid[:, 1])) / 1e3,
             rms_resid_Mz_kNm=rms(resid[:, 2]) / 1e3)
    return m, legs


def capability_metrics(res) -> dict:
    """Summary of a sim5 CapabilityResult (low-frequency motion, as the watch circle uses)."""
    return dict(cap_worst_pos_lf_m=float(np.max(res.max_pos_dev_lf)),
                cap_worst_psi_lf_deg=float(np.degrees(np.max(res.max_psi_dev_lf))),
                cap_mean_util_pct=float(np.mean(res.utilization)),
                cap_dirs_ok=int(np.sum(res.feasible_lf)), cap_dirs=int(len(res.directions_deg)))


# --------------------------------------------------------------------------
# Running one (simulation, T_b, zeta) task
# --------------------------------------------------------------------------
def _noop(*_a, **_k):
    return None


def _prepare_rc(task: dict):
    """Import run_case_part_2 and switch off everything that draws."""
    import matplotlib
    matplotlib.use("Agg")
    import run_case_part_2 as rc
    rc.standard_plots = _noop
    rc.plot_capability = _noop
    if not getattr(rc, "_tb_sweep_patched", False):
        rc.capability_sweep = functools.partial(rc.capability_sweep, verbose=False)
        rc._tb_sweep_patched = True
    if task.get("observer"):
        rc.SELECTED_OBSERVER = task["observer"]
    return rc


def _run_sim4(rc, use_observer: bool):
    """Simulation 4, stage (c): station keeping in the standard environment."""
    obs = rc.selected_observer() if use_observer else (rc.SELECTED_OBSERVER or "nonlinear_passive")
    cfg = rc.Part2SimConfig(dt=rc.DT, T=600.0, use_observer=use_observer, observer_type=obs)
    return rc.run(cfg, np.zeros(6), rc.environment())


def run_task(task: dict) -> dict:
    """Run one simulation with one tuning.  Never raises: failures become a status."""
    sim, T_b, zeta = task["sim"], tuple(task["T_b"]), float(task["zeta"])
    spec = SIMS[sim]
    row = dict(sim=sim, T_b_surge=T_b[0], T_b_sway=T_b[1], T_b_yaw=T_b[2], zeta=zeta,
               status="ok")
    legs, gains, source = [], None, ""
    t0 = time.time()
    built = []
    try:
        rc = _prepare_rc(task)

        def factory():
            ctrl, how = build_controller(T_b, [zeta] * 3)
            built.append((ctrl, how))
            return DivergenceGuard(ctrl, task["abort_m"])

        rc.DPController = factory
        umax = np.array([th.u_max for th in rc.default_thrusters_part_2()], float)
        if spec.get("capability"):
            result = rc.sim5(step_deg=task["sim5_step_deg"])
            row.update(capability_metrics(result))
        else:
            if "custom" in spec:
                logs = _run_sim4(rc, spec["use_observer"])
            else:
                logs = getattr(rc, spec["preset"])()
            metrics, legs = log_metrics(logs, spec, umax)
            row.update(metrics)
    except RunDiverged as exc:
        row["status"] = f"diverged ({exc})"
    except NotImplementedError as exc:
        row["status"] = f"skipped: {str(exc).splitlines()[0][:90]}"
    except Exception as exc:                                   # keep the sweep going
        row["status"] = f"error: {type(exc).__name__}: {exc}"
        if task.get("verbose"):
            traceback.print_exc()
    if built:
        gains, source = read_gains(built[-1][0]), built[-1][1]
    row["runtime_s"] = time.time() - t0
    return dict(row=row, legs=legs, gains=gains, source=source)


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
def _clean(v):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return ""
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.6g}"
    return v


def write_csv(path: Path, cols, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore", restval="")
        w.writeheader()
        for r in rows:
            w.writerow({k: _clean(r.get(k)) for k in cols})


def write_all(out: Path, results: list, order: dict):
    results = sorted(results, key=lambda r: order[(r["row"]["sim"], tuple(
        (r["row"]["T_b_surge"], r["row"]["T_b_sway"], r["row"]["T_b_yaw"])), r["row"]["zeta"])])
    summary, legs, gains, seen = [], [], [], set()
    for r in results:
        row = r["row"]
        key = (row["T_b_surge"], row["T_b_sway"], row["T_b_yaw"], row["zeta"])
        summary.append(row)
        for l in r["legs"]:
            legs.append({**{k: row[k] for k in ("sim", "T_b_surge", "T_b_sway", "T_b_yaw", "zeta")}, **l})
        if r["gains"] and key not in seen:
            seen.add(key)
            g = {k: row[k] for k in ("T_b_surge", "T_b_sway", "T_b_yaw", "zeta")}
            g["source"] = r["source"]
            for name, vals in r["gains"].items():
                for d, v in zip(("surge", "sway", "yaw"), vals):
                    g[f"{name}_{d}"] = v
            gains.append(g)
    write_csv(out / "sweep_summary.csv", SUMMARY_COLS, summary)
    write_csv(out / "sweep_legs.csv", LEG_COLS, legs)
    write_csv(out / "sweep_gains.csv", GAIN_COLS, gains)
    return summary


def _f(v, w=7, d=2):
    return f"{'-':>{w}}" if (v is None or v == "" or not math.isfinite(float(v))) else f"{float(v):{w}.{d}f}"


def print_tables(summary: list):
    for sim in dict.fromkeys(r["sim"] for r in summary):
        rows = [r for r in summary if r["sim"] == sim]
        print(f"\n=== {sim}: {SIMS[sim]['desc']} ===")
        if SIMS[sim].get("capability"):
            print(f"{'T_b [s]':>14} {'zeta':>5} | {'worst LF pos':>12} {'worst LF psi':>12} "
                  f"{'mean util':>9} {'dirs ok':>8} | status")
            for r in rows:
                tb = ",".join(f"{r[k]:g}" for k in ("T_b_surge", "T_b_sway", "T_b_yaw"))
                ok = f"{r['cap_dirs_ok']}/{r['cap_dirs']}" if "cap_dirs" in r else "-"
                print(f"{tb:>14} {r['zeta']:5.2f} | {_f(r.get('cap_worst_pos_lf_m'), 12)} "
                      f"{_f(r.get('cap_worst_psi_lf_deg'), 12)} {_f(r.get('cap_mean_util_pct'), 9, 1)} "
                      f"{ok:>8} | {r['status']}")
            continue
        print(f"{'T_b [s]':>14} {'zeta':>5} | {'RMS err [m]':>11} {'max [m]':>8} {'RMS N':>6} "
              f"{'RMS E':>6} {'RMS psi':>7} | {'OS% N':>6} {'E':>6} {'psi':>6} | "
              f"{'peak u%':>7} {'@lim%':>6} | status")
        for r in rows:
            tb = ",".join(f"{r[k]:g}" for k in ("T_b_surge", "T_b_sway", "T_b_yaw"))
            print(f"{tb:>14} {r['zeta']:5.2f} | {_f(r.get('rms_pos_m'), 11)} {_f(r.get('max_pos_m'), 8)} "
                  f"{_f(r.get('rms_N_m'), 6)} {_f(r.get('rms_E_m'), 6)} {_f(r.get('rms_psi_deg'), 7)} | "
                  f"{_f(r.get('os_N_pct'), 6, 1)} {_f(r.get('os_E_pct'), 6, 1)} {_f(r.get('os_psi_pct'), 6, 1)} | "
                  f"{_f(r.get('peak_util_pct'), 7, 0)} {_f(r.get('at_limit_pct'), 6, 1)} | {r['status']}")


def make_plots(summary: list, out: Path):
    import matplotlib.pyplot as plt
    ok = [r for r in summary if r["status"] == "ok"]
    if not ok:
        return
    labels = list(dict.fromkeys(
        (r["T_b_surge"], r["T_b_sway"], r["T_b_yaw"]) for r in summary))
    zetas = sorted({r["zeta"] for r in summary})

    def fig_for(metric_keys, ylabel, fname, sims):
        fig, ax = plt.subplots(figsize=(8.5, 4.5))
        for sim in sims:
            for z in zetas:
                pts = []
                for i, lab in enumerate(labels):
                    rr = [r for r in ok if r["sim"] == sim and r["zeta"] == z
                          and (r["T_b_surge"], r["T_b_sway"], r["T_b_yaw"]) == lab]
                    vals = [float(rr[0][k]) for k in metric_keys
                            if rr and rr[0].get(k) not in (None, "") and math.isfinite(float(rr[0][k]))] if rr else []
                    if vals:
                        pts.append((i, max(vals)))
                if pts:
                    ax.plot(*zip(*pts), marker="o",
                            label=sim + (f" (zeta={z:g})" if len(zetas) > 1 else ""))
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([",".join(f"{v:g}" for v in lab) for lab in labels], rotation=30, ha="right")
        ax.set_xlabel("T_b  [surge, sway, yaw]  [s]")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        if ax.has_data():
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(out / fname, dpi=110)
        plt.close(fig)

    track_sims = [s for s in dict.fromkeys(r["sim"] for r in ok) if not SIMS[s].get("capability")]
    fig_for(["rms_pos_m"], "RMS horizontal tracking error [m]", "tracking_error_vs_Tb.png", track_sims)
    fig_for(["os_N_pct", "os_E_pct", "os_psi_pct"], "worst overshoot over the DOFs [% of step]",
            "overshoot_vs_Tb.png", [s for s in track_sims if SIMS[s].get("preset") in ("sim2", "sim3")])


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------
def _probe(ctrl) -> np.ndarray:
    """Wrench [Fx, Fy, Mz] the controller commands for a fixed 5 m / 5 m / 0.1 rad error."""
    try:
        ctrl.reset(np.zeros(6))
    except TypeError:
        ctrl.reset()
    except AttributeError:
        pass
    eta, zero = np.zeros(6), np.zeros(6)
    eta[0], eta[1], eta[5] = 5.0, 5.0, 0.1
    return np.asarray(ctrl.compute(0.0, 0.1, eta, zero, zero, zero, zero), float)[list(DOF_IDX)]


def check_injection():
    """Make sure T_b really reaches the controller, before spending an hour on a sweep."""
    a, how = build_controller((30.0,) * 3, [1.0] * 3)
    b, _ = build_controller((60.0,) * 3, [1.0] * 3)
    ga, gb = read_gains(a), read_gains(b)
    if all(math.isnan(x) for x in ga["Kp"]):
        print("note: the controller exposes no 'Kp' attribute, so the gains actually used "
              "cannot be read back or logged", file=sys.stderr)
    elif np.allclose(ga["Kp"], gb["Kp"]):
        sys.exit("T_b has no effect on the controller: Kp is identical for T_b = 30 and 60.  "
                 "Give DPController.__init__ T_b and zeta keyword arguments (see the module "
                 "docstring).")
    try:
        pa, pb = _probe(a), _probe(b)
    except Exception as exc:                                   # probing is best effort
        print(f"note: could not probe the controller output ({type(exc).__name__}: {exc})",
              file=sys.stderr)
        return
    if np.any(np.isclose(pa, pb, rtol=1e-9, atol=1e-6)):
        sys.exit("T_b does not change what compute() outputs on every DOF "
                 f"(wrench for a fixed error: T_b=30 -> {np.round(pa)}, T_b=60 -> {np.round(pb)}).  "
                 "compute() must use the T_b-dependent gains; see the module docstring.")
    print(f"controller check ({how}): wrench for a fixed error, T_b=60 / T_b=30 = "
          f"{np.round(pb / pa, 3)} (about 0.25 if the loop is proportional-dominated)")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sims", nargs="+", choices=list(SIMS), default=DEFAULT_SIMS)
    p.add_argument("--tb", nargs="+", type=float, help="T_b [s], the same on all three DOFs")
    p.add_argument("--tb-vec", nargs="+", metavar="S,W,Y",
                   help="explicit T_b per DOF [surge,sway,yaw], e.g. 40,20,30")
    p.add_argument("--zeta", nargs="+", type=float, default=DEFAULT_ZETA)
    p.add_argument("--observer", choices=("nonlinear_passive", "kalman"),
                   help="observer for sim4obs/5/6/7 (default: SELECTED_OBSERVER in run_case_part_2.py)")
    p.add_argument("--sim5-step-deg", type=float, default=30.0,
                   help="direction spacing of the sim5 sweep (preset default is 10)")
    p.add_argument("--abort-m", type=float, default=ABORT_M,
                   help="horizontal tracking error [m] that aborts a run as diverged")
    p.add_argument("--jobs", type=int, default=1, help="parallel worker processes")
    p.add_argument("--out", default="tb_sweep_results")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--verbose", action="store_true", help="print tracebacks of failed runs")
    p.add_argument("--list", action="store_true", help="list the simulations and exit")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.list:
        for name, spec in SIMS.items():
            tag = "" if name in DEFAULT_SIMS else "  (opt-in)"
            print(f"{name:8s} {spec['desc']}{tag}")
        return 0

    tbs = [(40, 20, 30)]
    if args.tb is not None or not args.tb_vec:
        tbs += [(float(v),) * 3 for v in (args.tb if args.tb is not None else DEFAULT_TB)]
    for s in args.tb_vec or []:
        vals = tuple(float(x) for x in s.split(","))
        if len(vals) != 3:
            sys.exit(f"--tb-vec needs three comma-separated values, got {s!r}")
        tbs.append(vals)
    tbs = list(dict.fromkeys(tbs))
    if any(min(t) <= 0 for t in tbs):
        sys.exit("T_b must be positive")

    check_injection()

    tasks, order = [], {}
    for sim in args.sims:
        for tb in tbs:
            for z in args.zeta:
                order[(sim, tb, float(z))] = len(tasks)
                tasks.append(dict(sim=sim, T_b=tb, zeta=float(z), observer=args.observer,
                                  sim5_step_deg=args.sim5_step_deg, abort_m=args.abort_m,
                                  verbose=args.verbose))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(tasks)} runs: {len(args.sims)} simulations x {len(tbs)} T_b x "
          f"{len(args.zeta)} zeta -> {out}/")

    results = []

    def done(res, k):
        r = res["row"]
        print(f"[{k}/{len(tasks)}] {r['sim']:8s} T_b=({r['T_b_surge']:g},{r['T_b_sway']:g},"
              f"{r['T_b_yaw']:g}) zeta={r['zeta']:g}  {r['runtime_s']:6.1f} s  {r['status']}",
              flush=True)
        results.append(res)
        write_all(out, results, order)              # partial results survive a Ctrl-C

    t_start = time.time()
    try:
        if args.jobs > 1:
            with ProcessPoolExecutor(max_workers=args.jobs) as pool:
                futs = [pool.submit(run_task, t) for t in tasks]
                for k, fut in enumerate(as_completed(futs), 1):
                    done(fut.result(), k)
        else:
            for k, t in enumerate(tasks, 1):
                done(run_task(t), k)
    except KeyboardInterrupt:
        print("\ninterrupted: writing the runs that finished", file=sys.stderr)

    summary = write_all(out, results, order)
    print_tables(summary)
    if not args.no_plot:
        make_plots(summary, out)
    print(f"\nfinished in {time.time() - t_start:.0f} s; files in {out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
    