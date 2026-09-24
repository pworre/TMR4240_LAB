"""Mandatory Simulations 1-4 as closed-loop checks.

Each test wraps one simulation check from ``simulation/checks.py``.  A
simulation whose required subsystems are still template placeholders is
reported as skipped, not failed.
"""
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest

from simulation.checks import CheckResult, SIMULATION_CHECKS, plot_check_results, run_check


@pytest.mark.parametrize("key", list(SIMULATION_CHECKS), ids=list(SIMULATION_CHECKS))
def test_simulation(key):
    result = run_check(key)
    detail = "\n".join([result.name, *result.details])
    if result.status == "NOT IMPLEMENTED":
        pytest.skip(detail)
    assert result.passed, detail


def test_plot_check_results_creates_figures():
    t = [0.0, 0.1, 0.2]
    logs = SimpleNamespace(
        t=np.asarray(t, dtype=float),
        eta=np.asarray([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.2, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=float),
        nu=np.asarray([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.2, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=float),
        sp=np.asarray([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.2, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=float),
        cmd=np.asarray([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.2, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=float),
        nu_ref=np.asarray([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                           [0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                           [0.2, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=float),
        thruster_names=["T1", "T2", "T3"],
        u=np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float),
        alpha=np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=float),
        tau_d=np.asarray([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=float),
        tau_thr=np.asarray([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=float),
        tau_total=np.asarray([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=float),
        Uc=np.asarray([0.0, 0.0, 0.0], dtype=float),
        beta_c=np.asarray([0.0, 0.0, 0.0], dtype=float),
        cur_body=np.asarray([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=float),
        U_w=np.asarray([0.0, 0.0, 0.0], dtype=float),
        beta_w=np.asarray([0.0, 0.0, 0.0], dtype=float),
        alpha_w=np.asarray([0.0, 0.0, 0.0], dtype=float),
        tau_w6=np.asarray([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=float),
    )
    result = CheckResult("demo simulation", "PASS", ["ok"], {"demo": logs})

    figs = plot_check_results([result], show=False)

    assert len(figs) >= 2
    for fig in figs:
        assert fig.axes
    plt.close("all")
