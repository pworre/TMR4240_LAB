"""
Closed-loop pole check for the DP controller.

Builds the linearized closed-loop state matrix for the regulation problem
(eta_d = 0, nu_d = 0) at a fixed heading psi0, using the FULL (possibly
coupled, off-diagonal) mass and damping matrices -- not just the diagonal
approximation used to design Kp/Kd/Ki. Comparing the resulting eigenvalues
against your design targets (wn, zeta) tells you two separate things:

  1. How much surge/sway/yaw coupling in the real M, D matrices moves the
     poles away from the clean decoupled design (this is present even at
     psi0 = 0, and is unavoidable with diagonal gains -- it tells you how
     good the decoupling approximation is).
  2. How much the poles drift as psi0 sweeps away from 0 (this should be
     ~flat once the body/NED frame handling in compute() is correct; if it
     drifts a lot with heading, that's a sign a rotation is still wrong).

State vector: x = [eta (3,), nu (3,), xi (3,)]   where xi = integral of eta
Dynamics (regulation, J(psi0) held constant -- valid for the linear check):
    eta_dot = J @ nu
    nu_dot  = M^-1 ( -(D + Kd) @ nu  -  J.T @ Kp @ eta  -  J.T @ Ki @ xi )
    xi_dot  = eta
"""
import numpy as np
from controller import DPController


def Rz(psi: float) -> np.ndarray:
    c, s = np.cos(psi), np.sin(psi)
    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ])


def closed_loop_state_matrix(
    M: np.ndarray,
    D: np.ndarray,
    Kp: np.ndarray,
    Kd: np.ndarray,
    Ki: np.ndarray | None = None,
    psi0: float = 0.0,
) -> np.ndarray:
    """
    Build the linearized 9x9 (or 6x6 without integral action) closed-loop
    state matrix A such that xdot = A @ x, for the Fossen-style law

        tau = -J.T(psi0) @ (Kp @ eta + Ki @ xi) - Kd @ nu

    M, D, Kp, Kd, Ki must all be (3,3); pass Ki=None to check the PD-only
    (6-state) system.
    """
    M = np.asarray(M, dtype=float)
    D = np.asarray(D, dtype=float)
    Kp = np.asarray(Kp, dtype=float)
    Kd = np.asarray(Kd, dtype=float)
    Minv = np.linalg.inv(M)
    J = Rz(psi0)

    if Ki is None:
        # state = [eta, nu]  (6,)
        A = np.zeros((6, 6))
        A[0:3, 3:6] = J
        A[3:6, 0:3] = -Minv @ J.T @ Kp
        A[3:6, 3:6] = -Minv @ (D + Kd)
        return A

    Ki = np.asarray(Ki, dtype=float)
    # state = [eta, nu, xi]  (9,)
    A = np.zeros((9, 9))
    A[0:3, 3:6] = J
    A[3:6, 0:3] = -Minv @ J.T @ Kp
    A[3:6, 3:6] = -Minv @ (D + Kd)
    A[3:6, 6:9] = -Minv @ J.T @ Ki
    A[6:9, 0:3] = np.eye(3)
    return A


def report_poles(A: np.ndarray, label: str = "") -> None:
    """Print eigenvalues plus wn/zeta for each complex pair, time constant
    for each real pole, and flag any unstable (Re >= 0) mode."""
    eigs = np.linalg.eigvals(A)
    eigs = eigs[np.argsort(-eigs.real)]  # slowest-decaying (closest to instability) first

    print(f"--- {label} ---" if label else "---")
    seen = set()
    for i, p in enumerate(eigs):
        if i in seen:
            continue
        unstable_flag = "  <-- UNSTABLE" if p.real >= 0 else ""
        if abs(p.imag) > 1e-9:
            # find its conjugate partner to report as one mode
            for j, q in enumerate(eigs):
                if j != i and j not in seen and np.isclose(q, np.conj(p), atol=1e-6):
                    seen.add(j)
                    break
            wn = abs(p)
            zeta = -p.real / wn if wn > 0 else float("nan")
            print(f"  pole {p:+.5f}  ->  wn = {wn:.5f} rad/s ({2*np.pi/wn:.2f} s period), "
                  f"zeta = {zeta:.3f}{unstable_flag}")
        else:
            tau = -1.0 / p.real if p.real != 0 else float("inf")
            print(f"  pole {p.real:+.5f} (real)  ->  time constant = {tau:.2f} s{unstable_flag}")
        seen.add(i)


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # 1) Load the same vessel data your controller uses.
    # ------------------------------------------------------------------
    import pickle
    from importlib.resources import files

    pkl = files("mcsimpy.vessel_data.gunnerus") / "parV_RVG3DOF.pkl"
    with open(str(pkl), "rb") as f:
        data = pickle.load(f)
    M_full = data["Mrb"] + data["Ma"]   # full (3,3) or (6,6) -- see note below
    D_full = data["Dl"]

    # If these come back as 6x6, slice to the 3 DOF used [surge, sway, yaw]:
    if M_full.shape[0] == 6:
        ix = np.ix_([0, 1, 5], [0, 1, 5])
        M_full = M_full[ix]
        D_full = D_full[ix]

    # ------------------------------------------------------------------
    # 2) Get Kp, Kd, Ki. Easiest: import your actual controller and read
    #    the gains straight off the instance, so this stays in sync with
    #    whatever you tune. Edit this import to match your project layout:
    # ------------------------------------------------------------------
    try:
        ctrl = DPController()
        Kp, Kd, Ki = ctrl.Kp, ctrl.Kd, ctrl.Ki
        print("Loaded Kp/Kd/Ki from DPController() instance.\n")
    except Exception as e:
        # Fallback: recompute directly so the script still runs standalone.
        print(f"(Could not import DPController ({e}); using manual gains below.)\n")
        M_diag = np.diag(M_full)
        D_diag = np.diag(D_full)
        wn = np.array([1/15 * 2*np.pi, 1/15 * 2*np.pi, 1/15 * 2*np.pi])
        zeta = 0.6
        Kp = np.diag(M_diag * wn**2)
        Kd = np.diag(2 * zeta * wn * M_diag - D_diag)
        Ki = np.diag(wn / 10 * np.diag(Kp))

    # ------------------------------------------------------------------
    # 3) Check the poles at psi0 = 0 (should match your design closely).
    # ------------------------------------------------------------------
    A0 = closed_loop_state_matrix(M_full, D_full, Kp, Kd, Ki, psi0=0.0)
    report_poles(A0, label="psi0 = 0 deg, full PID (9 states)")

    A0_pd = closed_loop_state_matrix(M_full, D_full, Kp, Kd, Ki=None, psi0=0.0)
    print()
    report_poles(A0_pd, label="psi0 = 0 deg, PD only (6 states, no integrator)")

    # ------------------------------------------------------------------
    # 4) Sweep heading to check how much the poles drift with psi -- after
    #    your fix this should be close to flat. Large drift = a rotation
    #    somewhere is still wrong.
    # ------------------------------------------------------------------
    print("\n--- pole drift vs heading (dominant mode's wn, zeta) ---")
    for psi_deg in [0, 30, 60, 90, 135, 180]:
        A = closed_loop_state_matrix(M_full, D_full, Kp, Kd, Ki, psi0=np.deg2rad(psi_deg))
        eigs = np.linalg.eigvals(A)
        dominant = eigs[np.argsort(-eigs.real)][0]
        wn = abs(dominant)
        zeta = -dominant.real / wn if wn > 0 else float("nan")
        print(f"  psi0 = {psi_deg:>3} deg:  dominant pole = {dominant:+.5f}, "
              f"wn = {wn:.4f}, zeta = {zeta:.3f}")