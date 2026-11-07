import numpy as np
import cvxpy as cp

from part_2.observer import NonlinearPassiveObserver

observer = NonlinearPassiveObserver()

A, B, C = observer.passivity_matrices()

frequencies = np.logspace(-4, 3, 2000)
min_real = float("inf")
worst_frequency = None

for w in frequencies:
    H = C @ np.linalg.solve(
        1j * w * np.eye(A.shape[0]) - A, B
    )

    real_H = (H + H.conj().T) / 2
    smallest = np.linalg.eigvalsh(real_H).min()

    if smallest < min_real:
        min_real = smallest
        worst_frequency = w

print("Minimum real-part eigenvalue:", min_real)
print("Worst frequency:", worst_frequency)

# Scale the bias states
k4 = np.diag(observer.K4)

S = np.diag(np.r_[np.ones(9), k4])
W = np.diag(1.0 / k4)

A_s = np.linalg.solve(S, A @ S)
B_s = np.linalg.solve(S, B)
C_s = W @ C @ S

# Solve scaled KYP problem
n = A_s.shape[0]
P_s = cp.Variable((n, n), symmetric=True)

constraints = [
    P_s >> 1e-8 * np.eye(n),
    A_s.T @ P_s + P_s @ A_s << -1e-8 * np.eye(n),
    B_s.T @ P_s == C_s
]

problem = cp.Problem(cp.Minimize(0), constraints)
problem.solve(solver=cp.CLARABEL)

print("Scaled KYP status:", problem.status)

if P_s.value is not None:
    P_val = P_s.value
    Q_val = -(A_s.T @ P_val + P_val @ A_s)

    min_eig_P = np.linalg.eigvalsh(P_val).min()
    min_eig_Q = np.linalg.eigvalsh(Q_val).min()
    residual = np.linalg.norm(B_s.T @ P_val - C_s)

    passed = (
        problem.status == "optimal"
        and min_eig_P > 1e-8
        and min_eig_Q > 1e-8
        and residual < 1e-6
    )

    print("Min eigenvalue P:", min_eig_P)
    print("Min eigenvalue Q:", min_eig_Q)
    print("Equality residual:", residual)
    print("KYP numerical check passed:", passed)