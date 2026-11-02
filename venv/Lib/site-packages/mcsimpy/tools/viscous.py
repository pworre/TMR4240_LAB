import numpy as np

def natfrequency(vessel, dof, w0, flag=1):
    """
    Placeholder for natfrequency(vessel,dof,w0,flag).
    The original code picks or adjusts w4 for roll DOF=4, etc.
    For now, we just return w0 unmodified or do a minimal calculation.
    """
    # In your MATLAB code: w4 = natfrequency(vessel,4,w_0,1)
    # Possibly it refines or calculates the natural frequency from w0.
    # Here we just return w0.
    return w0

def plotBv(vessel):
    """
    Placeholder for plotBv(vessel).
    Possibly plots Bv vs frequency or similar.
    """
    print("[DEBUG] plotBv called")

def DPperiods(vessel, flag=1):
    """
    Placeholder for DPperiods(vessel,flag).
    Possibly computes and prints damping periods, natural periods, etc.
    """
    print("[DEBUG] DPperiods called with flag =", flag)

def viscous(vessel,
            T_input=None,
            B44v_input=None,
            alpha=1.0,
            do_plots=True):
    """
    Compute viscous damping matrix Bv in surge, sway, yaw, and additional
    roll damping. This is a Python translation of viscous.m.

    Parameters
    ----------
    vessel : dict
        The MSS vessel structure. Expects at least:
         - vessel['freqs']: 1D array of wave frequencies [rad/s].
         - vessel['MRB']:   6x6 inertia matrix.
         - vessel['A']:     4D array (6,6,Nfreqs,Nspeeds) of added mass
         - vessel['B']:     4D array (6,6,Nfreqs,Nspeeds) of potential damping
         - vessel['C']:     4D array (6,6,Nfreqs,Nspeeds) of restoring
         - (optionally) vessel['Bv']: for storing the result
    T_input : array-like of length 3, optional
        Time constants T for DOF 1,2,6 => surge, sway, yaw.
        If None, defaults to [50, 100, 20].
    B44v_input : float, optional
        Additional viscous ROLL damping at w4.
        If None, uses a default from the code's estimate (B44v_def).
    alpha : float, optional
        Exponential rate in Bvii(w) = beta_i * exp(- alpha * w). Default=1.0
    do_plots : bool, optional
        If True, calls plotBv(...) and DPperiods(...) to mimic the MATLAB end steps.

    Returns
    -------
    Bv : 3D NumPy array, shape (6,6,Nfreqs)
        The viscous damping matrix for each frequency.
    """

    # Unpack relevant data
    w = vessel['freqs']  # 1D array of frequencies
    MRB = vessel['MRB']  # 6x6
    A4D = vessel['A']    # shape [6,6,Nfreqs,Nspeeds]
    C4D = vessel['C']    # shape [6,6,Nfreqs,Nspeeds]

    # We'll assume speed=0 for this logic, i.e. A[..., k, 0], B[..., k, 0] for k in 0..Nfreqs-1
    # as done in the original MATLAB code (which uses "speed=1" in 1-based).
    speed0_index = 0

    Nfreqs = len(w)

    # Build A(k) and B(k) for zero speed from the 4D arrays
    # A(:,:,k,1) => A_ in Python => A[..., k, 0]
    A_ = np.zeros((6,6,Nfreqs))
    for k in range(Nfreqs):
        A_[:,:,k] = A4D[:,:,k,speed0_index]

    # For the restoring matrix G => vessel.C(:,:,Nfreqs,1)
    # => the last frequency, zero-based => C[..., Nfreqs-1, 0]
    # i.e. shape [6,6]
    G = C4D[:,:,Nfreqs-1,speed0_index]

    print("\n-------------------------------------------------------------------")
    print("VISCOUS DAMPING")
    print("-------------------------------------------------------------------")
    print("Viscous damping in surge, sway and yaw: Bvii(w) = beta_i * exp(- alpha * w)")

    if T_input is None:
        # Replicate the MATLAB default if user doesn't provide T_input
        T_input = [50, 100, 20]
    # If you want to emulate the interactive prompt:
    # T_str = input("Specify time constants Ti=[(MRBii+Aii(0))/Bvii(0)] for DOFs 1,2,6 (default=[50 100 20]): ")
    # if T_str.strip() == "":
    #     T_input = [50, 100, 20]
    # else:
    #     T_input = list(map(float, T_str.split()))

    # Compute beta_i for surge(1), sway(2), yaw(6) => zero-based => (0,1,5)
    #  MRB(1,1)+A(1,1,1) => Python => MRB[0,0] + A_[0,0,0]
    #  T_input(1) => T_input[0], etc.
    beta_i = np.zeros(3)

    beta_i[0] = (MRB[0,0] + A_[0,0,0]) / T_input[0]   # Surge
    beta_i[1] = (MRB[1,1] + A_[1,1,0]) / T_input[1]   # Sway
    beta_i[2] = (MRB[5,5] + A_[5,5,0]) / T_input[2]   # Yaw

    print("")
    print("Roll damping: B44_total(w4) = B44(w4) + B44v(w4) at w4.")
    print("The relative damping ratio zeta4 is specified such that:")
    print("B44_total(w4) = 2 * zeta4 * w4 * (I_x + A44(w4)).")

    #  Natural period for DOF=4 => w_0 = sqrt(G(4,4)/(MRB(4,4)+A_(4,4))).
    #  In Python => G[3,3], MRB[3,3], A_[3,3,k], but we pick k=some freq, or k=0?
    #  The code uses A(4,4) => let's pick A_(3,3,0] or do we do an average?
    #  The code picks A(4,4) from the zero freq or directly from "A(4,4)" => we interpret that as A_[3,3,0].
    w_0 = np.sqrt(G[3,3] / (MRB[3,3] + A_[3,3,0]) )

    # Then w4 = natfrequency(vessel,4,w_0,1).
    w4 = natfrequency(vessel, 4, w_0, 1)

    # Interpolate A44, B44 at w4 => A(4,4,:,1) => A_[3,3,:]
    A44_array = A_[3,3,:]  # shape (Nfreqs,)
    # In the original code, B(4,4,:,1) => vessel.B[3,3,k,0], but we can read from vessel.B or from a local B_ if we want.
    B_ = np.zeros((6,6,Nfreqs))
    for k in range(Nfreqs):
        B_[:,:,k] = vessel['B'][:,:,k,speed0_index]
    B44_array = B_[3,3,:]

    # Use np.interp for 1D interpolation
    A44 = np.interp(w4, w, A44_array)
    B44 = np.interp(w4, w, B44_array)

    # total damping zeta(4) => B44_total = 2*zeta4*w4*(Ix + A44(w4))
    M44 = MRB[3,3] + A44
    zeta4 = 0.5 * B44 / M44 * (1.0 / w4)

    # The code uses cutoffs: if zeta(4) < 0.05 => up to 0.25 => default
    if zeta4 < 0.05:
        zeta_new = 0.05
    elif zeta4 < 0.10:
        zeta_new = 0.10
    elif zeta4 < 0.15:
        zeta_new = 0.15
    elif zeta4 < 0.20:
        zeta_new = 0.20
    elif zeta4 < 0.25:
        zeta_new = 0.25
    else:
        zeta_new = zeta4

    B44v_def = 2.0 * (zeta_new - zeta4) * w4 * M44

    print("")
    print(f"Relative damping ratio in roll at w4 = {w4:.2f} rad/s is zeta4 = {zeta4:.4f}")
    print(f"This can be increased to {zeta_new:.2f} by adding viscous damping B44v = {B44v_def:.3g}")
    print("")
    print(f"Potential damping: B44(w4) = {B44:.3g}")

    # if B44v_input not provided => default to B44v_def
    if B44v_input is None:
        Bv44 = B44v_def
    else:
        Bv44 = B44v_input

    # Build the final 3D array Bv => shape (6,6,Nfreqs)
    Bv = np.zeros((6,6,Nfreqs))

    # Fill diagonal for surge(1), sway(2), roll(4), yaw(6)
    # in Python => indices (0,1,3,5)
    for k in range(Nfreqs):
        # Bv(1,1,k) => Bv[0,0,k]
        Bv[0,0,k] = beta_i[0] * np.exp(-alpha * w[k])   # surge
        Bv[1,1,k] = beta_i[1] * np.exp(-alpha * w[k])   # sway
        Bv[3,3,k] = Bv44                                 # roll
        Bv[5,5,k] = beta_i[2] * np.exp(-alpha * w[k])   # yaw

    # For debugging, we store Bv in vessel_new and do the same final steps:
    vessel_new = dict(vessel)  # shallow copy
    vessel_new['Bv'] = Bv

    print("")
    print("VISCOUS DAMPING:")
    if do_plots:
        plotBv(vessel_new)
        DPperiods(vessel_new, 1)

    # Then we do the same with Bv=0 => show baseline potential damping
    vessel_new['Bv'] = 0.0 * Bv
    print("DAMPING FROM HYDRODYNAMIC CODE:")
    if do_plots:
        DPperiods(vessel_new, 1)

    print("")

    return Bv
