import numpy as np

def viscous(vessel):
    """
    Placeholder function that computes or returns viscous damping (Bv)
    for the vessel. You must implement or replace with your actual code.
    """
    # For now, just return an empty structure or placeholder array.
    Bv_placeholder = {}
    return Bv_placeholder

def plot_speedterms(vessel, Anew, Bnew, Bv):
    """
    Placeholder function that plots the raw data in MSS axes.
    Implement or replace with your custom plotting routine.
    """
    print("[DEBUG] plot_speedterms(...) called")

def ABCtransform(vessel, striptheory='veres', plot_flag=0):
    """
    Python version of ABCtransform.m

    Parameters
    ----------
    vessel : dict
        The MSS vessel structure, which must contain at least:
          - vessel['freqs'] (array of wave frequencies)
          - vessel['velocities'] (array of forward speeds)
          - vessel['A'] and vessel['B'], each shaped [6,6,Nfreqs,Nspeeds]
          - vessel['main']['nabla'], vessel['main']['Lpp'], vessel['main']['rho']
    striptheory : str, optional
        E.g. 'veres'. If 'veres', we do the special transformations for A11/B11.
    plot_flag : int, optional
        1 to plot the raw data in MSS axes (calls plot_speedterms).
        0 (or anything else) => no plotting.

    Returns
    -------
    Anew : ndarray
        Transformed added-mass array.
    Bnew : ndarray
        Transformed damping array.
    Bv   : dict or ndarray
        Viscous damping structure (as returned by `viscous(...)`).
    """

    # Extract relevant data
    freqs = vessel['freqs']            # shape (Nfreqs,) or similar
    speeds = vessel['velocities']      # shape (Nspeeds,) or similar
    A = vessel['A']                    # shape (6,6,Nfreqs,Nspeeds)
    B = vessel['B']                    # shape (6,6,Nfreqs,Nspeeds)

    Nfreqs = len(freqs)
    Nspeeds = len(speeds)

    # We will create copies for Anew and Bnew
    Anew = np.copy(A)
    Bnew = np.copy(B)

    # ------------------------------------------------------------------
    # 1) Symmetrize zero-speed A( :, :, :, 0 ) and B( :, :, :, 0 )
    #    to remove numerical noise.
    #    In MATLAB: for i=1..Nfreqs => A_0(1:6,1:6,i) = 0.5*(A_0 + A_0')
    # ------------------------------------------------------------------
    speed0_index = 0  # zero-based => 'speed=1' in MATLAB

    for i in range(Nfreqs):
        # Symmetrize A( :,:, i, speed0_index )
        A_slice = Anew[:, :, i, speed0_index]
        A_sym   = 0.5 * (A_slice + A_slice.T)
        Anew[:, :, i, speed0_index] = A_sym

        # Symmetrize B( :,:, i, speed0_index )
        B_slice = Bnew[:, :, i, speed0_index]
        B_sym   = 0.5 * (B_slice + B_slice.T)
        Bnew[:, :, i, speed0_index] = B_sym

    # ------------------------------------------------------------------
    # 2) If striptheory == 'veres', scale A11, B11 from A22, B22
    #    A11_0 = 2.7*(rho / L^2)*nabla^(5/3)
    #    alpha = A11_0 / A22_0
    #    Then A11(w) = alpha * A22(w), B11(w) = alpha * B22(w)
    # ------------------------------------------------------------------
    if striptheory.lower() == 'veres':
        nabla = vessel['main']['nabla']
        Lpp   = vessel['main']['Lpp']
        rho   = vessel['main']['rho']

        # A11_0 => from the formula
        A11_0 = 2.7 * (rho / (Lpp**2)) * (nabla ** (5/3))

        # A22_0 => from A(2,2,1,1) in MATLAB => zero-based => A[1,1,0,0] in Python
        A22_0 = Anew[1, 1, 0, 0]  # freq=0, speed=0

        alpha = A11_0 / A22_0

        # Scale A(1,1,i,speed), B(1,1,i,speed)
        for s in range(Nspeeds):
            for i in range(Nfreqs):
                # A(1,1,i,s), B(1,1,i,s)
                Anew[0, 0, i, s] = alpha * Anew[1, 1, i, s]
                Bnew[0, 0, i, s] = alpha * Bnew[1, 1, i, s]

    # ------------------------------------------------------------------
    # 3) Viscous damping
    #    We create vessel_new with updated A & B and call `viscous(...)`.
    # ------------------------------------------------------------------
    vessel_new = dict(vessel)  # shallow copy
    vessel_new['A'] = Anew
    vessel_new['B'] = Bnew

    Bv = viscous(vessel_new)

    # ------------------------------------------------------------------
    # 4) Optional plotting
    # ------------------------------------------------------------------
    if plot_flag == 1:
        plot_speedterms(vessel, Anew, Bnew, Bv)

    return Anew, Bnew, Bv
