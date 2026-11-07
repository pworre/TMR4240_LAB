import numpy as np
import os

def read_veres_WD(filename, disp_flag=0):
    """
    read_veres_WD(filename, disp_flag=0) -> vessel (dict)

    Reads and processes wave-drift data from a Veres (*.re2) output file.

    Parameters
    ----------
    filename : str
        Path to the *.re2 file.
    disp_flag : int, optional
        0 => No console output
        1 => Print run info
        2 => (Plotting omitted in this translation)
        3 => (Plotting omitted in this translation)

    Returns
    -------
    vessel : dict
        A dictionary with fields:
          vessel['driftfrc']:
            - 'amp'[dof] => 3D array (nfreq, nhead_tot, nvel)
            - 'w'        => frequency array
          vessel['headings']   => array of headings (radians)
          vessel['velocities'] => array of speeds
        The dimension dof=1..3 corresponds to surge(1), sway(2), yaw(3).
        We allocate arrays up to dof=6 for consistency with MSS usage, but
        only dof 1..3 are populated.
    """

    vessel = {}
    vessel['driftfrc'] = {}

    try:
        fid = open(filename, 'r')
    except IOError:
        print(f"[ERROR] Unable to open file: {filename}")
        return vessel

    lines = fid.readlines()
    fid.close()

    if len(lines) < 9:
        print("[ERROR] File too short or invalid.")
        return vessel

    # ----------------------------------------------------------------------
    # 1) Parse header lines
    #    First 6 lines => textual run info
    #    Next 3 lines => numeric run info
    # ----------------------------------------------------------------------
    header_text = lines[0:6]
    header_num  = lines[6:9]

    # line 7 => rho, g
    temp_7 = header_num[0].split()
    rho = float(temp_7[0])
    g   = float(temp_7[1])

    # line 8 => LPP, B, T
    temp_8 = header_num[1].split()
    LPP = float(temp_8[0])
    B   = float(temp_8[1])
    T   = float(temp_8[2])

    # line 9 => nvel, nhead, nfreq
    temp_9 = header_num[2].split()
    nvel   = int(temp_9[0])
    nhead  = int(temp_9[1])
    nfreq  = int(temp_9[2])

    # ----------------------------------------------------------------------
    # 2) Read wave-drift data lines
    #    For each velocity, for each heading, we read one line => [vel, heading]
    #    Then for each frequency => one line => [freq, drift_surge, drift_sway, drift_yaw]
    # ----------------------------------------------------------------------
    current_line = 9  # we've consumed 9 lines so far

    # We'll track the velocities in an array
    vels = np.zeros(nvel)
    # We'll track the headings in a separate array,
    # but note in the code the same heading array is used across all velocities.
    # The code sets "headings(headno) = temp(2)" in the loop.
    headings_array = np.zeros(nhead)
    freqs_array    = np.zeros(nfreq)

    # We'll store intermediate data in a structure resp[velno].amp, shaped (nhead*nfreq, 3).
    # Because dof=1..3 => surge, sway, yaw. We only store 3 columns.
    resp = []
    for v_i in range(nvel):
        k = 0  # response index => 0..(nhead*nfreq-1)
        # allocate an array for amplitude data => shape (nhead*nfreq, 3)
        amp_data = np.zeros((nhead * nfreq, 3))

        for headno in range(nhead):
            line_h = lines[current_line].split()
            current_line += 1
            vel_val     = float(line_h[0])  # temp(1)
            heading_val = float(line_h[1])  # temp(2)

            # store velocity for this velno (it might be overwritten each heading, that's how Veres does it)
            if headno == 0:
                vels[v_i] = vel_val
            if v_i == 0:
                # store heading in a separate array (only needed once for all velocities if consistent)
                headings_array[headno] = heading_val

            # read frequencies
            for freqno in range(nfreq):
                line_f = lines[current_line].split()
                current_line += 1
                fr    = float(line_f[0])  # freq
                # drift_surge, drift_sway, drift_yaw
                d_surge = float(line_f[1])
                d_sway  = float(line_f[2])
                d_yaw   = float(line_f[3])

                if headno == 0:
                    # store freq in an array only the first heading block
                    freqs_array[freqno] = fr

                # amplitude scaling => from code:
                #    amp(k,1) = temp(2)*rho*g*B^2 / LPP  => surge
                #    amp(k,2) = temp(3)*rho*g*B^2 / LPP  => sway
                #    amp(k,3) = temp(4)*rho*g*B^2       => yaw
                # We'll store them in amp_data[k, 0..2].
                amp_data[k, 0] = d_surge * rho*g*(B**2)/LPP
                amp_data[k, 1] = d_sway  * rho*g*(B**2)/LPP
                amp_data[k, 2] = d_yaw   * rho*g*(B**2)
                k += 1

        resp.append({
            'amp': amp_data
        })

    # ----------------------------------------------------------------------
    # 3) Check headings => must be 0, 10, 20, ..., 180 for nhead=19
    # ----------------------------------------------------------------------
    if nhead == 19:
        for ii in range(1, 20):  # 1..19
            expected_hdg = 10.0*(ii-1)
            if abs(headings_array[ii-1] - expected_hdg) > 1e-6:
                print("[ERROR] MSS Hydro requires RAO data for headings 0:10:180 deg")
                return {}
    else:
        # If not 19, the code does not do anything special.
        # You can decide whether to enforce or skip.
        pass

    # ----------------------------------------------------------------------
    # 4) Transform data to MSS coordinate frame
    #    - Mirror headings for 180..360 deg
    #    - Change sign for surge (dof=1) and yaw (dof=3) => i.e. dof=0 or dof=2 in 0-based
    # ----------------------------------------------------------------------
    nhead_tot = (nhead - 2)*2 + 2   # e.g. 19 => 36

    # Preallocate: driftfrc.amp[dof] => each is shape (nfreq, nhead_tot, 10) for up to 10 velocities.
    # We'll store 6 dofs for consistency, though dofs 1..3 are the only ones used.
    drift_amp = [np.zeros((nfreq, nhead_tot, 10)) for _ in range(6)]

    # We loop dofno in [1..3] in MATLAB => zero-based => dof_index in [0..2].
    #   dof_index=0 => surge, dof_index=1 => sway, dof_index=2 => yaw
    for dof_index in range(3):  # surge, sway, yaw
        for v_i in range(nvel):
            # amp_data => shape (nhead*nfreq, 3)
            amp_vec = resp[v_i]['amp'][:, dof_index]
            # reshape => (nfreq, nhead) in row-major
            amp_mat = amp_vec.reshape((nfreq, nhead), order='C')

            # Re-order to get correct heading sequence (reverse heading dimension)
            # for head=1..nhead => amp(:, head) = amp_mat(:, nhead+1-head)
            # in Python => amp[:, head_i] = amp_mat[:, nhead-1 - head_i]
            amp_reord = np.zeros_like(amp_mat)
            for head_i in range(nhead):
                amp_reord[:, head_i] = amp_mat[:, nhead - 1 - head_i]

            # If dof in [1,3] => surge,yaw => dof_index=0 or 2 => sign change
            # (the code uses dofno==1||3 in 1-based => 0,2 in 0-based)
            if dof_index in [0, 2]:
                amp_reord = -amp_reord

            # Expand headings from 180..360 deg:
            #   for ii=2..(nhead-1), jj=nhead_tot+2-ii
            #   headings(jj)= -headings(ii)+360
            #   if dofno in [1,3,5] => no sign flip, else => sign flip
            # But in the code:
            #   if dofno==1||3||5 => amp(:,jj)=amp(:,ii)
            #   else => amp(:,jj)=-amp(:,ii)
            #
            # 1-based => dof=1,3,5 => 0-based => 0,2,4 => no sign flip
            # Else => sign flip. So if dof_index in [0,2,4] => no flip, else flip.
            # We'll replicate that carefully.

            amp_final = np.zeros((nfreq, nhead_tot))
            # Copy the first nhead directly
            amp_final[:, 0:nhead] = amp_reord

            for ii in range(1, nhead-1):
                jj = nhead_tot + 1 - (ii+1) - 1  # zero-based indexing
                # if dof_index in [0, 2, 4] => no flip
                if dof_index in [0, 2, 4]:
                    amp_final[:, jj] = amp_reord[:, ii]
                else:
                    amp_final[:, jj] = -amp_reord[:, ii]

            # store final array in drift_amp[dof_index]
            # shape => (nfreq, nhead_tot, v_i)
            drift_amp[dof_index][:, :, v_i] = amp_final

    # We place them in vessel['driftfrc']['amp'][dof], dof=0..5
    vessel['driftfrc']['amp'] = drift_amp
    vessel['driftfrc']['w']   = freqs_array

    # Convert final headings to radians
    # The code re-writes headings in the expansion loop; let's replicate that logic:
    headings_deg_expanded = np.zeros(nhead_tot)
    headings_deg_expanded[0:nhead] = headings_array
    for ii in range(1, nhead-1):
        jj = nhead_tot + 1 - (ii+1) - 1
        headings_deg_expanded[jj] = -headings_array[ii] + 360.0

    vessel['headings']   = headings_deg_expanded * np.pi / 180.0
    vessel['velocities'] = vels

    # ----------------------------------------------------------------------
    # 5) Print run info if disp_flag > 0
    # ----------------------------------------------------------------------
    if disp_flag > 0:
        print("")
        print("-------------------------------------------")
        print("       VERES WAVEDRIFT DATA")
        print("-------------------------------------------")
        print("File header:")
        for i in range(1, 6):
            print(header_text[i].strip())
        print("-------------------------------------------")
        print("Run info:")
        print(" Main particulars:")
        print(f" LPP     : {LPP:.2f} m")
        print(f" Breadth : {B:.2f} m")
        print(f" Draft   : {T:.2f} m")
        print("")
        print(f" Number of velocities  : {nvel}")
        print(f" Number of headings    : {nhead_tot}")
        print(f" Number of frequencies : {nfreq}")
        print("-------------------------------------------")
        print("Details:")
        hdg_str = " ".join([f"{h:.0f}" for h in headings_deg_expanded])
        print(f"Headings (deg)     : {hdg_str}")
        freq_str = " ".join([f"{fval:.2f}" for fval in freqs_array])
        print(f"Frequencies (rad/s): {freq_str}")
        vel_str = " ".join([f"{v:.2f}" for v in vels])
        print(f"Velocities (m/s)   : {vel_str}")
        print("-------------------------------------------")

    # ----------------------------------------------------------------------
    # 6) Omit plotting (disp_flag > 1) for wave drift
    # ----------------------------------------------------------------------

    return vessel

