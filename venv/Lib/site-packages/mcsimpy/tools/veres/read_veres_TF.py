import numpy as np
import os

def rad2pipi(angles):
    """
    Wrap angles (in radians) to the range [-pi, pi).
    Equivalent to MSS 'rad2pipi.m'.
    """
    # A common Pythonic approach:
    return (angles + np.pi) % (2.0 * np.pi) - np.pi


def read_veres_TF(filename, disp_flag=0):
    """
    read_veres_TF(filename, disp_flag=0) -> vessel (dict)

    Reads data from a ShipX (Veres) output file *.re1 or *.re8 (transfer
    function data for motions or forces).

    Parameters
    ----------
    filename : str
        Path to the *.re1 or *.re8 file.
    disp_flag : int, optional
        0 => no console output
        1 => display run info
        2 => (omitted plotting in this version)
        3 => (omitted further plotting in this version)

    Returns
    -------
    vessel : dict
        A dictionary with fields:
          vessel['main'] (rho, Lpp, B, T, CG, etc.)
          vessel['headings'], vessel['velocities']
          vessel['forceRAO'] or vessel['motionRAO']
             - each has .amp[dof], .phase[dof], .w
             - each is a 3D array indexed by [freq, heading, velocity]
    """

    # Initialize the vessel structure
    vessel = {}
    vessel['main'] = {}

    # Check file extension to determine if it's *.re1 or *.re8
    # We'll assume the last 3 characters are the extension.
    # (You can adapt this logic as needed for your environment.)
    ext = os.path.splitext(filename)[1].lower()  # e.g. ".re1"
    # Alternatively, in MATLAB they do: ext = filename(end-2:end)

    try:
        fid = open(filename, 'r')
    except IOError:
        print(f"[ERROR] Unable to open file: {filename}")
        return vessel

    lines = fid.readlines()
    fid.close()

    if len(lines) < 10:
        print("[ERROR] File too short or invalid.")
        return vessel

    # ----------------------------------------------------------------------
    # 1. Parse header lines
    #    The first 6 lines => textual info
    #    The next 4 lines => numeric run info
    # ----------------------------------------------------------------------
    header_text = lines[0:6]
    header_num  = lines[6:10]

    # line 7 => rho, g
    temp_7 = header_num[0].split()  # e.g. "1025 9.81"
    rho = float(temp_7[0])
    g   = float(temp_7[1])

    # line 8 => LPP, B, T
    temp_8 = header_num[1].split()  # e.g. "120.0 20.0 6.0"
    LPP = float(temp_8[0])
    B   = float(temp_8[1])
    T   = float(temp_8[2])

    # line 9 => LCG, VCG
    temp_9 = header_num[2].split()
    LCG = float(temp_9[0])
    VCG = float(temp_9[1])

    # line 10 => nvel, nhead, nfreq, ndof
    temp_10 = header_num[3].split()
    nvel   = int(temp_10[0])
    nhead  = int(temp_10[1])
    nfreq  = int(temp_10[2])
    ndof   = int(temp_10[3])

    # Store vessel main data
    vessel['main']['rho'] = rho
    vessel['main']['Lpp'] = LPP
    vessel['main']['B']   = B
    vessel['main']['T']   = T
    # CG is [-LCG, 0, VCG] in Fossen's coordinate (x-positive forward).
    vessel['main']['CG']  = [-LCG, 0.0, VCG]

    # ----------------------------------------------------------------------
    # 2. Read dynamic data lines
    #    For each velocity, we read:
    #      - one line => 5 floats => [vel, sink, trim, xmtn, zmtn]
    #      - for each heading => 1 line => heading
    #      - for each frequency => 1 line => freq, then ndof lines => dof, Real, Imag
    # ----------------------------------------------------------------------
    current_line = 10  # we've consumed 10 lines in total

    # We'll store velocity data in arrays:
    vel_vals   = np.zeros(nvel)
    sink_vals  = np.zeros(nvel)
    trim_vals  = np.zeros(nvel)
    xmtn_vals  = np.zeros(nvel)
    zmtn_vals  = np.zeros(nvel)

    headings_array = None  # we read them from the file
    freqs_array    = None

    # `resp[velno]` is a dictionary with 'amp' and 'phase' arrays.
    # We'll store them as lists of dicts (or we can do a 2D array).
    resp_list = []
    for velno in range(nvel):
        # read velocity line: [vel, sink, trim, xmtn, zmtn]
        line_vel = lines[current_line].split()
        current_line += 1

        # parse them
        vel_vals[velno]   = float(line_vel[0])
        sink_vals[velno]  = float(line_vel[1])
        trim_vals[velno]  = float(line_vel[2])
        xmtn_vals[velno]  = float(line_vel[3])
        zmtn_vals[velno]  = float(line_vel[4])

        # We'll store amplitude/phase in arrays that have shape (k_total, ndof).
        # k_total = nhead*nfreq. We'll fill them in the correct order.
        k_total = nhead * nfreq

        amp_ = np.zeros((k_total, ndof))
        phs_ = np.zeros((k_total, ndof))

        # We'll keep track of the combined index k = 0..(nhead*nfreq-1)
        k = 0

        # read headings/frequencies
        # we want to store the unique headings and frequencies eventually
        # as described in the code
        local_headings = []
        local_freqs = []

        for headno in range(nhead):
            # read heading
            line_head = lines[current_line].split()
            current_line += 1
            hdg = float(line_head[0])
            local_headings.append(hdg)

            for freqno in range(nfreq):
                # read freq
                line_freq = lines[current_line].split()
                current_line += 1
                fr = float(line_freq[0])
                if headno == 0:
                    # store freq in array only once
                    # (The code just overwrites the same array for each heading in MATLAB.)
                    pass
                local_freqs.append(fr)

                # read ndof lines => each has [DOF Real Imag]
                for dof_i in range(ndof):
                    line_dof = lines[current_line].split()
                    current_line += 1
                    dof_id = int(line_dof[0])    # 1..ndof
                    real_v = float(line_dof[1])
                    imag_v = float(line_dof[2])
                    complex_val = real_v + 1j*imag_v

                    # store amplitude and phase in amp_, phs_ at row k, column dof_id-1
                    amp_[k, dof_id - 1]  = np.abs(complex_val)
                    phs_[k, dof_id - 1]  = np.angle(complex_val)

                k += 1  # increment the combined index for each freq

        # Store results for this velocity
        resp_list.append({'amp': amp_, 'phase': phs_})

        # headings/freqs from the final iteration
        # In the original code, headings is the same for every velocity.
        if velno == 0:
            headings_array = np.array(local_headings)
            # the code sets `freqs(freqno)` in the loop, but effectively
            # one set of frequencies for all headings
            # We'll just keep the final chunk of local_freqs for each heading.
            # local_freqs is length nhead*nfreq => every group of nfreq is one heading.
            # We'll extract the unique frequencies from the first heading block:
            # local_freqs[0..(nfreq-1)]
            freqs_array = np.array(local_freqs[0:nfreq])

    # ----------------------------------------------------------------------
    # 3. Make sure headings are 0:10:180 deg for MSS Hydro
    #    The original code checks the first 19 headings to ensure they match
    #    0, 10, 20, ..., 180. If it fails, prints an error and returns.
    # ----------------------------------------------------------------------
    # For nhead=19, headings_array should be [0,10,20,...,180].
    # The code does:
    # for ii in range(1,20):
    #   if headings(ii) != 10*(ii-1):
    #       Error
    # We'll replicate that check if you want to be strict:
    if nhead == 19:
        for ii in range(1, 20):
            expected_hdg = 10.0 * (ii - 1)
            actual_hdg = headings_array[ii - 1]
            if abs(actual_hdg - expected_hdg) > 1e-6:
                print("[ERROR] MSS Hydro requires RAO data for headings 0:10:180 deg")
                return {}
    else:
        # If it's not 19, you can decide to skip or proceed.
        pass

    # ----------------------------------------------------------------------
    # 4. Build the final RAO data structures
    #    The code checks the file extension: if .re8 => forceRAO, else if .re1 => motionRAO
    #
    #    Then it constructs:
    #       vessel.forceRAO.amp{dofno}(freq, heading, vel)
    #       vessel.forceRAO.phase{dofno}(freq, heading, vel)
    #       vessel.forceRAO.w (1,freq)
    #    Similarly for motionRAO.
    #
    #    The code expands headings from 0..180 to 0..360 by symmetry.
    # ----------------------------------------------------------------------
    # The total number of headings in final is nhead_tot = (nhead - 2)*2 + 2
    # e.g. 19 => 36
    nhead_tot = (nhead - 2)*2 + 2

    # We'll store for up to 10 velocities => allocate a 3D array: [nfreq, nhead_tot, 10]
    if ext.endswith('re8'):
        # Force RAO
        vessel['forceRAO'] = {}
        # We store each DOF in separate arrays (like a list of arrays)
        # e.g. vessel.forceRAO.amp[dofno], dofno in 1..6
        # In Python, we might do 0..5 for indexing.
        vessel['forceRAO']['amp']   = [np.zeros((nfreq, nhead_tot, 10)) for _ in range(6)]
        vessel['forceRAO']['phase'] = [np.zeros((nfreq, nhead_tot, 10)) for _ in range(6)]
    else:
        # Motion RAO
        vessel['motionRAO'] = {}
        vessel['motionRAO']['amp']   = [np.zeros((nfreq, nhead_tot, 10)) for _ in range(6)]
        vessel['motionRAO']['phase'] = [np.zeros((nfreq, nhead_tot, 10)) for _ in range(6)]

    # We'll define a local function to do the “mirror headings” logic
    # and phase adjustments.
    def expand_headings_amp_phase(amp_mat, phs_mat):
        """
        Given amp_mat, phs_mat of shape (nfreq, nhead) in the order
        [freq, heading_index], create the mirrored version up to 360 deg,
        returning shape (nfreq, nhead_tot).
        """
        # Reorder headings from [0..nhead-1] in reversed order,
        # then apply sign changes to phase for dof=2,4,6 or dof=1,3,4,6?
        # The code has two separate transformations:
        # 1) phs += pi for dof=1,3,4,6
        # 2) Add data for 180-360 deg (symmetry).
        #    if dof=1,3,5 => phs[:,jj] = phs[:,ii]
        #    else => phs[:,jj] = rad2pipi(phs[:,ii] - pi)
        # We'll only do the final step here; the "phs + pi" was done earlier in MATLAB.
        # However, the code we see actually does that inside the same for-loop.
        # We'll replicate it exactly in our main loop.

        # Just return something so we can do transformations inline.
        return amp_mat.copy(), phs_mat.copy()

    # In the code, for dofno in range(ndof): loop velocities, we do:
    # "resp[velno].phase = unwrap(...)"
    # and "amp_mat = reshape(...)" => shape (nfreq, nhead)
    # then re-sequence headings in reverse order: for head in 1..nhead => amp(:,head) = amp_mat(:,nhead + 1 - head)
    # then if dof=1,3,4,6 => phs=phs+pi
    # then mirror headings for 180..360 => do the if dof=1 or 3 or 5 => no shift, else => shift - pi
    #
    # Let's replicate that carefully.

    # final headings array from 0..180 mirrored => shape (nhead_tot,):
    # headings 0..180 => we store them in ascending order from the file
    # then we mirror them for 180..360 => 2..(nhead-1)
    # so final heading array in degrees:
    headings_deg_expanded = np.zeros(nhead_tot)
    # copy the first nhead from the original:
    headings_deg_expanded[0:nhead] = headings_array

    # fill the mirrored headings
    # for ii in [2..(nhead-1)] => jj = nhead_tot+2-ii
    # but we have to be mindful of Python indexing vs MATLAB.
    # Let's do it carefully:
    # in MATLAB:
    #  for ii = 2:(nhead-1)
    #      jj = nhead_tot + 2 - ii
    #      headings(jj) = -headings(ii) + 360
    #
    # if nhead=19 => nhead_tot=36
    #  ii=2 => jj=36 => headings(36)= -headings(2)+360
    #  ii=3 => jj=35 => headings(35)= -headings(3)+360
    # ...
    # We can replicate that exactly. We'll store in zero-based arrays carefully.

    for ii in range(1, nhead-1):  # 1..(nhead-2) in 0-based => 1..17
        jj = nhead_tot + 1 - (ii + 1)  # shift from 1-based indexing
        headings_deg_expanded[jj-1] = -headings_array[ii] + 360.0

    # In practice, that yields 36 headings in ascending order?
    # Actually, it might end up reversed. This is exactly how the MATLAB code does it, though.

    # We'll store final headings in vessel['headings'] as radians
    # but let's do that after we fill the RAO arrays.

    # Prepare working arrays to hold final data
    # We'll fill them for each dof, vel in the loops.
    # The code references dofno in 1..ndof, but the final arrays are 1..6.
    # We'll treat dofno as 0..(ndof-1) in Python, but store in index dofno of the 6-element list.

    for dof_m in range(ndof):  # dof_m = 0..(ndof-1)
        dof_index = dof_m  # In MATLAB, dofno = dof_m+1
        # The code says: correct phases in surge(1), heave(3), roll(4), yaw(6).
        # Index-based => dof=0 => surge, dof=2 => heave, dof=3 => roll, dof=5 => yaw
        # We'll define a set for these DOFs:
        phase_shift_dofs = {0, 2, 3, 5}  # 1,3,4,6 in 1-based

        # The code also has a logic for the final half:
        # if dof=1 or 3 or 5 => no shift
        # else => shift by -pi
        # 1-based => dof=2,4,6 => shift
        # So for 0-based => dof=1,3,5 => shift by -pi. We'll check that logic below.

        for v_i in range(nvel):
            # 1) Unwrap
            phs_temp = np.unwrap(resp_list[v_i]['phase'][:, dof_index])
            resp_list[v_i]['phase'][:, dof_index] = phs_temp

            # 2) Reshape into (nfreq,nhead), but in the order freq major
            #    We read them in order: for head in 1..nhead => freq in 1..nfreq => total k = nhead*nfreq
            #    The code uses "reshape(resp{velno}.amp(:,dofno), nfreq, nhead)" in row-major
            #    In Python, we default to row-major. So we want the first dimension to vary fastest => freq?
            #    The code in MATLAB has freq as the first dimension => so we do reshape(..., (nfreq,nhead), order='C')
            amp_vec = resp_list[v_i]['amp'][:, dof_index]
            phs_vec = resp_list[v_i]['phase'][:, dof_index]
            amp_mat = amp_vec.reshape((nfreq, nhead), order='C')
            phs_mat = phs_vec.reshape((nfreq, nhead), order='C')

            # 3) Reverse heading order => in code:
            #    for head = 1..nhead => amp(:,head) = amp_mat(:,nhead +1 - head)
            # We'll build a new array:
            amp_reord = np.zeros_like(amp_mat)
            phs_reord = np.zeros_like(phs_mat)
            for head_i in range(nhead):
                amp_reord[:, head_i] = amp_mat[:, nhead - 1 - head_i]
                phs_reord[:, head_i] = phs_mat[:, nhead - 1 - head_i]

            # 4) If dof in [1,3,4,6] => add pi to phase => 0-based => dof_m in [0,2,3,5]
            if dof_m in phase_shift_dofs:
                phs_reord += np.pi

            # 5) Expand to 360 deg:
            #    for ii in 2..(nhead-1):
            #       jj = nhead_tot+2-ii
            #       headings(jj)= ...
            #       if dof=1 or 3 or 5 => phs(:,jj)=phs(:,ii)
            #       else => phs(:,jj)= rad2pipi(phs(:,ii)-pi)
            #
            # We'll build final arrays of shape (nfreq, nhead_tot)
            amp_final = np.zeros((nfreq, nhead_tot))
            phs_final = np.zeros((nfreq, nhead_tot))
            # first half => copy directly:
            amp_final[:, 0:nhead] = amp_reord
            phs_final[:, 0:nhead] = phs_reord

            # fill the mirror
            # "if dof=1 or 3 or 5" => 1-based => 0-based => dof_m in [0,2,4]
            # but the code says (dofno == 1 || dofno == 3 || dofno == 5)
            # => 1-based => dof=1,3,5 => 0-based => 0,2,4
            # => meaning SURGE(0), HEAVE(2), PITCH(4).
            # Actually, your snippet does two separate checks, so let's replicate exactly:
            # if dofno == 1 || dofno == 3 || dofno == 5 => no shift
            # else => shift by -pi
            # 1-based => 1,3,5 => do not shift => 2,4,6 => shift => 0-based => 1,3,5 => shift
            # It's slightly confusing because the code's dof references can differ from
            # the earlier "phase shift" set. We'll do exactly what's in the snippet:

            # snippet:
            # if dofno == 1 || dofno == 3 || dofno == 5
            #   amp(:,jj)=amp(:,ii)
            #   phs(:,jj)=phs(:,ii)
            # else
            #   amp(:,jj)=amp(:,ii)
            #   phs(:,jj)=rad2pipi(phs(:,ii)-pi)
            # end
            # => dofno in {1,3,5} => 1-based => dof_m in {0,2,4} => no shift
            # else => shift
            mirror_shift = (dof_m not in [0, 2, 4])  # True if dof_m is 1 or 3 or 5 => 1-based is 2,4,6 => confusing.
            # Actually, let's just code it directly:

            for ii in range(1, nhead-1):
                jj = nhead_tot + 1 - (ii + 1) - 1  # zero-based
                amp_final[:, jj] = amp_reord[:, ii]
                if dof_m in [0, 2, 4]:
                    # no shift
                    phs_final[:, jj] = phs_reord[:, ii]
                else:
                    # shift by -pi
                    phs_final[:, jj] = rad2pipi(phs_reord[:, ii] - np.pi)

            # Now store final (nfreq,nhead_tot) in the vessel dict
            if ext.endswith('re8'):
                # forceRAO
                vessel.setdefault('forceRAO', {})
                vessel['forceRAO']['amp'][dof_m][:, :, v_i]   = amp_final
                vessel['forceRAO']['phase'][dof_m][:, :, v_i] = phs_final
                vessel['forceRAO']['w'] = freqs_array
            else:
                # motionRAO
                vessel.setdefault('motionRAO', {})
                vessel['motionRAO']['amp'][dof_m][:, :, v_i]   = amp_final
                vessel['motionRAO']['phase'][dof_m][:, :, v_i] = phs_final
                vessel['motionRAO']['w'] = freqs_array

            # also store headings, velocities
            # We'll do this in the loop for each dof/vel, but it's the same each time.
            # So we can do it once after all loops if you prefer.
            vessel['headings']   = headings_deg_expanded * np.pi/180.0
            vessel['velocities'] = vel_vals

    # ----------------------------------------------------------------------
    # 5. Print run info if disp_flag > 0
    # ----------------------------------------------------------------------
    if disp_flag > 0:
        print("")
        print("-------------------------------------------")
        print("       VERES TRANSFER FUNCTION DATA")
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
        print(f" LCG     : {LCG:.2f} m (rel. LPP/2)")
        print(f" VCG     : {VCG:.2f} m (rel. baseline)")
        print("")
        print(f" Number of velocities  : {nvel}")
        print(f" Number of headings    : {nhead_tot}")
        print(f" Number of frequencies : {nfreq}")
        print(f" Number of DOF         : {ndof}")
        print("-------------------------------------------")
        print("Details:")
        hdg_str = " ".join([f"{h:.0f}" for h in headings_deg_expanded])
        print(f"Headings (deg)     : {hdg_str}")
        freq_str = " ".join([f"{fval:.2f}" for fval in freqs_array])
        print(f"Frequencies (rad/s): {freq_str}")
        print("")
        for m in range(nvel):
            print(f" Velocity {m+1}  : {vel_vals[m]:.2f} m/s")
            print(f"    Sinkage  : {sink_vals[m]:.2f} m")
            print(f"    Trim     : {trim_vals[m]:.2f} deg")
            print("    Motion defined in:")
            print(f"    X-Coord  : {xmtn_vals[m]:.2f} m (rel. LPP/2)")
            print("    Y-Coord  : 0.00 m (rel. centerline)")
            print(f"    Z-Coord  : {zmtn_vals[m]:.2f} m (rel. baseline)")
            print("")
        print("-------------------------------------------")

    return vessel
