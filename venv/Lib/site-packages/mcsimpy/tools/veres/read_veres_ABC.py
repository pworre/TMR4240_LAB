import numpy as np

import mcsimpy.tools.abc_transform as ABCtransform

# -------------------------------------------------------------------------
# 1) Transformation from Veres to Fossen axes: Tmtrx
# -------------------------------------------------------------------------
Tmtrx = np.diag([-1, 1, -1, -1, 1, -1])

# -------------------------------------------------------------------------
# 2) Placeholder for Hmtrx (shifts inertia matrix to new origin)
#    In MSS Toolbox, Hmtrx(r) is typically a 6x6 matrix that transforms
#    inertia terms for a shift in the center of gravity.
# -------------------------------------------------------------------------
def Hmtrx(r_cg):
    """
    Construct a 6x6 transformation matrix for shifting reference frame
    by r_cg = [x_cg, y_cg, z_cg].
    You must implement this to match the MSS 'Hmtrx' convention.
    """
    # Placeholder / example only:
    # In many marine systems codes, H looks like:
    #
    #    H = [[ I3,      0 ],
    #         [ S(r_cg), I3 ]]
    #
    # where S(r) is the skew-symmetric matrix of r. For a shift in CG
    # only along x, we’d have a non-zero S(r_cg).
    #
    # Adjust or replace with your actual function.
    H = np.eye(6)

    # Example 3D shift for the mass matrix:
    # top-left: 3x3 identity
    # bottom-right: 3x3 identity
    # bottom-left: skew(r_cg)
    # We define a small helper for the skew-symmetric part:
    def S(r):
        return np.array([
            [ 0.0,   -r[2],  r[1] ],
            [ r[2],   0.0,  -r[0] ],
            [-r[1],  r[0],   0.0 ]
        ])
    r = np.array(r_cg, dtype=float)
    # Fill block in H
    H[3:6, 0:3] = S(r)

    return H

# -------------------------------------------------------------------------
# 3) Placeholder for ABCtransform
#    In the MSS Toolbox, ABCtransform often adjusts A, B, and (optionally) C
#    for different reference points or coordinate transformations.
# -------------------------------------------------------------------------
# def ABCtransform(vessel):
#     """
#     Transform the A, B matrices (and possibly add friction, etc.).
#     This replicates the behavior of ABCtransform.m in MSS.

#     Must return:
#        Anew: transformed added mass
#        Bnew: transformed damping
#        Bv:   any additional viscous damping terms
#     """
#     # In your MATLAB code, you see:
#     #  [Anew, Bnew, Bv] = ABCtransform(vessel, 'veres', disp_flag);
#     # We'll do a placeholder that just returns vessel['A'], vessel['B']
#     # and vessel['roll']['Bv44'] as Bv or something similar.

#     Anew = vessel['A']
#     Bnew = vessel['B']

#     # Suppose Bv is a dict or array with the same shape as vessel['roll']['Bv44'].
#     # We'll gather it:
#     Bv = vessel['roll'].get('Bv44', None)

#     return Anew, Bnew, Bv

# -------------------------------------------------------------------------
# 4) The main function: read_veres_ABC
# -------------------------------------------------------------------------
def read_veres_ABC(filename):
    """
    read_veres_ABC(filename, disp_flag=0) -> vessel (dict)

    Reads data from a ShipX (Veres) output file *.re7. Populates a
    dictionary 'vessel' with hydrodynamic data.

    Parameters
    ----------
    filename : str
        Path to the *.re7 file.
    disp_flag : int (optional)
        1 to display run info, 0 to suppress.

    Returns
    -------
    vessel : dict
        A Python dictionary mimicking the MSS 'vessel' structure:
          vessel['main']       : main particulars (mass, CG, Lpp, B, T, etc.)
          vessel['MRB']        : 6x6 rigid-body inertia matrix
          vessel['A']          : 6x6xnfreqxnvel array (added mass)
          vessel['B']          : 6x6xnfreqxnvel array (damping)
          vessel['C']          : 6x6xnfreqxnvel array (restoring)
          vessel['roll']       : dict with roll info, e.g. 'veres', 'Bv44'
          vessel['freqs']      : array of wave frequencies
          vessel['headings']   : array of headings (in radians)
          vessel['velocities'] : array of forward speeds
          ...
    """

    vessel = {}
    vessel['main'] = {}

    try:
        fid = open(filename, 'r')
    except IOError:
        print(f"[ERROR] Could not open file: {filename}")
        return None

    # Read lines
    lines = fid.readlines()
    fid.close()

    # The first 6 lines => textual run info (header[0..5] in MATLAB)
    header_text = lines[0:6]
    # The next 5 lines => numeric run info (header[6..10] in MATLAB)
    header_num  = lines[6:11]

    # Convert these lines to strings as needed:
    # In MATLAB code:
    #   line 7 => header{7} => rhow, g
    #   line 8 => header{8} => Lpp, B, T
    #   line 9 => header{9} => LCG, VCG
    #   line 10 => header{10}
    #   line 11 => header{11} => might have 4 or 6 numbers depending on version
    #
    # We'll parse them accordingly.

    # Check version by looking at line 11
    temp_line_11 = header_num[4].split()
    # If line 11 has 6 columns => version = 1, else => version = 2
    if len(temp_line_11) == 6:
        version = 1
    else:
        version = 2

    # Parse line 7 => rhow, g
    temp_7 = header_num[0].split()  # line 7
    rhow = float(temp_7[0])
    g    = float(temp_7[1])

    # Parse line 8 => LPP, B, T_WL
    temp_8 = header_num[1].split()  # line 8
    LPP   = float(temp_8[0])
    B     = float(temp_8[1])
    T_WL  = float(temp_8[2])

    # Parse line 9 => LCG, VCG
    temp_9 = header_num[2].split()  # line 9
    LCG   = float(temp_9[0])
    VCG   = float(temp_9[1])

    # Next lines depend on version:
    # If version == 2, line 10 is "new" data we do not necessarily parse in detail
    # Then line 11 => nvel, nhead, nfreq, ndof
    if version == 2:
        # We can read line 10, but it's not stored in the original code
        # (the code calls "temp = str2num(header{10});" but doesn't store it).
        line_10 = header_num[3]  # we ignore or store if needed
        # line 11 => nvel, nhead, nfreq, ndof
        temp_11 = header_num[4].split()
        nvel   = int(temp_11[0])
        nhead  = int(temp_11[1])
        nfreq  = int(temp_11[2])
        ndof   = int(temp_11[3])

        # Next 6 lines in the file => rigid body inertia matrix MRB
        # We must read from lines[11..16]
        # Because we already consumed 0..10 for header, so the next 6 lines
        # start at index 11 in Python (where the first "mdata" line is).
        mdata_lines = lines[11:17]
        data_index_after_mdata = 17

    else:
        # version == 1
        # line 10 => nvel, nhead, nfreq, ndof
        temp_10 = header_num[3].split()  # line 10
        nvel   = int(temp_10[0])
        nhead  = int(temp_10[1])
        nfreq  = int(temp_10[2])
        ndof   = int(temp_10[3])

        # Now, line 11 => first line of MRB
        # in MATLAB code => mdata{1} = header{11}, then 5 more lines from file
        # That means the first line of MRB is header_num[4],
        # plus 5 more lines from the file = lines[11..15]
        mdata_lines = [header_num[4]] + lines[11:16]
        data_index_after_mdata = 16

    # Build MRB from those 6 lines
    # Each line has 6 floats
    mdata = []
    for md_line in mdata_lines:
        row = np.fromstring(md_line, sep=' ')
        mdata.append(row)
    MRB_raw = np.vstack(mdata)  # shape (6,6)

    # Transform MRB from Veres to Fossen axes
    MRB_ = Tmtrx @ MRB_raw @ Tmtrx  # (6,6)

    # Now shift MRB to CO midships (H matrix shift of -LCG along x)
    H = Hmtrx([-LCG, 0, 0])
    MRB = H.T @ MRB_ @ H

    # Populate vessel['MRB']
    vessel['MRB'] = MRB

    # Fill vessel['main'] fields
    vessel['main']['g']   = g
    vessel['main']['rho'] = rhow
    vessel['main']['Lpp'] = LPP
    vessel['main']['T']   = T_WL
    vessel['main']['B']   = B
    vessel['main']['CG']  = [-LCG, 0.0, VCG]

    # Mass, inertia, displacement
    m_val = MRB[0, 0]
    vessel['main']['m']     = m_val
    vessel['main']['k44']   = np.sqrt(MRB[3, 3] / m_val)
    vessel['main']['k55']   = np.sqrt(MRB[4, 4] / m_val)
    vessel['main']['k66']   = np.sqrt(MRB[5, 5] / m_val)
    vessel['main']['nabla'] = m_val / rhow

    # ---------------------------------------------------------------------
    # Next, parse the dynamic data (A, B, C, roll damping).
    # We have nvel velocities, nhead headings, nfreq frequencies.
    # Data starts at lines[data_index_after_mdata].
    # ---------------------------------------------------------------------
    current_line = data_index_after_mdata

    vels = np.zeros(nvel)
    headings_array = np.zeros(nhead)
    freqs = np.zeros(nfreq)

    # We will build placeholders for A, B, C, roll:
    # Atemp shape: 6x6
    # Then we store them in Atemp(:,:,freqno, headno, velno) => in Python
    # we might build a 5D array, but it's simpler to read them and
    # only store them in intermediate arrays, then later pick out
    # the heading=90 deg or do the re-shaping as in MATLAB code.
    Amtrx = np.zeros((6, 6, nfreq, nhead, nvel))
    Bmtrx = np.zeros((6, 6, nfreq, nhead, nvel))
    Cmtrx = np.zeros((6, 6, nfreq, nhead, nvel))
    Rollvect = np.zeros((1, 4, nfreq, nvel))  # your code uses 1,4 ?

    # Actually, in the MATLAB code:
    #   Rolltemp = str2num(fgets(fid)) => 4 numbers?
    #   Rollvect(:,:,freqno,velno) = Rolltemp
    # We'll assume it is shape (1,4,nfreq,nvel).

    for velno in range(nvel):
        # read velocity line
        line_vel = lines[current_line].split()
        current_line += 1
        vels[velno] = float(line_vel[0])  # only the first float is used

        for headno in range(nhead):
            line_head = lines[current_line].split()
            current_line += 1
            headings_array[headno] = float(line_head[0])  # single number

            for freqno in range(nfreq):
                line_freq = lines[current_line].split()
                current_line += 1
                freqs[freqno] = float(line_freq[0])

                # Atemp (6 lines, each with 6 floats)
                Atemp_list = []
                for _ in range(6):
                    row = np.fromstring(lines[current_line], sep=' ')
                    current_line += 1
                    Atemp_list.append(row)
                Atemp = np.vstack(Atemp_list)

                if version == 2:
                    # AtempADD also has 6 lines
                    for _ in range(6):
                        _ = lines[current_line]
                        current_line += 1
                    # The code stores them but does not appear to use them later
                    # in the final A. It's likely for some advanced features.

                # Btemp (6 lines)
                Btemp_list = []
                for _ in range(6):
                    row = np.fromstring(lines[current_line], sep=' ')
                    current_line += 1
                    Btemp_list.append(row)
                Btemp = np.vstack(Btemp_list)

                if version == 2:
                    # BtempADD (6 lines)
                    for _ in range(6):
                        _ = lines[current_line]
                        current_line += 1

                # Ctemp (6 lines)
                Ctemp_list = []
                for _ in range(6):
                    row = np.fromstring(lines[current_line], sep=' ')
                    current_line += 1
                    Ctemp_list.append(row)
                Ctemp = np.vstack(Ctemp_list)

                if version == 2:
                    # CtempADD (6 lines)
                    for _ in range(6):
                        _ = lines[current_line]
                        current_line += 1

                # Roll damping (1 line => 4 floats)
                roll_line = lines[current_line].split()
                current_line += 1
                roll_vals = np.array([float(x) for x in roll_line])  # shape (4,)

                # Store them in the big arrays
                Amtrx[:, :, freqno, headno, velno] = Atemp
                Bmtrx[:, :, freqno, headno, velno] = Btemp
                Cmtrx[:, :, freqno, headno, velno] = Ctemp
                # The MATLAB code: Rollvect(:,:,freqno,velno) = Rolltemp
                # We can do:
                Rollvect[0, :, freqno, velno] = roll_vals

    # The MATLAB code picks out heading = 90 deg for data such that w_o = w_e
    # Then it re-shapes A,B,C ignoring heading dimension since they are
    # effectively heading-independent.
    # headno = findstr(headings,90);
    # If none found, headno=1
    #
    # We can do something similar in Python:
    found_90 = np.where(headings_array == 90.0)[0]
    if found_90.size == 0:
        head_index = 0
    else:
        head_index = found_90[0]

    # Extract A, B, C at that heading only => shape (6,6,nfreq,nvel)
    Amtrx2 = Amtrx[:, :, :, head_index, :]
    Bmtrx2 = Bmtrx[:, :, :, head_index, :]
    Cmtrx2 = Cmtrx[:, :, :, head_index, :]

    # Transform to Fossen axes
    #   vessel.A(:,:,i,k) = H'*Tmtrx*Amtrx2(:,:,i,k)*Tmtrx*H;
    A_ = np.zeros_like(Amtrx2)
    B_ = np.zeros_like(Bmtrx2)
    C_ = np.zeros_like(Cmtrx2)

    # Bv44 structure
    roll_Bv44 = np.zeros((nfreq, nvel))

    for k in range(nvel):               # velocity index
        for i in range(nfreq):          # frequency index
            Atemp = Amtrx2[:, :, i, k]
            Btemp = Bmtrx2[:, :, i, k]
            Ctemp = Cmtrx2[:, :, i, k]

            A_[..., i, k] = H.T @ Tmtrx @ Atemp @ Tmtrx @ H
            B_[..., i, k] = H.T @ Tmtrx @ Btemp @ Tmtrx @ H
            C_[..., i, k] = H.T @ Tmtrx @ Ctemp @ Tmtrx @ H

            # include viscous roll damping => from Rollvect
            # Bv1 = Rolltemp(1,1,i,k); Bv2L = Rolltemp(1,3,i,k);
            # In Python, we have Rollvect[0, :, i, k], shape (4,)
            Bv1  = Rollvect[0, 0, i, k]  # linear damping
            Bv2L = Rollvect[0, 2, i, k]  # nonlinear damping (linearized)

            if Bv1 < 0:   # remove negative damping
                Bv1 = 0
            if Bv2L < 0:
                Bv2L = 0
            roll_Bv44[i, k] = Bv1 + Bv2L

    vessel['A'] = A_
    vessel['B'] = B_
    vessel['C'] = C_

    # Store the roll damping info
    vessel['roll'] = {}
    vessel['roll']['veres'] = Rollvect   # full 4-element data from Veres
    vessel['roll']['Bv44']  = roll_Bv44  # final linear+nonlinear damping

    # Frequencies, headings, velocities
    # The MATLAB code sets:
    #   vessel.freqs = freqs
    #   vessel.headings = [0:10:350]*pi/180
    #   vessel.velocities = vels
    #
    # But strictly, the .re7 file might define the headings differently.
    # The code sets headings to a generic 0:10:350 deg => 36 headings in radians.
    # We'll replicate the code literally:
    degs = np.arange(0, 360, 10)
    vessel['headings'] = degs * np.pi / 180.0
    vessel['freqs']    = freqs
    vessel['velocities'] = vels

    # ---------------------------------------------------------------------
    # ADDED MASS and DAMPING TRANSFORMATIONS
    # "adds A11 and B11 terms for Veres"
    # computes viscous friction Bv
    # In MATLAB: [Anew,Bnew,Bv] = ABCtransform(vessel,'veres',disp_flag);
    # ---------------------------------------------------------------------
    Anew, Bnew, Bv = ABCtransform(vessel)
    vessel['Bv'] = Bv
    vessel['A']  = Anew
    vessel['B']  = Bnew

    # ---------------------------------------------------------------------
    # If disp_flag > 0, print run info
    # ---------------------------------------------------------------------
    if False:
        print("*******************************************************************")
        print("SHIPX (VERES) DATA")
        print("*******************************************************************")
        # Lines 2..6 => header_text[1..5]
        for hline in header_text[1:6]:
            print(hline.strip())
        print("-------------------------------------------------------------------")
        print(" Run info:")
        print(" Main particulars:")
        print(f" Lpp    : {LPP:.2f} m")
        print(f" Breadth: {B:.2f} m")
        print(f" Draught: {T_WL:.2f} m")
        print("")
        print(f" Number of velocities : {nvel}")
        print(f" Number of headings   : {nhead}")
        print(f" Number of frequencies: {nfreq}")
        print("-------------------------------------------------------------------")
        # headings read from file => headings_array
        # but the code prints the ones from the file:
        head_str = " ".join([f"{h:.0f}" for h in headings_array])
        freq_str = " ".join([f"{f:.1f}" for f in freqs])
        vel_str  = " ".join([f"{v:.1f}" for v in vels])
        print(f" Headings (deg)     : {head_str}")
        print(f" Frequencies (rad/s): {freq_str}")
        print(f" Velocities (m/s)   : {vel_str}")
        print("-------------------------------------------------------------------")

    return vessel
