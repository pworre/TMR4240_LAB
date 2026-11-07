import os
import h5py
import numpy as np

from mcsimpy.tools.veres import read_veres_ABC, read_veres_TF, read_veres_WD

# -------------------------------------------------------------------------
# Helper function to recursively store nested Python dicts into HDF5
# -------------------------------------------------------------------------
def save_dict_to_hdf5(py_dict, h5_group):
    """
    Recursively save a nested dictionary (with arrays, scalars, or more
    dicts) into an HDF5 group.
    """
    for key, value in py_dict.items():
        if value is None:
            # If None, skip or store an empty placeholder.
            continue

        if isinstance(value, dict):
            # Create a subgroup and recurse
            subgroup = h5_group.create_group(key)
            save_dict_to_hdf5(value, subgroup)
        elif isinstance(value, (list, tuple, np.ndarray)):
            # Convert lists/tuples to numpy array
            arr = np.array(value)
            h5_group.create_dataset(key, data=arr)
        elif isinstance(value, str):
            # Store strings as variable-length UTF-8
            dt = h5py.string_dtype(encoding='utf-8')
            h5_group.create_dataset(key, data=value, dtype=dt)
        else:
            # Assume it's a scalar (int, float, etc.)
            h5_group.create_dataset(key, data=value)

# -------------------------------------------------------------------------
# Python version of `veres2vessel.m` that saves data in HDF5 format
# -------------------------------------------------------------------------
def veres_to_vessel(filename):
    """
    veres2vessel reads data from ShipX (Veres) output files and
    returns a dictionary (vessel) with hydrodynamic data. The data is
    then saved to an HDF5 file <filename>.h5 using h5py.

    Parameters
    ----------
    filename : str
        Base name of the ShipX (Veres) files, without extension.
        Expected files are:
            filename.re1  (motion RAOs)
            filename.re2  (wave drift data)
            filename.re7  (A, B, C matrices)
            filename.re8  (force RAOs)
            filename.hyd  (hydrostatics)

    Returns
    -------
    vessel : dict
        Dictionary containing the vessel hydrodynamic/hydrostatic data.
        The contents mimic the original MATLAB structure fields,
        e.g., vessel['main'], vessel['A'], vessel['B'], vessel['C'], etc.

    Notes
    -----
    - The sub-functions (read_veres_ABC, read_veres_TF, read_veres_WD,
      plotTF, plotWD) must be implemented or replaced by you.
    - The code saves to an HDF5 file using h5py.
    """

    print("")
    print("************************* MSS Hydro *******************************")
    print("vessel = veres2vessel(*) computes the MSS vessel structure from the")
    print("ShipX (VERES) output files *.reN (N=1,2,7,8). The results are stored")
    print("in an HDF5 file.")
    print("")
    print("Author: Thor I. Fossen (original MATLAB version)")
    print("Translated to Python (example), now storing in HDF5")
    print("*******************************************************************")

    vesselfile = filename  # If you wanted user input, you could prompt here

    print(f"Reading ShipX (VERES) data files: {filename}.re1, .re2, .re7, .re8, and .hyd ...")

    vessel = read_veres_ABC(f"{filename}.re7", 0)

    # Read wave excitation forces from *.re8
    data1 = read_veres_TF(f"{filename}.re8", 0)

    # Read motion RAOs from *.re1
    data2 = read_veres_TF(f"{filename}.re1", 0)

    # Read wave drift data from *.re2
    data3 = read_veres_WD(f"{filename}.re2", 0)

    # Populate top-level fields
    vessel['main']['name']   = vesselfile
    vessel['forceRAO']       = data1.get('forceRAO', {})
    vessel['motionRAO']      = data2.get('motionRAO', {})
    vessel['driftfrc']       = data3.get('driftfrc', {})

    # ---------------------------------------------------------------------
    # Read ShipX (Veres) hydrostatic data from *.hyd
    # ---------------------------------------------------------------------
    hyd_file = f"{filename}.hyd"
    KB   = None
    LCB  = None
    C_B  = None
    GM_L = None
    GM_T = None

    if os.path.isfile(hyd_file):
        with open(hyd_file, "r") as fid:
            lines = fid.readlines()

        i = 0
        while i < len(lines):
            txt = lines[i]
            if 'KB' in txt:
                # Adjust substring indices to match your actual .hyd file format
                KB = float(txt[49:].strip())

            elif 'LCB' in txt:
                LCB = float(txt[49:].strip())
                # Advance two more lines to get C_B in the original code
                i += 1
                if i < len(lines):
                    i += 1
                    if i < len(lines):
                        C_B_line = lines[i]
                        C_B = float(C_B_line[49:].strip())

            elif 'GMl' in txt:
                GM_L = float(txt[49:].strip())
                i += 1
                if i < len(lines):
                    GM_T_line = lines[i]
                    GM_T = float(GM_T_line[49:].strip())
            i += 1

        # Store them in the vessel dict
        if GM_L is not None:
            vessel['main']['GM_L'] = GM_L
        if GM_T is not None:
            vessel['main']['GM_T'] = GM_T
        if C_B is not None:
            vessel['main']['C_B'] = C_B

        # In MATLAB: vessel.main.CB(1) = LCB - Lpp/2
        # Python: index 0 -> x, index 1 -> y, index 2 -> z
        if LCB is not None and vessel['main']['Lpp'] is not None:
            vessel['main']['CB'][0] = LCB - (vessel['main']['Lpp'] / 2.0)
        if KB is not None:
            vessel['main']['CB'][2] = KB
        vessel['main']['CB'][1] = 0.0
    else:
        print(f"[WARNING] No .hyd file found: {hyd_file}")

    # ---------------------------------------------------------------------
    # Approximations from the MATLAB code
    # ---------------------------------------------------------------------
    if vessel['main']['Lpp'] is not None:
        vessel['main']['Lwl'] = vessel['main']['Lpp']
    if (vessel['main']['B'] is not None and
        vessel['main']['T'] is not None and
        vessel['main']['Lpp'] is not None):
        vessel['main']['S'] = vessel['main']['B'] * (
            vessel['main']['Lpp'] + 2.0 * vessel['main']['T']
        )

    # ---------------------------------------------------------------------
    # Save data to <vesselfile>.h5 (HDF5)
    # ---------------------------------------------------------------------
    out_hdf_file = f"{vesselfile}.h5"
    with h5py.File(out_hdf_file, 'w') as hdf:
        # Option 1: Store everything in the root group
        # save_dict_to_hdf5(vessel, hdf)

        # Option 2: Create a group named "vessel" for clarity
        vessel_group = hdf.create_group("vessel")
        save_dict_to_hdf5(vessel, vessel_group)

    print(f"Structure <vessel> has been saved in HDF5 format to: {out_hdf_file}")
    print("-------------------------------------------------------------------")

    # Return the Python dictionary
    return vessel

# -------------------------------------------------------------------------
# Example usage (uncomment to test if your read_veres_* stubs are ready):
# -------------------------------------------------------------------------
# if __name__ == "__main__":
#     v = veres2vessel("input", plot_flag="1111")
#     print("Vessel dictionary keys:", v.keys())
