"""
This reader scripts provide functions to read data from several files and deserialized them in
data classes.

This scripts contains three functions:
- crack_profile_reader
    Read crack profile from an xlsx file and create a CrackProfile instance
- abaqus_LD_reader
    Read load-displacement data from an rpt file. It creates a LoadDispalcement instance
- experiment_LD_reader
    Read load-displacement data from an experiment csv file. It creates a LoadDispalcement instance

Author
------
ROTUNNO Noah

Date
----
2026
"""

from tools.LoadDisplacement import LoadDisplacement
from tools.CrackProfile import CrackProfile
import numpy as np
import pandas as pd

def crack_profile_reader(file : str) -> CrackProfile:
    """
    Load data of a crack profile

    Parameters
    ----------
    file : str
        File name (and path)

    Returns
    -------
    CrackProfile
        A crack profile instance
    """
    df = pd.read_excel(file)                            # Read xlsx file
    l_i = df["Unnamed: 1"][0:9].to_numpy()              # Extract distance along the width
    a_i = df["length from microscope"][0:9].to_numpy()  # Extract crack length

    return CrackProfile(l_i, a_i)

def abaqus_LD_reader(file : str) -> LoadDisplacement:
    """
    Load LD data from an Abaqus output file

    Parameters
    ----------
    file : str
        File name (and path)

    Returns
    -------
    LoadDisplacement
        A load-displacement instance

    Raises
    ------
    ValueError
        If unable to open the file

    Warnings
    --------
    The file must be a .rpt file composed of only U2 (displacement) and RF2 (reaction force),
    meaning that the file will have three columns (time, displacement and load). The reader
    assume that the data are provided in this order.
    """
    # Open the file
    try:
        with open(file, "r") as f:
            lines = f.readlines()
    except:
        print("ERROR: Can not open file " + file)
        raise ValueError("Can not open file " + file)

    # Skip header lines (first 4 lines)
    data_lines = lines[4:]

    # Create numpy arrays
    t   = []
    RF2 = []
    U2  = []

    # Extract data
    for i in range(0, len(data_lines)):
        data = data_lines[i].split()
        if len(data) != 3:
            break
        t.append(float(data[0]))
        RF2.append(-float(data[1])*4) # WARNING: The simulation only simulate one quarter of the problem. The reaction force must be multiplied by 4 !!!
        U2.append(-float(data[2]))

    t   = np.array(t)   # Time given in seconds
    RF2 = np.array(RF2) # Forces given in micro Newton
    U2  = np.array(U2)  # Displacement given in micron

    ld = LoadDisplacement(t, RF2*1e-6, U2*1e-3)
    return ld

def experiment_LD_reader(file : str) -> LoadDisplacement:
    """
    Load LD data from an experiment output file

    Parameters
    ----------
    file : str
        File name (and path)

    Returns
    -------
    LoadDisplacement
        A load-displacement instance

    Raises
    ------
    ValueError
        If unable to open the file

    Warnings
    --------
    The file must be a .csv file, the the .dat file.
    """
    # Open the file
    try:
        with open(file, "r") as f:
            lines = f.readlines()
    except:
        print(f"ERROR: Can not open file " + file)
        raise ValueError("Can not open file " + file)

    # Skip header lines
    data_lines = lines[1:]

    # Create lists for data extraction
    t   = []
    RF2 = []
    U2  = []

    # Extract data
    for i in range(0, len(data_lines)):
        data = data_lines[i].strip()
        data = data.split(';')
        if len(data) != 5:
            break
        t.append(float(data[0]))
        RF2.append(-float(data[2]))
        U2.append(-float(data[1]))

    t   = np.array(t)   # Time given in seconds
    RF2 = np.array(RF2) # Forces given in kilo Newton
    U2  = np.array(U2)  # Displacement given in mm

    ld = LoadDisplacement(t, RF2*1000, U2)

    return ld