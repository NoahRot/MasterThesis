from tools.LoadDisplacement import LoadDisplacement
import numpy as np

"""
Load data from an Abaqus output file
Input:
 - file (str): File path + name
Output:
 - ld (LoadDisplacement): the class representing the load-displacement
"""
def abaqus_LD_reader(file : str) -> LoadDisplacement:
    # Open the file
    try:
        with open(file, "r") as f:
            lines = f.readlines()
    except:
        print(f"ERROR: Can not open file {file}")
        raise ValueError("Can not open file {file}")

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

"""
Load data from an experiment output file.
WARNING: The data are not treated, only loaded.
Input:
 - file (str): File path + name
Output:
 - ld (LoadDisplacement): the class representing the load-displacement
"""
def experiment_LD_reader(file : str) -> LoadDisplacement:
    # Open the file
    try:
        with open(file, "r") as f:
            lines = f.readlines()
    except:
        print(f"ERROR: Can not open file {file}")
        raise ValueError("Can not open file {file}")

    # Skip header lines
    data_lines = lines[1:]

    # Create numpy arrays
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