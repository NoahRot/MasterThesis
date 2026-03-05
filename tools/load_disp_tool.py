"""
This file contains function that provided the needed for computations related to load-displacement data and loading data
"""

import numpy as np
import scipy as sci
import matplotlib.pyplot as plt

"""
Determine the elastic region using the R^2 method.
Input:
 - load : (np.array[float]) array containing the load data
 - disp : (np.array[float]) array containing the displacement data
 - min_points : (int) minimum number of points used to compute the first linear regression. Must be >= 2
 -  r2_threshold : (float) threshold for R^2. If R^2 > threshold, then consider that it goes from elastic to plastic.
Output:
 - (int) index at which the elastic region stop 
"""
def elastic_region_determination_r2_method(load, disp, min_points : int = 3, r2_threshold : float = 0.995, debug_plot = False) -> list[int, float, float]:
    # Variables
    r2 = 0.0
    elastic_end = 0
    r2_list = []

    # Search where the criteria is satisfied (R^2 < threshold)
    for i in range(min_points, len(disp)):
        # Evaluate slope and compute R^2
        slope, intercept, r_value, p_value, std_err = sci.stats.linregress(disp[:i], load[:i])
        r2 = r_value**2
        r2_list.append(r2)

        # Check criteria
        if r2 < r2_threshold:
            elastic_end = i - 1
            break
    
    # Print error if can not find the elastic region
    if elastic_end == 0:
        print("ERROR: Can not determine the elastic region")
        quit()

    # Compute final stiffnes and intercept
    stiffness, intercept, r_value, p_value, std_err = sci.stats.linregress(disp[:elastic_end], load[:elastic_end])

    # Debug R^2 evolution
    if debug_plot:
        fig = plt.figure()
        ax = fig.subplots()
        ax.plot(r2_list, "b+")
        ax.set_ylabel("$R^2$")

    return elastic_end, stiffness, intercept

"""
Compute the geometric function for the stress intensity factor computation
Input:
 - a0 : (float) initial crack length
 - W : (float) width of the specimen
Output:
 - (float) geometric function
"""
def geometric_fnc_K(a0 : float, W : float) -> float:
    r = a0/W
    return 3.0*np.sqrt(r)*(1.99 - r*(1.0-r)*(2.15 - 3.93*r + 2.7*r**2))/(2.0*(1.0 + 2.0*r)*(1.0-r)**1.5)

"""
Compute the stress intensity factor
Input:
 - P : (float) load
 - S : (float) span
 - B : (float) thickness
 - B_N : (float) net thickness
 - W : (float) width
 - a0 : (float) initial crack length
Output:
 - (float) stress intensity factor
"""
def stress_intensity_factor(P : float, S : float, B : float, B_N : float, W : float, a0 : float) -> float:
    f_geom = geometric_fnc_K(a0, W)
    return P*S/(np.sqrt(B*B_N) * W**1.5) * f_geom

"""
Compute the elastic J-integral
Input:
 - K : (float) stress intensity factor
 - nu : (float) Poisson ratio
 - E : (float) Young modulus
Output:
 - (float) J-integral elastic
"""
def J_integral_el(K : float, nu : float, E : float) -> float:
    return K**2 * (1.0 - nu**2) / E

"""
Compute the elastic J-integral
Input:
 - eta_pl : (float) ??? parameters to compute the J-elastic
 - B_N : (float) net thickness
 - b0 : (float) width minus initial crack length
 - index_computation : (int) Index at which the J integral will be computed
 - load : (np.array(float)) load array
 - disp : (np.array(float)) displacement array
 - stiffness : (float) stiffness (slope of the curve)
Output:
 - (float) J-integral elastic
"""
def J_integral_pl(eta_pl : float, B_N : float, b0 : float, index_computation : int, load, disp, stiffness : float) -> float:
    # Linear regression in the elastic region
    intercept_2 = -stiffness*disp[index_computation] + load[index_computation]

    # Compute area of plasticity
    total_area = np.trapz(load, disp)
    A_pl = total_area - 0.5*(disp[index_computation] + intercept_2/stiffness)*load[index_computation]

    return eta_pl*A_pl/(B_N*b0)

"""
Load the data load-displacement from an .rpt file
Input:
 - file : (str) File path and name
 - file_type : (str) Type of the file (abaqus: Abaqus output file, experiment: Experiment output file)
Output:
 - t (np.array[float]) : time
 - RF2 (np.array[float]) : load
 - U2 (np.array[float]) : displacement
"""
def load_load_disp_data(file : str, file_type : str):
    # Open the file
    try:
        with open(file, "r") as f:
            lines = f.readlines()
    except:
        print(f"ERROR: Cannot open file {file}")
        quit()

    if file_type == "abaqus":
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

    elif file_type == "experiment":
        # Skip header lines (first 4 lines)
        data_lines = lines[1:]

        # Create numpy arrays
        t   = []
        RF2 = []
        U2  = []

        # Extract data
        for i in range(0, len(data_lines)):
            data = data_lines[i].strip()
            data = data.split(';')
            print(data)
            if len(data) != 5:
                break
            t.append(float(data[0]))
            RF2.append(-float(data[2]))
            U2.append(-float(data[1]))

        t   = np.array(t)   # Time given in seconds
        RF2 = np.array(RF2) # Forces given in micro Newton
        U2  = np.array(U2)  # Displacement given in micron

    else:
        print("ERROR: Unknown load file type. Please use 'abaqus' of 'experiment'")

    return t, RF2, U2