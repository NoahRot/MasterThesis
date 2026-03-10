from tools.LoadDisplacement import *
from tools.reader import *
from tools.plt_spec import *
import os
import numpy as np
import scipy as sci

"""
Detect slopes in a load-displacement curve
Input:
 - ld (LoadDisplacement): Load-displacement data
 - prominence (float): Peak prominence
 - min_points (int): minimum number of points used to compute linear regression
 - r_break (float): R^2 break point to limit number of linear regression necessary
 - debug_plot (bool): Debug plot
Output:
 - peaks (ndarray): array of the peaks index
 - stiffness_list (ndarray): array of stiffness
 - intercept_list (ndarray): array of interception with the y-axis
"""
def detect_slopes(ld : LoadDisplacement, prominence : float = 100, min_points : int = 30, r_break : float = 0.3, debug_plot : bool = False):
    # Detect peaks
    peaks, properties = sci.signal.find_peaks(-ld.load, prominence=prominence)
    x2 = np.linspace(0, len(ld.load)-1, len(ld.load))
    stiffness_list = []
    intercept_list = []

    # Plot
    if debug_plot:
        fig = plt.figure()
        ax = fig.subplots(1,2)
        ax[0].plot(x2, ld.load)
        ax[0].plot(x2[peaks], ld.load[peaks], linestyle=" ", marker="x")
        ax[0].set_xlabel("Index")
        ax[0].set_ylabel("$L$ [N]")
        ax[1].plot(ld.disp, ld.load)
        ax[1].plot(ld.disp[peaks], ld.load[peaks], linestyle=" ", marker="x")
        ax[1].set_xlabel("$\Delta$ [mm]")
        ax[1].set_ylabel("$L$ [N]")

    # Use R^2 maximum method to find the stiffness
    for p in peaks:

        # Use R^2 to find the stiffness
        r2 = 0.0
        elastic_end = 0
        r2_list = []

        # Make the list of the R^2 values
        for i in range(p + min_points, len(ld.disp)):

            # Evaluate slope and compute R^2
            slope, intercept, r_value, p_value, std_err = sci.stats.linregress(ld.disp[p:i], ld.load[p:i])
            r2 = r_value**2
            r2_list.append(r2)

            # Stop condition to avoid spending too much time
            if r2 < r_break: 
                break

        # Get the maximum of R^2
        r2_list = np.array(r2_list)
        elastic_end = np.argmax(r2_list)

        # Check if the maximum has been correctly obtained
        if elastic_end == 0:
            stiffness_list.append(None)
            intercept_list.append(None)
            continue

        # Compute the correct values for stiffness and intercept
        stiffness, intercept, r_value, p_value, std_err = sci.stats.linregress(ld.disp[p:p+elastic_end], ld.load[p:p+elastic_end])
        stiffness_list.append(stiffness)
        intercept_list.append(intercept)

        # Plot
        if debug_plot:
            x = np.array([ld.disp[p], ld.disp[p+elastic_end]])
            y = x*stiffness + intercept
            ax[1].plot(x, y, color="red", linestyle="--")

    return peaks, np.array(stiffness_list), np.array(intercept_list)

# -------------------------------
# Matplotlib specification
# -------------------------------
init_plt(latex=False)
create_sns_palette()

# -------------------------------
# Input parameters
# -------------------------------
# Speciemen parameters
W = 3 # specimen width [mm]
S = 4*W # span [mm]
B = 4 # specimen thickness [mm]
B_N = B # specimen net thickness [mm]
a0 = 1.5 # initial crack length [mm]

# Material data
nu = 0.3 # Poisson ratio
E = 210250 # Young modulus [MPa]

# IDK
eta_pl = 2.0 # ??? parameters to compute the J-elastic. It is 1.9 because we use the load-displacement curve

# -------------------------------
# Path
# -------------------------------
path = "C:\\Users\\rotunn_n\\Documents\\PDM\\data\\3_points_bending"
sample = "sample5_20C.csv"

full_path = os.path.join(path, sample)
ld = experiment_LD_reader(full_path)

fig, ax = plot_LD(ld)

stiffness_list = []
disp_list = []
load_list = []

# -------------------------------
# Stiffness computation
# -------------------------------
peaks, stiffness, intercept = detect_slopes(ld, debug_plot=True)

fig3 = plt.figure()
ax3 = fig3.subplots()
ax3.plot(ld.disp[peaks], stiffness, marker="+")
ax3.set_xlabel("$\Delta$ [mm]")
ax3.set_ylabel("$S$ [N/mm]")

plt.show()