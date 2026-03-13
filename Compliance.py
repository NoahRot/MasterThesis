from tools.LoadDisplacement import *
from tools.reader import *
from tools.plt_spec import *
from tools.Specimen import *
from tools.Fracture import *
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
    end_of_curve_list = []
    A_pl_list = []

    # For each peak detect end of curve
    nbr_point = 60
    for p in peaks:
        id = p
        while id > nbr_point and not np.all(ld.disp[id-nbr_point:id] <= ld.disp[id]):
            id -= 1
        end_of_curve_list.append(id)

    # Plot
    if debug_plot:
        fig = plt.figure()
        ax = fig.subplots(1,2)
        eoc = np.array(end_of_curve_list)
        ax[0].plot(x2, ld.load)
        ax[0].plot(x2[peaks], ld.load[peaks], linestyle=" ", marker="x")
        ax[0].plot(x2[eoc], ld.load[eoc], linestyle=" ", marker="*")
        ax[0].set_xlabel("Index")
        ax[0].set_ylabel("$L$ [N]")
        ax[1].plot(ld.disp, ld.load)
        ax[1].plot(ld.disp[peaks], ld.load[peaks], linestyle=" ", marker="x")
        ax[1].plot(ld.disp[eoc], ld.load[eoc], linestyle=" ", marker="*")
        ax[1].set_xlabel("$\Delta$ [mm]")
        ax[1].set_ylabel("$L$ [N]")

    # Compute stiffness
    for i in range(len(peaks)):
        b = end_of_curve_list[i]
        e = peaks[i]
        if b == 0:
            stiffness_list.append(None)
            intercept_list.append(None)
            continue
        
        stiffness, intercept, r_value, p_value, std_err = sci.stats.linregress(ld.disp[b:e], ld.load[b:e])

        stiffness_list.append(stiffness)
        intercept_list.append(intercept)

        # Plot
        if debug_plot:
            x = np.array([ld.disp[b], ld.disp[e]])
            y = x*stiffness + intercept
            ax[1].plot(x, y, color="red", linestyle="--")

    # Compute A_pl
    for i in range(len(peaks)):
        # Compute total area
        A_pl1 = np.trapezoid(ld.load[:end_of_curve_list[i]], ld.disp[:end_of_curve_list[i]])
        # Compute additionnal area
        A_pl2 = 0.25*((stiffness_list[0]**2)*(ld.disp[0])**2) + ld.load[0]**2/stiffness_list[0]
        A_pl2 = 0
        # Compute removed area
        A_rm = 0.5/stiffness_list[i]*ld.load[end_of_curve_list[i]]**2
        # A plastic
        A_pl = A_pl1 + A_pl2 - A_rm
        print("A_pl1 = ", A_pl1)
        print("A_pl2 = ", A_pl2)
        print("A_rm = ", A_rm) 

        A_pl_list.append(A_pl)

    return peaks, np.array(stiffness_list), np.array(intercept_list), np.array(end_of_curve_list), np.array(A_pl_list)

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
sigma_YS = 700

# IDK
eta_pl = 2.0 # ??? parameters to compute the J-elastic. It is 1.9 because we use the load-displacement curve

# -------------------------------
# Path
# -------------------------------
path = "C:\\Users\\rotunn_n\\Documents\\PDM\\data\\3_points_bending"
sample = "sample5_20C.csv"

full_path = os.path.join(path, sample)
ld = experiment_LD_reader(full_path)
specimen = Specimen(W, S, B, B_N, 1.5, nu, E, eta_pl, sigma_YS)

# -------------------------------
# Slopes detection and computation
# -------------------------------
peaks, stiffness, intercept, end_of_curve, A_pl = detect_slopes(ld, debug_plot=True)
compliance = 1/stiffness 

fig3 = plt.figure()
ax3 = fig3.subplots()
ax3.plot(ld.disp[peaks], compliance, marker="+")
ax3.set_xlabel("$\Delta$ [mm]")
ax3.set_ylabel("$C$ [mm/N]")

# -------------------------------
# J-a curve NOT WORKING PROPERLY
# -------------------------------
Be = specimen.B - (specimen.B - specimen.B_N)**2/specimen.B
u = 1/(np.sqrt((Be*specimen.W*specimen.E*compliance)/(0.25*specimen.S)) + 1)
crack_length = specimen.W*(0.999748 - 3.9504*u + 2.9821*u**2 - 3.21408*u**3 + 51.51564*u**4 - 113.031*u**5)

K = stress_intensity_factor(specimen, ld.load[end_of_curve])
J_el = J_integral_el(specimen, K)
J_pl = J_integral_pl(specimen, A_pl)
J_c = J_el + J_pl 
K_Jc = np.sqrt(J_c*specimen.E_plain_strain)

fig = plt.figure()
ax = fig.subplots()
ax.plot(crack_length, J_c*1e-3)
ax.set_xlabel("$a_i$ [mm]")
ax.set_ylabel("$J$ MPa m")

# TODO Obtain the J-a curve (need to understand link between the compliance and the crack length)

plt.show()