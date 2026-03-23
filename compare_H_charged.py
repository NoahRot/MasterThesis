from tools.reader import *
from tools.CrackProfile import *
from tools.LoadDisplacement import *
from tools.plt_spec import *
from tools.Specimen import *
from tools.ElasticRegion import *
from tools.Fracture import *
from tools.MonteCarlo import *
from tools.MasterCurve import *
import os

init_plt(latex=False, background_fig_color="white", background_axe_color="white")
create_sns_palette("bright", "seaborn", True)

# -------------------------------
# Input parameters
# -------------------------------
# Speciemen Geometry
W = 4 # specimen width [mm]
S = 4*W # span [mm]
B = 3 # specimen thickness [mm]
B_N = B # specimen net thickness [mm]
eta_pl = 1.9 # parameters to compute the J-elastic.
sigma_YS = 700

# Speciemen Material data
nu = 0.3 # Poisson ratio
E = 210250 # Young modulus [MPa]

id_computation = -1
nbr_sample = 100000
percentile = 5
show_tolerance = True
show_errorbar = False

T = -120 # Temperature of the tests [°C]

# -------------------------------
# Path to files
# -------------------------------
path = "C:\\Users\\rotunn_n\\Documents\\PDM\\data\\3_points_bending"
#path = "data_test"

list_test = [1, 3, 4, 6, 7, 8, 9, 10, 11, 12] # Path for the non Hydrogen charged samples
list_test_H = [13, 14] # Path for the Hydrogen charged samples
test_name = ["sample", "_m120C.csv"]
crack_name = ["EU97C", "_crack_length.xlsx"]

file_names = [] # LD data, crack data, report
for i in list_test:
    file_names.append([os.path.join(path, test_name[0]) + str(i) + test_name[1],  # load-disp data
                       os.path.join(path, crack_name[0]) + str(i) + crack_name[1]      # crack data
    ])

file_names_H = [] # LD data, crack data, report
for i in list_test_H:
    file_names_H.append([os.path.join(path, test_name[0]) + str(i) + test_name[1],  # load-disp data
                       os.path.join(path, crack_name[0]) + str(i) + crack_name[1]      # crack data
    ])

# -------------------------------
# Specimen
# -------------------------------
specimen_u = SpecimenDistribution(
    W = W,       # width [mm]
    W_u = 0.001,    # uncertainty in width [mm]
    S = S,      # span [mm] (4*W)
    S_u = 0.001,    # uncertainty in span [mm]
    B = B,       # thickness [mm]
    B_u = 0.001,    # uncertainty in thickness [mm]
    B_N = B_N,     # net thickness [mm]
    B_N_u = 0.001,  # uncertainty in net thickness [mm]
    nu = nu,      # Poisson ratio [-]
    E = E,    # Young modulus [MPa]
    E_u = 10,    # uncertainty in Young modulus [MPa]
    eta_pl = eta_pl,  # eta_pl [-],
    sigma_YS = sigma_YS, # Yield strength [MPa]
    sigma_YS_u = 1, # uncertainty in Yield strength [MPa]
    crack_profile_dist = None # Initial Crack profile 
)

# -------------------------------
# Listes
# -------------------------------
fractures = []
fractures_mc = []
fractures_H = []
fractures_H_mc = []

# -------------------------------
# Experimental without H
# -------------------------------
i = 0
for test in file_names:
    print("="*60)
    print(" Test " + str(list_test[i]) + " analysis ")
    print("="*60)

    # -------------------------------
    # Load data, treat and compute SIF and J-integral
    # -------------------------------
    print("Analysis of fracture")
    ld = experiment_LD_reader(test[0])
    crack_profile = crack_profile_reader(test[1])
    specimen = Specimen(W, S, B, B_N, crack_profile.initial_crack_length(), nu, E, eta_pl, sigma_YS)

    ld = experimental_LD_treatment(ld, 5, False)
    elastic_region = elastic_region_determination_r2_max(ld, 10, False)
    ld, elastic_region = offset_LD_according_to_stiffness(ld, elastic_region)

    fracture = Fracture(specimen, elastic_region, ld, id_computation, list_test[i])

    # -------------------------------
    # Monte-Carlo uncertainties evaluations
    # -------------------------------
    rng = np.random.default_rng()
    specimen_u.crack_profile_dist = crack_profile_distribution(crack_profile, 0.001, 0.001)
    specimen_mc = specimen_u.sample(nbr_sample, rng)
    elastic_mc = elastic_region_distribution(ld, elastic_region).sample(nbr_sample, rng)
    mc = Fracture(specimen_mc, elastic_mc, ld, id_computation)

    fractures.append(fracture)
    fractures_mc.append(mc)

    i += 1

# -------------------------------
# Experimental With H
# -------------------------------
i = 0
for test in file_names_H:
    print("="*60)
    print(" Test " + str(list_test_H[i]) + " analysis ")
    print("="*60)

    # -------------------------------
    # Load data, treat and compute SIF and J-integral
    # -------------------------------
    print("Analysis of fracture")
    ld = experiment_LD_reader(test[0])
    crack_profile = crack_profile_reader(test[1])
    specimen = Specimen(W, S, B, B_N, crack_profile.initial_crack_length(), nu, E, eta_pl, sigma_YS)

    ld = experimental_LD_treatment(ld, 5, False)
    elastic_region = elastic_region_determination_r2_max(ld, 10, False)
    ld, elastic_region = offset_LD_according_to_stiffness(ld, elastic_region)

    fracture = Fracture(specimen, elastic_region, ld, id_computation, list_test_H[i])

    # -------------------------------
    # Monte-Carlo uncertainties evaluations
    # -------------------------------
    rng = np.random.default_rng()
    specimen_u.crack_profile_dist = crack_profile_distribution(crack_profile, 0.001, 0.001)
    specimen_mc = specimen_u.sample(nbr_sample, rng)
    elastic_mc = elastic_region_distribution(ld, elastic_region).sample(nbr_sample, rng)
    mc = Fracture(specimen_mc, elastic_mc, ld, id_computation)

    fractures_H.append(fracture)
    fractures_H_mc.append(mc)

    i += 1

# -------------------------------
# Analysis and plot Master Curve
# -------------------------------

mc = MasterCurve(fractures, T, percentile, fractures_mc)
mc_H = MasterCurve(fractures_H, T, percentile, fractures_H_mc)

#mc.plot_ld_curves()
#mc.plot_list_K()
#mc.plot_master_curve(True, False)

#mc_H.plot_ld_curves()
#mc_H.plot_list_K()
#mc_H.plot_master_curve(True, False)

# -------------------------------
# Comparison of the results
# -------------------------------
palette = sns.color_palette()

# Load-displacement curves
fig = plt.figure()
ax = fig.subplots()
ax.set_xlabel("$\\Delta$ [mm]")
ax.set_ylabel("$L$ [N]")
for f in mc.fractures:
    l = ax.plot(f.ld.disp, f.ld.load, color=palette[0])
for f in mc_H.fractures:
    l_H = ax.plot(f.ld.disp, f.ld.load, color=palette[1])
l[0].set_label("Without Hydrogen")
l_H[0].set_label("With Hydrogen")
ax.legend()

# List of K_Jc
fig = plt.figure()
ax = fig.subplots()
ax.set_ylabel("$K_{Jc}$ [MPa $\sqrt{\\text{m}}$]")

bar_color = []
for i in range(len(mc.K_Jci)):
    if mc.K_Jci[i] < mc.K_Jc_lim[i]:
        bar_color.append("orange")
    else:
        bar_color.append("moccasin")
bar_color_H = []
for i in range(len(mc_H.K_Jci)):
    if mc_H.K_Jci[i] < mc_H.K_Jc_lim[i]:
        bar_color_H.append("deepskyblue")
    else:
        bar_color_H.append("skyblue")

bar_label = []
for f in mc.fractures:
    if f.test_nbr is None:
        bar_label.append("Test ??")
    else:
        bar_label.append("Test " + str(f.test_nbr))
bar_label_H = []
for f in mc_H.fractures:
    if f.test_nbr is None:
        bar_label_H.append("Test ??")
    else:
        bar_label_H.append("Test " + str(f.test_nbr))

p = ax.bar(bar_label, mc.K_Jci*10**-1.5, yerr=mc.K_Jc_err*10**-1.5, edgecolor = "black", color=bar_color, capsize=5)
ax.bar_label(p, label_type='center', rotation=90)
p_H = ax.bar(bar_label_H, mc_H.K_Jci*10**-1.5, yerr=mc_H.K_Jc_err*10**-1.5, edgecolor = "black", color=bar_color_H, capsize=5)
ax.bar_label(p_H, label_type='center', rotation=90)
ax.tick_params(axis='x', rotation=90)
ax.set_ylabel("$K_{Jc}$ [MPa $\sqrt{\\text{m}}$]")

# Master curve
T_master_curve = np.linspace(min(mc.T0, mc_H.T0) - 100.0, max(mc.T0, mc_H.T0) + 100.0, 2000)

c = "orange"
c_H = "deepskyblue"

K_Jc_master_curve = master_curve(T_master_curve, mc.T0)
K_Jc_master_curve_005 = master_curve(T_master_curve, mc.T0_low)
K_Jc_master_curve_095 = master_curve(T_master_curve, mc.T0_high)

K_Jc_master_curve_H = master_curve(T_master_curve, mc_H.T0)
K_Jc_master_curve_H_005 = master_curve(T_master_curve, mc_H.T0_low)
K_Jc_master_curve_H_095 = master_curve(T_master_curve, mc_H.T0_high)

fig = plt.figure()
ax = fig.subplots()
ax.set_xlabel("$T$ [°C]")
ax.set_ylabel("$K_{Jc(med)}$")

ax.axvline(mc.T0, color=c, linestyle="-", label="$T_0$ no H")
if show_tolerance:
    ax.axvline(mc.T0_low, color=c, linestyle="--")
    ax.axvline(mc.T0_high, color=c, linestyle="--")
ax.plot(T_master_curve, K_Jc_master_curve, label="Master curve no H", color=c)
if show_tolerance:
    ax.plot(T_master_curve, K_Jc_master_curve_005, linestyle="--", color=c)
    ax.plot(T_master_curve, K_Jc_master_curve_095, linestyle="--", color=c)
#if show_errorbar:
#    ax.errorbar(np.zeros_like(mc.K_Jc1T)+mc.T, mc.K_Jc1T*10**-1.5, np.array(mc.K_Jc_err*10**-1.5), label="$K_{Jc(1T)}$", linestyle=" ", marker="x", capsize=5)
#else:
#    ax.plot(np.zeros_like(mc.K_Jc1T)+mc.T, mc.K_Jc1T*10**-1.5, label="$K_{Jc(1T)}$ no H", linestyle=" ", marker="x")

ax.axvline(mc_H.T0, color=c_H, linestyle="-", label="$T_0$ with H")
if show_tolerance:
    ax.axvline(mc_H.T0_low, color=c_H, linestyle="--")
    ax.axvline(mc_H.T0_high, color=c_H, linestyle="--")
ax.plot(T_master_curve, K_Jc_master_curve_H, label="Master curve with H", color=c_H)
if show_tolerance:
    ax.plot(T_master_curve, K_Jc_master_curve_H_005, linestyle="--", color=c_H)
    ax.plot(T_master_curve, K_Jc_master_curve_H_095, linestyle="--", color=c_H)
#if show_errorbar:
#    ax.errorbar(np.zeros_like(mc.K_Jc1T)+mc.T, mc.K_Jc1T*10**-1.5, np.array(mc.K_Jc_err*10**-1.5), label="$K_{Jc(1T)}$", linestyle=" ", marker="x", capsize=5)
#else:
#    ax.plot(np.zeros_like(mc.K_Jc1T)+mc.T, mc.K_Jc1T*10**-1.5, label="$K_{Jc(1T)}$ no H", linestyle=" ", marker="x")

ax.legend()

plt.show()