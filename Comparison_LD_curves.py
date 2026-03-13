from tools.reader import *
from tools.CrackProfile import *
from tools.LoadDisplacement import *
from tools.plt_spec import *
from tools.Specimen import *
from tools.ElasticRegion import *
from tools.Fracture import *
from tools.MonteCarlo import *
import os

def single_T_master_curve_analysis(fractures : list[Fracture], T : float):
    K_min = 20
    Bx = 25.4
    nbr_uncencored_data = 0

    # Load data
    K_Jci = []
    K_Jc_lim = []
    for f in fractures:

        # Check limits
        if f.K_Jc < f.specimen.K_Jc_lim:
            nbr_uncencored_data += 1
            K_Jci.append(f.K_Jc*10**-1.5)
            K_Jc_lim.append(f.specimen.K_Jc_lim*10**-1.5)
        else:
            K_Jci.append(f.specimen.K_Jc_lim*10**-1.5)
            K_Jc_lim.append(f.specimen.K_Jc_lim*10**-1.5)

    # Check if it is possible to compute the master curve
    if nbr_uncencored_data == 0:
        raise ValueError("ERROR: No K_Jc in acceptable limit. Impossible to compute master curve.")

    K_Jci = np.array(K_Jci)
    K_Jc_lim = np.array(K_Jc_lim)

    # size-adjusted to 1T thickness
    K_Jc1T = K_min + (K_Jci - K_min)*(specimen_u.B/Bx)**0.25

    # Compute T0 of the master curve temperature
    K0 = (np.sum((K_Jc1T - K_min)**4/nbr_uncencored_data))**0.25 + K_min
    K_Jc_med =  K_min + 0.91*(K0 - K_min)
    T_0Q = T - (1/0.019)*np.log((K_Jc_med - 30)/70)

    # Check T0 validity
    valid_T0Q = T-T_0Q > -50 and  T-T_0Q < 50

    return K_Jc1T, K0, K_Jc_med, T_0Q, valid_T0Q, nbr_uncencored_data

def plot_master_curve(T_0, K_Jc1T, T):
    T_master_curve = np.linspace(T_0-100, T_0 + 100, 1000)
    K_Jc_master_curve = 30 + 70*np.exp(0.019*(T_master_curve - T_0))
    fig = plt.figure()
    ax = fig.subplots()
    ax.axvline(T_0, color="black", linestyle="--", label="$T_0$")
    ax.plot(T_master_curve, K_Jc_master_curve, label="Master curve")
    ax.plot(np.zeros_like(K_Jc1T)+T, K_Jc1T, label="$K_{Jc(1T)}$", linestyle=" ", marker="x")
    ax.set_xlabel("$T$ [°C]")
    ax.set_ylabel("$K_{Jc(med)}$")
    ax.legend()

    return fig, ax

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

T = -120 # Temperature of the tests [°C]

# -------------------------------
# Path to files
# -------------------------------
path = "C:\\Users\\rotunn_n\\Documents\\PDM\\data\\3_points_bending"

list_test = [1, 3, 4, 6, 7, 8, 9, 10]
test_name = ["sample", "_m120C.csv"]
crack_name = ["EU97C", "_crack_length.xlsx"]
report_name = ["report/test", ".txt"]
figure_name = ["fig/test", ".svg"]

file_names = [] # LD data, crack data, report
for i in list_test:
    file_names.append([os.path.join(path, test_name[0]) + str(i) + test_name[1],  # load-disp data
                       os.path.join(path, crack_name[0]) + str(i) + crack_name[1],      # crack data
                       report_name[0] + str(i) + report_name[1],                        # report path and name
                       figure_name[0] + str(i) + figure_name[1]                         # figure path and name
    ])

path_abaqus = "C:\\Users\\rotunn_n\\Documents\\Abaqus\\Charpy Models\\"
list_abaqus = ['01']
ld_abaqus = "load_disp.rpt"
abaqus_name = ["charpy_", "micron_L=1500micron_R=2mm"]
abaqus_crack_name = ["EU97C", "_crack_length.xlsx"]
abaqus_report_name = ["report/abaqus", ".txt"]
abaqus_figure_name = ["fig/abaqus", ".svg"]
abaqus_files = []
for i in list_abaqus:
    abaqus_files.append([os.path.join(path_abaqus, abaqus_name[0] + i + abaqus_name[1], ld_abaqus),   # load-disp data
                       1.5,                                                                         # crack data
                       abaqus_report_name[0] + i + abaqus_report_name[1],                           # report path and name
                       abaqus_figure_name[0] + i + abaqus_figure_name[1]                            # figure path and name
    ])

# -------------------------------
# Specimen
# -------------------------------
specimen_u = SpecimenDistribution(
    W = W,       # width [mm]
    W_u = 0.001,    # uncertainty in width [mm]
    S = S,      # span [mm] (4*W)
    S_u = 0.005,    # uncertainty in span [mm]
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
# Compute load-displacement
# -------------------------------
ld_list = []
legend_ld_list = []
K_Jc_list = []
K_Jc_lim_list = []
fractures = []

# -------------------------------
# Experimental
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
    #logger = Logger("txt", test[2])
    specimen = Specimen(W, S, B, B_N, crack_profile.initial_crack_length(), nu, E, eta_pl, sigma_YS)

    ld = experimental_LD_treatment(ld, 5, False)
    elastic_region = elastic_region_determination_r2_max(ld, 10, False)
    ld, elastic_region = offset_LD_according_to_stiffness(ld, elastic_region)

    fracture = Fracture(specimen, elastic_region, ld, id_computation)

    #fracture.plot_details()

    ld_list.append(ld)
    legend_ld_list.append("Test " + str(list_test[i]) + " $K_{Jc}$ = " + "{:.0f}".format(fracture.K_Jc*10**-1.5))
    K_Jc_list.append(fracture.K_Jc*10**-1.5)
    K_Jc_lim_list.append(fracture.specimen.K_Jc_lim*10**-1.5)

    fractures.append(fracture)

    i += 1

# -------------------------------
# Abaqus
# -------------------------------
print("="*60)
print(" Abaqus " + str(list_test[0]) + " analysis ")
print("="*60)

# -------------------------------
# Load data, treat and compute SIF and J-integral
# -------------------------------
print("Analysis of fracture")
ld = abaqus_LD_reader(path_abaqus + "charpy_" + list_abaqus[0] + "micron_L=1500micron_R=2mm\\load_disp.rpt")
elastic_region = elastic_region_determination_r2_method(ld, 3, 0.999, False)

fracture = Fracture(specimen, elastic_region, ld, id_computation)
#fracture.plot_details()

#ld_list.append(ld)
#legend_ld_list.append("Abaqus " + str(list_abaqus[0]) + " $K_{Jc}$ = " + "{:.0f}".format(fracture.K_Jc*10**-1.5))
#K_Jc_list.append(fracture.K_Jc*10**-1.5)

# -------------------------------
# Analysis and plot of global results
# -------------------------------
fig, ax = plot_comparison_LD(ld_list, legend_ld_list)
ax.plot(ld.disp, ld. load, color="black", linestyle = "-.", label="Abaqus $K_{Jc}$ = " + "{:.0f}".format(fracture.K_Jc*10**-1.5))
ax.legend()

K_Jc_lim = np.array(K_Jc_lim_list)
K_Jci = np.array(K_Jc_list)
K_Jc_mean = np.mean(K_Jci)
K_Jc_std = np.std(K_Jci)

bar_color = []
for i in range(len(K_Jci)):
    if K_Jci[i] < K_Jc_lim[i]:
        bar_color.append("lightgreen")
    else:
        bar_color.append("salmon")

bar_label = []
for i in list_test:
    bar_label.append("Test " + str(i))
#for i in list_abaqus:
#    bar_label.append("Abaqus " + i)
fig = plt.figure()
ax = fig.subplots()
p = ax.bar(bar_label, np.array(K_Jc_list), edgecolor = "black", color=bar_color, label="$K_{Jc}$ Tests")
p2 = ax.bar("Abaqus", fracture.K_Jc*10**-1.5, edgecolor = "black", color="skyblue", label="$K_{Jc}$ Abaqus")
ax.bar_label(p, label_type='center')
ax.bar_label(p2, label_type='center')
ax.axhline(K_Jc_mean, label="$K_{Jc}$ mean = "+"{:.0f}".format(K_Jc_mean), color="red", linestyle="--")
ax.axhline(K_Jc_mean + K_Jc_std, color="black", linestyle="-.", label="$K_{Jc}$ std = "+"{:.0f}".format(K_Jc_std))
ax.axhline(K_Jc_mean - K_Jc_std, color="black", linestyle="-.")
ax.set_ylabel("$K_{Jc}$ [MPa m$^{0.5}$]")
ax.legend()

# -------------------------------
# Single Temperature Analysis
# -------------------------------
print("="*60)
print(" Single Temperature Analysis")
print("="*60)

K_Jc1T, K0, K_Jc_med, T_0Q, valid_T0Q, nbr_uncencored_data = single_T_master_curve_analysis(fractures, T)

# Plot master curve
plot_master_curve(T_0Q, K_Jc1T, T)

if valid_T0Q:
    valid_T0Q_message = "Valid temperature"
else:
    valid_T0Q_message = "Unvalid temperature"

print(f" K0 = {K0:.3f} MPa m^0.5")
print(f" K_Jc_med = {K_Jc_med:.3f} MPa m^0.5")
print(f" T0 = {T_0Q:.3f} °C, " + valid_T0Q_message)
print(f" Number of uncensored data: {nbr_uncencored_data}")
print(f" Number of data: {len(fractures)}")




print("="*60)
print(" Analysis completed")
print("="*60)
plt.show()